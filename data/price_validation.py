"""Audit and validate daily OHLCV observations.

The audit keeps row-level reason codes so ingestion and research can report the
same data-quality rules without confusing invalid rows with provider failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class OHLCVAudit:
    """Result of auditing a frame; reason counts are non-mutually-exclusive."""

    valid_rows: pd.DataFrame
    invalid_rows: pd.DataFrame
    reason_counts: dict[str, int]
    missing_columns: tuple[str, ...] = ()

    @property
    def invalid_row_count(self) -> int:
        return len(self.invalid_rows)


def audit_ohlcv_frame(data: pd.DataFrame | None, ticker: str = "") -> OHLCVAudit:
    """Classify invalid OHLCV rows without mutating the source frame.

    Reason counts count row/reason pairs, not unique rows; one row can have
    multiple reasons. The invalid_rows frame includes a validation_reasons list.
    """
    if data is None:
        empty = pd.DataFrame()
        return OHLCVAudit(empty, empty, {}, ())

    frame = data.copy()
    if frame.empty:
        return OHLCVAudit(frame.copy(), frame.copy(), {}, ())

    missing = tuple(sorted(set(REQUIRED_OHLCV_COLUMNS) - set(frame.columns)))
    if missing:
        return OHLCVAudit(
            valid_rows=frame.iloc[0:0].copy(),
            invalid_rows=frame.iloc[0:0].copy(),
            reason_counts={"missing_required_columns": 1},
            missing_columns=missing,
        )

    row_reasons: list[list[str]] = [[] for _ in range(len(frame))]

    def add_reason(mask: pd.Series | np.ndarray, reason: str) -> None:
        values = np.asarray(mask, dtype=bool)
        for position in np.flatnonzero(values):
            row_reasons[int(position)].append(reason)

    numeric: dict[str, pd.Series] = {}
    for column in REQUIRED_OHLCV_COLUMNS:
        raw = frame[column]
        values = pd.to_numeric(raw, errors="coerce")
        numeric[column] = values
        missing_values = raw.isna().to_numpy()
        non_numeric = (raw.notna() & values.isna()).to_numpy()
        non_finite = (values.notna() & ~np.isfinite(values)).to_numpy()
        add_reason(missing_values, "missing_ohlc_values" if column in PRICE_COLUMNS else "missing_volume")
        add_reason(non_numeric, "non_numeric_ohlc_values" if column in PRICE_COLUMNS else "non_numeric_volume")
        add_reason(non_finite, "non_finite_ohlc_values" if column in PRICE_COLUMNS else "non_finite_volume")

    prices = pd.DataFrame({column: numeric[column] for column in PRICE_COLUMNS}, index=frame.index)
    price_matrix = prices.to_numpy(dtype=float)
    finite_prices = np.isfinite(price_matrix).all(axis=1)
    positive_prices = (price_matrix > 0).all(axis=1)
    add_reason(
        np.isfinite(price_matrix).all(axis=1) & ~(price_matrix > 0).all(axis=1),
        "non_positive_ohlc_values",
    )

    volume = numeric["volume"].to_numpy(dtype=float)
    add_reason(np.isfinite(volume) & (volume < 0), "negative_volume")

    if "adj_close" in frame.columns:
        raw_adj = frame["adj_close"]
        adj = pd.to_numeric(raw_adj, errors="coerce")
        add_reason((raw_adj.notna() & adj.isna()).to_numpy(), "non_numeric_adj_close")
        adj_values = adj.to_numpy(dtype=float)
        add_reason(
            np.isfinite(adj_values) & (adj_values <= 0),
            "non_positive_adj_close",
        )
        add_reason(
            adj.notna().to_numpy() & ~np.isfinite(adj_values),
            "non_finite_adj_close",
        )

    # Check OHLC geometry only when all four price fields are finite and positive.
    geometry_eligible = finite_prices & positive_prices
    if geometry_eligible.any():
        o = numeric["open"].to_numpy(dtype=float)
        h = numeric["high"].to_numpy(dtype=float)
        low = numeric["low"].to_numpy(dtype=float)
        close = numeric["close"].to_numpy(dtype=float)
        add_reason(
            geometry_eligible & (h < np.maximum.reduce([o, low, close])),
            "high_below_ohlc_max",
        )
        add_reason(
            geometry_eligible & (low > np.minimum.reduce([o, h, close])),
            "low_above_ohlc_min",
        )

    invalid_mask = np.asarray([bool(reasons) for reasons in row_reasons], dtype=bool)
    invalid = frame.iloc[np.flatnonzero(invalid_mask)].copy()
    invalid["validation_reasons"] = [
        row_reasons[position] for position in np.flatnonzero(invalid_mask)
    ]
    valid = frame.iloc[np.flatnonzero(~invalid_mask)].copy()

    counts: dict[str, int] = {}
    for reasons in row_reasons:
        for reason in set(reasons):
            counts[reason] = counts.get(reason, 0) + 1

    return OHLCVAudit(
        valid_rows=valid,
        invalid_rows=invalid,
        reason_counts=dict(sorted(counts.items())),
    )


def validate_ohlcv_frame(
    data: pd.DataFrame | None,
    ticker: str,
    *,
    context: str = "ingestion",
) -> pd.DataFrame | None:
    """Return only valid OHLCV rows and log actionable diagnostic reasons.

    Required columns are never fabricated. Prices must be finite and positive,
    volume finite and non-negative, adjusted close positive when supplied, and
    OHLC ranges internally consistent. The context labels ingestion versus
    Supabase reads so logs do not imply that evaluation is uploading data.
    """
    audit = audit_ohlcv_frame(data, ticker)

    if audit.missing_columns:
        logger.warning(
            "%s | %s: rejecting price frame; missing required columns: %s",
            context,
            ticker,
            ", ".join(audit.missing_columns),
        )
        return None

    if audit.invalid_row_count:
        example_rows = audit.invalid_rows.head(3)
        if "price_date" in example_rows.columns:
            examples = [
                {
                    "date": str(row["price_date"]),
                    "reasons": row["validation_reasons"],
                }
                for _, row in example_rows.iterrows()
            ]
        else:
            examples = [
                {"index": str(index), "reasons": row["validation_reasons"]}
                for index, row in example_rows.iterrows()
            ]
        logger.warning(
            "%s | %s: excluded %d invalid OHLCV row(s); "
            "reason_counts=%s; examples=%s",
            context,
            ticker,
            audit.invalid_row_count,
            audit.reason_counts,
            examples,
        )

    if audit.valid_rows.empty:
        if data is not None and not data.empty:
            logger.warning(
                "%s | %s: no valid OHLCV rows remain; no rows will be used",
                context,
                ticker,
            )
        return None

    return audit.valid_rows


__all__ = [
    "OHLCVAudit",
    "audit_ohlcv_frame",
    "validate_ohlcv_frame",
    "REQUIRED_OHLCV_COLUMNS",
    "PRICE_COLUMNS",
]
