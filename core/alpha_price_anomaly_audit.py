"""Compare stored and provider adjusted price histories for Alpha data QA.

The comparison is diagnostic only. Large moves and source disagreements are
flagged for review; they are never treated as proof of corruption or repaired
automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FIELDS = ("open", "high", "low", "close", "adj_close", "volume")
PRICE_VALUE_FIELDS = ("open", "high", "low", "close", "adj_close")
DEFAULT_LARGE_RETURN_THRESHOLD = 0.50
DEFAULT_SCALE_DEVIATION_THRESHOLD = 0.02


def _prepare_history(history: pd.DataFrame, source: str) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame):
        raise ValueError(f"{source} history must be a pandas DataFrame")
    missing = set(PRICE_FIELDS) - set(history.columns)
    if missing:
        raise ValueError(f"{source} history is missing columns: {sorted(missing)}")
    frame = history.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if index.tz is not None:
        index = index.tz_convert(None)
    frame.index = index.normalize()
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{source} history has duplicate session dates")
    for column in PRICE_FIELDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def compare_adjusted_price_histories(
    stored_history: pd.DataFrame,
    reference_history: pd.DataFrame,
    *,
    large_return_threshold: float = DEFAULT_LARGE_RETURN_THRESHOLD,
    scale_deviation_threshold: float = DEFAULT_SCALE_DEVIATION_THRESHOLD,
    ticker: str = "",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a date-level source comparison and a compact audit summary.

    One median stored/reference adjusted-close ratio is used to diagnose
    whether a constant scale difference could explain the two price bases.
    It is not a repair factor and must never be applied automatically.
    """
    if not np.isfinite(large_return_threshold) or large_return_threshold <= 0:
        raise ValueError("large_return_threshold must be finite and > 0")
    if not np.isfinite(scale_deviation_threshold) or scale_deviation_threshold < 0:
        raise ValueError("scale_deviation_threshold must be finite and >= 0")

    stored = _prepare_history(stored_history, "stored")
    reference = _prepare_history(reference_history, "reference")
    calendar = stored.index.union(reference.index).sort_values()

    stored_values = stored.reindex(calendar)
    reference_values = reference.reindex(calendar)
    stored_adj = stored_values["adj_close"]
    reference_adj = reference_values["adj_close"]
    overlap = (
        stored_adj.notna()
        & reference_adj.notna()
        & np.isfinite(stored_adj)
        & np.isfinite(reference_adj)
        & (stored_adj > 0)
        & (reference_adj > 0)
    )
    ratios = (stored_adj.loc[overlap] / reference_adj.loc[overlap]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    scale_factor = float(ratios.median()) if not ratios.empty else float("nan")
    relative_deviation = (
        ratios / scale_factor - 1.0
        if np.isfinite(scale_factor) and scale_factor > 0
        else pd.Series(dtype=float)
    )
    p95_dispersion = (
        float(relative_deviation.abs().quantile(0.95))
        if not relative_deviation.empty
        else float("nan")
    )

    stored_returns = stored_adj.pct_change(fill_method=None)
    reference_returns = reference_adj.pct_change(fill_method=None)
    stored_returns = stored_returns.reindex(calendar)
    reference_returns = reference_returns.reindex(calendar)

    stored_volume = stored_values["volume"]
    reference_volume = reference_values["volume"]
    dividends = (
        pd.to_numeric(reference["dividends"], errors="coerce").reindex(calendar).fillna(0.0)
        if "dividends" in reference.columns else pd.Series(0.0, index=calendar)
    )
    splits = (
        pd.to_numeric(reference["stock_splits"], errors="coerce").reindex(calendar).fillna(0.0)
        if "stock_splits" in reference.columns else pd.Series(0.0, index=calendar)
    )

    rows: list[dict[str, object]] = []
    for date in calendar:
        stored_present = date in stored.index
        reference_present = date in reference.index
        stored_value = stored_adj.loc[date]
        reference_value = reference_adj.loc[date]
        ratio = (
            float(stored_value / reference_value)
            if pd.notna(stored_value) and pd.notna(reference_value)
            and stored_value > 0 and reference_value > 0
            else float("nan")
        )
        deviation = (
            float(ratio / scale_factor - 1.0)
            if np.isfinite(ratio) and np.isfinite(scale_factor) and scale_factor > 0
            else float("nan")
        )
        flags: list[str] = []
        if stored_present and not reference_present:
            flags.append("missing_reference_session")
        elif reference_present and not stored_present:
            flags.append("missing_stored_session")
        if stored_present and pd.isna(stored_value):
            flags.append("invalid_stored_adjusted_close")
        if reference_present and pd.isna(reference_value):
            flags.append("invalid_reference_adjusted_close")
        if np.isfinite(deviation) and abs(deviation) > scale_deviation_threshold:
            flags.append("adjusted_close_basis_deviation")

        stored_return = stored_returns.loc[date]
        reference_return = reference_returns.loc[date]
        if pd.notna(stored_return) and abs(float(stored_return)) >= large_return_threshold:
            flags.append("large_stored_observation_return")
        if pd.notna(reference_return) and abs(float(reference_return)) >= large_return_threshold:
            flags.append("large_reference_observation_return")
        if pd.notna(stored_volume.loc[date]) and float(stored_volume.loc[date]) == 0:
            flags.append("zero_stored_volume")
        if pd.notna(reference_volume.loc[date]) and float(reference_volume.loc[date]) == 0:
            flags.append("zero_reference_volume")
        if float(splits.loc[date]) != 0.0:
            flags.append("reference_reports_stock_split")
        if float(dividends.loc[date]) != 0.0:
            flags.append("reference_reports_dividend")

        row: dict[str, object] = {
            "ticker": ticker.strip().upper(),
            "date": date,
            "stored_adj_close": stored_value,
            "reference_adj_close": reference_value,
            "stored_to_reference_adj_close_ratio": ratio,
            "calibrated_scale_factor": scale_factor,
            "relative_deviation_from_scale": deviation,
            "stored_return_since_previous_observation": stored_return,
            "reference_return_since_previous_observation": reference_return,
            "stored_volume": stored_volume.loc[date],
            "reference_volume": reference_volume.loc[date],
            "reference_dividends": dividends.loc[date],
            "reference_stock_splits": splits.loc[date],
            "flags": ";".join(flags),
        }
        for field in PRICE_VALUE_FIELDS:
            stored_field = stored_values[field].loc[date]
            reference_field = reference_values[field].loc[date]
            row[f"stored_{field}"] = stored_field
            row[f"reference_{field}"] = reference_field
        rows.append(row)

    comparison = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "ticker": ticker.strip().upper(),
        "stored_rows": int(len(stored)),
        "reference_rows": int(len(reference)),
        "overlap_rows": int(overlap.sum()),
        "calibrated_scale_factor": scale_factor,
        "p95_relative_scale_dispersion": p95_dispersion,
        "max_abs_stored_observation_return": (
            float(stored_returns.abs().max()) if stored_returns.notna().any() else float("nan")
        ),
        "max_abs_reference_observation_return": (
            float(reference_returns.abs().max()) if reference_returns.notna().any() else float("nan")
        ),
        "flagged_sessions": int(comparison["flags"].ne("").sum()),
        "start_date": calendar.min().date().isoformat() if len(calendar) else None,
        "end_date": calendar.max().date().isoformat() if len(calendar) else None,
    }
    return comparison, summary


__all__ = [
    "DEFAULT_LARGE_RETURN_THRESHOLD",
    "DEFAULT_SCALE_DEVIATION_THRESHOLD",
    "compare_adjusted_price_histories",
]
