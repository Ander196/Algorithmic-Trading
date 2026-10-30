"""Tickers deliberately removed from ingestion and research workflows.

Entries include known invalid price histories and symbols explicitly retired
from this research universe. Keep this list aligned with the Supabase cleanup.
"""

EXCLUDED_TICKERS = frozenset(
    {
        "ABI.BR",
        "ACA.PA",
        "ITX.MC",
        "UCG.MI",
        "AZN.L",
        "BATS.L",
        "BP.L",
        "BT-A.L",
        "GSK.L",
        "HSBA.L",
        "LSEG.L",
        "NG.L",
        "PRU.L",
        "REL.L",
        "RIO.L",
        "SHEL.L",
        "ULVR.L",
        "VOD.L",
        # Explicitly retired: no usable stored history; do not re-add it.
        # Retired after merger; exclude from live data, but keep its stored
        # history eligible for historical point-in-time Alpha research.
    }
)


HISTORICAL_ALPHA_INCLUDED_TICKERS = frozenset()


def normalize_ticker(ticker: object) -> str:
    """Normalize provider/database ticker text before comparisons."""
    return str(ticker).strip().upper()


def is_excluded_ticker(ticker: object) -> bool:
    """Return whether a ticker is excluded from live ingestion and selection."""
    return normalize_ticker(ticker) in EXCLUDED_TICKERS


def is_excluded_from_alpha_history(ticker: object) -> bool:
    """Return whether a ticker must be omitted from historical Alpha research.

    A retired symbol with preserved, usable prices can remain eligible in its
    historical period even though it is excluded from the live universe.
    """
    normalized = normalize_ticker(ticker)
    return (
        normalized in EXCLUDED_TICKERS
        and normalized not in HISTORICAL_ALPHA_INCLUDED_TICKERS
    )


def filter_excluded_ticker_rows(
    rows,
    ticker_column: str = "ticker",
    *,
    include_historical_retired: bool = False,
):
    """Return normalized rows without excluded tickers.

    By default, applies the full operational blocklist. Historical research
    callers may opt into symbols whose preserved history remains usable, such
    as QRVO, while still excluding symbols with known invalid price histories.
    """
    if ticker_column not in rows.columns:
        raise ValueError(f"missing ticker column: {ticker_column}")
    filtered = rows.copy()
    normalized = filtered[ticker_column].map(normalize_ticker)
    excluded_symbols = (
        EXCLUDED_TICKERS - HISTORICAL_ALPHA_INCLUDED_TICKERS
        if include_historical_retired
        else EXCLUDED_TICKERS
    )
    excluded_mask = normalized.isin(excluded_symbols)
    excluded = tuple(sorted(normalized.loc[excluded_mask].dropna().unique().tolist()))
    filtered[ticker_column] = normalized
    return filtered.loc[~excluded_mask].copy(), excluded
