"""Tickers deliberately removed from ingestion and research workflows.

Entries identify regional tickers intentionally outside the current S&P 500-only
research scope. Historical S&P 500 tickers must not be placed on this list.
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

    European symbols remain excluded from this S&P 500-focused research universe.
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

    Applies the regional-symbol blocklist. The point-in-time S&P 500 list is
    independently enforced by the membership helpers.
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
