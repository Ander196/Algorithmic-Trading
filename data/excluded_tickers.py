"""Operational ticker exclusions and Alpha-only data-quality quarantine.

EXCLUDED_TICKERS controls operational ingestion/selection. Alpha research has a
separate quarantine for historical price series confirmed to be unreliable.
Historical S&P 500 symbols must not be added to the operational blocklist just
because their preserved research history needs investigation.
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


# Historical series whose ticker identity/price quality is unreliable for
# Alpha research:
# - CPWR was reused: historical S&P 500 Compuware ceased trading in 2014, and a
#   different issuer adopted CPWR in 2017. Its audited 2006-2011 history also
#   contains extreme observation-to-observation jumps and long zero-volume runs
#   reproduced by Yahoo; source agreement does not certify correctness.
# - PARA has inconsistent price scales and volume observations in stored data.
# - EP and GR have extensive zero-volume sessions reported by both Supabase and
#   Yahoo, plus numerous >50% observation-to-observation adjusted-price moves.
#   Their provider agreement is not independent validation of the historical
#   OHLCV series; keep them out pending independent reconstruction.
# Keep this separate from the operational blocklist: it must not affect
# ingestion, daily allocation, or any live trading path.
ALPHA_DATA_QUALITY_EXCLUDED_TICKERS = frozenset({"CPWR", "EP", "GR", "PARA"})


HISTORICAL_ALPHA_INCLUDED_TICKERS = frozenset()


def normalize_ticker(ticker: object) -> str:
    """Normalize provider/database ticker text before comparisons."""
    return str(ticker).strip().upper()


def is_excluded_ticker(ticker: object) -> bool:
    """Return whether a ticker is excluded from live ingestion and selection."""
    return normalize_ticker(ticker) in EXCLUDED_TICKERS


def is_excluded_from_alpha_history(ticker: object) -> bool:
    """Return whether a ticker must be omitted from historical Alpha research.

    European symbols outside the research scope and Alpha-only data-quality
    quarantines are excluded. This does not change the operational blocklist.
    """
    normalized = normalize_ticker(ticker)
    alpha_exclusions = EXCLUDED_TICKERS | ALPHA_DATA_QUALITY_EXCLUDED_TICKERS
    return (
        normalized in alpha_exclusions
        and normalized not in HISTORICAL_ALPHA_INCLUDED_TICKERS
    )


def filter_excluded_ticker_rows(
    rows,
    ticker_column: str = "ticker",
    *,
    include_historical_retired: bool = False,
    include_alpha_quality_exclusions: bool = False,
):
    """Return normalized rows without excluded tickers.

    Applies the operational blocklist. Set include_alpha_quality_exclusions
    when preparing Alpha research inputs or downstream Alpha artifacts. The
    point-in-time S&P 500 list is independently enforced by membership helpers.
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
    if include_alpha_quality_exclusions:
        excluded_symbols = excluded_symbols | ALPHA_DATA_QUALITY_EXCLUDED_TICKERS
    excluded_mask = normalized.isin(excluded_symbols)
    excluded = tuple(sorted(normalized.loc[excluded_mask].dropna().unique().tolist()))
    filtered[ticker_column] = normalized
    return filtered.loc[~excluded_mask].copy(), excluded
