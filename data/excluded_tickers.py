"""Tickers excluded because their stored OHLC history is known to be invalid.

Keep this list aligned with the controlled Supabase cleanup. Exclusion is applied
at ingestion and research-universe boundaries so future loaders do not restore
these rows or reactivate the symbols.
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


def normalize_ticker(ticker: object) -> str:
    """Normalize provider/database ticker text before comparisons."""
    return str(ticker).strip().upper()


def is_excluded_ticker(ticker: object) -> bool:
    """Return whether a ticker is explicitly excluded from market data."""
    return normalize_ticker(ticker) in EXCLUDED_TICKERS
