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
        "WBD",
        # Retired after merger; preserve history but exclude from active data/research.
        "QRVO",
    }
)


def normalize_ticker(ticker: object) -> str:
    """Normalize provider/database ticker text before comparisons."""
    return str(ticker).strip().upper()


def is_excluded_ticker(ticker: object) -> bool:
    """Return whether a ticker is explicitly excluded from market data."""
    return normalize_ticker(ticker) in EXCLUDED_TICKERS



def filter_excluded_ticker_rows(rows, ticker_column: str = "ticker"):
    """Return a normalized copy without explicitly excluded ticker rows.

    Shared by CSV, audit and portfolio entry points so they honour the same
    blocklist as ingestion and the Alpha universe.
    """
    if ticker_column not in rows.columns:
        raise ValueError(f"missing ticker column: {ticker_column}")
    filtered = rows.copy()
    normalized = filtered[ticker_column].map(normalize_ticker)
    excluded_mask = normalized.isin(EXCLUDED_TICKERS)
    excluded = tuple(sorted(normalized.loc[excluded_mask].dropna().unique().tolist()))
    filtered[ticker_column] = normalized
    return filtered.loc[~excluded_mask].copy(), excluded
