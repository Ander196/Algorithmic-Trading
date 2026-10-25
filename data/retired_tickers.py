"""Curated Yahoo Finance symbols retained for historical-only research.

These tickers are no longer part of the live universe. The catalog loader checks
that Yahoo Finance currently returns non-empty history before registering each
symbol. Historical ingestion must still respect the availability and quality of
the upstream data.
"""

RETIRED_TICKERS: dict[str, dict[str, str]] = {
    "TWTR": {
        "name": "Twitter, Inc.",
        "exchange": "NMS",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
    },
    "ATVI": {
        "name": "Activision Blizzard, Inc.",
        "exchange": "NMS",
        "sector": "Communication Services",
        "industry": "Electronic Gaming & Multimedia",
    },
    "SPLK": {
        "name": "Splunk Inc.",
        "exchange": "NMS",
        "sector": "Technology",
        "industry": "Software—Infrastructure",
    },
    "CTXS": {
        "name": "Citrix Systems, Inc.",
        "exchange": "NMS",
        "sector": "Technology",
        "industry": "Software—Application",
    },
    "CELG": {
        "name": "Celgene Corporation",
        "exchange": "NMS",
        "sector": "Healthcare",
        "industry": "Biotechnology",
    },
    "TIF": {
        "name": "Tiffany & Co.",
        "exchange": "NYQ",
        "sector": "Consumer Cyclical",
        "industry": "Luxury Goods",
    },
    "APC": {
        "name": "Anadarko Petroleum Corporation",
        "exchange": "NYQ",
        "sector": "Energy",
        "industry": "Oil & Gas E&P",
    },
    "YHOO": {
        "name": "Yahoo! Inc.",
        "exchange": "NMS",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
    },
    "SIVB": {
        "name": "SVB Financial Group",
        "exchange": "NMS",
        "sector": "Financial Services",
        "industry": "Banks—Regional",
    },
    "SBNY": {
        "name": "Signature Bank",
        "exchange": "NYS",
        "sector": "Financial Services",
        "industry": "Banks—Regional",
    },
    "FRC": {
        "name": "First Republic Bank",
        "exchange": "NYS",
        "sector": "Financial Services",
        "industry": "Banks—Regional",
    },
}
