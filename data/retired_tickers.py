"""Curated Yahoo Finance symbols retained for historical-only research.

Each entry records the last date on which the original public equity traded.
Yahoo queries are bounded to that date to prevent symbol reuse or later OTC
quotes from being mistaken for the original listed equity.
"""

RETIRED_TICKERS: dict[str, dict[str, str]] = {
    "TWTR": {
        "name": "Twitter, Inc.",
        "exchange": "NMS",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "last_trade_date": "2022-10-27",
    },
    "ATVI": {
        "name": "Activision Blizzard, Inc.",
        "exchange": "NMS",
        "sector": "Communication Services",
        "industry": "Electronic Gaming & Multimedia",
        "last_trade_date": "2023-10-12",
    },
    "SPLK": {
        "name": "Splunk Inc.",
        "exchange": "NMS",
        "sector": "Technology",
        "industry": "Software—Infrastructure",
        "last_trade_date": "2024-03-15",
    },
    "CTXS": {
        "name": "Citrix Systems, Inc.",
        "exchange": "NMS",
        "sector": "Technology",
        "industry": "Software—Application",
        "last_trade_date": "2022-09-29",
    },
    "CELG": {
        "name": "Celgene Corporation",
        "exchange": "NMS",
        "sector": "Healthcare",
        "industry": "Biotechnology",
        "last_trade_date": "2019-11-20",
    },
    "TIF": {
        "name": "Tiffany & Co.",
        "exchange": "NYQ",
        "sector": "Consumer Cyclical",
        "industry": "Luxury Goods",
        "last_trade_date": "2021-01-06",
    },
    "APC": {
        "name": "Anadarko Petroleum Corporation",
        "exchange": "NYQ",
        "sector": "Energy",
        "industry": "Oil & Gas E&P",
        "last_trade_date": "2019-08-08",
    },
    "YHOO": {
        "name": "Yahoo! Inc.",
        "exchange": "NMS",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "last_trade_date": "2017-06-16",
    },
    "SIVB": {
        "name": "SVB Financial Group",
        "exchange": "NMS",
        "sector": "Financial Services",
        "industry": "Banks—Regional",
        "last_trade_date": "2023-03-09",
    },
    "SBNY": {
        "name": "Signature Bank",
        "exchange": "NYS",
        "sector": "Financial Services",
        "industry": "Banks—Regional",
        "last_trade_date": "2023-03-10",
    },
    "FRC": {
        "name": "First Republic Bank",
        "exchange": "NYS",
        "sector": "Financial Services",
        "industry": "Banks—Regional",
        "last_trade_date": "2023-04-28",
    },
}
