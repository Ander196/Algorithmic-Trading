#!/usr/bin/env python
"""Incrementally ingest the market index and active stock universe."""
from __future__ import annotations

import logging
import os
import sys
import time

from data.data_loader import getSupabaseClient, getActiveTickers, processTicker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    client = getSupabaseClient()
    market_ticker = os.getenv("MARKET_TICKER", "SPY").upper()
    tickers = [t.upper() for t in getActiveTickers(client) if t.upper() != market_ticker]

    universe = [market_ticker, *tickers]
    logger.info("Ingesting %d tickers including market=%s", len(universe), market_ticker)

    errors: list[str] = []
    inserted = 0
    for ticker in universe:
        result = processTicker(client, ticker)
        inserted += int(result["inserted"])
        if result["error"]:
            errors.append(f"{ticker}: {result['error']}")
        time.sleep(0.2)

    logger.info("Ingestion complete: inserted=%d errors=%d", inserted, len(errors))
    if errors:
        for error in errors:
            logger.error(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
