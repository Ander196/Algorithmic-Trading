#!/usr/bin/env python
"""
Daily update script for HMM regime detection.

This script is designed to be run on a schedule (cron, GitHub Actions, etc.)
to update regime signals for all registered tickers.

Usage:
    python -m hmm_regime.daily_update

Environment variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase service role or anon key
    MARKET_TICKER: Market index to use (default: SPY)
    MODEL_PATH: Path to saved model JSON (optional, will train if not provided)
"""
import argparse
import logging
import os
import sys

from data.data_loader import getSupabaseClient
from core.hmm_model import run_train_only

# Configure logging
def new_func(__name__):
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
    logger = logging.getLogger(__name__)
    return logger

logger = new_func(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run daily HMM regime detection update"
    )
    parser.add_argument(
        "--market-ticker",
        type=str,
        default=os.getenv("MARKET_TICKER", "SPY"),
        help="Market index ticker (default: SPY)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("MODEL_PATH", "models/hmm_market_model.json"),
        help="Path to saved model JSON",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger.info(f"Starting daily update for {args.market_ticker}")

    # Check environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set")
        sys.exit(1)

    try:
        result = run_train_only(getSupabaseClient(), args.market_ticker, market_model_path=args.model_path)

        logger.info("=" * 50)
        logger.info("Daily update complete")
        logger.info(f"  Stocks updated: {len(result['results'])}")
        logger.info(f"  Market regime: {result['market']['regime_label']}")
        logger.info("=" * 50)


    except Exception as e:
        logger.exception(f"Daily update failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
