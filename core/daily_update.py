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
from datetime import datetime, timedelta, timezone

from core.hmm_model import TwoLayerRegimeEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to process",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=504,
        help="Days of history to fetch (default: 504)",
    )
    parser.add_argument(
        "--fallback-tickers",
        type=str,
        default="SPY,QQQ,IWM",
        help="Fallback market tickers if primary fails (comma-separated)",
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

    # Parse tickers
    tickers = None
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]

    # Parse fallback market tickers
    fallback_tickers = [t.strip() for t in args.fallback_tickers.split(",")]

    # Create engine
    engine = TwoLayerRegimeEngine(market_ticker=args.market_ticker)

    # Try to load existing model or train new one
    if os.path.exists(args.model_path):
        logger.info(f"Loading model from {args.model_path}")
        engine.load_market_model(args.model_path)
    else:
        logger.warning(f"Model not found at {args.model_path}, will train from Supabase data")

    # Register tickers if provided
    if tickers:
        engine.register_tickers(tickers)
    else:
        logger.warning("No tickers provided, exiting")
        sys.exit(0)

    # Run daily update
    try:
        result = engine.run_daily_update(
            market_tickers=fallback_tickers,
            registered_tickers=tickers,
            lookback_days=args.lookback_days,
        )

        logger.info("=" * 50)
        logger.info("Daily update complete")
        logger.info(f"  Successes: {result['successes']}")
        logger.info(f"  Failures: {result['failures']}")
        logger.info(f"  Total: {result['total']}")
        logger.info(f"  Market regime: {result['market_state']}")
        logger.info("=" * 50)

        if result["failures"] > 0:
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Daily update failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()