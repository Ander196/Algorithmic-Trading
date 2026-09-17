
"""
Main Trading System Orchestrator

Entry point for the algorithmic trading system.
Handles startup, main trading loop, shutdown, and error handling.

Usage:
    python main.py [command]

Commands:
    dry-run     Full pipeline, no orders placed
    backtest    Walk-forward backtester
    train-only  Train HMM and exit
    stress-test Run stress tests
    compare     Benchmark comparisons
    dashboard   Show dashboard for running instance

Environment variables (in .env.secrets):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    ALPACA_PAPER=true
    SUPABASE_URL
    SUPABASE_ANON_KEY
"""

import argparse
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import supabase
from dotenv import load_dotenv

from broker.alpaca_client import (
    create_alpaca_client,
    AlpacaClient,
    AlpacaConnectionError,
    AlpacaHealthCheckFailed,
)
from broker.order_executor import (
    OrderExecutor,
    TradeContext,
    create_order_executor,
    OrderExecutionError,
)
from broker.position_tracker import (
    PositionTracker,
    create_position_tracker,
)
from core.hmm_model import (
    HMMVolatilityClassifier,
    train_hmm,
    RegimeState,
    run_train_only,
)
from core.hmm_model import SUPABASE_AVAILABLE as HMM_SUPPABASE_AVAILABLE
from core.regime_strategies import (
    Signal,
    Direction,
    StrategyOrchestrator,
    StrategyConfig,
)
from core.risk_manager import (
    RiskManager,
    Settings,
    PortfolioState,
    HMMSnapshot,
    CircuitBreakerLevel,
)
from data.data_loader import (
    getSupabaseClient,
    getActiveTickers,
)
from backtest.performance import calculate_core_metrics
from data.indicators import prepareFeaturesForHMM
from backtest.backtester import BacktestConfig, WalkForwardBacktester
from state import StateManager, SessionStats, create_state_manager

load_dotenv(".env.secrets")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


@dataclass
class TradingSystem:
    """Main trading system orchestrator."""
    settings: Settings
    alpaca_client: Optional[AlpacaClient] = None
    order_executor: Optional[OrderExecutor] = None
    position_tracker: Optional[PositionTracker] = None
    risk_manager: Optional[RiskManager] = None
    hmm_model: Optional[HMMVolatilityClassifier] = None
    orchestrator: Optional[StrategyOrchestrator] = None
    supabase_client: Optional[supabase.Client] = None
    state_manager: Optional[StateManager] = None
    tickers: list[str] = field(default_factory=list)
    current_regime: Optional[RegimeState] = None
    session_stats: SessionStats = field(default_factory=SessionStats)
    is_running: bool = False
    dry_run: bool = False
    last_bar_time: Optional[datetime] = None


def setupLogging(settings: Settings) -> None:
    """Configure logging based on settings."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)


def checkMarketHours(client: AlpacaClient, settings: Settings, dry_run: bool = False) -> bool:
    """Check if market is open. Wait or exit based on settings."""
    clock = client.get_clock()

    if clock.is_open:
        logger.info(f"Market is OPEN. Next close: {clock.next_close}")
        return True

    logger.warning(f"Market is CLOSED. Next open: {clock.next_open}")

    # In dry-run mode, proceed even if market closed
    if dry_run:
        logger.info("Proceeding in dry-run mode despite market closure")
        return True

    if settings.market_wait:
        logger.info("Waiting for market open...")
        wait_seconds = (clock.next_open - datetime.now(timezone.utc)).total_seconds()
        if wait_seconds > 0:
            logger.info(f"Waiting {wait_seconds/3600:.1f} hours for market open")
            time.sleep(min(wait_seconds, 8 * 3600))  # Max 8 hours
        return checkMarketHours(client, settings, dry_run)
    else:
        logger.info("Exiting due to market closure (market_wait=false)")
        return False


def loadOrTrainHMM(
    settings: Settings,
    supabase_client: supabase.Client,
    tickers: list[str],
) -> HMMVolatilityClassifier:
    """Load existing HMM or train new one if missing/old."""
    model_path = Path(settings.hmm_model_path)
    needs_training = True
    model = None

    # Check if model exists and is recent
    if model_path.exists():
        try:
            model = HMMVolatilityClassifier()
            model.load_model(str(model_path))

            # Make training_date timezone-aware if naive
            training_date = model.training_date
            if training_date.tzinfo is None:
                training_date = training_date.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - training_date).days

            if age_days <= settings.hmm_max_age_days:
                logger.info(f"Loaded existing HMM model (age: {age_days} days)")
                needs_training = False
            else:
                logger.info(f"HMM model is {age_days} days old, retraining (> {settings.hmm_max_age_days} days)")
        except Exception as e:
            logger.warning(f"Failed to load HMM model: {e}")

    if needs_training:
        logger.info("Training new HMM model...")
        # Fetch training data
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=settings.min_train_bars * 2)

        all_data = []
        for ticker in tickers[:20]:  # Limit to 20 tickers for training
            try:
                response = (
                    supabase_client.table("stock_prices")
                    .select("ticker, price_date, open, high, low, close, volume")
                    .eq("ticker", ticker)
                    .gte("price_date", start_date.strftime("%Y-%m-%d"))
                    .lte("price_date", end_date.strftime("%Y-%m-%d"))
                    .order("price_date")
                    .execute()
                )
                if response.data:
                    df = pd.DataFrame(response.data)
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch data for {ticker}: {e}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.sort_values("price_date")

            model = train_hmm(
                combined,
                min_periods=settings.min_train_bars,
                n_init=settings.n_init,
                confidence_threshold=settings.min_confidence,
                confirmation_bars=settings.stability_bars,
                flicker_threshold=settings.flicker_threshold,
                flicker_window=settings.flicker_window,
            )
            model.save_model(str(model_path))
            logger.info(f"HMM model trained and saved to {model_path}")

            # Store HMM results for each ticker in Supabase
            if HMM_SUPPABASE_AVAILABLE:
                logger.info("Storing HMM results to Supabase...")
                for ticker in tickers[:20]:
                    try:
                        # Fetch latest bars for this ticker
                        response = (
                            supabase_client.table("stock_prices")
                            .select("ticker, price_date, open, high, low, close, volume")
                            .eq("ticker", ticker)
                            .order("price_date", desc=True)
                            .limit(504)
                            .execute()
                        )
                        if response.data and len(response.data) >= 100:
                            df = pd.DataFrame(response.data)
                            df = df.sort_values("price_date")
                            features = prepareFeaturesForHMM(df, min_periods=100)
                            if not features.empty:
                                model.store_to_supabase(ticker, features)
                                logger.info(f"Stored HMM result for {ticker}")
                    except Exception as e:
                        logger.warning(f"Failed to store HMM result for {ticker}: {e}")
        else:
            raise RuntimeError("No training data available for HMM")

    return model


def fetchLatestBars(
    supabase_client: supabase.Client,
    tickers: list[str],
    since: Optional[datetime] = None,
) -> dict[str, pd.DataFrame]:
    """Fetch latest bars from Supabase."""
    bars_dict = {}

    for ticker in tickers:
        try:
            query = supabase_client.table("stock_prices").select(
                "ticker, price_date, open, high, low, close, volume"
            ).eq("ticker", ticker).order("price_date", desc=True).limit(100)

            if since:
                query = query.gte("price_date", since.strftime("%Y-%m-%d"))

            response = query.execute()

            if response.data:
                df = pd.DataFrame(response.data)
                df = df.sort_values("price_date")
                bars_dict[ticker] = df
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {ticker}: {e}")

    return bars_dict


def computeFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for HMM prediction."""
    if df.empty or len(df) < 20:
        return pd.DataFrame()

    df = df.copy()

    # Returns
    df["returns"] = df["close"].pct_change()

    # Volatility (rolling std of returns)
    df["volatility"] = df["returns"].rolling(20).std()

    # Price momentum
    df["momentum"] = df["close"].pct_change(10)

    # Volume SMA
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # High-Low range
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # Drop NaN
    df = df.dropna()

    # Feature columns for HMM
    features = ["returns", "volatility", "momentum", "volume_ratio", "hl_range"]

    return df[features]


def getPortfolioState(
    position_tracker: PositionTracker,
    risk_manager: RiskManager,
) -> PortfolioState:
    """Get current portfolio state from position tracker."""
    snapshot = position_tracker.get_portfolio_snapshot()

    return PortfolioState(
        equity=snapshot.equity,
        cash=snapshot.cash,
        buying_power=snapshot.buying_power,
        positions=snapshot.positions,
        daily_pnl=snapshot.daily_pnl,
        weekly_pnl=snapshot.weekly_pnl,
        peak_equity=snapshot.equity,
        drawdown=snapshot.daily_pnl / snapshot.equity if snapshot.equity > 0 else 0,
        circuit_breaker_status=CircuitBreakerLevel.NONE,
        flicker_rate=0,
        trades_today=0,
    )


def startup(system: TradingSystem) -> bool:
    """Execute startup sequence."""
    logger.info("=" * 60)
    logger.info("STARTING TRADING SYSTEM")
    logger.info("=" * 60)

    # Step 1: Load config (already done via Settings.get())
    settings = system.settings
    logger.info(f"Config loaded: bar_interval={settings.bar_interval}min, dry_run={system.dry_run}")

    # Step 2: Connect to Alpaca
    try:
        system.alpaca_client = create_alpaca_client(paper_trading=True)
        logger.info(f"Connected to Alpaca (Paper: {system.alpaca_client.is_paper_trading})")
    except (AlpacaConnectionError, AlpacaHealthCheckFailed) as e:
        logger.error(f"Failed to connect to Alpaca: {e}")
        return False

    # Step 3: Verify account
    account = system.alpaca_client.get_account()
    logger.info(f"Account: {account.status}, Equity: ${account.equity:,.2f}, Cash: ${account.cash:,.2f}")

    # Step 4: Check market hours
    if not checkMarketHours(system.alpaca_client, settings, dry_run=system.dry_run):
        return False

    # Step 5: Connect to Supabase
    try:
        system.supabase_client = getSupabaseClient()
        system.tickers = getActiveTickers(system.supabase_client)
        logger.info(f"Connected to Supabase, {len(system.tickers)} active tickers")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return False

    # Step 6: Load or train HMM
    try:
        system.hmm_model = loadOrTrainHMM(settings, system.supabase_client, system.tickers)
    except Exception as e:
        logger.error(f"Failed to load/train HMM: {e}")
        return False

    # Step 7: Initialize risk manager
    system.risk_manager = RiskManager()

    # Step 8: Initialize position tracker and sync
    from alpaca.trading import TradingClient
    from alpaca.common.enums import BaseURL

    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    base_url = BaseURL.TRADING_PAPER if paper else BaseURL.TRADING_LIVE

    trading_client = TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
        paper=paper,
        url_override=str(base_url.value),
    )

    system.position_tracker = create_position_tracker(trading_client)
    system.position_tracker.sync_with_broker()
    logger.info(f"Synced {len(system.position_tracker.get_all_positions())} positions from broker")

    # Step 9: Initialize order executor
    system.order_executor = create_order_executor(trading_client)

    # Step 10: Initialize strategy orchestrator
    config = StrategyConfig(
        min_confidence_threshold=settings.min_confidence,
        rebalance_threshold=settings.rebalance_threshold,
    )
    regime_infos = system.hmm_model.regime_info if system.hmm_model else {}
    system.orchestrator = StrategyOrchestrator(config, regime_infos)

    # Step 11: Initialize state manager
    system.state_manager = create_state_manager(settings.state_snapshot_path)

    # Step 12: Try to recover from snapshot
    saved_state = system.state_manager.load()
    if saved_state:
        logger.info(f"Recovered state from snapshot: regime={saved_state.current_regime_name}")
    else:
        logger.info("No state snapshot found, starting fresh")

    # Step 13: Initialize session stats
    system.session_stats = SessionStats(
        start_time=datetime.now(timezone.utc).isoformat(),
    )

    logger.info("=" * 60)
    logger.info("SYSTEM ONLINE")
    logger.info(f"  Mode: {'DRY-RUN' if system.dry_run else 'LIVE'}")
    logger.info(f"  Equity: ${account.equity:,.2f}")
    logger.info(f"  Positions: {len(system.position_tracker.get_all_positions())}")
    logger.info(f"  Tickers: {len(system.tickers)}")
    logger.info("=" * 60)

    system.is_running = True
    return True


def mainLoop(system: TradingSystem) -> None:
    """Main trading loop."""
    settings = system.settings
    poll_interval = settings.poll_interval

    logger.info(f"Starting main loop (poll every {poll_interval}s)")

    while system.is_running:
        try:
            # Fetch latest bars
            bars_dict = fetchLatestBars(
                system.supabase_client,
                system.tickers,
                since=system.last_bar_time,
            )

            if not bars_dict:
                time.sleep(poll_interval)
                continue

            # Get latest bar time
            latest_times = [df["price_date"].max() for df in bars_dict.values() if not df.empty]
            if latest_times:
                system.last_bar_time = pd.to_datetime(max(latest_times)).to_pydatetime()

            # Process each ticker
            for ticker, bars in bars_dict.items():
                if len(bars) < 20:
                    continue

                # Compute features
                features = computeFeatures(bars)
                if features.empty:
                    continue

                # Get current regime (forward-only prediction)
                regime = system.hmm_model.get_current_regime_state(features)
                system.current_regime = regime

                # Check regime stability (persistence)
                is_stable = regime.consecutive_bars >= settings.stability_bars

                # Check flicker rate
                is_flickering = regime.consecutive_bars < settings.flicker_threshold

                # Get portfolio state
                portfolio_state = getPortfolioState(system.position_tracker, system.risk_manager)

                # Create HMM snapshot for risk manager
                hmm_snapshot = HMMSnapshot(
                    regime_id=regime.state_id,
                    regime_name=regime.label,
                    regime_probability=regime.probability,
                    is_flickering=is_flickering,
                    flicker_count=regime.consecutive_bars,
                )

                # Generate signals
                signals = system.orchestrator.generate_signals(
                    symbols=[ticker],
                    bars_dict={ticker: bars},
                    regime_state=regime,
                    is_flickering=is_flickering,
                )

                # Process each signal
                for signal in signals:
                    # Validate with risk manager
                    decision = system.risk_manager.validate_signal(
                        signal=signal,
                        portfolio_state=portfolio_state,
                        hmm_snapshot=hmm_snapshot,
                        price_data=bars,
                    )

                    if decision.status.name == "APPROVED":
                        final_signal = decision.modified_signal or signal

                        if not system.dry_run:
                            # Submit order
                            try:
                                trade_context = TradeContext(
                                    trade_id=f"signal_{ticker}_{int(time.time())}",
                                    signal=final_signal,
                                    signal_timestamp=datetime.now(timezone.utc),
                                )
                                system.order_executor.submit_order(final_signal, trade_context)
                                logger.info(f"Order submitted: {ticker} {final_signal.direction.value}")

                                # Update session stats
                                system.session_stats.total_trades += 1
                                system.session_stats.regime_distribution[regime.label] = \
                                    system.session_stats.regime_distribution.get(regime.label, 0) + 1
                            except OrderExecutionError as e:
                                logger.error(f"Order failed: {e}")
                        else:
                            logger.info(f"[DRY-RUN] Would submit order: {ticker} {final_signal.direction.value}")

                    elif decision.status.name == "MODIFIED":
                        logger.info(f"Signal modified: {decision.modifications}")
                    else:
                        logger.info(f"Signal rejected: {decision.rejection_reason}")

            # Update positions
            system.position_tracker.update_all_positions()

            # Circuit breaker check
            portfolio_state = getPortfolioState(system.position_tracker, system.risk_manager)
            cb_level = system.risk_manager.check(portfolio_state)

            if cb_level != CircuitBreakerLevel.NONE:
                logger.warning(f"Circuit breaker triggered: {cb_level.name}")
                if cb_level in (CircuitBreakerLevel.HALT_DAY, CircuitBreakerLevel.HALT_WEEK, CircuitBreakerLevel.HALT_ALL):
                    logger.warning("Halting trading due to circuit breaker")
                    break

            # Save state periodically
            if system.state_manager:
                positions = system.position_tracker.get_all_positions()
                system.state_manager.save(
                    regime_id=system.current_regime.state_id if system.current_regime else 0,
                    regime_name=system.current_regime.label if system.current_regime else "UNKNOWN",
                    regime_probability=system.current_regime.probability if system.current_regime else 0.0,
                    is_flickering=is_flickering if system.current_regime else False,
                    flicker_count=system.current_regime.consecutive_bars if system.current_regime else 0,
                    last_bar_time=system.last_bar_time,
                    positions=positions,
                    session_stats=system.session_stats,
                    hmm_model_path=settings.hmm_model_path,
                    hmm_last_trained=system.hmm_model.training_date if system.hmm_model else None,
                )

            # Check for weekly retrain
            if settings.retrain_weeks > 0 and system.hmm_model:
                age = (datetime.now(timezone.utc) - system.hmm_model.training_date).days
                if age >= settings.retrain_weeks * 7:
                    logger.info("Retraining HMM (weekly schedule)...")
                    system.hmm_model = loadOrTrainHMM(settings, system.supabase_client, system.tickers)

            time.sleep(poll_interval)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.debug(traceback.format_exc())

            # Save state on error
            if system.state_manager:
                system.state_manager.save(
                    regime_id=system.current_regime.state_id if system.current_regime else 0,
                    regime_name=system.current_regime.label if system.current_regime else "UNKNOWN",
                    regime_probability=system.current_regime.probability if system.current_regime else 0.0,
                    last_bar_time=system.last_bar_time,
                    session_stats=system.session_stats,
                    hmm_model_path=settings.hmm_model_path,
                )

            #alpaca error handling - retry with backoff
            if "Alpaca" in str(type(e)).lower():
                logger.warning("Alpaca API error, retrying in 30s...")
                time.sleep(30)
            else:
                # Data feed drop - pause signals but keep running
                logger.warning("Data feed issue, continuing with existing state...")
                time.sleep(poll_interval * 2)


def shutdown(system: TradingSystem) -> None:
    """Execute shutdown sequence."""
    logger.info("=" * 60)
    logger.info("SHUTTING DOWN TRADING SYSTEM")
    logger.info("=" * 60)

    # Save state snapshot
    if system.state_manager:
        positions = []
        if system.position_tracker:
            positions = system.position_tracker.get_all_positions()

        system.state_manager.save(
            regime_id=system.current_regime.state_id if system.current_regime else 0,
            regime_name=system.current_regime.label if system.current_regime else "UNKNOWN",
            regime_probability=system.current_regime.probability if system.current_regime else 0.0,
            last_bar_time=system.last_bar_time,
            positions=positions,
            session_stats=system.session_stats,
            hmm_model_path=system.settings.hmm_model_path,
            hmm_last_trained=system.hmm_model.training_date if system.hmm_model else None,
        )
        logger.info(f"State saved to {system.settings.state_snapshot_path}")

    # Print session summary
    logger.info("=" * 60)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total Trades: {system.session_stats.total_trades}")
    logger.info(f"  Winning Trades: {system.session_stats.winning_trades}")
    logger.info(f"  Losing Trades: {system.session_stats.losing_trades}")
    logger.info(f"  Total P&L: ${system.session_stats.total_pnl:,.2f}")
    logger.info(f"  Regime Distribution: {system.session_stats.regime_distribution}")
    logger.info(f"  Start Time: {system.session_stats.start_time}")
    logger.info(f"  Last Update: {system.session_stats.last_update}")
    logger.info("=" * 60)

    # DO NOT close positions - stops remain active
    logger.info("Positions left open (stops remain active)")

    # Note: Supabase connections close automatically
    logger.info("Shutdown complete")


def runDryRun(settings: Settings) -> None:
    """Run in dry-run mode (full pipeline, no orders)."""
    logger.info("Starting DRY-RUN mode")

    system = TradingSystem(settings=settings, dry_run=True)

    if not startup(system):
        logger.error("Startup failed")
        return

    try:
        mainLoop(system)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(system)


def getTickersFromHMMResults(client, limit: int = 100) -> list[str]:
    """Get unique tickers that have HMM results in Supabase."""
    response = (
        client.table("hmm_results")
        .select("ticker")
        .order("result_date", desc=True)
        .limit(limit)
        .execute()
    )

    if not response.data:
        logger.warning("No HMM results found in database")
        return []

    # Get unique tickers
    tickers = list(set(row['ticker'] for row in response.data))
    logger.info(f"Found {len(tickers)} unique tickers with HMM results")
    return tickers


def fetchOHLCVFromSupabase(
    client,
    ticker: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 2000,
) -> pd.DataFrame:
    """Fetch OHLCV data for a ticker from Supabase."""
    query = client.table("stock_prices").select(
        "ticker, price_date, open, high, low, close, volume"
    ).eq("ticker", ticker)

    if start_date:
        query = query.gte("price_date", start_date.strftime("%Y-%m-%d"))
    if end_date:
        query = query.lte("price_date", end_date.strftime("%Y-%m-%d"))

    query = query.order("price_date").limit(limit)
    response = query.execute()

    if not response.data:
        return pd.DataFrame()

    df = pd.DataFrame(response.data)
    df['price_date'] = pd.to_datetime(df['price_date'])
    df = df.set_index('price_date')
    df = df.sort_index()

    return df


def ensureBacktestResultsTable(client: supabase.Client) -> None:
    """Check if backtest_results table exists, create if not."""
    try:
        # Check if table exists by querying it
        client.table("backtest_results").select("ticker").limit(1).execute()
        logger.info("backtest_results table already exists")
    except Exception as e:
        logger.warning(f"backtest_results table does not exist: {e}")
        logger.info("Creating backtest_results table...")

        # Create the table using Supabase SQL execution
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS backtest_results (
            ticker TEXT PRIMARY KEY,
            total_return_pct DOUBLE PRECISION,
            cagr_pct DOUBLE PRECISION,
            sharpe_ratio DOUBLE PRECISION,
            sortino_ratio DOUBLE PRECISION,
            calmar_ratio DOUBLE PRECISION,
            max_drawdown_pct DOUBLE PRECISION,
            max_drawdown_days INTEGER,
            total_trades INTEGER,
            win_rate_pct DOUBLE PRECISION,
            avg_win_pct DOUBLE PRECISION,
            avg_loss_pct DOUBLE PRECISION,
            profit_factor DOUBLE PRECISION,
            avg_holding_period_days DOUBLE PRECISION,
            worst_day_pct DOUBLE PRECISION,
            worst_week_pct DOUBLE PRECISION,
            worst_month_pct DOUBLE PRECISION,
            max_consecutive_losses INTEGER,
            longest_underwater_days INTEGER,
            volatility_annualized_pct DOUBLE PRECISION,
            skewness DOUBLE PRECISION,
            kurtosis DOUBLE PRECISION,
            backtest_date TIMESTAMPTZ,
            num_windows INTEGER,
            final_equity DOUBLE PRECISION
        );
        """

        try:
            # Try to execute raw SQL using the postgrest client
            # supabase-py v2 uses postgrest.execute_raw for raw SQL
            if hasattr(client.postgrest, 'execute_raw'):
                client.postgrest.execute_raw(create_table_sql)
                logger.info("backtest_results table created successfully")
            else:
                raise AttributeError("execute_raw not available")
        except Exception as create_error:
            # If automatic creation fails, try an alternative approach using HTTP
            try:
                import requests
                supabase_url = os.getenv("SUPABASE_URL")
                supabase_key = os.getenv("SUPABASE_ANON_KEY")
                if supabase_url and supabase_key:
                    # Try using the Supabase REST API directly
                    response = requests.post(
                        f"{supabase_url}/rest/v1/rpc/exec_sql",
                        json={"query": create_table_sql},
                        headers={
                            "apikey": supabase_key,
                            "Authorization": f"Bearer {supabase_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    if response.status_code in (200, 201):
                        logger.info("backtest_results table created successfully via API")
                    else:
                        raise Exception(f"API returned {response.status_code}")
                else:
                    raise Exception("SUPABASE_URL or SUPABASE_ANON_KEY not set")
            except ImportError:
                # requests not available
                logger.warning(f"Could not create table automatically: {create_error}")
                logger.info("Please create the table manually in Supabase SQL Editor with:")
                logger.info(create_table_sql)
            except Exception as api_error:
                logger.warning(f"Could not create table automatically: {api_error}")
                logger.info("Please create the table manually in Supabase SQL Editor with:")
                logger.info(create_table_sql)


def saveBacktestResultToSupabase(
    client: supabase.Client,
    ticker: str,
    result,
) -> None:
    """Save backtest result metrics to Supabase (upsert - one result per ticker)."""
    try:
        # Calculate core metrics using performance.py
        metrics = calculate_core_metrics(result.equity_curve, result.trade_log)

        # Prepare data for upsert
        data = {
            "ticker": ticker,
            "total_return_pct": metrics.total_return_pct,
            "cagr_pct": metrics.cagr_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "max_drawdown_days": metrics.max_drawdown_days,
            "total_trades": metrics.total_trades,
            "win_rate_pct": metrics.win_rate_pct,
            "avg_win_pct": metrics.avg_win_pct,
            "avg_loss_pct": metrics.avg_loss_pct,
            "profit_factor": metrics.profit_factor,
            "avg_holding_period_days": metrics.avg_holding_period_days,
            "worst_day_pct": metrics.worst_day_pct,
            "worst_week_pct": metrics.worst_week_pct,
            "worst_month_pct": metrics.worst_month_pct,
            "max_consecutive_losses": metrics.max_consecutive_losses,
            "longest_underwater_days": metrics.longest_underwater_days,
            "volatility_annualized_pct": metrics.volatility_annualized_pct,
            "skewness": metrics.skewness,
            "kurtosis": metrics.kurtosis,
            "backtest_date": datetime.now(timezone.utc).isoformat(),
            "num_windows": len(result.windows),
            "final_equity": result.final_equity,
        }

        # Upsert - will replace if ticker exists (due to primary key)
        client.table("backtest_results").upsert(data, on_conflict="ticker").execute()
        logger.info(f"Saved backtest results for {ticker} to Supabase")
    except Exception as e:
        logger.warning(f"Could not save backtest results to Supabase: {e}")


def runBacktest(settings: Settings) -> None:
    """Run walk-forward backtester using tickers from hmm_results."""
    import numpy as np

    from backtest.backtester import BacktestConfig, WalkForwardBacktester

    logger.info("Starting BACKTEST mode")

    # Connect to Supabase
    client = getSupabaseClient()

    # Get tickers from HMM results
    tickers = getTickersFromHMMResults(client)
    if not tickers:
        logger.error("No tickers found from HMM results")
        return

    logger.info(f"Running backtest for {len(tickers)} tickers: {tickers}")

    # Ensure backtest_results table exists
    ensureBacktestResultsTable(client)

    # Configure backtester
    # min_periods_for_hmm=150 works because with ~500 rows we get ~176 valid features
    config = BacktestConfig(
        initial_capital=settings.initial_capital,
        is_period_days=200,  # ~10 months for training
        oos_period_days=63,  # 3 months for evaluation
        step_days=63,  # 3 month step
        min_periods_for_hmm=150,  # 150 bars minimum for HMM
        rebalance_threshold=0.10,  # 10% allocation change triggers rebalance
        slippage_pct=0.0005,  # 0.05% slippage
    )

    backtester = WalkForwardBacktester(config)

    # Fetch all available data (no date restrictions)
    all_results = []

    for ticker in tickers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running backtest for {ticker}")
        logger.info(f"{'='*60}")

        # Fetch OHLCV data from stock_prices table with lookback
        # Fetch all available OHLCV data from stock_prices table
        df = fetchOHLCVFromSupabase(client, ticker, limit=5000)

        if df.empty:
            logger.warning(f"No data found for {ticker}")
            continue

        logger.info(f"Fetched {len(df)} bars for {ticker}")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

        # Check if we have enough data for backtest
        # Need enough for IS + OOS + lookback for features (SMA200)
        min_required = config.min_periods_for_hmm + config.oos_period_days + 100  # extra for feature lookback
        if len(df) < min_required:
            logger.warning(f"Insufficient data for {ticker}: {len(df)} bars, need at least {min_required}")
            continue

        try:
            # Run walk-forward backtest
            result = backtester.run(df, ticker)
            all_results.append(result)

            # Save results to Supabase
            saveBacktestResultToSupabase(client, ticker, result)

            # Print summary for this ticker
            logger.info(f"\nBacktest Results for {ticker}:")
            logger.info(f"  Total Return: {result.total_return_pct:.2f}%")
            logger.info(f"  Final Equity: ${result.final_equity:,.2f}")
            logger.info(f"  Total Trades: {len(result.trade_log) if len(result.trade_log) > 0 else 0}")
            logger.info(f"  Windows: {len(result.windows)}")

        except Exception as e:
            logger.error(f"Backtest failed for {ticker}: {e}")
            continue

    # Print aggregate summary
    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("AGGREGATE SUMMARY")
        logger.info(f"{'='*60}")

        total_returns = [r.total_return_pct for r in all_results]
        avg_return = np.mean(total_returns) if total_returns else 0

        logger.info(f"  Tickers Tested: {len(all_results)}")
        logger.info(f"  Average Return: {avg_return:.2f}%")
        logger.info(f"  Best Return: {max(total_returns):.2f}%")
        logger.info(f"  Worst Return: {min(total_returns):.2f}%")

    logger.info("Backtest complete")


def runTrainOnly(settings: Settings, market_ticker: str = "SPY") -> None:
    """Run the persisted two-layer market and per-stock HMM workflow."""
    logger.info("Starting TRAIN-ONLY mode")
    client = getSupabaseClient()
    try:
        result = run_train_only(client, market_ticker)
        logger.info("HMM train-only complete: %d stocks, market=%s, base=%.4f", len(result["results"]), result["market"]["regime_label"], result["market"]["base_multiplier"])
    except Exception as e:
        logger.error(f"HMM training failed: {e}")
        sys.exit(1)


def runStressTest(settings: Settings) -> None:
    """Run stress tests."""
    logger.info("Starting STRESS-TEST mode")
    # Delegate to stress test module
    import subprocess
    subprocess.run(["python", "backtest/stress_test.py", "--help"])
    logger.info("Run: python backtest/stress_test.py --data <path>")


def runCompare(settings: Settings) -> None:
    """Run benchmark comparisons."""
    logger.info("Starting COMPARE mode")
    logger.info("Benchmarking not yet implemented")


def runDashboard(settings: Settings) -> None:
    """Show dashboard for running instance."""
    logger.info("Starting DASHBOARD mode")
    import subprocess
    subprocess.run(["streamlit", "run", "monitoring/app.py"])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Algorithmic Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "command",
        choices=["dry-run", "backtest", "train-only", "stress-test", "compare", "dashboard"],
        help="Command to run",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (for dry-run command)",
    )
    parser.add_argument(
        "--market-ticker",
        default=os.getenv("MARKET_TICKER", "SPY"),
        help="Liquid market index used by train-only (default: SPY)",
    )

    args = parser.parse_args()

    # Load settings
    settings = Settings.get()
    setupLogging(settings)

    # Override dry-run from CLI if provided
    if args.command == "dry-run" or args.dry_run:
        settings.dry_run = True

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Execute command
    try:
        match args.command:
            case "dry-run":
                runDryRun(settings)
            case "backtest":
                runBacktest(settings)
            case "train-only":
                runTrainOnly(settings, args.market_ticker)
            case "stress-test":
                runStressTest(settings)
            case "compare":
                runCompare(settings)
            case "dashboard":
                runDashboard(settings)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
