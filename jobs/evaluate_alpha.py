"""Run the V1 Alpha model evaluation on the configured Supabase universe.

This job is research-only:
- universe comes from stocks.is_active = true;
- historical prices come from stock_prices;
- AlphaDatasetBuilder creates point-in-time features/labels;
- WalkForwardProtocol and AlphaEvaluator produce OOS metrics;
- no Strategy, RiskManager, execution, or production Top-N selection is performed.

V1 caveat:
the current active universe is applied to historical data. This is intentionally
not survivorship-bias-free and must be treated as a research limitation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from supabase import Client

from core.alpha_dataset import AlphaDatasetBuilder, AlphaDatasetConfig
from core.alpha_diagnostics import evaluate_alpha_diagnostics, extreme_return_rows
from core.alpha_evaluation import AlphaEvaluationConfig, AlphaEvaluationReport, AlphaEvaluator
from core.alpha_model import AlphaModelConfig
from core.walk_forward import WalkForwardConfig, WalkForwardProtocol
from jobs.common import fetch_price_history, get_supabase_client
from dotenv import load_dotenv

from data.excluded_tickers import is_excluded_ticker, normalize_ticker

load_dotenv(".env.secrets")


@dataclass(frozen=True)
class AlphaLoadSummary:
    requested_tickers: int
    loaded_tickers: int
    failed_tickers: tuple[str, ...]


def _to_ohlcv_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Convert the shared Supabase history format into Alpha OHLCV input."""
    required = {"price_date", "open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"price history missing required columns: {sorted(missing)}")

    frame = data.copy()
    frame["price_date"] = pd.to_datetime(frame["price_date"], utc=True).dt.tz_convert(None)
    return (
        frame.set_index("price_date")
        .sort_index()[["open", "high", "low", "close", "volume"]]
        .copy()
    )


def get_alpha_universe(
    client: Client,
    market_ticker: str,
) -> list[str]:
    """Return the configured execution universe, excluding the market benchmark."""
    response = (
        client.table("stocks")
        .select("ticker")
        .eq("is_active", True)
        .execute()
    )
    tickers = {
        normalize_ticker(row["ticker"])
        for row in (response.data or [])
        if row.get("ticker") and not is_excluded_ticker(row["ticker"])
    }
    tickers.discard(market_ticker.upper())
    return sorted(tickers)


def load_alpha_inputs(
    client: Client,
    tickers: list[str],
    market_ticker: str,
    history_limit: int | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, AlphaLoadSummary]:
    """Load market and stock histories needed by the Alpha dataset builder."""
    if history_limit is not None and history_limit <= 0:
        raise ValueError("history_limit must be greater than zero")

    market_raw = fetch_price_history(client, market_ticker, limit=history_limit)
    market_data = _to_ohlcv_frame(market_raw)

    stock_data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        try:
            stock_data[ticker] = _to_ohlcv_frame(
                fetch_price_history(client, ticker, limit=history_limit)
            )
        except Exception as exc:
            failed.append(ticker)
            print(f"WARNING: skipping {ticker}: {exc}")

    return (
        stock_data,
        market_data,
        AlphaLoadSummary(
            requested_tickers=len(tickers),
            loaded_tickers=len(stock_data),
            failed_tickers=tuple(failed),
        ),
    )


def evaluate_alpha(
    client: Client,
    market_ticker: str = "SPY",
    history_limit: int | None = None,
    max_tickers: int | None = None,
) -> tuple[AlphaEvaluationReport, AlphaLoadSummary, int]:
    """Build the real-data Alpha dataset and return its OOS evaluation report."""
    universe = get_alpha_universe(client, market_ticker)
    if max_tickers is not None:
        if max_tickers <= 0:
            raise ValueError("max_tickers must be greater than zero")
        universe = universe[:max_tickers]

    stock_data, market_data, load_summary = load_alpha_inputs(
        client,
        universe,
        market_ticker,
        history_limit=history_limit,
    )

    if not stock_data:
        raise RuntimeError("No stock histories were loaded from the configured universe")

    dataset_config = AlphaDatasetConfig(
        horizon_days=5,
        market_ticker=market_ticker,
    )
    dataset = AlphaDatasetBuilder(dataset_config).build(
        stock_data,
        market_data,
    )

    protocol = WalkForwardProtocol(
        WalkForwardConfig(
            train_period=756,
            test_period=126,
            step_period=126,
            horizon_days=dataset_config.horizon_days,
            min_train_rows=500,
        )
    )
    evaluator = AlphaEvaluator(
        AlphaEvaluationConfig(
            top_n_values=(5, 10, 20, 50),
            min_cross_section=20,
        )
    )
    report = evaluator.evaluate_walk_forward(
        dataset,
        protocol,
        model_config=AlphaModelConfig(),
    )

    return report, load_summary, len(dataset)


def _print_report(
    report: AlphaEvaluationReport,
    load_summary: AlphaLoadSummary,
    dataset_rows: int,
) -> None:
    print("\n=== Alpha V1 OOS Evaluation ===")
    print(f"Requested tickers: {load_summary.requested_tickers}")
    print(f"Loaded tickers:    {load_summary.loaded_tickers}")
    print(f"Dataset rows:      {dataset_rows}")
    print(f"Walk-forward folds:{report.fold_count}")
    print(f"Mean daily IC:     {report.mean_ic:.6f}")
    print(f"Median fold IC:    {report.median_ic:.6f}")
    print(f"Positive IC frac:  {report.ic_positive_fraction:.3f}")

    for fold in report.folds:
        print(
            f"Fold {fold.fold_id}: "
            f"test={fold.test_start.date()}..{fold.test_end.date()} "
            f"rows={fold.test_rows} "
            f"mean_ic={fold.mean_ic:.6f} "
            f"mean_cs={fold.mean_cross_section:.1f}"
        )
        for metric in fold.top_n_metrics:
            print(
                f"  Top-{metric.top_n}: "
                f"excess={metric.mean_excess_return:.6f} "
                f"vs_universe={metric.mean_top_vs_universe_spread:.6f} "
                f"vs_bottom={metric.mean_top_vs_bottom_spread:.6f} "
                f"hit_rate={metric.mean_hit_rate:.3f} "
                f"dates={metric.valid_dates}"
            )

    if load_summary.failed_tickers:
        print(f"Failed tickers: {', '.join(load_summary.failed_tickers)}")


def _print_diagnostics(
    report: AlphaEvaluationReport,
    *,
    top_n: int,
    random_repeats: int,
    random_seed: int,
    diagnostics_dir: Path | None,
) -> None:
    """Print out-of-sample baselines and optionally persist audit CSV files."""
    diagnostics = evaluate_alpha_diagnostics(
        report.oos_predictions,
        top_n=top_n,
        min_cross_section=20,
        random_repeats=random_repeats,
        random_seed=random_seed,
    )
    print("\n=== Alpha OOS Baseline Comparison ===")
    print(diagnostics.ranking_comparison.to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    print("\n=== Alpha OOS Outlier Sensitivity ===")
    print(diagnostics.outlier_sensitivity.to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))

    extremes = extreme_return_rows(report.oos_predictions, n_per_side=10)
    print("\n=== Largest OOS Excess-Return Observations ===")
    print(extremes.to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    print(
        "Diagnostic returns are overlapping five-session labels, not a "
        "compounded or transaction-cost-adjusted portfolio backtest."
    )

    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        report.oos_predictions.to_csv(diagnostics_dir / "alpha_oos_predictions.csv")
        diagnostics.ranking_comparison.to_csv(
            diagnostics_dir / "alpha_ranking_comparison.csv", index=False
        )
        diagnostics.outlier_sensitivity.to_csv(
            diagnostics_dir / "alpha_outlier_sensitivity.csv", index=False
        )
        extremes.to_csv(diagnostics_dir / "alpha_extreme_returns.csv", index=False)
        print(f"Diagnostics written to: {diagnostics_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the V1 Alpha model on Supabase history")
    parser.add_argument("--market-ticker", default="SPY")
    parser.add_argument(
        "--history-limit",
        type=int,
        default=None,
        help="Optional maximum number of rows per ticker; default loads all available history.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional cap for local/smoke runs; production evaluation uses the full active universe.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help="Optional directory for CSV exports of OOS predictions and diagnostics.",
    )
    parser.add_argument(
        "--diagnostics-top-n",
        type=int,
        default=5,
        help="Top-N size used in Alpha and baseline diagnostics (default: 5).",
    )
    parser.add_argument(
        "--random-baseline-repeats",
        type=int,
        default=20,
        help="Number of seeded random rankings averaged for the baseline (default: 20).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for reproducible random-baseline rankings.",
    )
    args = parser.parse_args()
    if args.diagnostics_top_n <= 0:
        parser.error("--diagnostics-top-n must be greater than zero")
    if args.random_baseline_repeats <= 0:
        parser.error("--random-baseline-repeats must be greater than zero")

    client = get_supabase_client()
    report, load_summary, dataset_rows = evaluate_alpha(
        client,
        market_ticker=args.market_ticker,
        history_limit=args.history_limit,
        max_tickers=args.max_tickers,
    )
    _print_report(report, load_summary, dataset_rows)
    _print_diagnostics(
        report,
        top_n=args.diagnostics_top_n,
        random_repeats=args.random_baseline_repeats,
        random_seed=args.random_seed,
        diagnostics_dir=args.diagnostics_dir,
    )


if __name__ == "__main__":
    main()
