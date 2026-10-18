from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.alpha_portfolio_backtest import (
    PortfolioBacktestConfig,
    run_alpha_portfolio_backtest,
)


def _prices(sessions: int = 100, tickers: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=sessions)
    data = {
        f"T{i:02d}": 20.0 * np.exp((0.0002 + i * 0.000015) * np.arange(sessions))
        for i in range(tickers)
    }
    data["SPY"] = 100.0 * np.exp(0.00025 * np.arange(sessions))
    return pd.DataFrame(data, index=dates)


def _predictions(
    prices: pd.DataFrame,
    first: int = 65,
    last: int = 80,
) -> pd.DataFrame:
    rows = []
    tickers = [column for column in prices.columns if column != "SPY"]
    for date in prices.index[first:last]:
        for ticker_idx, ticker in enumerate(tickers):
            rows.append({
                "date": date,
                "ticker": ticker,
                "expected_excess_return": float(ticker_idx) + (date.day % 3) / 100.0,
            })
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def test_backtest_returns_four_comparable_portfolios_and_metrics() -> None:
    prices = _prices()
    report = run_alpha_portfolio_backtest(
        _predictions(prices),
        prices,
        config=PortfolioBacktestConfig(
            top_n=5,
            min_cross_section=20,
            rebalance_every=5,
            momentum_lookback=60,
            transaction_cost_bps=5.0,
        ),
    )

    assert set(report.summary["strategy"]) == {
        "ridge_top_n",
        "momentum_top_n",
        "equal_weight_universe",
        "benchmark",
    }
    assert report.daily_returns.index.is_monotonic_increasing
    assert report.daily_returns.index.min() == prices.index[66]
    assert report.daily_returns.index.max() == prices.index[80]
    assert (report.daily_returns.filter(like="_equity") > 0).all().all()
    assert {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio_rf0",
        "max_drawdown",
        "average_session_turnover",
        "sum_cost_rate",
    }.issubset(report.summary.columns)


def test_transaction_cost_is_subtracted_and_initial_purchase_is_charged() -> None:
    prices = _prices()
    report = run_alpha_portfolio_backtest(
        _predictions(prices),
        prices,
        config=PortfolioBacktestConfig(
            top_n=5,
            min_cross_section=20,
            transaction_cost_bps=10.0,
        ),
    )
    daily = report.daily_returns
    first_date = daily.index[0]

    assert daily.loc[first_date, "ridge_top_n_turnover"] == pytest.approx(1.0)
    assert daily.loc[first_date, "ridge_top_n_cost_rate"] == pytest.approx(0.001)
    assert daily.loc[first_date, "benchmark_cost_rate"] == pytest.approx(0.001)
    for strategy in ("ridge_top_n", "momentum_top_n", "equal_weight_universe", "benchmark"):
        assert (
            daily[f"{strategy}_gross_return"] - daily[f"{strategy}_net_return"]
        ).to_numpy() == pytest.approx(daily[f"{strategy}_cost_rate"].to_numpy())


def test_portfolios_hold_positions_between_rebalances_and_keep_weights_normalized() -> None:
    prices = _prices()
    report = run_alpha_portfolio_backtest(
        _predictions(prices),
        prices,
        config=PortfolioBacktestConfig(
            top_n=5,
            min_cross_section=20,
            rebalance_every=5,
            transaction_cost_bps=0.0,
        ),
    )
    ridge = report.holdings.loc[report.holdings["strategy"] == "ridge_top_n"]
    total_weight = ridge.groupby("date")["weight"].sum()
    assert np.allclose(total_weight.to_numpy(), 1.0)
    assert report.daily_returns["ridge_top_n_turnover"].iloc[1:5].eq(0.0).all()
    assert (report.daily_returns["ridge_top_n_turnover"] >= 0.0).all()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"top_n": 0}, "top_n"),
        ({"rebalance_every": 0}, "rebalance_every"),
        ({"transaction_cost_bps": -1.0}, "transaction_cost_bps"),
        ({"min_cross_section": 4, "top_n": 5}, "min_cross_section"),
    ],
)
def test_configuration_rejects_invalid_values(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        PortfolioBacktestConfig(**kwargs)


def test_backtest_excludes_wbd_even_if_present_in_prediction_csv() -> None:
    prices = _prices()
    predictions = _predictions(prices)
    extra = predictions.reset_index()[["date"]].drop_duplicates().copy()
    extra["ticker"] = "WBD"
    extra["expected_excess_return"] = 1000.0
    extra = extra.set_index(["date", "ticker"])
    predictions = pd.concat([predictions, extra]).sort_index()
    prices["WBD"] = 10.0

    report = run_alpha_portfolio_backtest(predictions, prices)

    assert "WBD" not in set(report.holdings["ticker"])


def test_backtest_rejects_duplicate_prediction_keys() -> None:
    prices = _prices()
    predictions = _predictions(prices)
    duplicated = pd.concat([predictions, predictions.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        run_alpha_portfolio_backtest(duplicated, prices)


def test_backtest_requires_benchmark_price_column() -> None:
    prices = _prices().drop(columns=["SPY"])
    with pytest.raises(ValueError, match="benchmark"):
        run_alpha_portfolio_backtest(_predictions(_prices()), prices)


def test_signal_is_executed_on_next_session_not_same_close() -> None:
    prices = _prices()
    predictions = _predictions(prices)
    first_signal_date = prices.index[65]
    next_signal_date = prices.index[66]
    first_rows = predictions.loc[first_signal_date]
    predictions.loc[first_signal_date, "expected_excess_return"] = [
        -int(ticker[1:]) for ticker in first_rows.index
    ]
    next_rows = predictions.loc[next_signal_date]
    predictions.loc[next_signal_date, "expected_excess_return"] = [
        int(ticker[1:]) for ticker in next_rows.index
    ]

    report = run_alpha_portfolio_backtest(
        predictions,
        prices,
        config=PortfolioBacktestConfig(
            top_n=5,
            min_cross_section=20,
            rebalance_every=5,
            transaction_cost_bps=0.0,
        ),
    )
    first_execution_date = prices.index[66]
    first_holdings = report.holdings.loc[
        (report.holdings["date"] == first_execution_date)
        & (report.holdings["strategy"] == "ridge_top_n")
    ]

    assert set(first_holdings["ticker"]) == {"T00", "T01", "T02", "T03", "T04"}
    assert set(first_holdings["signal_date"]) == {first_signal_date}


def test_missing_next_session_price_for_held_asset_is_not_silently_zeroed() -> None:
    prices = _prices()
    prices.loc[prices.index[67], "T29"] = np.nan

    with pytest.raises(ValueError, match="missing next-session returns"):
        run_alpha_portfolio_backtest(
            _predictions(prices),
            prices,
            config=PortfolioBacktestConfig(
                top_n=5,
                min_cross_section=20,
                transaction_cost_bps=0.0,
                max_stale_sessions=0,
            ),
        )


def test_max_drawdown_includes_loss_from_initial_cash_value() -> None:
    prices = _prices()
    prices.loc[prices.index[67], "SPY"] = prices.loc[prices.index[66], "SPY"] * 0.9
    report = run_alpha_portfolio_backtest(
        _predictions(prices),
        prices,
        config=PortfolioBacktestConfig(
            top_n=5,
            min_cross_section=20,
            transaction_cost_bps=0.0,
        ),
    )
    benchmark = report.summary.set_index("strategy").loc["benchmark"]

    assert benchmark["max_drawdown"] <= -0.099


def test_backtest_rejects_prediction_ticker_without_price_series() -> None:
    prices = _prices()
    predictions = _predictions(prices)
    with pytest.raises(ValueError, match="missing predicted tickers"):
        run_alpha_portfolio_backtest(predictions, prices.drop(columns=["T29"]))


def test_backtest_rejects_oos_signal_date_missing_from_price_calendar() -> None:
    prices = _prices()
    predictions = _predictions(prices)
    with pytest.raises(ValueError, match="missing OOS signal dates"):
        run_alpha_portfolio_backtest(
            predictions,
            prices.drop(index=prices.index[70]),
        )


def test_single_session_price_gap_can_be_tolerated_when_explicitly_enabled() -> None:
    prices = _prices()
    prices.loc[prices.index[67], "T29"] = np.nan

    report = run_alpha_portfolio_backtest(
        _predictions(prices),
        prices,
        config=PortfolioBacktestConfig(
            top_n=5,
            min_cross_section=20,
            transaction_cost_bps=0.0,
            max_stale_sessions=1,
        ),
    )

    assert report.daily_returns["ridge_top_n_net_return"].notna().all()
    assert report.daily_returns["equal_weight_universe_net_return"].notna().all()
