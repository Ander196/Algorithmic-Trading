# Alpha OOS portfolio backtest

## Purpose

This research-only job compares four portfolios over the same out-of-sample
(OOS) window:

- **Ridge Top-N**: the highest `expected_excess_return` in each eligible OOS
  cross-section.
- **Momentum Top-N**: highest trailing 60-session adjusted return.
- **Equal-weight universe**: every eligible ticker in that OOS cross-section.
- **Benchmark**: buy-and-hold of SPY (or the configured benchmark).

Equity portfolios are equal weighted on rebalance dates. Signals are observed at
close t, positions are first set no earlier than close t+1, and returns begin
from that execution close to the next close. That avoids assuming we can trade
at the same closing price needed to calculate the signal. Portfolio weights
drift with asset returns between rebalances.

## Input CSVs

### OOS model scores

Use the CSV exported by the Alpha OOS diagnostics job, such as
`alpha_oos_predictions.csv`, or another OOS-only file with these columns:

- `date`
- `ticker`
- `expected_excess_return`

The predictions must come from chronological held-out/walk-forward folds. Do
not feed in-sample model predictions into the backtest.

### Adjusted prices

The price file must be **long format** with these required fields:

- `date`
- `ticker`
- `adjusted_close_base`

`adjusted_close_base` must be adjusted for splits and distributions and
converted to a **single base currency** on every date. For example, if the
base currency is USD, a European share price in EUR must also reflect the
EUR/USD movement before its USD portfolio return is computed. Include SPY in
the same file for the benchmark.

The price file must include at least 60 trading sessions before the first OOS
signal for the momentum calculation, as well as two sessions after the last
OOS signal for lagged execution and its realized return. Duplicate
(date, ticker) rows are rejected. Missing prices are not forward-filled by
default; a held asset without a valid next-session return causes an error
rather than a fabricated zero return. Optional stale-session filling is
available for controlled diagnostics, but the default should be preferred.

## Example

After producing OOS scores and the base-currency adjusted-price export:

```powershell
python -m jobs.backtest_alpha_portfolios --predictions-csv reports/alpha_diagnostics/alpha_oos_predictions.csv --prices-csv reports/adjusted_close_prices_base_currency.csv --output-dir reports/alpha_portfolio_backtest --benchmark-ticker SPY --top-n 5 --rebalance-every 5 --transaction-cost-bps 5
```

The prices CSV should resemble:

```csv
date,ticker,adjusted_close_base
2025-01-02,SPY,580.20
2025-01-02,AAPL,243.85
2025-01-03,SPY,583.10
2025-01-03,AAPL,245.10
```

These example numbers are illustrative only.

## Outputs

The job creates three local CSV files:

- `alpha_portfolio_summary.csv`: total and annualized return, annualized
  volatility, zero-risk-free-rate Sharpe ratio, maximum drawdown, average
  session turnover, and the sum of daily charged cost rates (not a compounded cost metric).
- `alpha_portfolio_daily_returns.csv`: gross return, net return, transaction
  cost rate, turnover, and equity curve for each portfolio.
- `alpha_portfolio_holdings.csv`: holdings and weights by execution date,
  including the signal date used.

`transaction_cost_bps` is charged against gross traded notional. Turnover is
`sum(abs(new_weight - old_weight))`, so a full rotation from one fully invested
portfolio into a disjoint set has turnover near 2.0 (all positions sold and new
positions bought). This is a simple proportional-cost model; set the bps to a
plausible all-in estimate including commissions, spread and slippage. Check
sensitivity to higher costs before interpreting results.

## Limitations and decision gate

This backtest is only as reliable as the OOS score history and adjusted-price
panel. It does not correct survivorship bias if the scores were generated from
the current active universe, does not model market impact or taxes, and does
not simulate actual order types, partial fills, liquidity or RiskManager limits.
The equal-weight universe is restricted to the model's eligible OOS rows; it is
not a point-in-time survivorship-free broad-market index.

Do not merge Alpha into Strategy or trade live based solely on this report.
Compare the strategies over identical dates after costs, inspect robustness
across folds/regimes, and reserve a final untouched holdout before drawing a
conclusion about Alpha's incremental economic value.
