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
not feed in-sample model predictions into the backtest. The evaluation job keeps
the established 756-session training window and 126-session test/step cadence,
but now includes a final partial test fold to bring the OOS report up to the
latest available history. Because the target uses a five-session forward return,
the final scored signal dates naturally stop before the last stored price date.

### Adjusted prices

The price file must be **long format** with these required fields:

- `date`
- `ticker`
- `adjusted_close_base`

`adjusted_close_base` must be adjusted for splits and distributions and
converted to a **single base currency** on every date. For example, if the
base currency is USD, a European share price in EUR must also reflect the
EUR/USD movement before its USD portfolio return is computed. Include SPY in
the same file for the benchmark. Align the panel to benchmark sessions: on a
foreign-market holiday, use the last local adjusted close translated at that
day's FX rate. Do not carry forward a stale local price without updating FX.

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
python -m jobs.backtest_alpha_portfolios --predictions-csv reports/alpha_diagnostics/alpha_oos_predictions.csv --prices-csv reports/adjusted_close_prices_base_currency.csv --output-dir reports/alpha_portfolio_backtest --benchmark-ticker SPY --top-n 5 --rebalance-every 5 --transaction-cost-bps 5 --max-stale-sessions 1
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
- `alpha_portfolio_cost_sensitivity.csv`: summary metrics across the
  `--cost-sensitivity-bps` scenarios (default `5,10,25`).

`transaction_cost_bps` is charged against gross traded notional. Turnover is
`sum(abs(new_weight - old_weight))`, so a full rotation from one fully invested
portfolio into a disjoint set has turnover near 2.0 (all positions sold and new
positions bought). This is a simple proportional-cost model; set the bps to a
plausible all-in estimate including commissions, spread and slippage. Check
sensitivity to higher costs before interpreting results.

## Limitations and decision gate

The OOS evaluation job now requests `adj_close`, scales all OHLC fields to
that per-row adjusted basis, and converts supported non-USD listing histories
to USD before Alpha features and excess-return labels are built. This makes the
training target currency-consistent with SPY for supported ticker suffixes.
FX uses daily reference/close data and bounded forward-fill of at most five
ticker sessions; leading and longer FX gaps remain missing rather than being
filled with future quotes. This is still a daily approximation to executable FX.

The current active universe is still applied to historical data, so survivorship
bias is not corrected. The model does not simulate actual order types, partial
fills, market impact, taxes, liquidity constraints or RiskManager limits. Keep
the results research-only and reserve a final untouched holdout before using
these backtest results to make a strategy decision.
The equal-weight universe is restricted to the model's eligible OOS rows; it is
not a point-in-time survivorship-free broad-market index.

Do not merge Alpha into Strategy or trade live based solely on this report.
Compare the strategies over identical dates after costs, inspect robustness
across folds/regimes, and reserve a final untouched holdout before drawing a
conclusion about Alpha's incremental economic value.


## Preparing the input files

Run the OOS evaluation and save its diagnostics. To compare the same currency-adjusted pipeline through a historical signal-date cutoff and the latest available data, use separate output directories:

```powershell
python -m jobs.evaluate_alpha --evaluation-end-date 2026-07-20 --diagnostics-dir reports/alpha_diagnostics_20260720_usd
python -m jobs.evaluate_alpha --diagnostics-dir reports/alpha_diagnostics_20261018_usd
```

The cutoff filters OOS signal dates, not realized label endpoints: five-session target returns may mature after the last signal date, as required for out-of-sample scoring. The two reports use the same point-in-time fold logic; the second adds the latest partial fold and later data.

Then export adjusted close prices from Supabase and convert supported foreign
listings to USD using historical FX closes:

```powershell
python -m jobs.export_alpha_adjusted_prices --predictions-csv reports/alpha_diagnostics/alpha_oos_predictions.csv --output-csv reports/adjusted_close_prices_base_currency.csv --benchmark-ticker SPY
```

Finally, run the portfolio backtest and cost sensitivity (default scenarios are
5, 10 and 25 basis points):

```powershell
python -m jobs.backtest_alpha_portfolios --predictions-csv reports/alpha_diagnostics/alpha_oos_predictions.csv --prices-csv reports/adjusted_close_prices_base_currency.csv --output-dir reports/alpha_portfolio_backtest --benchmark-ticker SPY --top-n 5 --rebalance-every 5 --transaction-cost-bps 5 --cost-sensitivity-bps 5,10,25
```

The price export leaves dates before the first valid FX quote missing rather than
backfilling with future exchange rates. Prolonged internal FX coverage gaps
remain missing in the export, and the Alpha evaluation stops if a prolonged
internal/trailing FX gap would distort its five-session training labels. Both
jobs read Supabase without modifying database records.


### FX data gaps

The exporter requests FX history through only the sessions used by the OOS
backtest. A leading gap or an internal gap longer than the bounded stale-price
allowance is preserved as a missing observation, not filled with a future quote.
The exporter prints a warning listing tickers with incomplete coverage. The
portfolio backtest must be stopped if a selected holding would have a missing
next-session return; do not relax that safety check merely to complete the run.


For a one-session missing daily close, the CLI example opts into at most one
stale benchmark-session per ticker via `--max-stale-sessions 1`. This carries
the last known close for that single session and therefore assigns no new price
move until the next available close. It is a bounded data-quality compromise,
not a substitute for fixing missing source data. Do not increase the limit
without inspecting the affected tickers and dates. A gap longer than the limit
must still stop the backtest if a held asset has no valid next-session return.
