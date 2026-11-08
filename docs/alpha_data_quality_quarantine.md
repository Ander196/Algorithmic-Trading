# Alpha historical price-quality quarantine

## Purpose

Alpha uses point-in-time S&P 500 constituents and adjusted OHLCV histories.
A corrupted series can affect both model training labels and OOS evaluation.
This document records the separate Alpha-only quality quarantine and the
read-only investigation workflow.

## Current quarantine

- **PARA** is excluded from Alpha research because the stored series contains
  inconsistent price scales and volume observations. The investigated examples
  include November 2023 and June 2025.
- The quarantine is separate from `EXCLUDED_TICKERS`; it does not change
  ingestion, daily allocation, Layer 1/Layer 2, or other operational paths.
- PARA remains in Supabase. This change does not delete, overwrite, or repair
  any stored rows.
- Re-admit PARA only after its complete historical series has been repaired
  from independently validated source data and revalidated.

The Alpha dataset builder and universe discovery honor the quarantine.
Stale OOS prediction files containing PARA are also filtered by Alpha downstream
readers: the adjusted-price exporter, coverage audit, and portfolio backtest.

## CPWR investigation

Earlier Ridge OOS diagnostics reported unusual five-session excess returns
for CPWR around April 2010 and May/June 2011, including observations where
`return_20` and `return_60` were both -90%. These are investigation leads,
not proof that stored prices are wrong.

Run the read-only source comparison locally with configured Supabase
credentials:

```powershell
python -m jobs.audit_alpha_ticker_history --ticker CPWR --output-dir reports/alpha_price_anomalies
```

The command downloads CPWR's historical Yahoo Finance OHLCV and corporate-action
fields, compares them with stored Supabase history, and creates:

- `cpwr_price_history_summary.csv`: overlapping observations, median constant
  scale ratio, scale dispersion, maximum one-observation returns, and count of
  flagged sessions.
- `cpwr_price_history_comparison.csv`: date-level stored/reference OHLCV,
  adjusted closes, returns, volume, Yahoo dividends/splits, and reason flags.

The reported constant scale is diagnostic only. It is never applied to prices.
Large moves, zero-volume sessions, provider disagreement and corporate actions
are flags for review; they do not automatically establish that a record is
invalid. Yahoo Finance is a comparison source, not an authoritative ground
truth. Check flagged periods against an independent source and corporate
actions before changing Supabase.

The audit is read-only: it neither updates Supabase nor generates repair SQL.

## Re-running Ridge

After this quarantine merges, regenerate the complete OOS dataset and all
walk-forward predictions:

```powershell
python -m jobs.evaluate_alpha --diagnostics-dir reports/alpha_diagnostics_ridge_no_para
```

Then use the newly generated prediction file for adjusted-price export and
portfolio backtesting. Do not simply remove PARA rows from an older prediction
CSV: the earlier Ridge models may already have trained on contaminated PARA
labels. Previous OOS reports remain historical diagnostics and are not the
final Ridge decision gate.
