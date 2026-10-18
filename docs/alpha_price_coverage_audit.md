# Alpha historical price coverage audit

Run this before interpreting Alpha portfolio backtest results:

\`\`\`powershell
python -m jobs.audit_alpha_price_coverage \`
  --predictions-csv reports/alpha_diagnostics/alpha_oos_predictions.csv \`
  --output-dir reports/alpha_price_coverage_audit \`
  --benchmark-ticker SPY
\`\`\`

The audit is read-only and checks all tickers in the OOS predictions plus the benchmark over the SPY-session window needed by the backtest: the momentum lookback before the first signal, the OOS signal dates, and two forward sessions after the last signal.

It compares Yahoo Finance price observations with Supabase, checks required OHLCV fields, and estimates a stable scale factor from overlapping adjusted closes. It does not automatically approve a repair if overlap is insufficient or the scale varies by more than the configured tolerance.

## Outputs

- \`price_coverage_summary.csv\`: ticker-level coverage, missing/invalid counts, price-basis calibration and status.
- \`price_coverage_gaps.csv\`: individual absent or invalid observations and whether they need manual review.
- \`price_repair_candidates.csv\`: candidate OHLCV rows and whether the SQL will insert a missing row or update an invalid existing row.
- \`price_repair.sql\`: generated SQL transaction; it is not executed by Python.

Review the reports before copying \`price_repair.sql\` into Supabase SQL Editor. It inserts only absent ticker/date keys and updates only rows that still fail the OHLCV validity checks. It does not overwrite valid existing rows.

The source uses Yahoo's adjusted-close factor to adjust OHLC consistently, then applies a constant per-ticker scale calibrated against overlapping Supabase adjusted closes. Tickers with insufficient overlap or inconsistent price basis are marked for review and excluded from the automatic SQL.

The audit evaluates dates in the SPY-session panel used by the backtest. A foreign market holiday is not treated as a missing database row when Yahoo has no bar for that date. The separate exporter still handles bounded local-price and FX alignment. Review any nonzero provider errors or manual-review rows before applying repairs or interpreting backtest metrics.
