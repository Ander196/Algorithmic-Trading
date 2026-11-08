# Alpha historical price-quality quarantine

## Purpose

Alpha uses point-in-time S&P 500 constituents and adjusted OHLCV histories.
A corrupted or misidentified series can affect both model training labels and
OOS evaluation. This document records the Alpha-only quarantine and the
read-only investigation workflow.

## Current quarantine

- **PARA** is excluded from Alpha research because the stored history contains
  inconsistent price scales and volume observations, including the investigated
  November 2023 and June 2025 periods.
- **CPWR** is excluded from Alpha research because its history is not reliable
  enough to establish an investable historical series. The 2006-2011 comparison
  showed 1,279 overlapping rows and an almost exact Supabase/Yahoo adjusted
  close match (median scale approximately 1.0), but 973 sessions had zero
  volume in both sources and 29/30 large observation-to-observation returns
  were reproduced by both sources. Agreement between providers therefore does
  not establish correctness.
- There is also a ticker-identity hazard. Compuware Corporation's CPWR stock
  ceased trading on Nasdaq when its take-private acquisition completed on
  2014-12-15. A different issuer later adopted CPWR from TDYS, beginning to
  trade under CPWR on 2017-06-21. Provider history keyed only by ticker can
  therefore join records from different issuers unless security identity and
  effective dates are handled explicitly.
- This quarantine is separate from `EXCLUDED_TICKERS`; it does not change
  operational ingestion, daily allocation, Layer 1/Layer 2, or any live path.
- PARA and CPWR remain in Supabase. This change does not delete, overwrite, or
  repair any stored rows.
- Re-admit either symbol only after its complete historical identity, OHLCV,
  corporate actions, and price basis have been independently validated.

References:
- Compuware acquisition completion and cessation of Nasdaq CPWR trading:
  https://www.sec.gov/Archives/edgar/data/859014/000114036114045515/ex99_1.htm
- Ocean Thermal Energy Corp. filing describing its change from TDYS to CPWR
  and first trading date:
  https://www.sec.gov/Archives/edgar/data/827099/000165495419000299/cpwr_posam.htm

The Alpha dataset builder and universe discovery honor the quarantine.
Stale OOS predictions are also filtered by Alpha downstream readers: the
adjusted-price exporter, price-coverage audit, and portfolio backtest.

## CPWR investigation

Earlier Ridge OOS diagnostics reported exceptionally large five-session
CPWR excess-return labels around April 2010 and May/June 2011. The audit
compared all 1,279 stored rows (2006-12-04 through 2011-12-30) to Yahoo
Finance:

- all 1,279 sessions overlap;
- median stored/reference adjusted-close ratio is approximately 1.0;
- 95th percentile dispersion around that constant ratio is approximately
  0.004%;
- both sources reproduce large one-observation returns, up to approximately
  +376%;
- both sources report zero volume on 973 sessions.

These observations do not prove that every CPWR price is wrong. They do show
that provider agreement is not independent validation, and that a ticker-only
history is unsafe where a symbol has been reused. CPWR is therefore excluded
from Alpha pending an issuer-identity-aware reconstruction and independent
verification of the historical series.

The read-only audit remains available to reproduce and inspect the comparison.
Run locally with configured Supabase credentials:

```powershell
python -m jobs.audit_alpha_ticker_history --ticker CPWR --output-dir reports/alpha_price_anomalies
```

The command creates:

- `cpwr_price_history_summary.csv`: overlap, scale dispersion, maximum
  observation-to-observation returns and flagged-session count.
- `cpwr_price_history_comparison.csv`: date-level stored/reference OHLCV,
  adjusted closes, returns, volume, Yahoo dividends/splits and reason flags.

The median scale is diagnostic only; it is never applied to prices. Yahoo
Finance is a comparison source, not authoritative ground truth. The audit
does not modify Supabase or generate repair SQL.

## Re-running Ridge

After this quarantine merges, regenerate the complete OOS dataset and all
walk-forward predictions:

```powershell
python -m jobs.evaluate_alpha --diagnostics-dir reports/alpha_diagnostics_ridge_clean_universe
```

Then use the newly generated prediction file for adjusted-price export and
portfolio backtesting. Do not simply remove PARA/CPWR rows from older
predictions: old Ridge folds may already have trained on contaminated labels.
Previous OOS reports remain historical diagnostics and are not the final Ridge
decision gate.
