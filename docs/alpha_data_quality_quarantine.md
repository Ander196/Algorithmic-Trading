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
- An independent identity cross-check also conflicts with the stored series:
  the CPWR comparison starts at an adjusted close of about 29.99 on 2006-12-04
  and ends around 0.346 on 2011-12-30, while Compuware's FY2011 Form 10-K
  reported a total-return index rising from 100 at FY2006 to 147.51 at FY2011.
  The SEC metric includes reinvested dividends and is not a direct daily-close
  series, but its direction is materially inconsistent with the vendor
  adjusted-price path. See
  https://www.sec.gov/Archives/edgar/data/859014/000114036111030346/form10k.htm.
- **EP (El Paso Corporation)** is excluded pending independent reconstruction
  of its historical OHLCV series. The read-only audit covered 3,119 overlapping
  Supabase/Yahoo observations from 2000-01-03 through 2012-05-24. Their
  adjusted-close scale factor was 1.0 (95th-percentile relative dispersion
  approximately 0.000004%), but both sources reported zero volume on 1,250
  sessions (about 40.1%). The audit also flagged 31 observation-to-observation
  adjusted-price moves greater than 50% (29 flagged by both sources and 2 by
  the stored series alone). Provider agreement does not independently validate
  these unusually sparse-volume and discontinuous observations.
- **GR (Goodrich Corporation)** is excluded for the same conservative
  data-quality reason. The audit covered 2,870 overlapping Supabase/Yahoo
  observations from 2001-01-02 through 2012-07-26. Their adjusted-close scale
  factor was 1.0 (95th-percentile relative dispersion approximately
  0.000004%), but both sources reported zero volume on 1,021 sessions (about
  35.6%). Fourteen observation-to-observation adjusted-price moves exceeded
  the 50% review threshold (11 flagged by both sources, 2 by Yahoo alone, and
  1 by the stored series alone). The maximum absolute observation-to-
  observation return was approximately 118.2% in both sources. These flags
  require issuer/exchange-level verification; they are not individually proof
  that each move is erroneous.
- The EP and GR audit compares adjusted OHLCV, not raw exchange closes. The
  quarantine is therefore not based on a direct comparison between adjusted
  closes and unadjusted figures quoted in issuer filings. Re-admission requires
  an independent reconstruction of price basis, trading volume, corporate
  actions, and issuer identity across the complete history.
- This quarantine is separate from `EXCLUDED_TICKERS`; it does not change
  operational ingestion, daily allocation, Layer 1/Layer 2, or any live path.
- PARA, CPWR, EP, and GR remain in Supabase. This change does not delete,
  overwrite, or repair any stored rows.
- Re-admit a symbol only after its complete historical identity, OHLCV,
  corporate actions, and price basis have been independently validated.

References:
- Compuware acquisition completion and cessation of Nasdaq CPWR trading:
  https://www.sec.gov/Archives/edgar/data/859014/000114036114045515/ex99_1.htm
- Ocean Thermal Energy Corp. filing describing its change from TDYS to CPWR
  and first trading date:
  https://www.sec.gov/Archives/edgar/data/827099/000165495419000299/cpwr_posam.htm
- El Paso Corporation historical common-stock filing:
  https://www.sec.gov/Archives/edgar/data/19617/000089109211006036/e45230fwp.htm
- Goodrich Corporation 2009 Annual Report:
  https://media.corporate-ir.net/media_files/irol/60/60759/reports/2009_AR.pdf
- Goodrich merger cash consideration and final option settlement:
  https://infomemo.theocc.com/infomemos?date=201207&number=30993

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

## EP and GR audit commands

Reproduce the read-only checks with configured Supabase credentials:

```powershell
python -m jobs.audit_alpha_ticker_history --ticker EP --output-dir reports/alpha_price_anomalies
python -m jobs.audit_alpha_ticker_history --ticker GR --output-dir reports/alpha_price_anomalies
```

The command creates one summary CSV and one date-level comparison CSV per
ticker. It never modifies Supabase or generates repair SQL. Review these
reports together with independent issuer/exchange evidence before any
re-admission decision.

## Re-running Ridge

After this quarantine merges, regenerate the complete OOS dataset and all
walk-forward predictions:

```powershell
python -m jobs.evaluate_alpha --diagnostics-dir reports/alpha_diagnostics_ridge_clean_universe
```

Then use the newly generated prediction file for adjusted-price export and
portfolio backtesting. Do not simply remove PARA/CPWR rows from older
predictions: old Ridge folds may already have trained on contaminated labels.
The same applies to EP and GR. Previous OOS reports remain historical
diagnostics and are not the final Ridge decision gate.
