# Alpha missing historical price coverage audit

## Purpose

The Alpha evaluation uses the point-in-time S&P 500 membership file and loads
historical prices from Supabase. Some former constituents have no usable
stored price history, so they are skipped before dataset construction. This
audit probes whether Yahoo Finance currently exposes observations for those
symbols inside their original membership windows.

The audit is read-only. It does not change Supabase, generate repair SQL, or
automatically map a predecessor ticker to a successor ticker.

## Input

The job expects a CSV with a `ticker` column. For the 8 November 2026
evaluation, the missing-symbol list was exported as `alpha_missing_tickers.csv`
(461 symbols). It was derived from the `Failed tickers:` line in the evaluation
log, not from the OOS predictions file.

## Run

With the project virtual environment and configured Python dependencies:

```powershell
python -m jobs.audit_alpha_missing_history --tickers-csv reports/alpha_missing_tickers.csv --output-dir reports/alpha_missing_history_audit
```

Optional controls:

```powershell
python -m jobs.audit_alpha_missing_history --tickers-csv reports/alpha_missing_tickers.csv --output-dir reports/alpha_missing_history_audit --max-tickers 25
python -m jobs.audit_alpha_missing_history --tickers-csv reports/alpha_missing_tickers.csv --output-dir reports/alpha_missing_history_audit --retries 3 --delay-seconds 0.3
python -m jobs.audit_alpha_missing_history --tickers-csv reports/alpha_missing_tickers.csv --output-dir reports/alpha_missing_history_audit --control-ticker SPY --control-start 2026-10-01 --control-end 2026-10-30
```

### Provider control and errors

Before probing any missing symbol, the job requests a known liquid control
(`AAPL` by default) over 2026-10-01 through 2026-10-30. The control must return
at least one valid price observation. If it fails or returns no prices, the job
aborts before classifying any missing ticker. A control result is saved to
`alpha_missing_history_control.csv`, including the yfinance version and
observed date range.

The Yahoo request uses `raise_errors=True` so yfinance exceptions are surfaced
instead of silently converted into empty DataFrames. The argument is deprecated
in newer yfinance releases in favor of changing the library's global debug
configuration, but remains supported by the versions allowed in this project.
Yahoo "no price data" exceptions and ordinary empty frames are counted as
empty/no-data responses and retried. Network, rate-limit, and other exceptions
are counted separately as provider errors. Provider errors must not be
interpreted as missing security history.

### Historical windows

The default probe is serial to reduce provider throttling. For each S&P
membership interval, Yahoo is queried with that interval's exact start date
and exclusive end date. The query is therefore bounded to the period during
which the canonical ticker belonged to the index. Symbols with multiple
membership periods are probed separately.

For dotted share-class symbols, when the canonical query returns no history,
the job also tries a dot-to-hyphen spelling (for example, `CIT.A` then
`CIT-A`). Existing project aliases in `data.sp500_universe` are respected.
The alternate spelling is a candidate only; it is not adopted automatically.

## Outputs

- `alpha_missing_history_audit.csv`: one row per input ticker, with membership
  windows, curated final trading date when available, original Yahoo symbol,
  returned observation counts/date ranges, empty/no-data responses, provider
  errors, dividend/split event counts, optional share-class alias probe, and a
  triage status.
- `alpha_missing_history_summary.csv`: counts by triage status.
- `alpha_missing_history_control.csv`: control ticker, date window, returned
  sessions/date range, yfinance version, and pass status.

Statuses:

- `primary_history_available_identity_review`: Yahoo returned observations
  under the original/canonical Yahoo symbol inside a membership window.
- `primary_history_available_with_provider_issues`: some observations were
  returned, but at least one window also had empty responses or errors.
- `possible_share_class_alias_identity_review`: the original spelling had
  no observations, but the dot-to-hyphen candidate did.
- `share_class_alias_available_with_provider_issues`: the alias returned some
  observations but also had empty responses or errors.
- `no_history_for_tested_yahoo_symbols`: tested symbols returned no history
  after retries, while the liquid control passed. This is not proof that the
  security did not trade or that no other archive contains its history.
- `provider_errors_no_conclusion`: one or more Yahoo requests failed for
  reasons other than a no-data response; retry rather than classify as absent.
- `not_in_sp500_membership_file`: input ticker not found in the configured
  historical universe.

The `ticker_shape_review` field adds weak triage hints for ticker strings
ending in Q, dotted share classes, and one-character symbols. These are not
issuer classifications and must not be used as automatic exclusion rules.

## Initial structural finding (evaluation 2026-11-08)

The evaluation requested 1,098 tickers and loaded 637, leaving 461 with no
usable Supabase history. Cross-referencing the failure list with
`data/sp500_membership.csv` found:

- all 461 have an entry in the membership file;
- all 461 have a closed final membership interval;
- none has an open final membership interval in the current membership file;
- membership end-year buckets: 112 by 2004; 112 in 2005-2009; 62 in
  2010-2014; 96 in 2015-2019; 61 in 2020-2024; and 18 in 2025-2026;
- 9 of the 461 are represented in the current curated `RETIRED_TICKERS`
  metadata: APC, ATVI, CELG, CTXS, FRC, SBNY, SIVB, TIF and TWTR;
- 6 use a dotted share-class spelling: AFS.A, AZA.A, CIT.A, COC.B, RDS.A and
  TMC.A;
- 30 end in Q. This can be associated with an OTC/distress symbol convention,
  but the suffix alone is not enough to determine bankruptcy or issuer identity.

The first Yahoo probe returned no prices for all 461 canonical symbols and all
six dotted-class alternatives, with no exceptions recorded. That result is
not sufficient to conclude that all 467 symbol spellings lack history: the
first version did not validate access with a liquid control and allowed
yfinance's default behavior of logging some failures and returning empty
DataFrames. Repeat the audit with this control-enabled version before
classifying any provider-coverage gap.

## Review and repair policy

Yahoo coverage is not ground truth. A non-empty response must be independently
matched to issuer and security identity, exchange, effective dates, share
class, corporate actions and adjusted-price basis before any backfill proposal.
A current or successor-company ticker must never be used as a replacement
merely because the businesses are related. Do not copy the successor's prices
back into an earlier security history.

No data repair or new Alpha exclusion should be made solely from the probe
status. Prioritize recent former constituents and share-class candidates,
then manually review no-history results against exchange/issuer records.
