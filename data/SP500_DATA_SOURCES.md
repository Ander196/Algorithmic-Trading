# S&P 500 point-in-time membership data

This repository file contains one row per uninterrupted S&P 500 membership interval.
`start_date` is inclusive and `end_date_exclusive` is exclusive. A blank end means
the symbol is a constituent in the current snapshot used to build this file.
Multiple rows for one ticker are intentional when it left and later re-entered.

## Sources and update policy

- Historical ticker intervals through 2026-08-18: Farrell J. Aultman's
  [sp500 repository](https://github.com/fja05680/sp500), file
  [sp500_ticker_start_end.csv](https://github.com/fja05680/sp500/blob/master/sp500_ticker_start_end.csv).
  Its README explains that the history was assembled from a historical component
  list and verified index changes, and cautions that early history may be incomplete.
  This project clips the available source to 2000-01-01.
- Changes after 2026-08-18: [chinobing/historical_sp500_constituents](https://github.com/chinobing/historical_sp500_constituents),
  file [sp500_changes_since_1996.csv](https://github.com/chinobing/historical_sp500_constituents/blob/main/sp500_changes_since_1996.csv),
  reconciled against that repository's current-constituents snapshot dated 2026-10-30.

Both upstream repositories publish these source files under the MIT License.
Their attribution and full license texts are retained in `SP500_DATA_SOURCE_LICENSE.txt`
and `SP500_CHANGE_SOURCE_LICENSE.txt`.

For the curated retired symbols in `retired_tickers.py`, the last actual trading
session is applied as an additional limit when it precedes the index removal date.
This is relevant for halted securities such as SBNY.

This is a maintained research dataset, not an official S&P Dow Jones Indices feed.
Ticker identity changes, special share classes and implementation dates must be
reviewed before production use. The file controls what this project attempts to
ingest; it does not guarantee Yahoo Finance still provides valid OHLCV for a
delisted or renamed security.
