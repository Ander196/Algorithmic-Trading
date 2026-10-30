from datetime import date

import pandas as pd
import pytest

from data.sp500_universe import (
    filter_sp500_price_rows,
    get_current_sp500_tickers,
    get_sp500_membership_periods,
    get_sp500_price_periods,
    get_sp500_tickers,
    is_sp500_member,
    load_sp500_membership,
    get_yfinance_symbol,
)


def test_membership_file_has_valid_non_overlapping_intervals():
    periods = load_sp500_membership()
    assert len(periods) > 1_000
    assert len(get_sp500_tickers()) > 1_000
    assert len(get_current_sp500_tickers(date(2026, 10, 30))) == 503


@pytest.mark.parametrize(
    ("ticker", "start", "end"),
    [
        ("APC", "2000-01-01", "2019-08-09"),
        ("SBNY", "2021-12-20", "2023-03-15"),
        ("QRVO", "2015-06-12", "2024-12-23"),
    ],
)
def test_known_membership_windows(ticker, start, end):
    periods = get_sp500_membership_periods(ticker)
    assert len(periods) == 1
    assert periods[0].start_date.isoformat() == start
    assert periods[0].end_date_exclusive.isoformat() == end


def test_repeated_membership_is_preserved():
    # AMD left the index in 2013 and returned in 2017.
    periods = get_sp500_membership_periods("AMD")
    assert len(periods) == 2
    assert periods[0].end_date_exclusive == date(2013, 9, 23)
    assert periods[1].start_date == date(2017, 3, 20)


def test_price_window_uses_original_last_trade_date_not_only_index_removal():
    sbny = get_sp500_price_periods("SBNY")
    assert len(sbny) == 1
    assert sbny[0].end_date_exclusive == date(2023, 3, 11)
    assert not is_sp500_member("SBNY", "2023-03-14")


def test_membership_boundaries_are_start_inclusive_end_exclusive():
    assert is_sp500_member("SBNY", "2021-12-20")
    assert not is_sp500_member("SBNY", "2023-03-15")
    assert not get_sp500_price_periods("SBNY")[0].end_date_exclusive > date(2023, 3, 11)
    assert is_sp500_member("APC", "2019-08-08")
    assert not is_sp500_member("APC", "2019-08-09")


def test_price_filter_keeps_only_constituent_dates_and_reentries():
    frame = pd.DataFrame({
        "price_date": ["2012-01-03", "2014-01-03", "2018-01-03", "2019-01-03"],
        "close": [1, 2, 3, 4],
    })
    filtered = filter_sp500_price_rows(frame, "AMD")
    assert filtered["price_date"].tolist() == ["2012-01-03", "2018-01-03"]


def test_yahoo_share_class_symbols_are_mapped_without_changing_db_ticker():
    assert get_yfinance_symbol("BRK.B") == "BRK-B"
    assert get_yfinance_symbol("BF.B") == "BF-B"
    assert get_yfinance_symbol("AAPL") == "AAPL"
