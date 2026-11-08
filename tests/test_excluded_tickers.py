from data.excluded_tickers import (
    ALPHA_DATA_QUALITY_EXCLUDED_TICKERS,
    EXCLUDED_TICKERS,
    filter_excluded_ticker_rows,
    is_excluded_from_alpha_history,
    is_excluded_ticker,
    normalize_ticker,
)
from data.data_loader import processTicker as process_incremental_ticker
from data.data_loader_historical import processTicker as process_historical_ticker
from data.sp500_universe import get_sp500_tickers, get_sp500_membership_periods
from jobs.evaluate_alpha import get_alpha_universe


EUROPEAN_TICKERS = {
    "ABI.BR", "ACA.PA", "ITX.MC", "UCG.MI",
    "AZN.L", "BATS.L", "BP.L", "BT-A.L", "GSK.L", "HSBA.L",
    "LSEG.L", "NG.L", "PRU.L", "REL.L", "RIO.L", "SHEL.L",
    "ULVR.L", "VOD.L",
}


def test_exclusion_list_is_limited_to_deliberately_unsupported_european_symbols():
    assert EXCLUDED_TICKERS == frozenset(EUROPEAN_TICKERS)


def test_historical_sp500_symbols_qrvo_and_wbd_are_not_excluded():
    historical_universe = set(get_sp500_tickers())
    assert "QRVO" in historical_universe
    assert "WBD" in historical_universe
    assert get_sp500_membership_periods("QRVO")
    assert get_sp500_membership_periods("WBD")
    assert not is_excluded_ticker("QRVO")
    assert not is_excluded_ticker("WBD")


def test_ticker_normalization_is_whitespace_and_case_safe():
    assert normalize_ticker(" abi.br ") == "ABI.BR"
    assert is_excluded_ticker(" abi.br ")
    assert not is_excluded_ticker("AAPL")
    assert not is_excluded_ticker("WBD")


def test_incremental_loader_skips_non_sp500_ticker_before_database_access():
    result = process_incremental_ticker(client=object(), ticker=" ulvr.l ")
    assert result["skipped"] == 1
    assert result["fetched"] == 0
    assert result["inserted"] == 0
    assert result["error"] is None


def test_historical_loader_skips_non_sp500_ticker_before_database_access():
    result = process_historical_ticker(client=object(), ticker=" ulvr.l ")
    assert result["skipped"] == 1
    assert result["fetched"] == 0
    assert result["inserted"] == 0
    assert result["error"] is None


def test_excluded_row_filter_does_not_remove_valid_historical_sp500_symbols():
    import pandas as pd

    raw = pd.DataFrame({
        "ticker": ["aapl", " WBD ", " qrvo ", "ULVR.L", "MSFT"],
        "score": [1, 2, 3, 4, 5],
    })
    filtered, excluded = filter_excluded_ticker_rows(raw)

    assert filtered["ticker"].tolist() == ["AAPL", "WBD", "QRVO", "MSFT"]
    assert excluded == ("ULVR.L",)



def test_alpha_data_quality_symbols_are_quarantined_only_from_alpha_research():
    assert ALPHA_DATA_QUALITY_EXCLUDED_TICKERS == frozenset({"PARA", "CPWR"})
    assert is_excluded_from_alpha_history(" para ")
    assert not is_excluded_ticker("PARA")
    assert is_excluded_from_alpha_history("CPWR")
    assert not is_excluded_ticker("CPWR")


def test_alpha_universe_omits_para_and_cpwr():
    universe = set(get_alpha_universe(client=object(), market_ticker="SPY"))
    assert "PARA" not in universe
    assert "CPWR" not in universe


def test_alpha_row_filter_applies_quality_quarantine_only_when_requested():
    import pandas as pd

    rows = pd.DataFrame({"ticker": ["PARA", "CPWR", "ULVR.L"]})
    default_rows, default_excluded = filter_excluded_ticker_rows(
        rows, include_historical_retired=True
    )
    alpha_rows, alpha_excluded = filter_excluded_ticker_rows(
        rows,
        include_historical_retired=True,
        include_alpha_quality_exclusions=True,
    )

    assert default_rows["ticker"].tolist() == ["PARA", "CPWR"]
    assert default_excluded == ("ULVR.L",)
    assert alpha_rows.empty
    assert alpha_excluded == ("CPWR", "PARA", "ULVR.L")
