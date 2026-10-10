from data.excluded_tickers import EXCLUDED_TICKERS, is_excluded_ticker, normalize_ticker
from data.data_loader import processTicker as process_incremental_ticker
from data.data_loader_historical import processTicker as process_historical_ticker
from data import ticker_loader


def test_exclusion_list_contains_all_18_confirmed_bad_tickers():
    expected = {
        "ABI.BR", "ACA.PA", "ITX.MC", "UCG.MI",
        "AZN.L", "BATS.L", "BP.L", "BT-A.L", "GSK.L", "HSBA.L",
        "LSEG.L", "NG.L", "PRU.L", "REL.L", "RIO.L", "SHEL.L",
        "ULVR.L", "VOD.L",
    }
    assert EXCLUDED_TICKERS == frozenset(expected)


def test_ticker_normalization_is_whitespace_and_case_safe():
    assert normalize_ticker(" abi.br ") == "ABI.BR"
    assert is_excluded_ticker(" abi.br ")
    assert not is_excluded_ticker("AAPL")


def test_incremental_loader_skips_excluded_ticker_before_database_access():
    result = process_incremental_ticker(client=object(), ticker=" abi.br ")

    assert result["skipped"] == 1
    assert result["fetched"] == 0
    assert result["inserted"] == 0
    assert result["error"] is None


def test_historical_loader_skips_excluded_ticker_before_database_access():
    result = process_historical_ticker(client=object(), ticker=" abi.br ")

    assert result["skipped"] == 1
    assert result["fetched"] == 0
    assert result["inserted"] == 0
    assert result["error"] is None

def test_ticker_catalog_fetch_filters_excluded_symbols(monkeypatch):
    looked_up = []

    def fake_fetch(ticker):
        looked_up.append(ticker)
        return {
            "ticker": ticker,
            "name": ticker,
            "exchange": "TEST",
            "sector": "TEST",
            "industry": "TEST",
        }

    monkeypatch.setattr(ticker_loader, "fetchTickerInfo", fake_fetch)

    frame = ticker_loader.fetchAllTickers(
        [" abi.br ", "AAPL", "ULVR.L"],
        batchSize=50,
        delay=0,
    )

    assert looked_up == ["AAPL"]
    assert frame["ticker"].tolist() == ["AAPL"]
