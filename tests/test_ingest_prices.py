from jobs import ingest_prices


def test_main_exits_nonzero_when_a_ticker_has_ingestion_error(monkeypatch):
    class FakeClient:
        pass

    monkeypatch.setattr(ingest_prices, "getSupabaseClient", lambda: FakeClient())
    monkeypatch.setattr(ingest_prices, "getActiveTickers", lambda client: ["AAA", "BBB"])
    monkeypatch.setattr(ingest_prices.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ingest_prices,
        "processTicker",
        lambda client, ticker: {
            "ticker": ticker,
            "inserted": 0,
            "error": "database unavailable" if ticker == "AAA" else None,
        },
    )

    try:
        ingest_prices.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("ingest_prices.main() must fail if any ticker fails")
