import inspect
import sys
from types import SimpleNamespace

import pytest

import jobs.evaluate_alpha as evaluate_alpha_job
from jobs.evaluate_alpha import AlphaLoadSummary, evaluate_alpha, load_alpha_inputs


def test_alpha_history_api_defaults_to_unlimited_history():
    assert inspect.signature(load_alpha_inputs).parameters["history_limit"].default is None
    assert inspect.signature(evaluate_alpha).parameters["history_limit"].default is None


@pytest.mark.parametrize(
    ("extra_args", "expected_limit"),
    [([], None), (["--history-limit", "1500"], 1500)],
)
def test_alpha_cli_defaults_to_unlimited_history_but_keeps_explicit_cap(
    monkeypatch, extra_args, expected_limit
) -> None:
    captured = {}

    def fake_evaluate_alpha(client, market_ticker, history_limit, max_tickers):
        captured["market_ticker"] = market_ticker
        captured["history_limit"] = history_limit
        captured["max_tickers"] = max_tickers
        return (
            SimpleNamespace(
                fold_count=0,
                mean_ic=0.0,
                median_ic=0.0,
                ic_positive_fraction=0.0,
                folds=[],
            ),
            AlphaLoadSummary(0, 0, ()),
            0,
        )

    monkeypatch.setattr(sys, "argv", ["evaluate_alpha.py", *extra_args])
    monkeypatch.setattr(evaluate_alpha_job, "get_supabase_client", lambda: object())
    monkeypatch.setattr(evaluate_alpha_job, "evaluate_alpha", fake_evaluate_alpha)

    evaluate_alpha_job.main()

    assert captured["history_limit"] == expected_limit
