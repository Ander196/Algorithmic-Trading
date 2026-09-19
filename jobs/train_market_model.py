#!/usr/bin/env python
"""Weekly training job for the Layer 1 market regime model."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from core.hmm_market import MarketRegimeClassifier
from jobs.common import fetch_price_history, get_supabase_client, utc_now

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _artifact_version(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def main() -> None:
    market_ticker = os.getenv("MARKET_TICKER", "SPY").upper()
    output_dir = Path(os.getenv("MARKET_MODEL_DIR", "models/market"))
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_supabase_client()
    history = fetch_price_history(
        client,
        market_ticker,
        limit=int(os.getenv("MARKET_TRAINING_BARS", "1500")),
    )

    logger.info("Training Layer 1 on %s (%d bars)", market_ticker, len(history))
    model = MarketRegimeClassifier(
        n_init=int(os.getenv("HMM_N_INIT", "10")),
        stability_bars=int(os.getenv("HMM_STABILITY_BARS", "5")),
        flicker_window=int(os.getenv("HMM_FLICKER_WINDOW", "20")),
        flicker_threshold=int(os.getenv("HMM_FLICKER_THRESHOLD", "4")),
    )
    model.fit(history.set_index("price_date"), market_ticker=market_ticker)

    temp_path = output_dir / ".market_regime_training.json"
    model.save_model(str(temp_path))
    payload = json.loads(temp_path.read_text(encoding="utf-8"))

    # The artifact hash is calculated from the complete model payload before
    # adding the hash itself, so the identifier is stable and reproducible.
    version = _artifact_version(payload)
    payload["metadata"]["model_version"] = version

    text = json.dumps(payload, indent=2) + "\n"
    versioned_path = output_dir / f"market_regime_{version}.json"
    current_path = output_dir / "current.json"
    versioned_path.write_text(text, encoding="utf-8")
    current_path.write_text(text, encoding="utf-8")
    temp_path.unlink(missing_ok=True)

    client.table("market_regime_models").upsert(
        {
            "model_version": version,
            "model_type": "MarketRegimeClassifier",
            "schema_version": 1,
            "market_ticker": market_ticker,
            "training_date": model.training_date.isoformat() if model.training_date else utc_now().isoformat(),
            "n_states": model.n_states,
            "bic_score": model.bic_score,
            "artifact": payload,
            "status": "ARCHIVED",
        },
        on_conflict="model_version",
    ).execute()

    client.table("market_regime_models").update({"status": "ARCHIVED"}).eq(
        "market_ticker", market_ticker
    ).neq("model_version", version).execute()
    client.table("market_regime_models").update({"status": "ACTIVE"}).eq(
        "model_version", version
    ).execute()

    logger.info(
        "Layer 1 trained successfully: version=%s states=%d BIC=%.2f",
        version, model.n_states, model.bic_score,
    )


if __name__ == "__main__":
    main()
