"""Fly the live decision models (Jev, other decision models, OpenAI, Gemini) on the live suite.

Uses your API keys from .env and is hard-capped by BUDGET_USD (the budget ledger stops the
run). Arms whose key is missing are skipped. Latency is each model's own measured wall-clock
time, so run this on an otherwise idle machine. Constants at the top instead of CLI args.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "scripts"))

from run_experiment import run_one

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.schemas.experiment import load_config

CONFIG_PATH = ROOT_DIR / "configs" / "experiments" / "live.yaml"
PROVIDER_KEYS = {
    "decisions_endpoint": "openrouter_api_key",
    "openrouter_chat": "openrouter_api_key",
    "openai_chat": "openai_api_key",
    "gemini": "gemini_api_key",
}


def fly(config_path: Path) -> None:
    """Run ``config_path``, skipping arms whose API key is not configured."""

    logger = get_logger("run_live")
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    kept = []
    for arm in cfg.arms:
        field = PROVIDER_KEYS.get(arm.provider)
        if field is not None and not getattr(settings, field).get_secret_value():
            logger.warning("skipping arm %s: %s is not set", arm.id, field.upper())
            continue
        kept.append(arm)
    cfg.arms = kept
    logger.info("live run: %d arms, budget cap $%.2f", len(kept), settings.budget_usd)
    run_one(cfg)
    logger.info("live run finished in %.1fs", time.perf_counter() - t0)


def main() -> None:
    configure_logging()
    fly(CONFIG_PATH)


if __name__ == "__main__":
    main()
