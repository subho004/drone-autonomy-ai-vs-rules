"""Run the rule-based latency sweeps of the Group A simulations (no network, no spend).

Constants at the top instead of CLI args. Run: ``.venv/bin/python scripts/run_sims.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "scripts"))

from run_experiment import run_one

from app.core.logging import configure_logging, get_logger
from app.schemas.experiment import load_config

CONFIG_DIR = ROOT_DIR / "configs" / "experiments"
CONFIG_NAMES = (
    "sim_follow",
    "sim_delivery",
    "sim_pursuit",
    "sim_pursuit_approach",
    "sim_landing",
    "sim_gates",
    "sim_canyon",
)


def main() -> None:
    configure_logging()
    logger = get_logger("run_sims")
    t0 = time.perf_counter()
    for name in CONFIG_NAMES:
        run_one(load_config(CONFIG_DIR / f"{name}.yaml"))
    logger.info("all simulation sweeps finished in %.0fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
