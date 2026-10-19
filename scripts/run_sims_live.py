"""Fly the live decision models on the Group A tasks (follow, landing, delivery, gates).

Small by design (12 scenarios per task, 4 models) and hard-capped by BUDGET_USD through the
budget ledger. Latency is each model's own measured wall-clock time, so run this on an
otherwise idle machine. Constants at the top instead of CLI args.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "scripts"))

from run_live import fly

from app.core.logging import configure_logging, get_logger

CONFIG_DIR = ROOT_DIR / "configs" / "experiments"
TASKS = ("gates", "follow", "landing", "delivery")


def main() -> None:
    configure_logging()
    logger = get_logger("run_sims_live")
    t0 = time.perf_counter()
    for task in TASKS:
        fly(CONFIG_DIR / f"sim_live_{task}.yaml")
    logger.info("all live task runs finished in %.0fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
