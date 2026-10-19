"""Prepare and run an experiment end to end (generate -> enqueue -> run pool -> report).

Constants at the top instead of CLI args. Run: ``uv run python scripts/run_experiment.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.harness.experiment import prepare_experiment
from app.harness.store import Store
from app.harness.supervisor import run_pool
from app.schemas.experiment import ExperimentConfig, load_config
from app.sim.world import DroneWorld

CONFIG_PATHS = (
    ROOT_DIR / "configs" / "experiments" / "pilot.yaml",
    ROOT_DIR / "configs" / "experiments" / "accuracy_x_latency.yaml",
)
DB_PATH = str(ROOT_DIR / settings.db_path)
CPU_WORKERS = 6
API_WORKERS = 4


def run_one(cfg: ExperimentConfig) -> None:
    logger = get_logger("run_experiment")
    t0 = time.perf_counter()
    logger.info("experiment %s (sha %s) starting", cfg.name, cfg.sha())
    store = Store(DB_PATH)
    world = DroneWorld()
    exp_id = prepare_experiment(store, world, cfg)
    world.close()
    store.close()
    run_pool(DB_PATH, "cpu", CPU_WORKERS, exp_id)
    if any(a.pool == "api" for a in cfg.arms):
        run_pool(DB_PATH, "api", API_WORKERS, exp_id)
    logger.info("experiment %s finished in %.1fs", cfg.name, time.perf_counter() - t0)


def main() -> None:
    configure_logging()
    for path in CONFIG_PATHS:
        run_one(load_config(path))


if __name__ == "__main__":
    main()
