"""A6: do predictive decisions (extrapolating by the expected latency) recover the latency penalty?

No network, no spend. Constants at the top instead of CLI args.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "scripts"))

from run_experiment import run_one

from app.core.logging import configure_logging
from app.schemas.experiment import load_config

CONFIG_NAMES = ("sim_ghost_cv", "sim_ghost_pursuit")


def main() -> None:
    configure_logging()
    for name in CONFIG_NAMES:
        run_one(load_config(ROOT_DIR / "configs" / "experiments" / f"{name}.yaml"))


if __name__ == "__main__":
    main()
