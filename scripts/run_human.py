"""A5: the FSM rule flown with human-like reaction times on the live-model scenarios.

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

CONFIG_PATH = ROOT_DIR / "configs" / "experiments" / "human_baseline.yaml"


def main() -> None:
    configure_logging()
    run_one(load_config(CONFIG_PATH))


if __name__ == "__main__":
    main()
