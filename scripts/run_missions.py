"""Model audition on multi-task missions (stations, obstacles, landing pad).

Flies ``configs/experiments/mission_audition.yaml`` with the live models under the same
hard USD cap as ``run_live.py``. Constants at the top instead of CLI args.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "scripts"))

from run_live import fly

from app.core.logging import configure_logging

CONFIG_PATH = ROOT_DIR / "configs" / "experiments" / "mission_audition.yaml"


def main() -> None:
    configure_logging()
    fly(CONFIG_PATH)


if __name__ == "__main__":
    main()
