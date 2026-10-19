"""Build the static HTML report for one experiment (edit the constants below)."""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.analysis.report import build_report
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.harness.store import connect

EXPERIMENT_NAMES = ("pilot", "accuracy_x_latency", "live")
DB_PATH = ROOT_DIR / settings.db_path
REPORTS_DIR = ROOT_DIR / "reports"

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    start = time.perf_counter()
    logger.info("Report job started: experiments=%s db=%s", EXPERIMENT_NAMES, DB_PATH)
    conn = connect(DB_PATH)
    try:
        for name in EXPERIMENT_NAMES:
            row = conn.execute(
                "SELECT id FROM experiments WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                logger.warning(
                    "experiment '%s' not found in %s; skipping", name, DB_PATH
                )
                continue
            for scope, suffix in (("challenge", ""), ("controls", "-controls")):
                out_path = REPORTS_DIR / f"{name}{suffix}.html"
                if scope == "controls" and not _has_controls(conn, int(row["id"])):
                    continue
                build_report(conn, int(row["id"]), out_path, scope)
    except Exception:
        logger.exception("Report job failed after %.3fs", time.perf_counter() - start)
        raise
    finally:
        conn.close()
    logger.info("Report job completed in %.3fs", time.perf_counter() - start)


def _has_controls(conn: sqlite3.Connection, experiment_id: int) -> bool:
    """True when the experiment flew any impossible-control (B < 0) scenario."""

    row = conn.execute(
        "SELECT 1 FROM flights f JOIN scenarios s ON s.id = f.scenario_id "
        "WHERE f.experiment_id = ? AND s.budget_s < 0 LIMIT 1",
        (experiment_id,),
    ).fetchone()
    return row is not None


if __name__ == "__main__":
    main()
