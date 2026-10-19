"""Summarise the live-model runs on the Group A tasks (follow, landing, delivery, gates).

Writes ``assets/sims/<slug>/live.json``. Each model flew the first 12 certified scenarios at
its own measured latency; the rule-based pilot (about 0 ms) is the ceiling. Rates carry
Wilson 95% intervals; the model-vs-pilot comparison is paired by scenario (exact McNemar).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.analysis.stats import paired_mcnemar_from_outcomes, wilson
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.harness.store import connect

DB_PATH = ROOT_DIR / settings.db_path
OUT_DIR = ROOT_DIR / "assets" / "sims"
TASKS: dict[str, tuple[str, str]] = {  # slug -> (experiment, graded metric key)
    "gates": ("sim_live_gates", "stations_scanned"),
    "follow": ("sim_live_follow", "in_band_frac"),
    "landing": ("sim_live_landing", "miss_m"),
    "delivery": ("sim_live_delivery", "max_breach_m"),
}
PILOTS = {"follow_pilot", "site_best", "zone_pilot", "gate_pilot"}
logger = get_logger("analyze_sims_live")


def main() -> None:
    configure_logging()
    conn = connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    for slug, (exp, metric) in TASKS.items():
        rows = conn.execute(
            """
            SELECT f.arm_id, f.shield, f.scenario_id, f.success, f.collision, f.outcome,
                   f.lat_mean_ms, f.cost_usd, f.stations_scanned, f.task_json
            FROM flights f JOIN experiments e ON e.id = f.experiment_id
            WHERE e.name = ? AND f.status = 'complete'
            """,
            (exp,),
        ).fetchall()
        if not rows:
            logger.warning("no live data for %s", slug)
            continue
        cells: dict[tuple[str, int], list[sqlite3.Row]] = {}
        for r in rows:
            cells.setdefault((r["arm_id"], int(r["shield"])), []).append(r)
        pilot = next((c for (a, s), c in cells.items() if a in PILOTS and s == 0), [])
        pilot_ok = {r["scenario_id"]: int(r["success"]) for r in pilot}
        out: list[dict[str, Any]] = []
        for (arm, shield), rs in sorted(cells.items()):
            k = sum(int(r["success"]) for r in rs)
            lo, hi = wilson(k, len(rs))
            vals = [
                float(r["stations_scanned"] or 0)
                if metric == "stations_scanned"
                else float(json.loads(r["task_json"] or "{}").get(metric, 0.0))
                for r in rs
            ]
            outcomes: dict[str, int] = {}
            for r in rs:
                outcomes[str(r["outcome"])] = outcomes.get(str(r["outcome"]), 0) + 1
            cell: dict[str, Any] = {
                "arm": arm,
                "shield": shield,
                "n": len(rs),
                "success": k / len(rs),
                "successes": k,
                "lo": lo,
                "hi": hi,
                "collision": sum(int(r["collision"] or 0) for r in rs) / len(rs),
                "metric": metric,
                "metric_mean": sum(vals) / len(vals),
                "latency_ms": sum(float(r["lat_mean_ms"] or 0.0) for r in rs) / len(rs),
                "cost_usd": sum(float(r["cost_usd"] or 0.0) for r in rs),
                "outcomes": outcomes,
            }
            if arm not in PILOTS and pilot_ok:
                ids = [r["scenario_id"] for r in rs if r["scenario_id"] in pilot_ok]
                res = paired_mcnemar_from_outcomes(
                    [pilot_ok[i] for i in ids],
                    [int(r["success"]) for r in rs if r["scenario_id"] in pilot_ok],
                )
                cell.update(
                    {
                        "pilot_only": res.b,
                        "model_only": res.c,
                        "p_vs_pilot": res.p_value,
                    }
                )
            out.append(cell)
            logger.info(
                "%-8s %-18s sh=%d n=%2d succ=%.2f [%.2f,%.2f] coll=%.2f %s=%.2f lat=%4.0f ms cost=$%.3f %s",
                slug, arm, shield, len(rs), cell["success"], lo, hi, cell["collision"],
                metric, cell["metric_mean"], cell["latency_ms"], cell["cost_usd"], outcomes,
            )  # fmt: skip
        (OUT_DIR / slug).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / slug / "live.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
