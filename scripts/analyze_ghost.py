"""A6: summarise the predictive-decision experiments (extrapolating by the expected latency).

Writes ``assets/sims/ghost/results.json`` and ``predictive_vs_raw.png``. Collisions are
counted on obstacle and intruder scenarios; every comparison is paired by scenario id.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.analysis.stats import paired_mcnemar_from_outcomes
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.harness.store import connect

DB_PATH = ROOT_DIR / settings.db_path
OUT_DIR = ROOT_DIR / "assets" / "sims" / "ghost"
EVENTS = ("obstacle_static", "obstacle_approach", "intruder_cross", "pursuit")
PAIRS = (
    ("predict_fsm_200", "fsm", 200),
    ("predict_fsm_400", "fsm", 400),
    ("predict_fsm_700", "fsm", 700),
    ("predict_reflex_ttc_400", "reflex_ttc", 400),
)
logger = get_logger("analyze_ghost")


def load(
    conn: sqlite3.Connection, exp: str
) -> dict[tuple[str, int], dict[str, tuple[int, float]]]:
    q = """
        SELECT f.arm_id, f.latency_ms, f.scenario_id, f.collision, s.budget_s
        FROM flights f JOIN experiments e ON e.id = f.experiment_id
        JOIN scenarios s ON s.id = f.scenario_id
        WHERE e.name = ? AND f.status = 'complete'
          AND s.event_type IN (?, ?, ?, ?)
        """
    out: dict[tuple[str, int], dict[str, tuple[int, float]]] = {}
    for r in conn.execute(q, (exp, *EVENTS)):
        out.setdefault((r["arm_id"], int(r["latency_ms"])), {})[r["scenario_id"]] = (
            int(r["collision"] or 0),
            float(r["budget_s"]),
        )
    return out


def main() -> None:
    configure_logging()
    conn = connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    result: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (key, exp) in zip(
        axes,
        (("constant_velocity", "sim_ghost_cv"), ("steering", "sim_ghost_pursuit")),
        strict=True,
    ):
        table = load(conn, exp)
        rows: list[dict[str, Any]] = []
        for arm, raw, ms in PAIRS:
            a, b = table[(arm, ms)], table[(raw, ms)]
            ids = sorted(set(a) & set(b))
            res = paired_mcnemar_from_outcomes(
                [a[i][0] for i in ids], [b[i][0] for i in ids]
            )
            ka, kb = sum(a[i][0] for i in ids), sum(b[i][0] for i in ids)
            # share of the predictor's collisions that were physically unavoidable (budget < latency)
            hard = sum(1 for i in ids if a[i][0] and a[i][1] < ms / 1000.0)
            rows.append(
                {"arm": arm, "raw": raw, "latency_ms": ms, "n": len(ids),
                 "predictive_collision": ka / len(ids), "raw_collision": kb / len(ids),
                 "predictive_only": res.b, "raw_only": res.c, "p": res.p_value,
                 "predictive_collisions_under_budget": hard, "predictive_collisions": ka}
            )  # fmt: skip
            logger.info("%s %s", key, rows[-1])
        result[key] = rows
        labels = [f"{r['arm'].replace('predict_', '')}" for r in rows]
        xs = range(len(rows))
        ax.bar(
            [x - 0.2 for x in xs],
            [r["raw_collision"] for r in rows],
            0.4,
            label="raw (no prediction)",
        )
        ax.bar(
            [x + 0.2 for x in xs],
            [r["predictive_collision"] for r in rows],
            0.4,
            label="predictive",
        )
        ax.set_xticks(list(xs), labels, rotation=20, fontsize=7)
        ax.set_title(
            {
                "constant_velocity": "constant-velocity hazards",
                "steering": "steering intruder",
            }[key],
            fontsize=9,
        )
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("collision rate")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "predictive_vs_raw.png", dpi=130)
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
