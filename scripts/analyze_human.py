"""A5: place a synthetic human on the live-model latency curve.

Compares the FSM rule flown with human-like latency (experiment ``human_baseline``) against
the live models (experiment ``live``) on the same obstacle and intruder scenarios, paired by
scenario id. Writes ``assets/sims/human/results.json`` and ``human_vs_models.png``.
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

from app.analysis.stats import paired_mcnemar_from_outcomes, wilson
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.harness.store import connect

DB_PATH = ROOT_DIR / settings.db_path
OUT_DIR = ROOT_DIR / "assets" / "sims" / "human"
EVENTS = ("obstacle_static", "obstacle_approach", "intruder_cross")
LIVE_ARMS = ("jev", "mercury", "gpt-4o-mini", "gemini-flash-lite", "fsm")
logger = get_logger("analyze_human")


def load(
    conn: sqlite3.Connection, exp: str
) -> dict[tuple[str, str, int, int], dict[str, int]]:
    """(arm, mode, latency_ms, shield) -> {scenario_id: collision}"""

    q = """
        SELECT f.arm_id, f.latency_mode, f.latency_ms, f.shield, f.scenario_id, f.collision,
               f.lat_mean_ms FROM flights f JOIN experiments e ON e.id = f.experiment_id
        JOIN scenarios s ON s.id = f.scenario_id
        WHERE e.name = ? AND f.status = 'complete' AND s.event_type IN (?, ?, ?)
        """
    out: dict[tuple[str, str, int, int], dict[str, int]] = {}
    for r in conn.execute(q, (exp, *EVENTS)):
        key = (r["arm_id"], r["latency_mode"], int(r["latency_ms"]), int(r["shield"]))
        out.setdefault(key, {})[r["scenario_id"]] = int(r["collision"] or 0)
    return out


def main() -> None:
    configure_logging()
    conn = connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    live = load(conn, "live")
    human = load(conn, "human_baseline")
    rows: list[dict[str, Any]] = []
    for src, table in (("live", live), ("human_baseline", human)):
        for (arm, mode, ms, shield), d in sorted(table.items()):
            if src == "live" and arm not in LIVE_ARMS:
                continue
            k, n = sum(d.values()), len(d)
            lo, hi = wilson(k, n)
            rows.append(
                {"source": src, "arm": arm, "mode": mode, "latency_ms": ms, "shield": shield,
                 "n": n, "collision": k / n, "lo": lo, "hi": hi}
            )  # fmt: skip
    jev = {s: d for (a, m, _, s), d in live.items() if a == "jev" and m == "measured"}
    pairs: list[dict[str, Any]] = []
    for (arm, mode, ms, shield), d in sorted(human.items()):
        if shield not in jev:
            continue
        ids = sorted(set(d) & set(jev[shield]))
        res = paired_mcnemar_from_outcomes(
            [d[i] for i in ids], [jev[shield][i] for i in ids]
        )
        pairs.append(
            {"arm": arm, "mode": mode, "latency_ms": ms, "shield": shield, "n": len(ids),
             "human_only_collides": res.b, "jev_only_collides": res.c, "p": res.p_value}
        )  # fmt: skip
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(
        json.dumps({"cells": rows, "vs_jev": pairs}, indent=2)
    )
    for r in rows:
        logger.info(
            "%-15s %-18s %-8s L=%5d sh=%d n=%d coll=%.2f [%.2f,%.2f]",
            r["source"],
            r["arm"],
            r["mode"],
            r["latency_ms"],
            r["shield"],
            r["n"],
            r["collision"],
            r["lo"],
            r["hi"],
        )
    for p in pairs:
        logger.info("vs jev: %s", p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, shield in zip(axes, (0, 1), strict=True):
        pts = [
            r
            for r in rows
            if r["shield"] == shield and r["source"] == "live" and r["arm"] != "fsm"
        ]
        for r in pts:
            ax.errorbar(
                r["latency_ms"]
                if r["mode"] != "measured"
                else _measured(conn, r["arm"]),
                r["collision"],
                yerr=[[r["collision"] - r["lo"]], [r["hi"] - r["collision"]]],
                fmt="s",
                label=r["arm"],
            )
        for arm, marker in (("fsm_machine", "o"), ("fsm_human", "^")):
            cs = sorted(
                (r for r in rows if r["arm"] == arm and r["shield"] == shield),
                key=lambda r: r["latency_ms"],
            )
            ax.plot(
                [c["latency_ms"] for c in cs],
                [c["collision"] for c in cs],
                marker=marker,
                label={
                    "fsm_machine": "FSM rule, fixed latency",
                    "fsm_human": "FSM rule, human-like latency (median)",
                }[arm],
            )
        ax.set(
            title=f"reflex shield {'on' if shield else 'off'}",
            xlabel="decision latency (ms)",
        )
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("collision rate")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "human_vs_models.png", dpi=130)


def _measured(conn: sqlite3.Connection, arm: str) -> float:
    r = conn.execute(
        "SELECT AVG(f.lat_mean_ms) FROM flights f JOIN experiments e ON e.id=f.experiment_id "
        "WHERE e.name='live' AND f.arm_id=? AND f.latency_mode='measured' AND f.lat_mean_ms>0",
        (arm,),
    ).fetchone()
    return float(r[0] or 0.0)


if __name__ == "__main__":
    main()
