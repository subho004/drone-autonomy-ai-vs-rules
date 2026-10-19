"""Summarise the rule-based latency sweeps of the Group A simulations.

Writes ``assets/sims/<slug>/results.json`` and ``latency_curve.png`` for each simulation.
Constants at the top instead of CLI args.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
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
OUT_DIR = ROOT_DIR / "assets" / "sims"
# slug -> (experiment names, graded metric key inside task_json, axis label)
SIMS: dict[str, tuple[tuple[str, ...], str | None, str]] = {
    "follow": (
        ("sim_follow",),
        "in_band_frac",
        "fraction of time in the allowed range",
    ),
    "delivery": (("sim_delivery",), "max_breach_m", "deepest no-fly-zone breach (m)"),
    "pursuit": (("sim_pursuit", "sim_pursuit_approach"), None, ""),
    "landing": (("sim_landing",), "miss_m", "distance from the chosen site edge (m)"),
    "gates": (("sim_gates",), "stations_scanned", "gates passed (of 5)"),
    "canyon": (("sim_canyon",), None, ""),
}
logger = get_logger("analyze_sims")


def cells(
    conn: sqlite3.Connection, exp: str, metric: str | None
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT f.arm_id, f.shield, f.latency_ms, f.success, f.collision, f.stations_scanned,
               f.task_json, f.scenario_id, s.spec_json
        FROM flights f JOIN experiments e ON e.id = f.experiment_id
        JOIN scenarios s ON s.id = f.scenario_id
        WHERE e.name = ? AND f.status = 'complete'
        """,
        (exp,),
    ).fetchall()
    groups: dict[tuple[str, int, float, str], list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        walls = '"boxes":[]' not in r["spec_json"] and '"boxes"' in r["spec_json"]
        variant = "block" if (walls and '"obstacle":{' in r["spec_json"]) else ""
        groups[(r["arm_id"], r["shield"], r["latency_ms"], variant)].append(r)
    out = []
    for (arm, shield, ms, variant), rs in sorted(groups.items()):
        k = sum(int(r["success"]) for r in rs)
        lo, hi = wilson(k, len(rs))
        vals = []
        for r in rs:
            if metric == "stations_scanned":
                vals.append(float(r["stations_scanned"] or 0))
            elif metric:
                vals.append(float(json.loads(r["task_json"] or "{}").get(metric, 0.0)))
        out.append(
            {
                "experiment": exp,
                "arm": arm,
                "shield": shield,
                "latency_ms": ms,
                "variant": variant,
                "n": len(rs),
                "success": k / len(rs),
                "success_lo": lo,
                "success_hi": hi,
                "collision": sum(int(r["collision"] or 0) for r in rs) / len(rs),
                "metric_mean": sum(vals) / len(vals) if vals else None,
            }
        )
    return out


# (slug, label, (experiment, arm, shield), (experiment, arm, shield), paired on "seed" or "id")
PAIRED: tuple[tuple[str, str, tuple[str, str, int], tuple[str, str, int]], ...] = (
    ("landing", "site_best vs site_nearest", ("sim_landing", "site_best", 0), ("sim_landing", "site_nearest", 0)),
    ("pursuit", "oracle: straight vs steering intruder", ("sim_pursuit_approach", "oracle", 0), ("sim_pursuit", "oracle", 0)),
    ("pursuit", "fsm: straight vs steering intruder", ("sim_pursuit_approach", "fsm", 0), ("sim_pursuit", "fsm", 0)),
    ("delivery", "zone_pilot: geofence off vs on", ("sim_delivery", "zone_pilot", 0), ("sim_delivery", "zone_pilot", 1)),
)  # fmt: skip


def paired_rows(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    """Per latency: paired exact McNemar of success for two cells on the same seeds."""

    def load(exp: str, arm: str, shield: int) -> dict[float, dict[int, int]]:
        out: dict[float, dict[int, int]] = defaultdict(dict)
        for r in conn.execute(
            "SELECT f.latency_ms, s.seed, f.success FROM flights f JOIN experiments e ON "
            "e.id = f.experiment_id JOIN scenarios s ON s.id = f.scenario_id WHERE e.name = ? "
            "AND f.arm_id = ? AND f.shield = ? AND f.status = 'complete' AND f.latency_ms >= 0",
            (exp, arm, shield),
        ):
            out[float(r["latency_ms"])][int(r["seed"])] = int(r["success"])
        return out

    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slug, label, a, b in PAIRED:
        da, db = load(*a), load(*b)
        for ms in sorted(set(da) & set(db)):
            seeds = sorted(set(da[ms]) & set(db[ms]))
            res = paired_mcnemar_from_outcomes(
                [da[ms][i] for i in seeds], [db[ms][i] for i in seeds]
            )
            result[slug].append(
                {"comparison": label, "latency_ms": ms, "n": len(seeds), "first_only": res.b, "second_only": res.c, "p": res.p_value}
            )  # fmt: skip
    return result


def plot(slug: str, data: list[dict[str, Any]], metric: str | None, label: str) -> None:
    series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in data:
        key = (
            f"{c['experiment'].replace('sim_', '')}: {c['arm']}"
            + (" (shield/geofence)" if c["shield"] else "")
            + (f" [{c['variant']}]" if c["variant"] else "")
        )
        series[key].append(c)
    panels = 2 if metric else 1
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 3.6), squeeze=False)
    for key, cs in series.items():
        cs = sorted(cs, key=lambda c: c["latency_ms"])
        xs = [c["latency_ms"] for c in cs]
        axes[0][0].plot(xs, [c["success"] for c in cs], marker="o", label=key)
        if metric:
            axes[0][1].plot(xs, [c["metric_mean"] for c in cs], marker="o", label=key)
    axes[0][0].set(
        xlabel="injected decision latency (ms)",
        ylabel="success rate",
        ylim=(-0.03, 1.03),
    )
    axes[0][0].grid(alpha=0.3)
    axes[0][0].legend(fontsize=6)
    if metric:
        axes[0][1].set(xlabel="injected decision latency (ms)", ylabel=label)
        axes[0][1].grid(alpha=0.3)
    fig.tight_layout()
    (OUT_DIR / slug).mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / slug / "latency_curve.png", dpi=130)
    plt.close(fig)


def main() -> None:
    configure_logging()
    conn = connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    paired = paired_rows(conn)
    for slug, (exps, metric, label) in SIMS.items():
        data: list[dict[str, Any]] = []
        for exp in exps:
            data += [c for c in cells(conn, exp, metric) if c["latency_ms"] >= 0]
        if not data:
            logger.warning("no data for %s", slug)
            continue
        (OUT_DIR / slug).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / slug / "results.json").write_text(
            json.dumps({"cells": data, "paired": paired.get(slug, [])}, indent=2)
        )
        plot(slug, data, metric, label)
        logger.info("%s: %d cells", slug, len(data))
        for c in data:
            logger.info(
                "  %-22s %-14s sh=%d var=%-5s L=%5.0f n=%2d succ=%.2f coll=%.2f metric=%s",
                c["experiment"],
                c["arm"],
                c["shield"],
                c["variant"],
                c["latency_ms"],
                c["n"],
                c["success"],
                c["collision"],
                "-" if c["metric_mean"] is None else f"{c['metric_mean']:.2f}",
            )


if __name__ == "__main__":
    main()
