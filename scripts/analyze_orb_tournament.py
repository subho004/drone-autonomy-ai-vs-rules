"""Summarise the Orb Dodge round robin: every dodger against every chaser on the same
randomised scenarios. Writes ``assets/sims/orbs/tournament/results.json`` and
``heatmap.png``. Survival = the drone lands on the pad; a hit = the orb touched the drone.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.analysis.stats import holm_correction, paired_mcnemar_from_outcomes, wilson
from app.core.logging import configure_logging, get_logger

RESULTS = ROOT_DIR / "assets" / "sims" / "orbs" / "results.json"
OUT_DIR = ROOT_DIR / "assets" / "sims" / "orbs" / "tournament"
DODGER_NAMES = {
    "pilot": "rule pilot (own dodge, 0 ms)",
    "navreflex": "rule nav + reflex (0 ms)",
    "mercury": "Mercury Decide + reflex",
    "jev": "Jev 1.13 + reflex",
}
CHASER_NAMES = {
    "pn": "homing (rule)",
    "pursuit": "pursuit (rule)",
    "llm": "gpt-4o-mini",
    "llm-gemini": "Gemini 3.5 Flash-Lite",
}
logger = get_logger("analyze_orb_tournament")


def main() -> None:
    configure_logging()
    rows: list[dict[str, Any]] = json.loads(RESULTS.read_text())["T"]
    cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cell[(r["drone"], r["orb"])].append(r)
    dodgers = [d for d in DODGER_NAMES if any(k[0] == d for k in cell)]
    chasers = [c for c in CHASER_NAMES if any(k[1] == c for k in cell)]
    out: dict[str, Any] = {
        "cells": [],
        "dodger_rank": [],
        "chaser_rank": [],
        "paired": {},
    }
    for (d, c), rs in sorted(cell.items()):
        k = sum(int(r["success"]) for r in rs)
        lo, hi = wilson(k, len(rs))
        out["cells"].append(
            {
                "dodger": d, "chaser": c, "n": len(rs), "survived": k, "lo": lo, "hi": hi,
                "hits": sum(1 for r in rs if r["outcome"] == "collision"),
                "outcomes": dict(Counter(r["outcome"] for r in rs)),
                "dodger_latency_ms": sum(r["drone_latency_ms"] or 0 for r in rs) / len(rs),
                "chaser_latency_ms": (
                    sum(r["orb_latency_ms"] or 0 for r in rs) / len(rs)
                    if any(r["orb_latency_ms"] for r in rs)
                    else None
                ),
                "cost_usd": sum(r["cost_usd"] for r in rs),
            }
        )  # fmt: skip
    for d in dodgers:
        rs = [r for c in chasers for r in cell[(d, c)]]
        k = sum(int(r["success"]) for r in rs)
        out["dodger_rank"].append(
            {"dodger": d, "survived": k, "n": len(rs), "rate": k / len(rs)}
        )
    for c in chasers:
        rs = [r for d in dodgers for r in cell[(d, c)]]
        h = sum(1 for r in rs if r["outcome"] == "collision")
        out["chaser_rank"].append(
            {"chaser": c, "hits": h, "n": len(rs), "rate": h / len(rs)}
        )
    out["dodger_rank"].sort(key=lambda x: -x["rate"])
    out["chaser_rank"].sort(key=lambda x: -x["rate"])

    def pooled(
        who: str, key: str, others: list[str], pred: str
    ) -> dict[str, dict[tuple[str, int], int]]:
        res: dict[str, dict[tuple[str, int], int]] = defaultdict(dict)
        for r in rows:
            val = (
                int(r["success"])
                if pred == "survive"
                else int(r["outcome"] == "collision")
            )
            res[r[key]][(r["orb" if key == "drone" else "drone"], r["seed"])] = val
        return res

    surv = pooled("d", "drone", chasers, "survive")
    hit = pooled("c", "orb", dodgers, "hit")
    for name, table, labels in (("dodgers", surv, dodgers), ("chasers", hit, chasers)):
        comps: list[dict[str, Any]] = []
        for a, b in combinations(labels, 2):
            ids = sorted(set(table[a]) & set(table[b]))
            res = paired_mcnemar_from_outcomes(
                [table[a][i] for i in ids], [table[b][i] for i in ids]
            )
            comps.append(
                {
                    "a": a,
                    "b": b,
                    "n": len(ids),
                    "a_only": res.b,
                    "b_only": res.c,
                    "p": res.p_value,
                }
            )
        for comp, adj in zip(
            comps, holm_correction([c["p"] for c in comps]), strict=True
        ):
            comp["p_holm"] = adj
        out["paired"][name] = comps
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    grid = [
        [
            next(x for x in out["cells"] if x["dodger"] == d and x["chaser"] == c)
            for c in chasers
        ]
        for d in dodgers
    ]
    im = ax.imshow(
        [[g["survived"] / g["n"] for g in row] for row in grid],
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
    )
    ax.set_xticks(
        range(len(chasers)), [CHASER_NAMES[c] for c in chasers], rotation=15, fontsize=8
    )
    ax.set_yticks(range(len(dodgers)), [DODGER_NAMES[d] for d in dodgers], fontsize=8)
    for i, row in enumerate(grid):
        for j, g in enumerate(row):
            ax.text(
                j, i, f"{g['survived']}/{g['n']}", ha="center", va="center", fontsize=10
            )
    ax.set_xlabel("chaser (who steers the orbs)")
    ax.set_title("Dodger survival: lands on the pad", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap.png", dpi=130)
    for g in out["cells"]:
        logger.info(
            "%-9s vs %-10s %2d/%2d survived, %2d hits %s",
            g["dodger"],
            g["chaser"],
            g["survived"],
            g["n"],
            g["hits"],
            g["outcomes"],
        )
    logger.info(
        "dodgers %s", [(x["dodger"], round(x["rate"], 2)) for x in out["dodger_rank"]]
    )
    logger.info(
        "chasers %s", [(x["chaser"], round(x["rate"], 2)) for x in out["chaser_rank"]]
    )


if __name__ == "__main__":
    main()
