"""Two-drone airspace sweep: (L_A, L_B) injected latencies, collision heat maps, predictions.

Idea A7 (docs/SIMULATION-IDEAS.md). Two drones fly head-on in a corridor, each with its own
decision arm and its own injected latency. Main grid: ReflexArm on both drones, L in
{0, 100, 300, 600, 1000} ms for each drone, N_SEEDS seeds per cell (the same seeds in every
cell, so cells can be compared pairwise). Supplements (all in the same JSON):

* equal-sum cells: (L, 0), (0, L) and (L/2, L/2), the direct test of "risk depends on L_A+L_B"
* shared handedness: drone B's "left" is the same world side as A's (symmetric-controller case)
* FsmArm on both drones (it can brake and stall, so it can deadlock)
* sense-range sensitivity and a perpendicular-crossing grid

Predictions tested (written before running):
  (a) collision probability is a function of L_A + L_B, so (L, 0) equals (L/2, L/2)
  (b) equal-latency symmetric controllers deadlock or mirror in >= 10% of head-on cases at >= 400 ms

Outputs: assets/sims/two-drone/results.json plus the figures. Constants at the top instead of
CLI args; set REUSE_RESULTS to re-analyse an existing results.json without flying again.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy import stats as sps
from scipy.optimize import minimize
from scipy.special import expit

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.analysis.stats import exact_mcnemar, holm_correction, wilson
from app.controllers.arms import FsmArm, ReflexArm
from app.core.logging import configure_logging, get_logger
from app.sim.two_drone import (
    DRONE_RADIUS,
    Pilot,
    TwoDroneSpec,
    TwoDroneWorld,
    build_layouts,
    fixed_latency_pilot,
    run_pair,
)

logger = get_logger(__name__)

# ------------------------------------------------------------------ constants
OUT_DIR = ROOT_DIR / "assets" / "sims" / "two-drone"
RESULTS_PATH = OUT_DIR / "results.json"
REUSE_RESULTS = False  # True: skip flying, re-analyse the existing results.json
LATENCIES_MS = (0, 100, 300, 600, 1000)
N_SEEDS = 100
SEED_BASE = 1000
SENSE_RANGE_M = 2.0
SENSE_SENSITIVITY_M = (4.0, 10.0)
EQUAL_SUM_TOTALS_MS = (600, 800, 1000, 1200, 1600)
FSM_LATENCIES_MS = (0, 100, 300, 600, 1000)
PERPENDICULAR_LATENCIES_MS = (0, 300, 1000)
N_WORKERS = 6
PREDICTION_B_MIN_LATENCY_MS = 400
PREDICTION_B_RATE = 0.10
ALPHA = 0.05
KEEP = (
    "outcome",
    "collision",
    "near_miss",
    "min_separation_m",
    "min_clearance_m",
    "both_reached",
    "deadlock",
    "dodged_same_way",
    "t_end",
)

Cell = dict[str, Any]


def cell_key(c: Cell) -> str:
    return (
        f"{c['scenario']}|{c['handedness']}|{c['arm']}|{c['sense']:g}|"
        f"{c['la']:g}|{c['lb']:g}"
    )


def make_cell(
    la: float,
    lb: float,
    *,
    scenario: str = "head_on",
    handedness: str = "own",
    arm: str = "reflex",
    sense: float = SENSE_RANGE_M,
) -> Cell:
    return {
        "scenario": scenario,
        "handedness": handedness,
        "arm": arm,
        "sense": sense,
        "la": float(la),
        "lb": float(lb),
    }


def build_blocks() -> dict[str, list[Cell]]:
    """Every block of cells to fly; cells shared between blocks are flown once."""

    grid = [make_cell(a, b) for a in LATENCIES_MS for b in LATENCIES_MS]
    equal_sum: list[Cell] = []
    for total in EQUAL_SUM_TOTALS_MS:
        equal_sum += [
            make_cell(total, 0),
            make_cell(0, total),
            make_cell(total / 2, total / 2),
        ]
    return {
        "grid": grid,
        "equal_sum": equal_sum,
        "shared": [
            make_cell(a, b, handedness="shared")
            for a in LATENCIES_MS
            for b in LATENCIES_MS
        ],
        "fsm_own": [make_cell(v, v, arm="fsm") for v in FSM_LATENCIES_MS],
        "fsm_shared": [
            make_cell(v, v, arm="fsm", handedness="shared") for v in FSM_LATENCIES_MS
        ],
        "sense": [
            make_cell(v, v, sense=s) for s in SENSE_SENSITIVITY_M for v in LATENCIES_MS
        ],
        "perpendicular": [
            make_cell(a, b, scenario="perpendicular")
            for a in PERPENDICULAR_LATENCIES_MS
            for b in PERPENDICULAR_LATENCIES_MS
        ],
    }


# -------------------------------------------------------------------- flying
_WORLD: TwoDroneWorld | None = None


def _pilot(arm: str, latency_ms: float) -> Pilot:
    provider = ReflexArm() if arm == "reflex" else FsmArm()
    return fixed_latency_pilot(provider, latency_ms)


def run_cell(cell: Cell) -> Cell:
    """Fly one (L_A, L_B) cell over all seeds; returns the cell with trimmed per-run rows."""

    global _WORLD
    if _WORLD is None:
        _WORLD = TwoDroneWorld()
    runs: list[dict[str, Any]] = []
    for k in range(N_SEEDS):
        spec = TwoDroneSpec(
            seed=SEED_BASE + k,
            scenario=cell["scenario"],
            handedness=cell["handedness"],
            sense_range_m=cell["sense"],
        )
        res = run_pair(
            _WORLD,
            spec,
            (_pilot(cell["arm"], cell["la"]), _pilot(cell["arm"], cell["lb"])),
        )
        row = {key: res.metrics[key] for key in KEEP}
        row["seed"] = spec.seed
        runs.append(row)
    return {**cell, "runs": runs}


def fly_all(blocks: dict[str, list[Cell]]) -> dict[str, Any]:
    unique: dict[str, Cell] = {}
    for cells in blocks.values():
        for c in cells:
            unique.setdefault(cell_key(c), c)
    logger.info(
        "flying %d cells x %d seeds with %d workers", len(unique), N_SEEDS, N_WORKERS
    )
    t0 = time.perf_counter()
    done: dict[str, Cell] = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS) as pool:
        for i, out in enumerate(
            pool.imap_unordered(run_cell, list(unique.values())), 1
        ):
            done[cell_key(out)] = out
            if i % 10 == 0 or i == len(unique):
                logger.info(
                    "%d/%d cells, %.0f s", i, len(unique), time.perf_counter() - t0
                )
    return {
        "config": {
            "n_seeds": N_SEEDS,
            "seed_base": SEED_BASE,
            "latencies_ms": list(LATENCIES_MS),
            "sense_range_m": SENSE_RANGE_M,
            "drone_radius_m": DRONE_RADIUS,
            "wall_s": time.perf_counter() - t0,
        },
        "blocks": {
            name: [cell_key(c) for c in cells] for name, cells in blocks.items()
        },
        "cells": done,
    }


# ------------------------------------------------------------------ analysis
def count(cell: Cell, field: str) -> int:
    return int(sum(int(r[field]) for r in cell["runs"]))


def rate_ci(k: int, n: int) -> tuple[float, float, float]:
    lo, hi = wilson(k, n)
    return k / n, lo, hi


def get(data: dict[str, Any], **kw: Any) -> Cell:
    return data["cells"][cell_key(make_cell(**kw))]


def success_count(cell: Cell) -> int:
    return int(sum(1 for r in cell["runs"] if r["both_reached"] and not r["collision"]))


def paired_mcnemar(
    a: Cell, b: Cell, field: str = "collision"
) -> tuple[int, int, float]:
    """Exact McNemar on the same seeds: (a only, b only, p)."""

    bb = sum(
        1
        for x, y in zip(a["runs"], b["runs"], strict=True)
        if x[field] and not y[field]
    )
    cc = sum(
        1
        for x, y in zip(a["runs"], b["runs"], strict=True)
        if y[field] and not x[field]
    )
    return bb, cc, exact_mcnemar(bb, cc) if bb + cc else 1.0


def no_reaction_ceiling() -> float:
    """Share of seeds whose lanes are close enough that a no-dodge pass is a contact."""

    n = hits = 0
    for k in range(N_SEEDS):
        a, b = build_layouts(TwoDroneSpec(seed=SEED_BASE + k))
        hits += int(abs(float(b.start[1] - a.start[1])) < 2 * DRONE_RADIUS)
        n += 1
    return hits / n


def analyse_equal_sum(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Prediction (a): (L, 0) and (0, L) versus (L/2, L/2), same seeds."""

    rows: list[dict[str, Any]] = []
    for total in EQUAL_SUM_TOTALS_MS:
        asym = [get(data, la=total, lb=0), get(data, la=0, lb=total)]
        sym = get(data, la=total / 2, lb=total / 2)
        k_asym = sum(count(c, "collision") for c in asym)
        k_sym = count(sym, "collision")
        n_sym = len(sym["runs"])
        _, p_fisher = sps.fisher_exact(
            [[k_asym, 2 * n_sym - k_asym], [k_sym, n_sym - k_sym]]
        )
        b = c = 0
        for a_cell in asym:
            bb, cc, _ = paired_mcnemar(a_cell, sym)
            b, c = b + bb, c + cc
        rows.append(
            {
                "total_ms": total,
                "asym_rate": k_asym / (2 * n_sym),
                "sym_rate": k_sym / n_sym,
                "asym_only": b,
                "sym_only": c,
                "p_mcnemar": exact_mcnemar(b, c) if b + c else 1.0,
                "p_fisher": float(p_fisher),
            }
        )
    adj = holm_correction([r["p_mcnemar"] for r in rows])
    for r, a in zip(rows, adj, strict=True):
        r["p_mcnemar_holm"] = a
    return rows


def _nll(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood of p = ceiling * sigmoid(a + b x)."""

    ceil, a, b = float(expit(theta[0])), theta[1], theta[2]
    p = np.clip(ceil * expit(a + b * x), 1e-9, 1 - 1e-9)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def predictor_fits(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Which function of (L_A, L_B) explains collisions best: sum, min, max or mean-of-two?"""

    keys = set(data["blocks"]["grid"]) | set(data["blocks"]["equal_sum"])
    feats: dict[str, list[float]] = {"sum": [], "min": [], "max": []}
    y: list[float] = []
    for key in sorted(keys):
        cell = data["cells"][key]
        for r in cell["runs"]:
            y.append(float(r["collision"]))
            feats["sum"].append(cell["la"] + cell["lb"])
            feats["min"].append(min(cell["la"], cell["lb"]))
            feats["max"].append(max(cell["la"], cell["lb"]))
    yy = np.array(y)
    out: dict[str, dict[str, float]] = {}
    for name, xs in feats.items():
        x = np.array(xs) / 1000.0
        best = None
        for start in ((0.0, -2.0, 3.0), (-1.0, -4.0, 8.0), (0.5, -1.0, 1.0)):
            res = minimize(_nll, np.array(start), args=(x, yy), method="Nelder-Mead")
            if best is None or res.fun < best.fun:
                best = res
        assert best is not None
        out[name] = {"nll": float(best.fun), "aic": float(2 * best.fun + 6)}
    return out


def equal_latency_rates(data: dict[str, Any], block: str) -> dict[str, Any]:
    """Prediction (b): deadlock and same-way dodge rates, equal-latency cells >= 400 ms."""

    cells = [data["cells"][k] for k in data["blocks"][block]]
    cells += [
        data["cells"][k] for k in data["blocks"].get("equal_sum", []) if block == "grid"
    ]
    seen: set[str] = set()
    pool = []
    for c in cells:
        if (
            c["la"] == c["lb"]
            and c["la"] >= PREDICTION_B_MIN_LATENCY_MS
            and cell_key(c) not in seen
        ):
            seen.add(cell_key(c))
            pool.append(c)
    n = sum(len(c["runs"]) for c in pool)
    res: dict[str, Any] = {"cells_ms": sorted(c["la"] for c in pool), "n": n}
    for field in ("deadlock", "dodged_same_way", "collision"):
        k = sum(count(c, field) for c in pool)
        rate, lo, hi = rate_ci(k, n) if n else (0.0, 0.0, 0.0)
        p = (
            float(sps.binomtest(k, n, PREDICTION_B_RATE, alternative="greater").pvalue)
            if n
            else 1.0
        )
        res[field] = {"k": k, "rate": rate, "ci": [lo, hi], "p_greater_than_10pct": p}
    return res


def analyse(data: dict[str, Any]) -> dict[str, Any]:
    ceiling = no_reaction_ceiling()
    grid_rates = {
        cell_key(make_cell(a, b)): get(data, la=a, lb=b)
        for a in LATENCIES_MS
        for b in LATENCIES_MS
    }
    sym_pairs = []
    for a in LATENCIES_MS:
        for b in LATENCIES_MS:
            if a < b:
                x, y = (
                    grid_rates[cell_key(make_cell(a, b))],
                    grid_rates[cell_key(make_cell(b, a))],
                )
                bb, cc, p = paired_mcnemar(x, y)
                sym_pairs.append({"la": a, "lb": b, "a_only": bb, "b_only": cc, "p": p})
    out: dict[str, Any] = {
        "no_reaction_ceiling": ceiling,
        "equal_sum": analyse_equal_sum(data),
        "predictor_fits": predictor_fits(data),
        "role_symmetry_pairs": sym_pairs,
        "prediction_b": {
            name: equal_latency_rates(data, name)
            for name in ("grid", "shared", "fsm_own", "fsm_shared")
        },
    }
    return out


# ------------------------------------------------------------------- figures
def _matrix(
    data: dict[str, Any], field: str, block: str = "grid", denom: str = "n"
) -> np.ndarray:
    lat = LATENCIES_MS
    m = np.zeros((len(lat), len(lat)))
    hand = "shared" if block == "shared" else "own"
    for i, lb in enumerate(lat):
        for j, la in enumerate(lat):
            cell = get(data, la=la, lb=lb, handedness=hand)
            m[i, j] = count(cell, field) / len(cell["runs"])
    return m


def heatmap(
    data: dict[str, Any],
    field: str,
    block: str,
    title: str,
    path: Path,
    ceiling: float,
) -> None:
    m = _matrix(data, field, block)
    vmax = max(0.4, float(m.max()))
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    fig.patch.set_facecolor("white")
    im = ax.imshow(m, origin="lower", cmap="YlOrRd", norm=Normalize(0.0, vmax))
    n = len(LATENCIES_MS)
    for i in range(n):
        for j in range(n):
            v = m[i, j]
            ax.text(
                j,
                i,
                f"{100 * v:.0f}%",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if v > 0.6 * vmax else "#222222",
                fontweight="bold",
            )
    ax.set_xticks(range(n), [str(v) for v in LATENCIES_MS])
    ax.set_yticks(range(n), [str(v) for v in LATENCIES_MS])
    ax.set_xlabel("Drone A decision latency L_A (ms)")
    ax.set_ylabel("Drone B decision latency L_B (ms)")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("share of runs")
    fig.text(
        0.5,
        0.01,
        f"{N_SEEDS} seeds per cell, ReflexArm on both drones, same seeds in every cell. "
        f"A pass with no dodge touches in {100 * ceiling:.0f}% of seeds (lanes under 0.12 m apart).",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def sum_figure(data: dict[str, Any], analysis: dict[str, Any], path: Path) -> None:
    keys = set(data["blocks"]["grid"]) | set(data["blocks"]["equal_sum"])
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), dpi=150, gridspec_kw={"width_ratios": [1.5, 1]}
    )
    fig.patch.set_facecolor("white")
    sc = None
    for key in sorted(keys):
        c = data["cells"][key]
        total = c["la"] + c["lb"]
        asym = abs(c["la"] - c["lb"]) / total if total else 0.0
        k, n = count(c, "collision"), len(c["runs"])
        r, lo, hi = rate_ci(k, n)
        ax.errorbar(
            total,
            100 * r,
            yerr=[[max(100 * (r - lo), 0.0)], [max(100 * (hi - r), 0.0)]],
            fmt="none",
            ecolor="#bbbbbb",
            zorder=1,
        )
        sc = ax.scatter(
            total,
            100 * r,
            c=[asym],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=60,
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
    ax.set_xlabel("L_A + L_B (ms)")
    ax.set_ylabel("collision rate (%)")
    ax.set_title("Collision rate against total latency", fontsize=12)
    ax.grid(alpha=0.3)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="asymmetry |L_A - L_B| / (L_A + L_B)")
    fits = analysis["predictor_fits"]
    names = ["sum", "min", "max"]
    aics = [fits[n]["aic"] for n in names]
    best = min(aics)
    ax2.bar(names, [a - best for a in aics], color=["#d95f02", "#1b9e77", "#7570b3"])
    ax2.set_ylabel("AIC minus best (lower is better)")
    ax2.set_title("Which function of (L_A, L_B) fits?", fontsize=12)
    ax2.set_xticks(range(3), ["L_A + L_B", "min(L_A, L_B)", "max(L_A, L_B)"])
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# -------------------------------------------------------------------- report
def log_report(data: dict[str, Any], analysis: dict[str, Any]) -> None:
    logger.info(
        "no-reaction ceiling (share of seeds with lanes < 0.12 m): %.2f",
        analysis["no_reaction_ceiling"],
    )
    m = _matrix(data, "collision")
    logger.info("collision rate, rows L_B, cols L_A=%s", list(LATENCIES_MS))
    for i, lb in enumerate(LATENCIES_MS):
        logger.info("L_B=%5d  %s", lb, " ".join(f"{100 * v:5.1f}" for v in m[i]))
    for r in analysis["equal_sum"]:
        logger.info(
            "sum %4d: asym %.2f sym %.2f (asym-only %d, sym-only %d) McNemar p=%.3g Holm p=%.3g Fisher p=%.3g",
            r["total_ms"],
            r["asym_rate"],
            r["sym_rate"],
            r["asym_only"],
            r["sym_only"],
            r["p_mcnemar"],
            r["p_mcnemar_holm"],
            r["p_fisher"],
        )
    sig = [p for p in analysis["role_symmetry_pairs"] if p["p"] < ALPHA]
    logger.info(
        "role symmetry (A,B) vs (B,A): %d of %d pairs differ at p<%.2f",
        len(sig),
        len(analysis["role_symmetry_pairs"]),
        ALPHA,
    )
    for name, f in analysis["predictor_fits"].items():
        logger.info("predictor %s: AIC %.1f", name, f["aic"])
    for name, res in analysis["prediction_b"].items():
        logger.info(
            "prediction b [%s] cells %s n=%d: deadlock %.3f (p>10%%=%.3g), same-way dodge %.3f (p>10%%=%.3g), collision %.3f",
            name,
            res["cells_ms"],
            res["n"],
            res["deadlock"]["rate"],
            res["deadlock"]["p_greater_than_10pct"],
            res["dodged_same_way"]["rate"],
            res["dodged_same_way"]["p_greater_than_10pct"],
            res["collision"]["rate"],
        )


def main() -> None:
    configure_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if REUSE_RESULTS and RESULTS_PATH.exists():
        data = json.loads(RESULTS_PATH.read_text())
        logger.info("re-analysing %s", RESULTS_PATH)
    else:
        data = fly_all(build_blocks())
    analysis = analyse(data)
    data["analysis"] = analysis
    RESULTS_PATH.write_text(json.dumps(data, indent=1))
    heatmap(
        data,
        "collision",
        "grid",
        "Head-on collision rate by decision latency of each drone",
        OUT_DIR / "heatmap-collision.png",
        analysis["no_reaction_ceiling"],
    )
    heatmap(
        data,
        "near_miss",
        "grid",
        "Head-on near-miss rate (clearance under 0.3 m, no contact)",
        OUT_DIR / "heatmap-near-miss.png",
        analysis["no_reaction_ceiling"],
    )
    heatmap(
        data,
        "collision",
        "shared",
        "Collision rate when both drones dodge toward the same world side",
        OUT_DIR / "heatmap-collision-shared.png",
        analysis["no_reaction_ceiling"],
    )
    sum_figure(data, analysis, OUT_DIR / "sum-vs-collision.png")
    log_report(data, analysis)
    logger.info("wrote %s and figures to %s", RESULTS_PATH.name, OUT_DIR)


if __name__ == "__main__":
    main()
