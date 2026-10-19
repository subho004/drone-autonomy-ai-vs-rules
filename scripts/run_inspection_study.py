"""Resolution study: how much image resolution does a vision model need to spot a defect?

Renders seeded stills of a wall panel (about half with a crack and/or corrosion, the rest
clean with distractor marks) at 64, 128 and 224 px, asks each model "Does this panel show a
defect (crack or corrosion)? Answer YES or NO.", and compares accuracy, latency and cost.
A text-only control (a neutral written description instead of the image) isolates what the
image adds to latency. Results go to assets/sims/inspection/study.json plus three charts.

Paid calls are capped in code (COST_CAP_USD, persisted in spend.json across runs). Keys are
read from .env through ``settings`` and only ever sent in request headers. Constants at the
top instead of CLI args; re-running resumes from study.json.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.analysis.stats import wilson
from app.core.logging import configure_logging, get_logger
from app.providers.vision import (
    CostCap,
    CostCapReachedError,
    FatalApiError,
    VisionClient,
)
from app.sim.inspection import (
    DEFECT_SIZES_M,
    IMAGE_SIZES,
    InspectionScene,
    describe_panel,
    draw_structure,
    encode_png,
    sample_pose,
)

BASE_SEED = 4100
N_STRUCTURES = 10
ROWS, COLS = (
    2,
    3,
)  # 6 panels per structure, 3 of them defective: 60 stills, 30 defective
RESOLUTIONS: tuple[int, ...] = IMAGE_SIZES
MODELS: tuple[str, ...] = ("gpt-4o-mini", "gemini-3.5-flash-lite", "gpt-6-luna")
COST_CAP_USD = 1.50
SHUFFLE_SEED = 11
CHECKPOINT_EVERY = 20
SAT_TOLERANCE = 0.05  # "saturated": 224 px gains less than this over 128 px
LATENCY_RISE_MS = (
    20.0  # "keeps rising": p50 at 224 px exceeds p50 at 64 px by this much
)
TEXT_FASTER_MS = 100.0  # the prediction's text-only margin
N_BOOT = 2000
OUT_DIR = ROOT_DIR / "assets" / "sims" / "inspection"
STUDY_JSON = OUT_DIR / "study.json"
SPEND_JSON = OUT_DIR / "spend.json"
COLORS = {
    "gpt-4o-mini": "#0072B2",
    "gemini-3.5-flash-lite": "#D55E00",
    "gpt-6-luna": "#009E73",
}
SMALL_DEFECT_M = 0.12

logger = get_logger("run_inspection_study")


@dataclass(slots=True)
class Still:
    still_id: str
    structure_seed: int
    panel_index: int
    truth: bool
    label: str
    smallest_defect_m: float
    description: str
    pngs: dict[int, bytes] = field(default_factory=dict)


def build_stills() -> tuple[list[Still], dict[int, float]]:
    """60 seeded stills at every resolution, plus mean render ms per resolution."""

    stills: list[Still] = []
    render_ns: dict[int, list[int]] = {r: [] for r in RESOLUTIONS}
    for s in range(N_STRUCTURES):
        seed = BASE_SEED + s
        structure = draw_structure(seed, ROWS, COLS, n_defective=ROWS * COLS // 2)
        scene = InspectionScene(structure)
        for panel in structure.panels:
            still = Still(
                still_id=f"s{seed}p{panel.index}",
                structure_seed=seed,
                panel_index=panel.index,
                truth=panel.has_defect,
                label=panel.label,
                smallest_defect_m=panel.smallest_defect_m,
                description=describe_panel(panel),
            )
            pose = sample_pose(structure, panel.index)
            for size in RESOLUTIONS:
                t0 = time.perf_counter_ns()
                rgb = scene.render_camera(pose, panel.index, size)
                render_ns[size].append(time.perf_counter_ns() - t0)
                still.pngs[size] = encode_png(rgb)
            stills.append(still)
        scene.close()
    render_ms = {r: float(np.mean(v)) / 1e6 for r, v in render_ns.items()}
    logger.info(
        "built %d stills (%d defective); render ms %s",
        len(stills),
        sum(s.truth for s in stills),
        {k: round(v, 1) for k, v in render_ms.items()},
    )
    return stills, render_ms


# ------------------------------------------------------------------------------ calls
def _load_calls() -> list[dict[str, Any]]:
    if STUDY_JSON.exists():
        data = json.loads(STUDY_JSON.read_text())
        return list(data.get("calls", []))
    return []


def _call_key(model: str, kind: str, res: int | None, still_id: str) -> tuple[Any, ...]:
    return (model, kind, res, still_id)


def run_calls(
    client: VisionClient, stills: list[Still], calls: list[dict[str, Any]], save: Any
) -> None:
    done = {_call_key(c["model"], c["kind"], c["res"], c["still"]) for c in calls}
    rng = np.random.default_rng(SHUFFLE_SEED)
    for model in MODELS:
        tasks: list[tuple[str, int | None, Still]] = [
            ("image", r, s) for s in stills for r in RESOLUTIONS
        ] + [("text", None, s) for s in stills]
        if all(_call_key(model, k, r, s.still_id) in done for k, r, s in tasks):
            continue  # nothing left for this model: no warm-up call either
        warm = client.ask(
            model, None, "A plain concrete panel with no marks."
        )  # opens the connection; discarded
        logger.info("%s warm-up %.0f ms (discarded)", model, warm.latency_ms)
        order = rng.permutation(len(tasks))
        for n, idx in enumerate(order):
            kind, res, still = tasks[int(idx)]
            if _call_key(model, kind, res, still.still_id) in done:
                continue
            if kind == "image":
                assert res is not None
                result = client.ask(model, still.pngs[res])
            else:
                result = client.ask(model, None, still.description)
            calls.append(
                {
                    "model": model,
                    "kind": kind,
                    "res": res,
                    "still": still.still_id,
                    "truth": still.truth,
                    "label": still.label,
                    "smallest_defect_m": still.smallest_defect_m,
                    "answer": result.answer,
                    "verdict": result.verdict,
                    "correct": result.verdict is not None
                    and result.verdict == still.truth,
                    "latency_ms": round(result.latency_ms, 1),
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "cost_usd": result.cost_usd,
                    "status": result.status,
                    "error": result.error,
                    "image_bytes": len(still.pngs[res]) if res is not None else 0,
                }
            )
            if (n + 1) % CHECKPOINT_EVERY == 0:
                save()
                logger.info("%s: %d/%d calls", model, n + 1, len(tasks))
        save()


# -------------------------------------------------------------------------- analysis
def _bootstrap_median_diff(
    a: list[float], b: list[float]
) -> tuple[float, float, float]:
    """Median(a) - median(b) with a percentile bootstrap 95% CI (independent samples)."""

    rng = np.random.default_rng(0)
    x, y = np.asarray(a), np.asarray(b)
    diffs = [
        float(np.median(rng.choice(x, x.size)) - np.median(rng.choice(y, y.size)))
        for _ in range(N_BOOT)
    ]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.median(x) - np.median(y)), float(lo), float(hi)


def _cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ok = sum(1 for r in rows if r["correct"])
    lo, hi = wilson(ok, n)
    lat = [r["latency_ms"] for r in rows if r["status"] == 200]
    defective = [r for r in rows if r["truth"]]
    clean = [r for r in rows if not r["truth"]]
    small = [r for r in defective if r["smallest_defect_m"] <= SMALL_DEFECT_M]
    return {
        "n": n,
        "accuracy": ok / n if n else None,
        "ci_lo": lo,
        "ci_hi": hi,
        "recall": _rate(defective),
        "recall_small": _rate(small),
        "n_small": len(small),
        "specificity": _rate(clean, positive=False),
        "invalid": sum(1 for r in rows if r["verdict"] is None),
        "lat_p50_ms": float(np.percentile(lat, 50)) if lat else None,
        "lat_p95_ms": float(np.percentile(lat, 95)) if lat else None,
        "lat_mean_ms": float(np.mean(lat)) if lat else None,
        "tokens_in_mean": float(np.mean([r["tokens_in"] for r in rows]))
        if rows
        else None,
        "cost_per_call_usd": float(np.mean([r["cost_usd"] for r in rows]))
        if rows
        else None,
        "image_bytes_mean": float(np.mean([r["image_bytes"] for r in rows]))
        if rows
        else None,
    }


def _rate(rows: list[dict[str, Any]], positive: bool = True) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if r["verdict"] is positive) / len(rows)


def summarise(calls: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODELS:
        mine = [c for c in calls if c["model"] == model]
        if not mine:
            continue
        cells: dict[str, dict[str, Any]] = {}
        for res in RESOLUTIONS:
            rows = [c for c in mine if c["kind"] == "image" and c["res"] == res]
            if rows:
                cells[str(res)] = _cell(rows)
        text = [c for c in mine if c["kind"] == "text"]
        if text:
            cells["text"] = _cell(text)
        out[model] = cells
    return out


def evaluate_prediction(
    calls: list[dict[str, Any]], summary: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    """Score each clause of the prediction per model, with the numbers behind it."""

    verdicts: dict[str, Any] = {}
    for model, cells in summary.items():
        if not all(str(r) in cells for r in RESOLUTIONS) or "text" not in cells:
            continue
        acc = {r: cells[str(r)]["accuracy"] for r in RESOLUTIONS}
        lo_r, mid_r, hi_r = RESOLUTIONS[0], RESOLUTIONS[1], RESOLUTIONS[-1]
        gain_hi = acc[hi_r] - acc[mid_r]
        gain_lo = acc[mid_r] - acc[lo_r]
        lat = {
            r: [
                c["latency_ms"]
                for c in calls
                if c["model"] == model
                and c["kind"] == "image"
                and c["res"] == r
                and c["status"] == 200
            ]
            for r in RESOLUTIONS
        }
        text_lat = [
            c["latency_ms"]
            for c in calls
            if c["model"] == model and c["kind"] == "text" and c["status"] == 200
        ]
        rise, rise_lo, rise_hi = _bootstrap_median_diff(lat[hi_r], lat[lo_r])
        text_gap = {r: _bootstrap_median_diff(lat[r], text_lat) for r in RESOLUTIONS}
        verdicts[model] = {
            "accuracy": acc,
            "gain_mid_to_hi": gain_hi,
            "gain_lo_to_mid": gain_lo,
            "saturates_by_mid": gain_hi < SAT_TOLERANCE and gain_lo > gain_hi,
            "accuracy_still_rising_at_hi": gain_hi >= SAT_TOLERANCE,
            "latency_rise_p50_ms": {"diff": rise, "ci": [rise_lo, rise_hi]},
            "latency_keeps_rising": rise > LATENCY_RISE_MS and rise_lo > 0,
            "image_minus_text_p50_ms": {
                str(r): {"diff": d, "ci": [lo, hi]}
                for r, (d, lo, hi) in text_gap.items()
            },
            "text_faster_by_margin": all(
                d >= TEXT_FASTER_MS for d, _, _ in text_gap.values()
            ),
        }
    return verdicts


# ---------------------------------------------------------------------------- charts
def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#8a8f98",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e8ec",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
        }
    )


def chart_accuracy(summary: dict[str, dict[str, dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = np.arange(len(RESOLUTIONS))
    models = list(summary)
    for k, model in enumerate(models):
        off = (k - (len(models) - 1) / 2) * 0.07
        cells = [summary[model][str(r)] for r in RESOLUTIONS]
        acc = np.array([c["accuracy"] for c in cells])
        lo = np.array([c["ci_lo"] for c in cells])
        hi = np.array([c["ci_hi"] for c in cells])
        ax.errorbar(
            xs + off,
            acc,
            yerr=[acc - lo, hi - acc],
            color=COLORS[model],
            marker="o",
            capsize=4,
            lw=2,
            label=model,
        )
    ax.axhline(0.5, color="#8a8f98", ls=":", lw=1)
    ax.text(
        len(RESOLUTIONS) - 1.0, 0.51, "chance", color="#8a8f98", ha="right", fontsize=9
    )
    ax.set_xticks(xs, [f"{r}x{r}" for r in RESOLUTIONS])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Accuracy (n = 60 per point)")
    ax.set_xlabel("Image size sent to the model")
    ax.set_title("Defect-spotting accuracy vs resolution (Wilson 95% CI)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "study-accuracy.png", dpi=160)
    plt.close(fig)


def chart_latency(summary: dict[str, dict[str, dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    modes = ["text", *[str(r) for r in RESOLUTIONS]]
    xs = np.arange(len(modes))
    models = list(summary)
    for k, model in enumerate(models):
        off = (k - (len(models) - 1) / 2) * 0.07
        cells = [summary[model].get(m) for m in modes]
        p50 = np.array([c["lat_p50_ms"] if c else np.nan for c in cells])
        p95 = np.array([c["lat_p95_ms"] if c else np.nan for c in cells])
        ax.errorbar(
            xs + off,
            p50,
            yerr=[np.zeros_like(p50), p95 - p50],
            color=COLORS[model],
            marker="o",
            capsize=4,
            lw=2,
            label=model,
        )
    ax.set_xticks(xs, ["text only", *[f"{r}x{r}" for r in RESOLUTIONS]])
    ax.set_ylabel("Round-trip latency, ms (median, whisker to p95)")
    ax.set_xlabel("Input")
    ax.set_title("Latency vs input size")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "study-latency.png", dpi=160)
    plt.close(fig)


def chart_cost(summary: dict[str, dict[str, dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    modes = [str(r) for r in RESOLUTIONS] + ["text"]
    models = list(summary)
    width = 0.8 / len(models)
    xs = np.arange(len(modes))
    for k, model in enumerate(models):
        vals = [
            (summary[model][m]["cost_per_call_usd"] or 0.0) * 1000
            if m in summary[model]
            else 0.0
            for m in modes
        ]
        bars = ax.bar(
            xs + (k - (len(models) - 1) / 2) * width,
            vals,
            width * 0.92,
            color=COLORS[model],
            label=model,
        )
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(xs, [f"{m}x{m}" if m != "text" else "text only" for m in modes])
    ax.set_ylabel("USD per 1,000 calls")
    ax.set_title("Cost per call (from reported token usage)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "study-cost.png", dpi=160)
    plt.close(fig)


def sample_sheet(stills: list[Still]) -> None:
    """One column per still (small, medium, large defect, then two clean), one row per size."""

    picks: list[Still] = []
    for size_m in DEFECT_SIZES_M:
        picks.append(
            next(s for s in stills if s.truth and s.smallest_defect_m == size_m)
        )
    picks += [s for s in stills if not s.truth][:2]
    cell = RESOLUTIONS[-1]
    rows = []
    for res in RESOLUTIONS:
        tiles = [
            np.asarray(
                Image.open(io.BytesIO(s.pngs[res]))
                .convert("RGB")
                .resize((cell, cell), Image.Resampling.NEAREST)
            )
            for s in picks
        ]
        rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    Image.fromarray(sheet).save(OUT_DIR / "study-samples.png")


# ------------------------------------------------------------------------------ main
def main() -> None:
    configure_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = CostCap(COST_CAP_USD, SPEND_JSON)
    logger.info(
        "study start: spent so far $%.4f of $%.2f", cap.spent_usd, cap.limit_usd
    )
    stills, render_ms = build_stills()
    calls = _load_calls()

    def save() -> None:
        summary = summarise(calls)
        STUDY_JSON.write_text(
            json.dumps(
                {
                    "config": {
                        "models": list(MODELS),
                        "resolutions": list(RESOLUTIONS),
                        "n_stills": len(stills),
                        "n_defective": sum(s.truth for s in stills),
                        "base_seed": BASE_SEED,
                        "cost_cap_usd": COST_CAP_USD,
                        "render_ms_mean": render_ms,
                    },
                    "summary": summary,
                    "prediction": evaluate_prediction(calls, summary),
                    "spent_usd": cap.spent_usd,
                    "calls": calls,
                },
                indent=1,
            )
        )

    client = VisionClient(cap)
    stopped: str | None = None
    try:
        run_calls(client, stills, calls, save)
    except (CostCapReachedError, FatalApiError) as exc:
        stopped = f"{type(exc).__name__}: {exc}"
        logger.error("stopped early: %s", stopped)
    finally:
        client.close()
        save()
    summary = summarise(calls)
    sample_sheet(stills)
    _style()
    if summary:
        chart_accuracy(summary)
        chart_latency(summary)
        chart_cost(summary)
    for model, cells in summary.items():
        for mode, c in cells.items():
            logger.info(
                "%-22s %-5s n=%d acc=%.3f [%.2f,%.2f] p50=%.0f p95=%.0f invalid=%d $/call=%.6f",
                model,
                mode,
                c["n"],
                c["accuracy"],
                c["ci_lo"],
                c["ci_hi"],
                c["lat_p50_ms"] or 0,
                c["lat_p95_ms"] or 0,
                c["invalid"],
                c["cost_per_call_usd"],
            )
    logger.info(
        "done in %.0fs, spent $%.4f of $%.2f%s",
        time.perf_counter() - t0,
        cap.spent_usd,
        cap.limit_usd,
        f" (stopped: {stopped})" if stopped else "",
    )


if __name__ == "__main__":
    main()
