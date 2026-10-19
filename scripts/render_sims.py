"""Render the Group A simulations: a side-by-side comparison video, a highlight GIF and
screenshots per simulation, saved under ``assets/sims/<slug>/``.

Each case compares recorded flights from the experiments on the *same* scenario. The
scenario is the median-ranked one among those where the ``pass_idx`` sides succeeded and the
other sides failed, so it is a typical contrast case rather than the most dramatic one.
Constants at the top; set the environment variable ``RENDER_SLUGS`` (comma separated) to
render a subset.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.harness.store import connect
from app.render.mission_video import FPS, encode_video, make_gif
from app.render.sim_video import SimRenderer, load_sim_run

DB_PATH = ROOT_DIR / settings.db_path
MEDIA_DIR = ROOT_DIR / "assets" / "sims"
FULL = (960, 540)  # each renderer's native size (screenshots)
PANEL_BY_N = {2: (720, 406), 3: (560, 316)}  # per-panel size in the comparison video
GIF_SECONDS = 12.0
GIF_WIDTH_BY_N = {2: 640, 3: 900}
END_HOLD_S = 2.5
OBSTACLE_EVENTS = ("obstacle_static", "obstacle_approach", "intruder_cross")

logger = get_logger("render_sims")


@dataclass(frozen=True)
class Side:
    experiment: str
    arm: str
    latency_ms: float  # -1 = the arm's own measured latency
    shield: int = 0
    label: str = ""


@dataclass(frozen=True)
class Case:
    slug: str
    sides: tuple[Side, ...]
    pass_idx: tuple[int, ...] = (0,)  # sides that must succeed (the rest must fail)
    name: str = "compare"
    block_only: bool = False  # canyon: only scenarios with a block to steer around
    events: tuple[str, ...] = ()  # restrict to these scenario event types
    title: str | None = None
    ghost_ms: float | None = None


CASES: tuple[Case, ...] = (
    Case(
        "follow",
        (
            Side("sim_follow", "follow_pilot", 0),
            Side("sim_follow", "follow_pilot", 1500),
        ),
    ),
    Case(
        "delivery",
        (
            Side("sim_delivery", "zone_pilot", 0, label="zone-aware pilot, 0 ms"),
            Side("sim_delivery", "mission_fsm", 0, label="zone-blind pilot, 0 ms"),
        ),
    ),
    Case(
        "pursuit",
        (Side("sim_pursuit", "oracle", 0), Side("sim_pursuit", "oracle", 300)),
    ),
    Case(
        "pursuit",
        (
            Side(
                "sim_pursuit_approach", "oracle", 300, label="straight intruder, 300 ms"
            ),
            Side("sim_pursuit", "oracle", 300, label="steering intruder, 300 ms"),
        ),
        name="paired",
    ),
    Case(
        "landing",
        (Side("sim_landing", "site_best", 0), Side("sim_landing", "site_best", 1000)),
    ),
    Case(
        "gates",
        (Side("sim_gates", "gate_pilot", 0), Side("sim_gates", "gate_pilot", 200)),
    ),
    Case(
        "canyon",
        (
            Side("sim_canyon", "fsm", 0, label="block ahead, 0 ms"),
            Side("sim_canyon", "fsm", 100, label="block ahead, 100 ms"),
        ),
        block_only=True,
    ),
    Case(
        "human",
        (
            Side("human_baseline", "fsm_machine", 0, label="reflex machine, 0 ms"),
            Side("human_baseline", "fsm_human", 300, label="human-like, median 300 ms"),
            Side("live", "jev", -1, label="Jev, measured ~600 ms"),
        ),
        pass_idx=(0, 1),
        events=OBSTACLE_EVENTS,
        title="HUMAN vs JEV vs REFLEX",
    ),
    Case(
        "ghost",
        (
            Side("sim_ghost_cv", "fsm", 400, label="raw, 400 ms"),
            Side("sim_ghost_cv", "predict_fsm_400", 400, label="predictive, 400 ms"),
        ),
        pass_idx=(1,),
        events=OBSTACLE_EVENTS,
        title="GHOST: WHERE THINGS WILL BE",
        ghost_ms=400.0,
    ),
)


def _flights(conn: sqlite3.Connection, side: Side) -> dict[str, sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT f.id, f.scenario_id, f.success, s.seed, s.spec_json
        FROM flights f JOIN experiments e ON e.id = f.experiment_id
        JOIN scenarios s ON s.id = f.scenario_id
        WHERE e.name = ? AND f.arm_id = ? AND f.latency_ms = ? AND f.shield = ?
          AND f.status = 'complete' AND f.trajectory_uri IS NOT NULL ORDER BY s.seed
        """,
        (side.experiment, side.arm, side.latency_ms, side.shield),
    ).fetchall()
    return {str(r["seed"]): r for r in rows}  # paired suites share seeds, not ids


def pick(conn: sqlite3.Connection, case: Case) -> tuple[list[int], int]:
    """Flight ids (one per side) of the median-ranked contrast scenario, and its seed."""

    per_side = [_flights(conn, side) for side in case.sides]
    shared = sorted(set.intersection(*(set(d) for d in per_side)))
    if case.block_only:
        shared = [s for s in shared if '"obstacle":{' in per_side[0][s]["spec_json"]]
    if case.events:
        shared = [
            s
            for s in shared
            if any(
                f'"event_type":"{e}"' in per_side[0][s]["spec_json"]
                for e in case.events
            )
        ]

    def ok(sid: str, i: int) -> bool:
        return bool(per_side[i][sid]["success"])

    contrast = [
        s
        for s in shared
        if all(ok(s, i) == (i in case.pass_idx) for i in range(len(case.sides)))
    ]
    pool = (
        contrast
        or [s for s in shared if all(ok(s, i) for i in case.pass_idx)]
        or shared
    )
    sid = pool[len(pool) // 2]
    logger.info(
        "%s/%s: %d shared scenarios, %d contrast; using %s",
        case.slug,
        case.name,
        len(shared),
        len(contrast),
        sid,
    )
    return [int(d[sid]["id"]) for d in per_side], int(per_side[0][sid]["seed"])


def _label(side: Side) -> str:
    if side.label:
        return side.label
    return f"injected latency {side.latency_ms:.0f} ms"


def render_case(conn: sqlite3.Connection, case: Case) -> dict[str, object]:
    out_dir = MEDIA_DIR / case.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, seed = pick(conn, case)
    n = len(case.sides)
    panel = PANEL_BY_N[n]
    sims = [
        load_sim_run(conn, fid, ROOT_DIR, _label(side))
        for fid, side in zip(ids, case.sides, strict=True)
    ]
    renderers = [SimRenderer(s, *FULL, title=case.title) for s in sims]
    for r in renderers:
        r.ghost_ms = case.ghost_ms
    total = max(s.run.t_end for s in sims) + END_HOLD_S
    names = [f"side{k + 1}" for k in range(n)]
    shots = {
        (k, f"{case.name}-{names[k]}-{tag}"): round(frac * s.run.t_end * FPS)
        for k, s in enumerate(sims)
        for tag, frac in (("early", 0.25), ("mid", 0.6), ("end", 0.995))
    }

    def frames() -> Iterator[tuple[float, Image.Image]]:
        for f in range(int(total * FPS)):
            canvas = Image.new("RGB", (panel[0] * n, panel[1]))
            for k, (r, s) in enumerate(zip(renderers, sims, strict=True)):
                img = r.frame_at(min(f / FPS, s.run.t_end))
                for (side_k, name), idx in shots.items():
                    if side_k == k and idx == f:
                        img.save(out_dir / f"{name}.jpg", quality=88)
                canvas.paste(
                    img.resize(panel, Image.Resampling.LANCZOS), (panel[0] * k, 0)
                )
            yield f / FPS, canvas

    mp4 = out_dir / f"{case.name}.mp4"
    frames_written = encode_video(frames(), mp4, size=(panel[0] * n, panel[1]))
    for r in renderers:
        r.close()
    make_gif(
        mp4,
        out_dir / f"{case.name}.gif",
        width=GIF_WIDTH_BY_N[n],
        duration_s=GIF_SECONDS,
    )
    meta = {
        "scenario_seed": seed,
        "frames": frames_written,
        "sides": [
            {
                "flight_id": fid,
                "label": _label(side),
                "outcome": s.run.outcome,
                "success": s.run.success,
                "task": s.task,
            }
            for fid, side, s in zip(ids, case.sides, sims, strict=True)
        ],
    }
    (out_dir / f"{case.name}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    configure_logging()
    t0 = time.perf_counter()
    wanted = set(filter(None, os.environ.get("RENDER_SLUGS", "").split(",")))
    conn = connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    for case in CASES:
        if wanted and case.slug not in wanted:
            continue
        t1 = time.perf_counter()
        meta = render_case(conn, case)
        outcomes = [s["outcome"] for s in meta["sides"]]  # type: ignore[attr-defined]
        logger.info(
            "%s/%s: %s in %.0fs",
            case.slug,
            case.name,
            outcomes,
            time.perf_counter() - t1,
        )
    logger.info("done in %.0fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
