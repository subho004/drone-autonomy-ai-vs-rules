"""Render two flights of the two-drone sweep: a collision and a clean pass on the same seed.

Selection rule (stated before looking at the footage): from assets/sims/two-drone/results.json
take BAD_CELL (both drones at 600 ms, the first cell of the grid where the collision rate
reaches its ceiling) and GOOD_CELL (both at 0 ms). Among seeds that collide in BAD_CELL and
pass with at least NEAR_MISS_CLEARANCE_M to spare in GOOD_CELL, render the seed with the
*median* collision time, not the most dramatic one. Both videos therefore show the same
encounter (same lanes, same start offsets), flown with different decision latencies.

Outputs to assets/sims/two-drone/: flight-collision.mp4 (the collision run), flight-clean.mp4,
highlight.gif (from the collision run) and six JPEG screenshots. Needs ffmpeg. Constants at the
top instead of CLI args.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.controllers.arms import ReflexArm
from app.core.logging import configure_logging, get_logger
from app.render.mission_video import FPS, encode_video, make_gif
from app.render.two_drone_video import PairRun, TwoDroneRenderer
from app.sim.two_drone import (
    DRONE_RADIUS,
    DT,
    NEAR_MISS_CLEARANCE_M,
    PairResult,
    TwoDroneSpec,
    TwoDroneWorld,
    fixed_latency_pilot,
    run_pair,
)

logger = get_logger(__name__)

OUT_DIR = ROOT_DIR / "assets" / "sims" / "two-drone"
RESULTS_PATH = OUT_DIR / "results.json"
BAD_CELL_MS = (600.0, 600.0)
GOOD_CELL_MS = (0.0, 0.0)
SENSE_RANGE_M = 2.0
VIDEO_MAX_MB = 5.0
GIF_MAX_MB = 6.0
GIF_WIDTH, GIF_FPS = 560, 8
GIF_LEAD_S, GIF_LENGTH_S = (
    2.0,
    4.6,
)  # video seconds before / around the closest approach
APPROACH_CLEARANCE_M = 5.0
CLOSING_CLEARANCE_M = 1.5
LATERAL_ACTIONS = (1, 2, 7, 8)  # LEFT, RIGHT, AVOID_LEFT, AVOID_RIGHT
SHOT_AFTER_AVOID_S = 0.3
SHOT_AFTER_CLOSEST_S = 0.8
JPEG_QUALITY = 88


def cell_runs(
    data: dict[str, Any], latencies: tuple[float, float]
) -> list[dict[str, Any]]:
    key = f"head_on|own|reflex|{SENSE_RANGE_M:g}|{latencies[0]:g}|{latencies[1]:g}"
    runs: list[dict[str, Any]] = data["cells"][key]["runs"]
    return runs


def select_seed(data: dict[str, Any]) -> tuple[int, int]:
    """Median-collision-time seed among those that collide when late and pass when fast."""

    bad = {r["seed"]: r for r in cell_runs(data, BAD_CELL_MS)}
    good = {r["seed"]: r for r in cell_runs(data, GOOD_CELL_MS)}
    eligible = sorted(
        (bad[s]["t_end"], s)
        for s in bad
        if bad[s]["collision"]
        and good[s]["outcome"] == "success"
        and good[s]["min_clearance_m"] >= NEAR_MISS_CLEARANCE_M
    )
    if not eligible:
        raise RuntimeError(
            "no seed collides in BAD_CELL and passes cleanly in GOOD_CELL"
        )
    seed = eligible[len(eligible) // 2][1]
    logger.info(
        "%d eligible seeds; median collision time -> seed %d", len(eligible), seed
    )
    return seed, len(eligible)


def fly(world: TwoDroneWorld, seed: int, latencies: tuple[float, float]) -> PairRun:
    spec = TwoDroneSpec(seed=seed, sense_range_m=SENSE_RANGE_M)
    pilots = (
        fixed_latency_pilot(ReflexArm(), latencies[0]),
        fixed_latency_pilot(ReflexArm(), latencies[1]),
    )
    res: PairResult = run_pair(world, spec, pilots)
    return PairRun(spec, res.traj, res.decisions, res.metrics, latencies)


def first_time_below(run: PairRun, clearance_m: float) -> float:
    """First sim time the surface clearance is at or below ``clearance_m``."""

    idx = np.nonzero(run.traj["sep"] - 2 * DRONE_RADIUS <= clearance_m)[0]
    return float(run.traj["t"][idx[0]]) if idx.size else run.t_end


def shot_times(run: PairRun, clean: bool) -> dict[str, float]:
    """Sim times of the screenshots, defined by events, not by eyeballing."""

    t_closest = float(run.metrics["min_clearance_t"])
    if clean:
        first_lateral = min(
            (
                r["apply_tick"] * DT
                for rows in run.decisions
                for r in rows
                if r["action"] in LATERAL_ACTIONS
            ),
            default=t_closest - 0.5,
        )
        return {
            "avoid": first_lateral + SHOT_AFTER_AVOID_S,
            "closest": t_closest,
            "pass": t_closest + SHOT_AFTER_CLOSEST_S,
        }
    return {
        "approach": first_time_below(run, APPROACH_CLEARANCE_M),
        "closing": first_time_below(run, CLOSING_CLEARANCE_M),
        "collision": run.t_end,
    }


def render(
    run: PairRun, mp4: Path, shots: dict[str, float], names: dict[str, str]
) -> TwoDroneRenderer:
    """Render ``run`` to ``mp4`` and save the screenshots on the way."""

    renderer = TwoDroneRenderer(run)
    times = renderer.frame_times()
    wanted: dict[int, str] = {}
    for key, t in shots.items():
        idx = next((i for i, ft in enumerate(times) if ft >= t - 1e-6), len(times) - 1)
        wanted[idx] = names[key]

    def stream() -> Iterator[tuple[float, Image.Image]]:
        for i, (t, img) in enumerate(renderer.frames()):
            if i in wanted:
                img.save(OUT_DIR / wanted[i], quality=JPEG_QUALITY)
            yield t, img

    n = encode_video(stream(), mp4, FPS)
    logger.info("%s: %d frames, %.2f MB", mp4.name, n, mp4.stat().st_size / 1e6)
    return renderer


def main() -> None:
    configure_logging()
    data = json.loads(RESULTS_PATH.read_text())
    seed, n_eligible = select_seed(data)
    world = TwoDroneWorld()
    bad_run, good_run = fly(world, seed, BAD_CELL_MS), fly(world, seed, GOOD_CELL_MS)
    world.close()
    logger.info(
        "seed %d: bad cell -> %s (min clearance %.3f), good cell -> %s (min clearance %.2f)",
        seed,
        bad_run.metrics["outcome"],
        bad_run.metrics["min_clearance_m"],
        good_run.metrics["outcome"],
        good_run.metrics["min_clearance_m"],
    )
    recorded = {r["seed"]: r for r in cell_runs(data, BAD_CELL_MS)}[seed]["outcome"]
    if bad_run.metrics["outcome"] != recorded:
        raise RuntimeError(
            "re-flown run differs from results.json (non-deterministic?)"
        )

    bad_shots = shot_times(bad_run, clean=False)
    renderer = render(
        bad_run,
        OUT_DIR / "flight-collision.mp4",
        bad_shots,
        {
            "approach": "shot-1-approach.jpg",
            "closing": "shot-2-closing.jpg",
            "collision": "shot-3-collision.jpg",
        },
    )
    times = renderer.frame_times()
    closest = float(bad_run.metrics["min_clearance_t"])
    idx = next(i for i, t in enumerate(times) if t >= closest)
    renderer.close()
    start_s = max(idx / FPS - GIF_LEAD_S, 0.0)
    make_gif(
        OUT_DIR / "flight-collision.mp4",
        OUT_DIR / "highlight.gif",
        GIF_WIDTH,
        GIF_FPS,
        start_s,
        GIF_LENGTH_S,
    )

    clean_renderer = render(
        good_run,
        OUT_DIR / "flight-clean.mp4",
        shot_times(good_run, clean=True),
        {
            "avoid": "shot-4-clean-avoid.jpg",
            "closest": "shot-5-clean-closest.jpg",
            "pass": "shot-6-clean-pass.jpg",
        },
    )
    clean_renderer.close()

    for path, limit in (
        (OUT_DIR / "flight-collision.mp4", VIDEO_MAX_MB),
        (OUT_DIR / "flight-clean.mp4", VIDEO_MAX_MB),
        (OUT_DIR / "highlight.gif", GIF_MAX_MB),
    ):
        mb = path.stat().st_size / 1e6
        logger.info("%s %.2f MB (limit %.0f)", path.name, mb, limit)
        if mb > limit:
            raise RuntimeError(
                f"{path.name} is {mb:.1f} MB, over the {limit} MB budget"
            )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,pix_fmt",
            "-of",
            "csv=p=0",
            str(OUT_DIR / "flight-clean.mp4"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    logger.info(
        "clean video stream: %s (eligible seeds %d)", probe.stdout.strip(), n_eligible
    )


if __name__ == "__main__":
    main()
