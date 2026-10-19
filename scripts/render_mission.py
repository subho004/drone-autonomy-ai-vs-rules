"""Pick the best decision model from the mission audition and render its flight.

Selection (printed and saved to assets/mission/selection.json): among live models, the
(model, shield) cell with the highest mission success rate (all stations scanned and landed
on the pad), then most stations scanned, then fewest collisions. The rendered flight is the
cell's *median-duration successful* run, not its best one. Outputs 3D screenshots and a
video to assets/mission/<winner|jev>/. Constants at the top instead of CLI args.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.harness.store import connect
from app.render.mission_video import (
    FPS,
    MissionRenderer,
    MissionRun,
    encode_video,
    load_run,
    make_gif,
)

EXPERIMENT = "mission_audition"
FLIGHT_ID: int | None = (
    None  # set to render a specific flight instead of the selected one
)
MEDIA_DIR = ROOT_DIR / "assets" / "mission"  # one subfolder per rendered flight
DB_PATH = ROOT_DIR / settings.db_path
# Rendered videos: the audition winner, plus Jev (the model this project is about).
EXTRA_MODELS: tuple[tuple[str, int], ...] = (("jev", 1),)
SHOT_LEADS_S = {"scan": 0.9, "obstacle": 1.4}
GIF_LEAD_S = 1.5
GIF_SECONDS = 14.0
LIVE_ONLY = True  # candidates are the live models, not the rule-based reference

logger = get_logger("render_mission")


def audition_table(conn: sqlite3.Connection) -> list[dict[str, float | int | str]]:
    rows = conn.execute(
        """
        SELECT f.arm_id, f.shield, a.provider, COUNT(*) AS n, AVG(f.success) AS success,
               AVG(f.stations_scanned) AS scanned, AVG(f.collision) AS collision,
               AVG(f.lat_mean_ms) AS lat_ms, AVG(f.t_end) AS t_end
        FROM flights f JOIN arms a ON a.id = f.arm_id JOIN experiments e ON e.id = f.experiment_id
        WHERE e.name = ? AND f.status = 'complete'
        GROUP BY f.arm_id, f.shield ORDER BY success DESC, scanned DESC, collision ASC
        """,
        (EXPERIMENT,),
    ).fetchall()
    return [dict(r) for r in rows]


def median_success_flight(conn: sqlite3.Connection, arm_id: str, shield: int) -> int:
    """The median-duration successful run of a (model, shield) cell (not its best one)."""

    runs = conn.execute(
        """
        SELECT f.id, f.success, f.stations_scanned, f.t_end FROM flights f
        JOIN experiments e ON e.id = f.experiment_id
        WHERE e.name = ? AND f.arm_id = ? AND f.shield = ? AND f.status = 'complete'
        ORDER BY f.success DESC, f.stations_scanned DESC
        """,
        (EXPERIMENT, arm_id, shield),
    ).fetchall()
    wins = [r for r in runs if r["success"]] or list(runs)
    wins.sort(key=lambda r: r["t_end"])
    return int(wins[len(wins) // 2]["id"])


def pick_winner(
    table: list[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    """Pre-declared rule: success rate, then stations scanned, then fewest collisions."""

    candidates = [r for r in table if not LIVE_ONLY or r["provider"] != "classical"]
    return candidates[0]


def first_obstacle_time(run: MissionRun) -> float:
    """Sim time the first obstacle appears (the GIF highlight starts just before it)."""

    obstacles = run.spec.mission.obstacles if run.spec.mission else []
    times = []
    for o in obstacles:
        idx = np.nonzero(run.traj["x"] >= o.trigger_x)[0]
        if idx.size:
            times.append((int(idx[0]) + 1) * 0.0208333)
    return min(times) if times else 0.0


def screenshot_times(run: MissionRun) -> dict[str, float]:
    times = {"takeoff": 1.2}
    for e in run.events:
        if e["kind"] == "scanned":
            times[f"scan-station-{int(e['station']) + 1}"] = float(e["t"]) + 0.15
    for i, o in enumerate(run.spec.mission.obstacles if run.spec.mission else []):
        idx = np.nonzero(run.traj["x"] >= o.trigger_x)[0]
        if idx.size:
            kind = "intruder" if o.kind == "cross" else "obstacle"
            times[f"{kind}-{i + 1}"] = (int(idx[0]) + 1) * 0.0208333 + SHOT_LEADS_S[
                "obstacle"
            ]
    times["landing"] = max(run.t_end - 1.0, 0.0)
    return times


def render_one(
    conn: sqlite3.Connection, flight_id: int, slug: str, *, shots: bool
) -> None:
    run = load_run(conn, flight_id, ROOT_DIR)
    logger.info(
        "rendering flight %d as %s (%s shield=%s %s)",
        flight_id,
        slug,
        run.arm_id,
        run.shield,
        run.outcome,
    )
    wanted = (
        {round(t * FPS): name for name, t in screenshot_times(run).items()}
        if shots
        else {}
    )
    renderer = MissionRenderer(run)
    out_dir = MEDIA_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "flight.mp4"

    def frames() -> Iterator[tuple[float, Image.Image]]:
        for k, (t, img) in enumerate(renderer.frames()):
            if k in wanted:
                img.save(out_dir / f"{wanted[k]}.jpg", quality=90)
            yield t, img

    n = encode_video(frames(), mp4)
    renderer.close()
    start = max(first_obstacle_time(run) - GIF_LEAD_S, 0.0)
    if (
        slug == "winner"
    ):  # one highlight GIF is enough; Jev gets the MP4 and screenshots
        make_gif(
            mp4,
            out_dir / "highlight.gif",
            start_s=start,
            duration_s=GIF_SECONDS,
        )
    logger.info("%s: %d frames, %d screenshots", slug, n, len(wanted))


def main() -> None:
    configure_logging()
    t0 = time.perf_counter()
    conn = connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    table = audition_table(conn)
    for r in table:
        logger.info(
            "audition %-18s shield=%d n=%d success=%.2f stations=%.2f collision=%.2f lat=%4.0f ms",
            r["arm_id"],
            r["shield"],
            r["n"],
            r["success"],
            r["scanned"],
            r["collision"],
            r["lat_ms"],
        )
    winner = pick_winner(table)
    targets = {"winner": (str(winner["arm_id"]), int(winner["shield"]))}
    for arm, shield in EXTRA_MODELS:
        if (arm, shield) != targets["winner"]:
            targets[arm] = (arm, shield)
    picked: dict[str, int] = {}
    for slug, (arm, shield) in targets.items():
        fid = (
            FLIGHT_ID
            if FLIGHT_ID is not None
            else median_success_flight(conn, arm, shield)
        )
        picked[slug] = fid
        render_one(conn, fid, slug, shots=True)
    (MEDIA_DIR / "selection.json").write_text(
        json.dumps(
            {"winner": targets["winner"], "flights": picked, "audition": table},
            indent=2,
        )
    )
    logger.info("done in %.0fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
