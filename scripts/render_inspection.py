"""Fly an inspection route and render it: the rule-based pilot flies, a vision model reads.

A wall of panels runs parallel to the flight line. The mission pilot scans three stations,
each facing one panel (defective, clean, defective, fixed by seed before any model call).
At each scan moment the drone-camera frame goes to the best study model (the cell with the
highest accuracy in assets/sims/inspection/study.json, ties broken by lower p50 latency),
at most MAX_CALLS_PER_FLIGHT paid calls, cached in verdicts.json so re-rendering is free.
Outputs (assets/sims/inspection/): flight.mp4, highlight.gif and six JPEG screenshots.
Constants at the top instead of CLI args.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.controllers.arms import MissionFsmArm
from app.core.logging import configure_logging, get_logger
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.providers.vision import CostCap, VisionClient
from app.render.inspection_video import InspectionRenderer, Verdict
from app.render.mission_video import FPS, MissionRun, encode_video, make_gif
from app.scenarios.spec import MissionSpec, ScenarioSpec, StationSpec
from app.sim.inspection import (
    COL_PITCH_M,
    FIRST_COL_X_M,
    StructureSpec,
    draw_structure,
    encode_png,
)
from app.sim.world import DroneWorld

STRUCTURE_SEED = 5200
ROWS, COLS = 2, 8
STATION_COLS = (2, 4, 6)  # lower-row panels the stations face (station x = panel x)
N_OTHER_DEFECTIVE = 5
FORCED_DEFECTS = {2: True, 4: False, 6: True}  # panel index (row 0) -> has a defect
STATION_Z_M = 1.0
PAD_AFTER_LAST_M = 3.5
V_CRUISE = 1.2
MAX_TIME_S = 60.0
FLIGHT_SEED = 5200
MAX_CALLS_PER_FLIGHT = 3
COST_CAP_USD = 1.50
GIF_LEAD_S = 3.0
GIF_SECONDS = 12.0
TAKEOFF_SHOT_S = 1.2
APPROACH_LEAD_S = 2.0
SETTLE_PAD_S = 0.8  # screenshot this long after the verdict appears
OUT_DIR = ROOT_DIR / "assets" / "sims" / "inspection"
STUDY_JSON = OUT_DIR / "study.json"
VERDICTS_JSON = OUT_DIR / "verdicts.json"
SPEND_JSON = OUT_DIR / "spend.json"

logger = get_logger("render_inspection")


def build_structure() -> StructureSpec:
    """The wall: station panels forced (defect, clean, defect), the rest drawn by seed."""

    return draw_structure(
        STRUCTURE_SEED, ROWS, COLS, n_defective=N_OTHER_DEFECTIVE, forced=FORCED_DEFECTS
    )


def build_spec() -> ScenarioSpec:
    stations = [
        StationSpec(x=FIRST_COL_X_M + c * COL_PITCH_M, y=0.0, z=STATION_Z_M)
        for c in STATION_COLS
    ]
    pad_x = stations[-1].x + PAD_AFTER_LAST_M
    return ScenarioSpec(
        seed=FLIGHT_SEED,
        suite="inspection",
        event_type="mission",
        tier="M",
        budget_s=99.0,
        v_cruise=V_CRUISE,
        goal_x=pad_x,
        max_time_s=MAX_TIME_S,
        mission=MissionSpec(stations=stations, pad_x=pad_x, pad_y=0.0, obstacles=[]),
    )


def fly(spec: ScenarioSpec) -> MissionRun:
    """Run the rule-based pilot (no network) and wrap the result for the renderer."""

    world = DroneWorld()
    try:
        cfg = FlightConfig(
            LatencyPolicy(LatencyMode.MEASURED),
            record_trajectory=True,
            label_decisions=False,
        )
        result = run_flight(world, spec, MissionFsmArm(), cfg)
    finally:
        world.close()
    m = result.metrics
    assert result.trajectory is not None
    return MissionRun(
        flight_id=0,
        arm_id="mission_fsm",
        model="Rule-based pilot",
        shield=False,
        outcome=m["outcome"],
        success=bool(m["success"]),
        spec=spec,
        traj=result.trajectory,
        decisions=result.decisions,
        events=json.loads(m["events_json"]),
        mean_latency_ms=float(m["lat_mean_ms"] or 0.0),
        stations_scanned=int(m["stations_scanned"]),
        stations_total=int(m["stations_total"]),
        t_end=float(m["t_end"]),
    )


def choose_cell() -> tuple[str, int]:
    """Best (model, resolution) in the study: highest accuracy, then lower p50 latency."""

    summary = json.loads(STUDY_JSON.read_text())["summary"]
    cells = [
        (model, int(mode), c)
        for model, modes in summary.items()
        for mode, c in modes.items()
        if mode != "text"
    ]
    model, res, _ = min(
        cells, key=lambda x: (-x[2]["accuracy"], x[2]["lat_p50_ms"], x[1])
    )
    return model, res


def _load_cache() -> dict[str, dict[str, Any]]:
    return json.loads(VERDICTS_JSON.read_text()) if VERDICTS_JSON.exists() else {}


def get_verdicts(
    renderer: InspectionRenderer,
    structure: StructureSpec,
    panels: list[int],
    model: str,
    resolution: int,
) -> list[Verdict]:
    """One verdict per scanned station; cached answers are reused, new ones are paid calls."""

    cache = _load_cache()
    client: VisionClient | None = None
    paid = 0
    out: list[Verdict] = []
    try:
        for e in renderer.run.events:
            if e["kind"] != "scanned":
                continue
            k, t_event = int(e["station"]), float(e["t"])
            rgb = renderer.camera.render_camera(
                renderer.pose_at(t_event), panels[k], resolution
            )
            png = encode_png(rgb)
            key = f"{model}|{resolution}|{structure.seed}|{k}|{sha256(png).hexdigest()[:16]}"
            truth = structure.panels[panels[k]].has_defect
            if key not in cache:
                if paid >= MAX_CALLS_PER_FLIGHT:
                    raise RuntimeError("would exceed the per-flight call limit")
                client = client or VisionClient(CostCap(COST_CAP_USD, SPEND_JSON))
                r = client.ask(model, png)
                paid += 1
                cache[key] = {
                    "answer": r.answer,
                    "verdict": r.verdict,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                }
                VERDICTS_JSON.write_text(json.dumps(cache, indent=1))
                logger.info(
                    "station %d: %r in %.0f ms (paid call)",
                    k + 1,
                    r.answer,
                    r.latency_ms,
                )
            c = cache[key]
            out.append(
                Verdict(
                    station=k,
                    panel_index=panels[k],
                    t_event=t_event,
                    model=model,
                    resolution=resolution,
                    answer=str(c["answer"]),
                    verdict=c["verdict"],
                    truth=truth,
                    latency_ms=float(c["latency_ms"]),
                    cost_usd=float(c["cost_usd"]),
                    tokens_in=int(c["tokens_in"]),
                    tokens_out=int(c["tokens_out"]),
                )
            )
    finally:
        if client is not None:
            client.close()
    return out


def shot_times(run: MissionRun, verdicts: list[Verdict]) -> dict[str, float]:
    """Moments for the six screenshots (sim seconds)."""

    settle = {
        v.station: v.t_event + max(v.latency_ms / 1000.0, 0.8) + SETTLE_PAD_S
        for v in verdicts
    }
    first_defect = next((v for v in verdicts if v.verdict), verdicts[0])
    first_clean = next((v for v in verdicts if v.verdict is False), verdicts[-1])
    first_scan = min(v.t_event for v in verdicts)
    return {
        "takeoff": TAKEOFF_SHOT_S,
        "camera-approach": max(first_scan - APPROACH_LEAD_S, 0.5),
        "panel-defect-verdict": settle[first_defect.station],
        "panel-clean-verdict": settle[first_clean.station],
        "last-panel": settle[verdicts[-1].station],
        "summary": max(run.t_end - 0.2, 0.0),
    }


def render(
    run: MissionRun, renderer: InspectionRenderer, verdicts: list[Verdict]
) -> None:
    shots = {round(t * FPS): name for name, t in shot_times(run, verdicts).items()}
    mp4 = OUT_DIR / "flight.mp4"

    def frames() -> Iterator[tuple[float, Image.Image]]:
        for k, (t, img) in enumerate(renderer.frames()):
            if k in shots:
                img.save(OUT_DIR / f"{shots[k]}.jpg", quality=90)
            yield t, img

    n = encode_video(frames(), mp4)
    start = max(min(v.t_event for v in verdicts) - GIF_LEAD_S, 0.0)
    make_gif(mp4, OUT_DIR / "highlight.gif", start_s=start, duration_s=GIF_SECONDS)
    logger.info("%d frames, %d screenshots", n, len(shots))


def main() -> None:
    configure_logging()
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    structure = build_structure()
    panels = list(STATION_COLS)
    model, resolution = choose_cell()
    logger.info("model %s at %dx%d (best study cell)", model, resolution, resolution)
    run = fly(build_spec())
    logger.info(
        "flight: outcome=%s stations=%d/%d t_end=%.1fs",
        run.outcome,
        run.stations_scanned,
        run.stations_total,
        run.t_end,
    )
    renderer = InspectionRenderer(run, structure, panels, [], resolution, model)
    try:
        verdicts = get_verdicts(renderer, structure, panels, model, resolution)
        renderer.verdicts = sorted(verdicts, key=lambda v: v.t_event)
        (OUT_DIR / "flight-summary.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "resolution": resolution,
                    "outcome": run.outcome,
                    "t_end": run.t_end,
                    "events": run.events,
                    "verdicts": [asdict(v) for v in verdicts],
                },
                indent=1,
            )
        )
        render(run, renderer, verdicts)
    finally:
        renderer.close()
    logger.info("done in %.0fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
