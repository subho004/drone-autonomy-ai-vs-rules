from __future__ import annotations

import numpy as np

from app.controllers.arms import MissionFsmArm
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.render.mission_video import MissionRenderer, MissionRun
from app.scenarios.missions import draw_mission
from app.sim.world import DroneWorld


def _run(world: DroneWorld) -> MissionRun:
    import json

    spec = draw_mission(3004)
    spec.max_time_s = 12.0
    cfg = FlightConfig(
        LatencyPolicy(LatencyMode.MEASURED),
        record_trajectory=True,
        label_decisions=False,
    )
    r = run_flight(world, spec, MissionFsmArm(), cfg)
    assert r.trajectory is not None
    return MissionRun(
        flight_id=1,
        arm_id="mission_fsm",
        model="mission_fsm",
        shield=False,
        outcome="timeout",
        success=False,
        spec=spec,
        traj=r.trajectory,
        decisions=[
            {
                "obs_tick": d["obs_tick"],
                "apply_tick": d["apply_tick"],
                "applied_action": d["applied_action"],
                "latency_ms": d["latency_ms"],
                "source": d["source"],
                "valid": d["valid"],
            }
            for d in r.decisions
        ],
        events=json.loads(r.metrics["events_json"]),
        mean_latency_ms=0.0,
        stations_scanned=int(r.metrics["stations_scanned"]),
        stations_total=3,
        t_end=r.metrics["t_end"],
    )


def test_renderer_produces_frames_with_hud_and_scene_content(world: DroneWorld) -> None:
    renderer = MissionRenderer(_run(world), width=480, height=270)
    try:
        frames = [renderer.frame_at(t) for t in (0.5, 1.0, 7.0)]
    finally:
        renderer.close()
    for f in frames:
        arr = np.asarray(f)
        assert arr.shape == (270, 480, 3) and arr.std() > 10  # not blank
    assert not np.array_equal(
        np.asarray(frames[0]), np.asarray(frames[2])
    )  # the scene moves
    top_left = np.asarray(frames[0])[20:60, 20:200].mean()
    assert top_left < 120  # the dark HUD panel is drawn


def test_renderer_marks_scanned_stations_and_schedules_toasts(
    world: DroneWorld,
) -> None:
    run = _run(world)
    renderer = MissionRenderer(run, width=320, height=180)
    try:
        if run.events:
            assert any("SCANNED" in t.text for t in renderer.toasts)
        status, pad = renderer._status(run.t_end)
        assert status.count("done") == run.stations_scanned and pad in (
            "todo",
            "active",
        )
    finally:
        renderer.close()
