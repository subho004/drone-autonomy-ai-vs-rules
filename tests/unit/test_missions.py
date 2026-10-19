from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.controllers.arms import MissionFsmArm
from app.domain.actions import Action
from app.domain.observation import Observation
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.harness.store import connect
from app.providers.encoding import (
    MISSION_RULES,
    ObservationEncoder,
    align_alt,
    align_lat,
)
from app.scenarios.missions import draw_mission, generate_missions
from app.scenarios.spec import ObstacleSpec, ScenarioSpec
from app.sim.world import DroneWorld


def _cfg() -> FlightConfig:
    return FlightConfig(
        LatencyPolicy(LatencyMode.MEASURED),
        record_trajectory=True,
        label_decisions=False,
    )


def test_mission_draw_is_deterministic_and_well_formed() -> None:
    a, b = draw_mission(42), draw_mission(42)
    assert a.scenario_id == b.scenario_id != draw_mission(43).scenario_id
    m = a.mission
    assert m is not None and len(m.stations) == 3 and len(m.obstacles) == 3
    xs = [s.x for s in m.stations]
    assert xs == sorted(xs) and m.pad_x > xs[-1]
    assert {s.y for s in m.stations} <= {-1.0, 0.0, 1.0}


def test_generate_missions_keeps_only_certified_draws() -> None:
    kept = generate_missions(3, 100, "t", certify=lambda s: s.seed % 2 == 0)
    assert [s.seed for s in kept] == [100, 102, 104] and all(s.certified for s in kept)


def test_world_handles_several_obstacles_and_reports_the_nearest(
    world: DroneWorld,
) -> None:
    spec = ScenarioSpec(seed=5, event_type="mission", mission=draw_mission(5).mission)
    spec.mission.obstacles = [  # type: ignore[union-attr]
        ObstacleSpec(kind="static", trigger_x=0.0, d_spawn=6.0),
        ObstacleSpec(kind="static", trigger_x=0.0, d_spawn=3.0, lateral_offset=0.2),
    ]
    world.reset(spec)
    world.step(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    assert world.obstacle_active
    assert abs(world.obstacle_pos[0] - 3.0) < 0.2  # nearest of the two is exposed
    _rays, fwd, _ = world.sense()
    assert 2.0 < fwd < 3.2  # the ray ring sees the nearer one first


def test_mission_flight_reports_progress_and_attaches_mission_state(
    world: DroneWorld,
) -> None:
    spec = draw_mission(3004)
    spec.max_time_s = 15.0
    result = run_flight(world, spec, MissionFsmArm(), _cfg())
    m = result.metrics
    assert m["stations_total"] == 3 and m["stations_scanned"] >= 1
    events = json.loads(m["events_json"])
    assert events[0]["kind"] == "scanned" and events[0]["station"] == 0
    assert 3.0 < events[0]["t"] < 9.0


def test_full_mission_can_be_completed_by_the_rule_based_pilot(
    world: DroneWorld,
) -> None:
    completed = 0
    for seed in (3004, 3010):
        r = run_flight(world, draw_mission(seed), MissionFsmArm(), _cfg())
        completed += r.metrics["success"]
        if r.metrics["success"]:
            assert r.metrics["outcome"] == "landed_on_pad"
            assert r.metrics["stations_scanned"] == r.metrics["stations_total"]
    assert completed >= 1


def test_mission_hold_keeps_an_in_progress_sidestep() -> None:
    from app.controllers.executor import ActionExecutor

    plain = ActionExecutor(np.array([0.0, 0.0, 1.0]), 1 / 48, 1.2, (0, 0, 1))
    keep = ActionExecutor(
        np.array([0.0, 0.0, 1.0]), 1 / 48, 1.2, (0, 0, 1), keep_lateral_on_stop=True
    )
    for ex in (plain, keep):
        ex.apply(Action.LEFT, 0.0)
        ex.apply(Action.HOVER, 0.1)
    assert plain.state.lat_target is None
    assert keep.state.lat_target is not None


def _obs(**extras: float) -> Observation:
    return Observation(
        tick=0,
        t_sim=1.0,
        pos=(2.0, 0.0, 1.0),
        vel=(0.0, 0.0, 0.0),
        goal_dx=14.0,
        home_dist=2.0,
        rays=(10.0,) * 16,
        sectors=(10.0,) * 8,
        nearest_range=10.0,
        nearest_bearing_deg=0.0,
        closing_speed=0.0,
        ttc=99.0,
        battery_pct=80.0,
        gps_ok=True,
        last_action=0,
        extras=extras,
    )


def test_prompt_has_no_mission_block_for_ordinary_observations() -> None:
    assert "mission:" not in ObservationEncoder().encode(_obs())


def test_prompt_mission_block_uses_semantic_buckets() -> None:
    text = ObservationEncoder().encode(
        _obs(stn_i=2, stn_n=3, st_dx=3.4, st_dy=1.1, st_dz=-0.5, st_phase=0, moving=0.0)
    )
    assert "station 2/3 phase=FAR moving=NO" in text
    assert "lat=LEFT alt=DOWN" in text and MISSION_RULES in text
    pad = ObservationEncoder().encode(
        _obs(stn_i=3, stn_n=3, st_dx=0.2, st_dy=0.0, st_dz=0.0, st_phase=5, moving=1.0)
    )
    assert "landing pad phase=PAD_HERE moving=YES" in pad


def test_alignment_buckets_have_hysteresis_free_thresholds() -> None:
    assert (
        align_lat(0.51) == "LEFT"
        and align_lat(-0.51) == "RIGHT"
        and align_lat(0.49) == "OK"
    )
    assert (
        align_alt(0.3) == "UP" and align_alt(-0.3) == "DOWN" and align_alt(0.1) == "OK"
    )


def test_store_migration_adds_mission_columns_to_an_old_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.db"
    conn = connect(path)
    for col in ("stations_scanned", "stations_total", "events_json"):
        conn.execute(
            f"ALTER TABLE flights DROP COLUMN {col}"
        )  # what a pre-mission DB looks like
    conn.close()
    cols = {r["name"] for r in connect(path).execute("PRAGMA table_info(flights)")}
    assert {"stations_scanned", "stations_total", "events_json"} <= cols
    connect(path)  # running it again is harmless
