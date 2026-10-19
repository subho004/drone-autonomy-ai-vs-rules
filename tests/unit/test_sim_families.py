from __future__ import annotations

import json
from typing import Any

import numpy as np

from app.controllers.arms import build_classical_arm
from app.domain.observation import Observation
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.providers.encoding import ObservationEncoder
from app.scenarios import sims
from app.scenarios.spec import BoxSpec, ObstacleSpec, ScenarioSpec
from app.sim.world import DT, DroneWorld


def _fly(
    world: DroneWorld,
    spec: ScenarioSpec,
    arm: str,
    ms: float = 0.0,
    shield: bool = False,
) -> dict[str, Any]:
    mode = LatencyMode.FIXED if ms > 0 else LatencyMode.MEASURED
    cfg = FlightConfig(
        LatencyPolicy(mode, fixed_ms=ms),
        shield=shield,
        record_trajectory=False,
        label_decisions=False,
    )
    return run_flight(world, spec, build_classical_arm(arm), cfg).metrics


def test_families_are_deterministic_and_distinct() -> None:
    for name, draw in sims.DRAWERS.items():
        a, b, c = draw(5, name), draw(5, name), draw(6, name)
        assert a.scenario_id == b.scenario_id != c.scenario_id, name


def test_follow_pilot_holds_the_band_at_zero_latency_and_loses_it_when_slow(
    world: DroneWorld,
) -> None:
    ok = _fly(world, sims.draw_follow(1), "follow_pilot")
    assert ok["success"] == 1 and json.loads(ok["task_json"])["in_band_frac"] > 0.9
    slow = _fly(world, sims.draw_follow(1), "follow_pilot", ms=1500.0)
    assert (
        json.loads(slow["task_json"])["in_band_frac"]
        < json.loads(ok["task_json"])["in_band_frac"]
    )


def test_delivery_pilot_avoids_zones_and_a_blind_pilot_breaches(
    world: DroneWorld,
) -> None:
    spec = sims.draw_delivery(2)
    ok = _fly(world, spec, "zone_pilot")
    assert ok["success"] == 1 and json.loads(ok["task_json"])["max_breach_m"] == 0.0
    blind = _fly(world, spec, "mission_fsm")  # ignores zones
    assert json.loads(blind["task_json"])["max_breach_m"] > 0.2


def test_geofence_layer_reduces_breach_for_a_zone_blind_pilot(
    world: DroneWorld,
) -> None:
    spec = sims.draw_delivery(2)
    off = json.loads(_fly(world, spec, "mission_fsm")["task_json"])["max_breach_m"]
    on = json.loads(_fly(world, spec, "mission_fsm", shield=True)["task_json"])[
        "max_breach_m"
    ]
    assert on < off


def test_pursuer_steers_toward_the_drone(world: DroneWorld) -> None:
    spec = ScenarioSpec(
        seed=1,
        event_type="pursuit",
        obstacle=ObstacleSpec(
            kind="pursue", trigger_x=0.0, d_spawn=5.0, lateral_offset=1.5, speed=1.0
        ),
    )
    world.reset(spec)
    hold = np.array([0.0, 0.0, 1.0])
    ys = []
    for _ in range(120):
        world.step(hold, np.zeros(3))
        ys.append(float(world.obstacle_pos[1]))
    assert ys[0] > 1.0 and ys[-1] < ys[0] - 0.4  # it curved toward y = 0


def test_landing_pilots_pick_different_sites_and_big_sites_forgive_latency(
    world: DroneWorld,
) -> None:
    spec = sims.draw_landing(3)
    best = _fly(world, spec, "site_best")
    near = _fly(world, spec, "site_nearest")
    assert best["success"] == 1
    assert (
        json.loads(best["task_json"])["chosen_site"]
        != json.loads(near["task_json"])["chosen_site"]
    )
    hits = sum(
        _fly(world, sims.draw_landing(s), "site_nearest", ms=1200.0)["success"]
        for s in range(6)
    )
    big = sum(
        _fly(world, sims.draw_landing(s), "site_best", ms=1200.0)["success"]
        for s in range(6)
    )
    assert big >= hits


def test_gates_are_flyable_at_zero_latency_and_collapse_with_latency(
    world: DroneWorld,
) -> None:
    done = [
        _fly(world, sims.draw_gates(s), "gate_pilot")["stations_scanned"]
        for s in range(4)
    ]
    slow = [
        _fly(world, sims.draw_gates(s), "gate_pilot", ms=500.0)["stations_scanned"]
        for s in range(4)
    ]
    assert min(done) == sims.N_GATES and np.mean(slow) < np.mean(done) - 2


def test_walls_are_solid_and_seen_by_the_rays(world: DroneWorld) -> None:
    spec = ScenarioSpec(
        seed=2,
        event_type="canyon",
        boxes=[BoxSpec(x=3.0, y=0.0, z=1.0, hx=0.2, hy=2.0, hz=1.5)],
    )
    world.reset(spec)
    _, fwd, _ = world.sense()
    assert 2.4 < fwd < 3.2  # wall face at x = 2.8
    hit = False
    ref = np.array([0.0, 0.0, 1.0])
    for _ in range(300):
        ref[0] += 1.5 * DT
        if world.step(ref, np.array([1.5, 0.0, 0.0])).collision:
            hit = True
            break
    assert hit


def test_gust_pushes_the_drone_sideways(world: DroneWorld) -> None:
    spec = ScenarioSpec(seed=3, event_type="canyon")
    spec.wind.gust_at_s, spec.wind.gust_dur_s, spec.wind.gust_y = 0.5, 1.5, 4.0
    world.reset(spec)
    hold = np.array([0.0, 0.0, 1.0])
    for _ in range(96):
        world.step(hold, np.zeros(3))
    assert abs(float(world.true_state()[0][1])) > 0.2


def _obs(**extras: float) -> Observation:
    return Observation(
        tick=0,
        t_sim=1.0,
        pos=(2.0, 0.0, 1.0),
        vel=(0.5, 0.0, 0.0),
        goal_dx=5.0,
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


def test_prompt_blocks_for_each_task() -> None:
    enc = ObservationEncoder()
    follow = enc.encode(
        _obs(
            fol_dx=3.4,
            fol_dy=-1.0,
            fol_rvx=0.5,
            fol_rvy=0.0,
            fol_in=1.0,
            fol_bmin=1.5,
            fol_bmax=5.0,
            moving=0.0,
        )
    )
    assert "follow: target dx=3.4 gap=OK lat=RIGHT trend=OPENING moving=NO" in follow
    zones = enc.encode(_obs(zn_dx=2.0, zn_blocked=1.0, zn_dir=1.0, zn_inside=0.0))
    assert "zones: nearest dx=2 state=BLOCKED step=LEFT" in zones
    sites = enc.encode(
        _obs(
            ls_n=3,
            ls1_dx=3.0,
            ls1_dy=0.2,
            ls1_r=0.6,
            ls1_hz=0.0,
            ls2_dx=2.0,
            ls2_dy=-1.0,
            ls2_r=1.6,
            ls2_hz=1.0,
            moving=1.0,
        )
    )
    assert (
        "1) dx=3 lat=OK size=SMALL hazard=NO" in sites
        and "2) dx=2 lat=RIGHT size=BIG hazard=YES" in sites
    )
    gate = enc.encode(
        _obs(
            stn_i=1,
            stn_n=5,
            st_dx=3.0,
            st_dy=0.0,
            st_dz=0.0,
            st_phase=0,
            moving=0.0,
            gate=1.0,
        )
    )
    assert "Never answer HOVER or BRAKE just because a gate is near" in gate
