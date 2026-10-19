"""Two-drone world: determinism, contact, per-drone latency, reflex behaviour, renderer."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from app.controllers.arms import HoverArm, ReflexArm, WaypointArm
from app.render.two_drone_video import PairRun, TwoDroneRenderer
from app.sim.two_drone import (
    DRONE_RADIUS,
    DT,
    Pilot,
    TwoDroneSpec,
    TwoDroneWorld,
    build_layouts,
    fixed_latency_pilot,
    ray_ring,
    run_pair,
)


@pytest.fixture(scope="module")
def world() -> Iterator[TwoDroneWorld]:
    w = TwoDroneWorld()
    yield w
    w.close()


def _pair(
    latency_a: float, latency_b: float, arm: type = ReflexArm
) -> tuple[Pilot, Pilot]:
    return fixed_latency_pilot(arm(), latency_a), fixed_latency_pilot(arm(), latency_b)


def _lane_offset(seed: int) -> float:
    a, b = build_layouts(TwoDroneSpec(seed=seed))
    return abs(float(b.start[1] - a.start[1]))


def _seed_where(predicate: Callable[[float], bool]) -> int:
    return next(s for s in range(500) if predicate(_lane_offset(s)))


def test_ray_ring_hits_forward_ray_only_when_ahead() -> None:
    rays = ray_ring(3.0, 0.0, True)
    assert rays[0] == pytest.approx(3.0 - 0.45)
    assert rays[8] == 10.0  # behind
    assert np.all(ray_ring(3.0, 0.0, False) == 10.0)


def test_run_is_deterministic(world: TwoDroneWorld) -> None:
    spec = TwoDroneSpec(seed=7)
    first = run_pair(world, spec, _pair(300, 0))
    second = run_pair(world, spec, _pair(300, 0))
    a, b = dict(first.metrics), dict(second.metrics)
    a.pop("sim_wall_s"), b.pop("sim_wall_s")
    assert a == b
    assert np.array_equal(first.traj["pos"], second.traj["pos"])


def test_other_drone_is_in_rays_extras_and_frames(world: TwoDroneWorld) -> None:
    world.reset(TwoDroneSpec(seed=3))
    assert world.observe(0, 0).extras["other_visible"] == 0.0  # 10 m away, out of range
    for _ in range(300):  # both cruise toward each other with no decisions
        if world.step().separation < 1.8:
            break
    for i in (0, 1):
        obs = world.observe(i, 0)
        local_pos, local_vel = world.local_state(i)
        assert (
            local_vel[0] > 1.0 and local_pos[0] > 3.0
        )  # own frame: forward is positive
        assert obs.extras["other_visible"] == 1.0
        assert obs.fwd_range < 2.0 and obs.rays[0] < 2.0
        assert obs.closing_speed > 1.5 and obs.ttc < 2.0
        assert abs(obs.extras["other_bearing_deg"]) < 15.0
    assert world.world_state(1)[1][0] < -1.0  # B really flies toward -x in the world


def test_contact_is_detected(world: TwoDroneWorld) -> None:
    spec = TwoDroneSpec(seed=1, separation_m=2.0, max_offset_m=0.0)
    res = run_pair(world, spec, _pair(0, 0, WaypointArm))
    assert res.metrics["outcome"] == "collision"
    assert res.metrics["min_clearance_m"] <= 0.02
    assert res.metrics["min_separation_m"] <= 2 * DRONE_RADIUS + 0.02
    assert res.traj["t"][-1] < 2.0


def test_no_contact_with_wide_lanes_and_no_avoidance(world: TwoDroneWorld) -> None:
    seed = _seed_where(lambda off: off > 0.25)
    res = run_pair(world, TwoDroneSpec(seed=seed), _pair(0, 0, WaypointArm))
    assert res.metrics["outcome"] == "success"
    assert res.metrics["collision"] == 0
    assert res.metrics["min_clearance_m"] > 0.1


def test_latency_ticks_are_applied_per_drone(world: TwoDroneWorld) -> None:
    spec = TwoDroneSpec(seed=2, max_time_s=4.0)
    res = run_pair(world, spec, _pair(100, 600, WaypointArm))
    rows_a, rows_b = res.decisions
    assert {r["apply_tick"] - r["obs_tick"] for r in rows_a} == {math.ceil(0.1 / DT)}
    assert {r["apply_tick"] - r["obs_tick"] for r in rows_b} == {math.ceil(0.6 / DT)}
    assert len(rows_a) > 4 * len(rows_b) > 0
    assert rows_a[0]["obs_tick"] == rows_b[0]["obs_tick"] == 0
    assert (
        rows_b[1]["obs_tick"] == rows_b[0]["apply_tick"]
    )  # sequential: next request on arrival


def test_zero_latency_reflex_passes_cleanly_in_most_seeds(world: TwoDroneWorld) -> None:
    results = [
        run_pair(world, TwoDroneSpec(seed=s), _pair(0, 0)).metrics for s in range(8)
    ]
    clean = [
        m for m in results if m["outcome"] == "success" and m["min_clearance_m"] > 0.3
    ]
    assert len(clean) >= 7
    assert all(m["collision"] == 0 for m in results)


def test_collision_when_both_latencies_are_huge(world: TwoDroneWorld) -> None:
    seed = _seed_where(lambda off: off < 0.05)
    res = run_pair(world, TwoDroneSpec(seed=seed), _pair(5000, 5000))
    assert res.metrics["outcome"] == "collision"


def test_one_fast_drone_is_enough_to_avoid(world: TwoDroneWorld) -> None:
    seed = _seed_where(lambda off: off < 0.05)
    res = run_pair(world, TwoDroneSpec(seed=seed), _pair(0, 5000))
    assert res.metrics["collision"] == 0


def test_perpendicular_scenario_runs(world: TwoDroneWorld) -> None:
    spec = TwoDroneSpec(seed=4, scenario="perpendicular")
    layouts = build_layouts(spec)
    assert float(np.dot(layouts[0].heading, layouts[1].heading)) == 0.0
    res = run_pair(world, spec, _pair(0, 0))
    assert res.metrics["outcome"] in {"success", "collision", "timeout", "deadlock"}
    assert res.traj["pos"].shape[1:] == (2, 3)


def test_renderer_produces_non_blank_frames(world: TwoDroneWorld) -> None:
    spec = TwoDroneSpec(seed=5)
    res = run_pair(world, spec, _pair(300, 300))
    run = PairRun(spec, res.traj, res.decisions, res.metrics, (300.0, 300.0))
    renderer = TwoDroneRenderer(run, 640, 360)
    try:
        early = np.asarray(renderer.frame_at(0.5))
        late = np.asarray(renderer.frame_at(min(3.0, run.t_end)))
    finally:
        renderer.close()
    assert early.shape == (360, 640, 3)
    assert early.std() > 10.0 and late.std() > 10.0
    assert not np.array_equal(early, late)
    assert renderer.n_frames > 24


def test_stalled_pair_is_flagged_as_deadlock(world: TwoDroneWorld) -> None:
    spec = TwoDroneSpec(seed=6, separation_m=2.5, max_time_s=12.0)
    res = run_pair(world, spec, _pair(0, 0, HoverArm))
    assert res.metrics["outcome"] == "deadlock"
    assert res.metrics["deadlock"] == 1 and res.metrics["collision"] == 0
    assert 4.0 < res.metrics["deadlock_t"] < 8.0
