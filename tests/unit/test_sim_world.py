from __future__ import annotations

import hashlib

import numpy as np

from app.domain.observation import MAX_RANGE_M
from app.scenarios.spec import ObstacleSpec, ScenarioSpec
from app.sim.world import DT, DroneWorld


def _cruise(world: DroneWorld, ticks: int, v: float = 2.0) -> list[np.ndarray]:
    ref = np.array([0.0, 0.0, world.spec.start_z])
    out = []
    for _ in range(ticks):
        ref[0] += v * DT
        world.step(ref, np.array([v, 0.0, 0.0]))
        out.append(world.true_state()[0])
    return out


def test_hover_is_stable(world: DroneWorld) -> None:
    world.reset(ScenarioSpec(seed=1, event_type="clear"))
    pos = np.array([0.0, 0.0, 1.0])
    for _ in range(96):
        world.step(pos, np.zeros(3))
    p, v, _, _ = world.true_state()
    assert np.allclose(p, pos, atol=0.03)
    assert np.linalg.norm(v) < 0.05


def test_cruise_reaches_two_m_per_s(world: DroneWorld) -> None:
    world.reset(ScenarioSpec(seed=1, event_type="clear"))
    _cruise(world, 120)
    assert abs(world.true_state()[1][0] - 2.0) < 0.15


def test_obstacle_spawns_at_trigger_and_collides(world: DroneWorld) -> None:
    spec = ScenarioSpec(seed=2, obstacle=ObstacleSpec(trigger_x=4.0, d_spawn=3.0))
    world.reset(spec)
    hit = False
    for _ in range(400):
        _cruise_one = np.array([world.true_state()[0][0] + 0.5, 0.0, 1.0])
        info = world.step(_cruise_one, np.array([2.0, 0, 0]))
        if info.collision:
            hit = True
            break
    assert hit and world.spawn_tick is not None
    assert abs(world.obstacle_pos[0] - (world.true_state()[0][0])) < 1.0


def test_rays_measure_obstacle_range(world: DroneWorld) -> None:
    spec = ScenarioSpec(
        seed=3, obstacle=ObstacleSpec(trigger_x=0.0, d_spawn=4.0, radius=0.3)
    )
    world.reset(spec)
    world.step(
        np.array([0.0, 0.0, 1.0]), np.zeros(3)
    )  # spawn happens on the first tick
    rays, fwd, _ = world.sense()
    expected = 4.0 - 0.3  # centre 4 m ahead, surface 0.3 m closer
    assert abs(fwd - expected) < 0.2
    assert rays[8] == MAX_RANGE_M or rays[8] > 9.0  # nothing behind


def test_observation_flags_and_symmetry(world: DroneWorld) -> None:
    spec = ScenarioSpec(
        seed=4, obstacle=ObstacleSpec(trigger_x=0.0, d_spawn=3.0, lateral_offset=0.8)
    )
    world.reset(spec)
    world.step(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    obs = world.observe(0)
    assert obs.nearest_bearing_deg > 0  # obstacle on the left (+y)
    assert obs.gps_ok and obs.battery_pct < 91


def test_gps_loss_freezes_observed_position(world: DroneWorld) -> None:
    spec = ScenarioSpec(seed=5, event_type="gps_loss")
    spec.sensors.gps_loss_at_s = 1.0
    world.reset(spec)
    _cruise(world, 120)
    obs = world.observe(0)
    assert not obs.gps_ok
    assert obs.pos[0] < world.true_state()[0][0] - 0.5


def test_flight_is_deterministic_across_resets(world: DroneWorld) -> None:
    spec = ScenarioSpec(seed=7, obstacle=ObstacleSpec(trigger_x=3.0, d_spawn=3.5))
    spec.sensors.ray_noise_sigma = 0.05
    hashes = []
    for _ in range(2):
        world.reset(spec)
        traj = np.array(_cruise(world, 100))
        obs = world.observe(0)
        hashes.append(
            hashlib.sha256(traj.tobytes() + str(obs.rays).encode()).hexdigest()
        )
    assert hashes[0] == hashes[1]


def test_wind_pushes_the_drone(world: DroneWorld) -> None:
    spec = ScenarioSpec(seed=8, event_type="clear")
    spec.wind.mean_y = 1.0
    spec.wind.step_at_s = 0.0
    world.reset(spec)
    hold = np.array([0.0, 0.0, 1.0])
    for _ in range(48):
        world.step(hold, np.zeros(3))
    assert world.true_state()[0][1] > 0.02
