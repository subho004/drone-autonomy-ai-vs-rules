"""Mission scenarios: visit and scan stations, dodge obstacles on the way, land on a pad.

Drawn from a seed; feasibility is certified by flying the rule-based mission pilot with no
latency, so every mission in a suite is known to be completable by a competent controller.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.scenarios.spec import MissionSpec, ObstacleSpec, ScenarioSpec, StationSpec
from app.sim.rng import channel_generator

MISSION_V_CRUISE = 1.2
MISSION_MAX_TIME_S = 80.0
STATION_X = (4.0, 8.5, 13.0)
STATION_X_JITTER = 0.4
STATION_Y_CHOICES = (-1.0, 0.0, 1.0)
STATION_Z_CHOICES = (1.0, 1.5)
PAD_AFTER_LAST_M = 3.5


def _pad_intruder(gen: np.random.Generator, last_station_x: float) -> ObstacleSpec:
    """An intruder crossing the approach to the pad, timed to meet the drone there."""

    speed = float(gen.uniform(0.6, 0.9))
    d_spawn = float(gen.uniform(1.8, 2.3))
    return ObstacleSpec(
        kind="cross",
        trigger_x=last_station_x + 0.2,
        d_spawn=d_spawn,
        speed=speed,
        cross_side=1 if gen.random() < 0.5 else -1,
        cross_start_y=max(speed * d_spawn / MISSION_V_CRUISE, 0.3),
        radius=0.28,
    )


def draw_mission(seed: int, suite: str = "mission") -> ScenarioSpec:
    """One mission for ``seed`` (deterministic)."""

    gen = channel_generator(seed, "mission")
    stations: list[StationSpec] = []
    prev_y = 0.0
    for x in STATION_X:
        y = float(gen.choice([c for c in STATION_Y_CHOICES if c != prev_y]))
        z = float(gen.choice(STATION_Z_CHOICES))
        stations.append(
            StationSpec(
                x=x + float(gen.uniform(-STATION_X_JITTER, STATION_X_JITTER)), y=y, z=z
            )
        )
        prev_y = y
    pad_x = stations[-1].x + PAD_AFTER_LAST_M
    obstacles = [
        ObstacleSpec(  # a static block between the first and second station
            kind="static",
            trigger_x=stations[0].x + 0.4,
            d_spawn=float(gen.uniform(2.6, 3.4)),
            lateral_offset=float(gen.uniform(-0.25, 0.25)),
            radius=0.3,
        ),
        ObstacleSpec(  # something drifting toward the drone before the last station
            kind="approach",
            trigger_x=stations[1].x + 0.5,
            d_spawn=float(gen.uniform(3.6, 4.6)),
            lateral_offset=float(gen.uniform(-0.3, 0.3)),
            speed=float(gen.uniform(0.5, 0.9)),
            radius=0.28,
        ),
        _pad_intruder(gen, stations[2].x),
    ]
    spec = ScenarioSpec(
        seed=seed,
        suite=suite,
        event_type="mission",
        tier="M",
        budget_s=99.0,
        v_cruise=MISSION_V_CRUISE,
        goal_x=pad_x,
        max_time_s=MISSION_MAX_TIME_S,
        mission=MissionSpec(
            stations=stations, pad_x=pad_x, pad_y=0.0, obstacles=obstacles
        ),
    )
    spec.sensors.ray_noise_sigma = 0.03
    return spec


def generate_missions(
    n: int,
    base_seed: int,
    suite: str,
    certify: Callable[[ScenarioSpec], bool],
    max_draws: int = 1000,
) -> list[ScenarioSpec]:
    """``n`` missions that pass ``certify`` (seeds advance past rejected draws)."""

    out: list[ScenarioSpec] = []
    seed = base_seed
    while len(out) < n and seed - base_seed < max_draws:
        spec = draw_mission(seed, suite)
        seed += 1
        if certify(spec):
            spec.certified = True
            out.append(spec)
    return out


__all__ = ["draw_mission", "generate_missions"]
