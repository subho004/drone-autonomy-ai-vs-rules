"""Scenario families for the extra simulations, drawn from a seed.

Each family has a ``draw_*`` function (deterministic given the seed) and is certified
feasible by flying a zero-latency reference pilot, exactly like the missions: a scenario that
the reference cannot complete is rejected, so every kept scenario is completable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.scenarios.spec import (
    BoxSpec,
    MissionSpec,
    ObstacleSpec,
    ScenarioSpec,
    SiteSpec,
    StationSpec,
    TargetSpec,
    ZoneSpec,
)
from app.sim.rng import channel_generator

FOLLOW_V_CRUISE = 1.5
GATE_V_CRUISE = 1.2
LAND_V_CRUISE = 1.5
CANYON_V_CRUISE = 1.5
GATE_X0 = 3.5
GATE_SPACING = 3.6
N_GATES = 5


def _u(gen: np.random.Generator, lo: float, hi: float) -> float:
    return float(gen.uniform(lo, hi))


def draw_follow(seed: int, suite: str = "follow") -> ScenarioSpec:
    """Follow a target for 30 s: weaving, stop-and-go or a straight line."""

    gen = channel_generator(seed, "follow")
    path = str(gen.choice(["weave", "weave", "stopgo", "line"]))
    target = TargetSpec(
        path=path,  # type: ignore[arg-type]
        speed=_u(gen, 0.6, 1.1),
        weave_amp=_u(gen, 0.6, 1.2),
        weave_period_s=_u(gen, 6.0, 10.0),
        start_dx=_u(gen, 3.0, 4.0),
    )
    return ScenarioSpec(
        seed=seed,
        suite=suite,
        event_type="follow",
        tier="F",
        budget_s=99.0,
        v_cruise=FOLLOW_V_CRUISE,
        goal_x=90.0,  # keeps the arena long enough to follow a target for the whole run
        max_time_s=target.duration_s + 6.0,
        follow=target,
    )


def draw_delivery(seed: int, suite: str = "delivery") -> ScenarioSpec:
    """Fly to the pad without entering two restricted zones that block the straight line."""

    gen = channel_generator(seed, "delivery")
    za_x = _u(gen, 4.0, 5.0)
    zb_x = _u(gen, 9.0, 10.0)
    zones = [
        ZoneSpec(x0=za_x, x1=za_x + 2.0, y0=-1.4, y1=0.5),
        ZoneSpec(x0=zb_x, x1=zb_x + 2.0, y0=-0.5, y1=1.4),
    ]
    if gen.random() < 0.5:  # mirror the layout half of the time
        zones = [ZoneSpec(x0=z.x0, x1=z.x1, y0=-z.y1, y1=-z.y0) for z in zones]
    pad_x = zb_x + 5.0
    return ScenarioSpec(
        seed=seed,
        suite=suite,
        event_type="delivery",
        tier="D",
        budget_s=99.0,
        v_cruise=1.2,
        goal_x=pad_x,
        max_time_s=70.0,
        mission=MissionSpec(stations=[], pad_x=pad_x, pad_y=0.0, zones=zones),
    )


def draw_pursuit(
    seed: int, suite: str = "pursuit", kind: str = "pursue"
) -> ScenarioSpec:
    """One intruder coming at the drone: ``pursue`` steers by proportional navigation,
    ``approach`` flies a straight line (same geometry, used as the comparison)."""

    gen = channel_generator(seed, "pursuit")
    ob = ObstacleSpec(
        kind=kind,  # type: ignore[arg-type]
        trigger_x=_u(gen, 4.0, 6.0),
        d_spawn=_u(gen, 4.5, 7.0),
        lateral_offset=_u(gen, -1.2, 1.2),
        speed=1.2,
        radius=0.3,
        max_accel=1.5,
        nav_gain=3.0,
    )
    spec = ScenarioSpec(
        seed=seed,
        suite=suite,
        event_type="pursuit",
        tier="P",
        budget_s=99.0,
        v_cruise=2.0,
        obstacle=ob,
    )
    spec.sensors.ray_noise_sigma = 0.03
    return spec


ORB_COUNT = 3
ORB_SPEED = 1.0
ORB_ACCEL = 1.1
ORB_SENSE_RADIUS = 0.5  # the drone's radar sees an orb as a disc this big


def draw_orbs(
    seed: int,
    suite: str = "orbs",
    kind: str = "pursue",
    ext: bool = False,
    speed: float = ORB_SPEED,
    accel: float = ORB_ACCEL,
    count: int = ORB_COUNT,
    wave: int = 1,
) -> ScenarioSpec:
    """Game arena: homing orbs appear in waves of ``wave`` (a swarm when > 1); dodge them
    all, then land."""

    gen = channel_generator(seed, "orbs")
    orbs = [
        ObstacleSpec(
            kind=kind,  # type: ignore[arg-type]
            trigger_x=3.5 + 4.0 * (k // wave) + _u(gen, -0.4, 0.4),
            d_spawn=_u(gen, 4.5, 6.0 + 0.4 * (k % wave)),
            lateral_offset=_u(gen, -1.4, 1.4) + 0.9 * ((k % wave) - (wave - 1) / 2),
            speed=speed,
            radius=0.25,
            max_accel=accel,
            nav_gain=3.0,
            sense_radius=ORB_SENSE_RADIUS,
            ext_control=ext,
        )
        for k in range(count)
    ]
    pad_x = 3.5 + 4.0 * ((count - 1) // wave + 1) + 4.0
    return ScenarioSpec(
        seed=seed,
        suite=suite,
        event_type="orbs",
        tier="G",
        budget_s=99.0,
        v_cruise=1.4,
        goal_x=pad_x,
        max_time_s=60.0,
        mission=MissionSpec(stations=[], pad_x=pad_x, pad_y=0.0, obstacles=orbs),
    )


SWARM_COUNT = 6
SWARM_WAVE = 2


def draw_swarm(
    seed: int, suite: str = "swarm", kind: str = "pursue", ext: bool = False
) -> ScenarioSpec:
    """Orb Dodge with six orbs arriving in waves of two."""

    return draw_orbs(seed, suite, kind, ext, count=SWARM_COUNT, wave=SWARM_WAVE)


def draw_orbs_random(
    seed: int, suite: str = "orbs-random", kind: str = "pursue", ext: bool = False
) -> ScenarioSpec:
    """Randomised Orb Dodge for the tournament: 2 to 5 orbs, one or two per wave, random orb
    speed and turn rate, a little steady wind and turbulence. Not certified: some of these
    are unwinnable, and every combination of dodger and chaser sees the same set."""

    gen = channel_generator(seed, "orbs-random")
    count = int(gen.integers(2, 6))
    wave = int(gen.integers(1, 3))
    spec = draw_orbs(
        seed,
        suite,
        kind,
        ext,
        speed=_u(gen, 0.8, 1.3),
        accel=_u(gen, 0.9, 1.5),
        count=count,
        wave=wave,
    )
    spec.wind.mean_x = _u(gen, -0.25, 0.25)
    spec.wind.mean_y = _u(gen, -0.25, 0.25)
    spec.wind.turb_sigma = _u(gen, 0.0, 0.15)
    return spec


def draw_landing(seed: int, suite: str = "landing") -> ScenarioSpec:
    """Battery goes critical in flight; pick a landing site before the motors fade."""

    gen = channel_generator(seed, "landing")
    drop_t = _u(gen, 3.0, 4.0)
    x0 = LAND_V_CRUISE * drop_t  # where the drone is when the battery drops
    side = 1.0 if gen.random() < 0.5 else -1.0
    sites = [
        SiteSpec(
            x=x0 + _u(gen, 2.6, 3.2), y=side * _u(gen, 0.2, 0.7), radius=0.6
        ),  # near, small
        SiteSpec(
            x=x0 + _u(gen, 2.0, 2.5),
            y=-side * _u(gen, 1.0, 1.6),
            radius=1.0,
            hazard=True,
        ),
        SiteSpec(
            x=x0 + _u(gen, 5.8, 6.8), y=-side * _u(gen, 0.0, 0.8), radius=1.6
        ),  # far, big
    ]
    spec = ScenarioSpec(
        seed=seed,
        suite=suite,
        event_type="landing",
        tier="L",
        budget_s=99.0,
        v_cruise=LAND_V_CRUISE,
        max_time_s=30.0,
        sites=sites,
    )
    spec.sensors.battery_drop_at_s = drop_t
    spec.sensors.battery_death_after_s = 8.0
    return spec


def draw_gates(seed: int, suite: str = "gates") -> ScenarioSpec:
    """Fly through a row of gates at cruise speed without stopping."""

    gen = channel_generator(seed, "gates")
    stations: list[StationSpec] = []
    prev_y, prev_z = 0.0, 1.0
    for k in range(N_GATES):
        # adjacent gates differ by exactly one 1.0 m sidestep and one 0.5 m height step
        y = float(
            gen.choice([c for c in (-1.0, 0.0, 1.0) if 0.0 < abs(c - prev_y) <= 1.0])
        )
        z = 1.5 if prev_z < 1.25 else 1.0
        stations.append(
            StationSpec(x=GATE_X0 + GATE_SPACING * k + _u(gen, -0.2, 0.2), y=y, z=z)
        )
        prev_y, prev_z = y, z
    pad_x = stations[-1].x + 3.0
    return ScenarioSpec(
        seed=seed,
        suite=suite,
        event_type="gates",
        tier="G",
        budget_s=99.0,
        v_cruise=GATE_V_CRUISE,
        goal_x=pad_x,
        max_time_s=45.0,
        mission=MissionSpec(
            stations=stations, pad_x=pad_x, gate_mode=True, tol_y=0.5, tol_z=0.3
        ),
    )


def draw_canyon(seed: int, suite: str = "canyon", block: bool = False) -> ScenarioSpec:
    """A corridor between two walls with a sideways gust; optionally a block to steer around."""

    gen = channel_generator(seed, "canyon")
    half = _u(gen, 2.2, 2.6) if block else _u(gen, 1.0, 1.3)
    length = 16.0
    walls = [
        BoxSpec(
            x=length / 2, y=sgn * (half + 0.15), z=1.25, hx=length / 2, hy=0.15, hz=1.25
        )
        for sgn in (-1.0, 1.0)
    ]
    side = 1.0 if gen.random() < 0.5 else -1.0
    spec = ScenarioSpec(
        seed=seed,
        suite=suite,
        event_type="canyon",
        tier="C" if not block else "K",
        budget_s=99.0,
        v_cruise=CANYON_V_CRUISE,
        goal_x=length,
        max_time_s=30.0,
        boxes=walls,
    )
    spec.wind.gust_at_s = _u(gen, 4.0, 6.0)
    spec.wind.gust_dur_s = 1.0
    spec.wind.gust_y = side * _u(gen, 1.6, 3.0)
    if block:
        spec.obstacle = ObstacleSpec(
            kind="static",
            trigger_x=_u(gen, 4.0, 5.0),
            d_spawn=_u(gen, 3.5, 4.5),
            lateral_offset=_u(gen, -0.2, 0.2),
            radius=0.3,
        )
    return spec


DRAWERS: dict[str, Callable[[int, str], ScenarioSpec]] = {
    "follow": draw_follow,
    "delivery": draw_delivery,
    "pursuit": draw_pursuit,
    "landing": draw_landing,
    "gates": draw_gates,
    "canyon": draw_canyon,
    "orbs": draw_orbs,
}

__all__ = [
    "DRAWERS",
    "draw_canyon",
    "draw_delivery",
    "draw_follow",
    "draw_gates",
    "draw_landing",
    "draw_orbs",
    "draw_orbs_random",
    "draw_pursuit",
    "draw_swarm",
]
