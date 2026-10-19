"""Shared helpers for provider tests (no tests of its own)."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

from app.domain.observation import Observation
from app.harness.store import connect

SECTOR_SWAP = (0, 7, 6, 5, 4, 3, 2, 1)


def make_obs(**overrides: object) -> Observation:
    base = Observation(
        tick=3,
        t_sim=0.15,
        pos=(1.0, 0.0, 1.5),
        vel=(2.3, 0.4, 0.0),
        goal_dx=14.2,
        home_dist=6.0,
        rays=(),
        sectors=(3.4, 7.8, 9.9, 10.0, 10.0, 10.0, 3.6, 3.4),
        nearest_range=3.4,
        nearest_bearing_deg=-20.0,
        closing_speed=2.1,
        ttc=1.6,
        battery_pct=43.0,
        gps_ok=True,
        last_action=0,
        decision_age_ms=180.0,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def mirror(obs: Observation) -> Observation:
    """Swap left and right: sectors, lateral velocity, nearest bearing."""

    return replace(
        obs,
        sectors=tuple(obs.sectors[i] for i in SECTOR_SWAP),
        vel=(obs.vel[0], -obs.vel[1], obs.vel[2]),
        nearest_bearing_deg=-obs.nearest_bearing_deg,
    )


def memory_db() -> sqlite3.Connection:
    return connect(":memory:")
