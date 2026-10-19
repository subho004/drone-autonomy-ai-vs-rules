"""Observation handed to every decision arm.

All geometry is in one frame: world axes with yaw held at zero, so the body
frame equals the world frame (x forward toward the goal, y left, z up). The
harness precomputes every derived quantity (ranges, time-to-collision, ...)
so decision arms never have to do arithmetic on raw floats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_RAYS = 16
N_SECTORS = 8
MAX_RANGE_M = 10.0


@dataclass(frozen=True, slots=True)
class Observation:
    """One sensed snapshot of the world at control tick ``tick``."""

    tick: int
    t_sim: float
    pos: tuple[float, float, float]
    vel: tuple[float, float, float]
    goal_dx: float
    home_dist: float
    rays: tuple[float, ...]
    sectors: tuple[float, ...]
    nearest_range: float
    nearest_bearing_deg: float
    closing_speed: float
    ttc: float
    battery_pct: float
    gps_ok: bool
    last_action: int
    decision_age_ms: float = 0.0
    fwd_range: float = MAX_RANGE_M  # nearest return inside the forward flight corridor
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vel[0], self.vel[1]))


@dataclass(frozen=True, slots=True)
class TruthState:
    """Privileged ground truth, available only to oracle/shield/labeller."""

    tick: int
    pos: tuple[float, float, float]
    vel: tuple[float, float, float]
    obstacle_pos: tuple[float, float, float] | None
    obstacle_vel: tuple[float, float, float] | None
    obstacle_radius: float
    clearance: float


__all__ = ["MAX_RANGE_M", "N_RAYS", "N_SECTORS", "Observation", "TruthState"]
