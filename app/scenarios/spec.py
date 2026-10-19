"""Seeded, serialisable scenario specification.

A scenario is pure data: every arm that flies it faces the identical world.
``scenario_id`` is a hash of the canonical JSON of everything that matters.
Event timing is in seconds / world coordinates, never ticks, so a scenario is
independent of the control rate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "obstacle_static",
    "obstacle_approach",
    "intruder_cross",
    "battery_critical",
    "gps_loss",
    "clear",
    "mission",
    "follow",
    "delivery",
    "landing",
    "gates",
    "canyon",
    "pursuit",
    "orbs",
]


class ObstacleSpec(BaseModel):
    """A sphere that appears when the drone crosses ``trigger_x``."""

    kind: Literal["static", "approach", "cross", "pursue"] = "static"
    trigger_x: float = 5.0  # drone x at which the obstacle appears
    d_spawn: float = 2.5  # range ahead of the drone (m) at the spawn instant
    lateral_offset: float = 0.0  # y of the obstacle centre at spawn (static/approach)
    radius: float = 0.30
    speed: float = 0.0  # approach: speed along -x; cross: lateral speed
    cross_side: int = 1  # cross: +1 starts at +y moving toward -y, -1 the reverse
    cross_start_y: float = 2.5  # cross: initial |y|
    max_accel: float = 1.5  # pursue: lateral acceleration limit (m/s^2)
    nav_gain: float = 3.0  # pursue: proportional-navigation gain
    ext_control: bool = (
        False  # pursue: steered by an external controller (L/R/straight)
    )
    sense_radius: float = (
        0.0  # >0: sensed as a disc of this radius (radar), not by rays alone
    )


class WindSpec(BaseModel):
    gust_at_s: float | None = None  # a short strong burst on top of the mean wind
    gust_dur_s: float = 1.0
    gust_x: float = 0.0  # m/s^2
    gust_y: float = 0.0
    mean_x: float = 0.0  # m/s^2 steady wind acceleration
    mean_y: float = 0.0
    step_at_s: float = 4.0  # time the mean wind switches on
    turb_sigma: float = 0.0  # OU turbulence std (m/s^2)
    turb_tau_s: float = 1.0


class TargetSpec(BaseModel):
    """A moving target the drone must keep within a range band (follow-me)."""

    path: Literal["weave", "line", "stopgo"] = "weave"
    speed: float = 1.0  # forward speed (m/s)
    weave_amp: float = 1.0  # lateral weave amplitude (m)
    weave_period_s: float = 8.0
    start_dx: float = 3.5  # initial distance ahead of the drone (m)
    start_y: float = 0.0
    alt: float = 1.0
    duration_s: float = 30.0
    band_min: float = 1.5  # in-band when the target is this far ahead...
    band_max: float = 5.0  # ...to this far (m)
    band_half_width: float = 1.2  # and within this lateral offset (m)
    lost_after_s: float = 2.5  # out of the band this long = lock lost
    stop_s: float = 2.0  # stopgo: pause length
    move_s: float = 4.0  # stopgo: move length


class ZoneSpec(BaseModel):
    """A restricted (no-fly) box, full height. Breach depth = distance inside the boundary."""

    x0: float
    x1: float
    y0: float
    y1: float


class SiteSpec(BaseModel):
    """A candidate landing site on the ground."""

    x: float
    y: float = 0.0
    radius: float = 0.8
    hazard: bool = False  # e.g. people or water: landing here is unsafe


class BoxSpec(BaseModel):
    """A static box (wall, building). Centre and half extents, in metres."""

    x: float
    y: float
    z: float
    hx: float
    hy: float
    hz: float


class SensorSpec(BaseModel):
    ray_noise_sigma: float = 0.03  # m
    ray_dropout_p: float = 0.0
    gps_loss_at_s: float | None = None
    gps_flagged: bool = True
    battery_start_pct: float = 90.0
    battery_drain_pct_per_s: float = 0.2
    battery_drop_at_s: float | None = None
    battery_drop_to_pct: float = 7.0
    battery_death_after_s: float = (
        8.0  # motors fade this long after CRITICAL if not landed
    )


class StationSpec(BaseModel):
    """A point of interest the drone must reach, hold over and scan."""

    x: float
    y: float = 0.0
    z: float = 1.0


class MissionSpec(BaseModel):
    """A multi-task flight: visit and scan stations, dodge obstacles, land on a pad."""

    stations: list[StationSpec]
    pad_x: float = 16.0
    pad_y: float = 0.0
    obstacles: list[ObstacleSpec] = Field(default_factory=list)
    dwell_s: float = 1.0  # time to hold inside the tolerance box to "scan" a station
    tol_x: float = 0.8
    tol_y: float = (
        0.6  # wider than one 1.0 m LEFT/RIGHT step so alignment always converges
    )
    tol_z: float = 0.3
    scan_speed_max: float = 0.6  # must be slower than this while scanning
    miss_margin: float = 1.0  # passing a station by this much marks it missed
    zones: list[ZoneSpec] = Field(default_factory=list)
    gate_mode: bool = False  # stations are gates: fly through, no hold or speed limit
    pad_tol: float = 1.0  # landing counts within this distance of the pad centre


class ScenarioSpec(BaseModel):
    seed: int
    suite: str = "dev"
    event_type: EventType = "obstacle_static"
    tier: str = "T2"
    budget_s: float = 1.0  # reaction budget B (s); negative = impossible control
    v_cruise: float = 2.0
    goal_x: float = 14.0
    start_z: float = 1.0
    max_time_s: float = 20.0
    obstacle: ObstacleSpec | None = None
    mission: MissionSpec | None = None
    follow: TargetSpec | None = None
    sites: list[SiteSpec] = Field(default_factory=list)
    boxes: list[BoxSpec] = Field(default_factory=list)
    wind: WindSpec = Field(default_factory=WindSpec)
    sensors: SensorSpec = Field(default_factory=SensorSpec)
    deadline_s: float | None = (
        None  # for battery/gps events: latest acceptable response time
    )
    certified: bool = False
    label_version: str = "label_v1"

    def canonical_json(self) -> str:
        data = self.model_dump(mode="json", exclude={"certified"})
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @property
    def scenario_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]


__all__ = [
    "BoxSpec",
    "EventType",
    "MissionSpec",
    "ObstacleSpec",
    "ScenarioSpec",
    "SensorSpec",
    "SiteSpec",
    "StationSpec",
    "TargetSpec",
    "WindSpec",
    "ZoneSpec",
]
