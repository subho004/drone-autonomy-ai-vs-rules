"""Pydantic response models for the read-side API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Page[T](BaseModel):
    """Generic pagination envelope."""

    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int


class ExperimentOut(BaseModel):
    id: int
    name: str
    created_at: str
    status: str
    config_sha: str
    n_flights: int
    n_complete: int


class ArmProgress(BaseModel):
    arm_id: str
    total: int
    complete: int


class ProgressOut(BaseModel):
    experiment_id: int
    status: str
    total: int
    counts: dict[str, int]
    done: int
    remaining: int
    fraction: float
    rate_per_min: float | None
    eta_s: float | None
    complete: bool
    per_arm: list[ArmProgress]


class ArmOut(BaseModel):
    id: str
    kind: str
    provider: str
    model: str


class ScenarioOut(BaseModel):
    id: str
    suite: str
    seed: int
    event_type: str
    tier: str
    budget_s: float
    certified: bool


class ScenarioSpecOut(ScenarioOut):
    spec: dict[str, Any]


class CellKey(BaseModel):
    """A latency cell: one configuration an arm was flown under."""

    arm_id: str
    latency_mode: str
    latency_ms: float
    shield: int
    cadence: str


class ResultRow(CellKey):
    n: int
    collisions: int
    collision_rate: float
    collision_ci_low: float
    collision_ci_high: float
    successes: int
    success_rate: float
    success_ci_low: float
    success_ci_high: float
    lat_mean_ms: float | None
    lat_p95_ms: float | None
    path_length: float | None
    time_to_goal: float | None
    invalid_rate: float | None


class CurvePoint(BaseModel):
    latency_ms: float
    latency_measured: bool = Field(
        description="True when the x value is the measured mean (sampled mode)."
    )
    n: int
    collisions: int
    rate: float
    ci_low: float
    ci_high: float


class CurveSeries(BaseModel):
    arm_id: str
    latency_mode: str
    shield: int
    cadence: str
    label: str
    points: list[CurvePoint]


class BreakdownRow(CellKey):
    group: str
    n: int
    collisions: int
    rate: float
    ci_low: float
    ci_high: float


class LatencyCurvesOut(BaseModel):
    experiment_id: int
    series: list[CurveSeries]
    by_tier: list[BreakdownRow]
    by_event: list[BreakdownRow]


class EcdfPoint(BaseModel):
    ms: float
    p: float


class LatencyDistribution(BaseModel):
    provider: str
    model: str
    label: str
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    ecdf: list[EcdfPoint]


class LatencyDistributionOut(BaseModel):
    experiment_id: int
    series: list[LatencyDistribution]


class FlightOut(BaseModel):
    id: int
    experiment_id: int
    scenario_id: str
    arm_id: str
    latency_mode: str
    latency_ms: float
    shield: int
    cadence: str
    rep: int
    status: str
    outcome: str | None
    success: int | None
    collision: int | None
    t_end: float | None
    time_to_goal: float | None
    path_length: float | None
    min_clearance: float | None
    reaction_time_s: float | None
    n_decisions: int | None
    lat_mean_ms: float | None
    lat_p95_ms: float | None
    invalid_rate: float | None
    fallback_rate: float | None
    shield_overrides: int | None
    cost_usd: float | None
    event_type: str
    tier: str
    budget_s: float
    has_trajectory: bool


class DecisionOut(BaseModel):
    seq: int
    obs_tick: int
    apply_tick: int
    t_obs: float
    t_apply: float
    action: int | None
    applied_action: int
    applied_action_name: str
    source: str
    valid: int
    latency_ms: float
    latency_ticks: int
    confidence: float | None
    acceptable: int | None
    from_cache: int
    cost_usd: float


class TrajectoryColumns(BaseModel):
    """Column arrays (NaN becomes null), downsampled to at most 24 Hz."""

    t: list[float | None] = []
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    rx: list[float | None] = []
    ry: list[float | None] = []
    rz: list[float | None] = []
    action: list[int | None] = []
    shield: list[int | None] = []
    ox: list[float | None] = []
    oy: list[float | None] = []
    oz: list[float | None] = []
    orad: list[float | None] = []
    clearance: list[float | None] = []


class ReplayEvent(BaseModel):
    t: float
    kind: str
    label: str


class ReplayOut(BaseModel):
    flight: FlightOut
    scenario: ScenarioSpecOut
    hz: float
    trajectory: TrajectoryColumns
    decisions: list[DecisionOut]
    events: list[ReplayEvent]
