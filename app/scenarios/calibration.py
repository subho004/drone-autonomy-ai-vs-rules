"""Empirical avoidance distance d_min0 per obstacle family.

For each (kind, closing speed) we bisect the smallest spawn range at which the
zero-latency privileged oracle (one control tick) flies the scenario without
contact. The reaction budget of a scenario is then
``B = (d_spawn - d_min0 * (1 + KAPPA_MARGIN)) / v_closing``, so a generated
scenario is avoidable by construction and its difficulty is stated in seconds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.controllers.arms import OracleArm
from app.core.logging import get_logger
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.scenarios.spec import ObstacleSpec, ScenarioSpec
from app.sim.world import DroneWorld

logger = get_logger(__name__)

V_CRUISE = 2.0
KAPPA_MARGIN = 0.15
D_LO, D_HI = 0.4, 9.0
BISECT_STEPS = 7
CAL_VERSION = "cal_v1"

# (kind, obstacle speed) cells; closing speed = V_CRUISE + approach speed, or the
# lateral crossing speed for "cross" (time-to-conflict is set by the cross geometry).
CELLS: tuple[tuple[str, float], ...] = (
    ("static", 0.0),
    ("approach", 1.0),
    ("approach", 2.0),
    ("cross", 1.0),
    ("cross", 2.0),
)


def cell_key(kind: str, speed: float) -> str:
    return f"{kind}:{speed:.1f}"


def crossing_start_y(speed: float, d_spawn: float) -> float:
    """Start offset so a crosser reaches y=0 when a cruising drone reaches it."""

    return max(speed * d_spawn / V_CRUISE, 0.3)


def _spec_for(kind: str, speed: float, d_spawn: float, seed: int) -> ScenarioSpec:
    ob = ObstacleSpec(
        kind=kind,  # type: ignore[arg-type]
        trigger_x=4.0,
        d_spawn=d_spawn,
        speed=speed,
        cross_side=1,
        # cross: the obstacle reaches the flight path just as the drone arrives
        cross_start_y=crossing_start_y(speed, d_spawn) if kind == "cross" else 0.0,
    )
    event = {
        "static": "obstacle_static",
        "approach": "obstacle_approach",
        "cross": "intruder_cross",
    }[kind]
    return ScenarioSpec(seed=seed, event_type=event, obstacle=ob, v_cruise=V_CRUISE)  # type: ignore[arg-type]


def _oracle_avoids(
    world: DroneWorld, kind: str, speed: float, d_spawn: float, seed: int
) -> bool:
    spec = _spec_for(kind, speed, d_spawn, seed)
    cfg = FlightConfig(
        LatencyPolicy(LatencyMode.MEASURED),
        shield=False,
        record_trajectory=False,
        label_decisions=False,
    )
    res = run_flight(world, spec, OracleArm(V_CRUISE), cfg)
    return res.metrics["collision"] == 0


def calibrate_cell(world: DroneWorld, kind: str, speed: float, seed: int = 11) -> float:
    """Smallest d_spawn at which the oracle is collision-free (bisection)."""

    lo, hi = D_LO, D_HI
    if not _oracle_avoids(world, kind, speed, hi, seed):
        return hi  # even the widest spawn fails: treat as unreachable
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if _oracle_avoids(world, kind, speed, mid, seed):
            hi = mid
        else:
            lo = mid
    return hi


def calibrate(world: DroneWorld) -> dict[str, float]:
    """Run all cells; returns {cell_key: d_min0} (raw, before the margin)."""

    out: dict[str, float] = {}
    for kind, speed in CELLS:
        d = calibrate_cell(world, kind, speed)
        out[cell_key(kind, speed)] = round(d, 3)
        logger.info("calibrated %s -> d_min0=%.3f m", cell_key(kind, speed), d)
    return out


def calibration_id(table: dict[str, float]) -> str:
    blob = json.dumps({"v": CAL_VERSION, "t": table}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def save_calibration(table: dict[str, float], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cid = calibration_id(table)
    path.write_text(
        json.dumps({"id": cid, "version": CAL_VERSION, "d_min0": table}, indent=2)
    )
    return cid


def load_calibration(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    table: dict[str, float] = data["d_min0"]
    return table


def closing_speed(kind: str, speed: float) -> float:
    """Closing speed used to convert distance into time for the budget."""

    if kind == "static":
        return V_CRUISE
    if kind == "approach":
        return V_CRUISE + speed
    return max(speed, 0.5)  # cross: the lateral approach rate


__all__ = [
    "CELLS",
    "KAPPA_MARGIN",
    "V_CRUISE",
    "calibrate",
    "calibrate_cell",
    "calibration_id",
    "cell_key",
    "closing_speed",
    "crossing_start_y",
    "load_calibration",
    "save_calibration",
]
