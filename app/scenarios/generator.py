"""Budget-first scenario generation and oracle certification.

A scenario's difficulty is its *reaction budget* ``B`` (seconds): the extra
time beyond the minimum avoidance distance that a decision layer has before
avoidance becomes physically impossible. We sample ``B`` directly, then
derive the spawn distance from the calibrated minimum, so every uncertified
draw is avoidable by construction, and certification (zero-latency oracle
replay on the full spec incl. wind/noise) proves it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from app.controllers.arms import OracleArm
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.scenarios.calibration import (
    CELLS,
    KAPPA_MARGIN,
    V_CRUISE,
    cell_key,
    closing_speed,
    crossing_start_y,
)
from app.scenarios.spec import ObstacleSpec, ScenarioSpec, WindSpec
from app.sim.rng import channel_generator
from app.sim.world import DroneWorld

B_MIN_S, B_MAX_S = 0.03, 3.0  # default headline budget range (s)
IMPOSSIBLE_B_RANGE = (-0.30, -0.05)
MAX_WIND_ACCEL = 1.5  # m/s^2, below the 0.5 * a_brake authority cap
TIERS: tuple[tuple[str, float, float], ...] = (
    ("T0", 2.0, math.inf),
    ("T1", 1.0, 2.0),
    ("T2", 0.5, 1.0),
    ("T3", 0.25, 0.5),
    ("T4", 0.1, 0.25),
    ("T5", 0.0, 0.1),
)
DEFAULT_MIX: dict[str, float] = {
    "obstacle_static": 0.35,
    "obstacle_approach": 0.30,
    "intruder_cross": 0.15,
    "battery_critical": 0.06,
    "gps_loss": 0.06,
    "clear": 0.08,
}
CHALLENGE_EVENTS = ("obstacle_static", "obstacle_approach", "intruder_cross")


def tier_for(budget_s: float) -> str:
    if budget_s < 0.0:
        return "T-1"
    for name, lo, hi in TIERS:
        if lo <= budget_s < hi:
            return name
    return "T5"


@dataclass(slots=True)
class GenerationStats:
    drawn: int = 0
    rejected: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)


class ScenarioGenerator:
    """Seeded generator over the calibrated avoidance table."""

    def __init__(self, calibration: dict[str, float]) -> None:
        self._cal = calibration

    # ----------------------------------------------------------------- draws
    def draw(
        self,
        seed: int,
        suite: str,
        *,
        mix: dict[str, float] | None = None,
        tier: str | None = None,
        impossible: bool = False,
        events: tuple[str, ...] | None = None,
        budget_range: tuple[float, float] | None = None,
    ) -> ScenarioSpec:
        """One uncertified scenario for ``seed`` (deterministic).

        ``budget_range`` narrows the log-uniform reaction-budget draw (seconds), e.g. to
        the 0.4-4 s band where a ~0.5 s decision layer has a fair chance.
        """

        gen = channel_generator(seed, "gen")
        weights = mix or DEFAULT_MIX
        names = [n for n in weights if events is None or n in events]
        probs = np.array([weights[n] for n in names], dtype=float)
        event = str(gen.choice(names, p=probs / probs.sum()))
        wind, noise_sigma, dropout = self._modifiers(gen)
        spec = ScenarioSpec(seed=seed, suite=suite, event_type=event)  # type: ignore[arg-type]
        spec.wind = wind
        spec.sensors.ray_noise_sigma = noise_sigma
        spec.sensors.ray_dropout_p = dropout
        if event in CHALLENGE_EVENTS:
            return self._obstacle(spec, gen, tier, impossible, budget_range)
        if event == "battery_critical":
            t = float(gen.uniform(3.0, 6.0))
            spec.sensors.battery_drop_at_s = t
            spec.sensors.battery_death_after_s = 8.0
            spec.deadline_s = 6.0
            spec.budget_s = 6.0
            spec.tier = "D"
            spec.max_time_s = 24.0
        elif event == "gps_loss":
            spec.sensors.gps_loss_at_s = float(gen.uniform(3.0, 6.0))
            spec.deadline_s = 2.0
            spec.budget_s = 2.0
            spec.tier = "D"
        else:
            spec.budget_s = 99.0
            spec.tier = "C"
        return spec

    def _modifiers(self, gen: np.random.Generator) -> tuple[WindSpec, float, float]:
        wind = WindSpec()
        if gen.random() < 0.25:
            ang = float(gen.uniform(0, 2 * math.pi))
            mag = float(gen.uniform(0.3, MAX_WIND_ACCEL))
            wind = WindSpec(
                mean_x=mag * math.cos(ang),
                mean_y=mag * math.sin(ang),
                step_at_s=float(gen.uniform(2.0, 5.0)),
                turb_sigma=float(gen.uniform(0.0, 0.4)),
            )
        noise = float(gen.uniform(0.02, 0.06))
        dropout = 0.02 if gen.random() < 0.3 else 0.0
        return wind, noise, dropout

    def _obstacle(
        self,
        spec: ScenarioSpec,
        gen: np.random.Generator,
        tier: str | None,
        impossible: bool,
        budget_range: tuple[float, float] | None = None,
    ) -> ScenarioSpec:
        kind = {
            "obstacle_static": "static",
            "obstacle_approach": "approach",
            "intruder_cross": "cross",
        }[spec.event_type]
        cells = [c for c in CELLS if c[0] == kind]
        _, speed = cells[int(gen.integers(len(cells)))]
        d0 = self._cal[cell_key(kind, speed)] * (1.0 + KAPPA_MARGIN)
        v_c = closing_speed(kind, speed)
        if impossible:
            budget = float(gen.uniform(*IMPOSSIBLE_B_RANGE))
        elif tier is not None:
            lo, hi = next((a, b) for n, a, b in TIERS if n == tier)
            hi = min(hi, B_MAX_S)
            budget = float(
                np.exp(gen.uniform(math.log(max(lo, B_MIN_S)), math.log(hi)))
            )
        else:
            lo_b, hi_b = budget_range or (B_MIN_S, B_MAX_S)
            budget = float(np.exp(gen.uniform(math.log(lo_b), math.log(hi_b))))
        d_spawn = max(d0 + v_c * budget, 0.3)
        spec.budget_s = budget
        spec.tier = tier_for(budget)
        spec.obstacle = ObstacleSpec(
            kind=kind,  # type: ignore[arg-type]
            trigger_x=float(4.0 + gen.uniform(0.0, 2.0)),
            d_spawn=d_spawn,
            lateral_offset=float(gen.uniform(-0.1, 0.1)) if kind != "cross" else 0.0,
            speed=speed,
            cross_side=1 if gen.random() < 0.5 else -1,
            cross_start_y=crossing_start_y(speed, d_spawn) if kind == "cross" else 0.0,
        )
        return spec

    # ----------------------------------------------------------- certification
    def certify(self, world: DroneWorld, spec: ScenarioSpec) -> tuple[bool, str]:
        """Replay with the zero-latency oracle; (ok, reason)."""

        cfg = FlightConfig(
            LatencyPolicy(LatencyMode.MEASURED),
            shield=False,
            record_trajectory=False,
            label_decisions=False,
        )
        res = run_flight(world, spec, OracleArm(V_CRUISE), cfg)
        m = res.metrics
        if spec.budget_s < 0.0 and spec.event_type in CHALLENGE_EVENTS:
            return (m["collision"] == 1, "impossible-not-a-collision")
        if m["success"] == 1:
            return True, "ok"
        return False, f"oracle-{m['outcome']}"

    def generate_certified(
        self,
        world: DroneWorld,
        n: int,
        base_seed: int,
        suite: str,
        *,
        mix: dict[str, float] | None = None,
        tier: str | None = None,
        impossible: bool = False,
        events: tuple[str, ...] | None = None,
        budget_range: tuple[float, float] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[ScenarioSpec], GenerationStats]:
        """Draw until ``n`` scenarios pass certification (seeds advance on rejection)."""

        out: list[ScenarioSpec] = []
        stats = GenerationStats()
        seed = base_seed
        while len(out) < n:
            spec = self.draw(
                seed,
                suite,
                mix=mix,
                tier=tier,
                impossible=impossible,
                events=events,
                budget_range=budget_range,
            )
            seed += 1
            stats.drawn += 1
            ok, reason = self.certify(world, spec)
            if ok:
                spec.certified = True
                out.append(spec)
                if progress:
                    progress(len(out), n)
            else:
                stats.rejected += 1
                stats.by_reason[reason] = stats.by_reason.get(reason, 0) + 1
        return out, stats


__all__ = [
    "CHALLENGE_EVENTS",
    "DEFAULT_MIX",
    "TIERS",
    "GenerationStats",
    "ScenarioGenerator",
    "tier_for",
]
