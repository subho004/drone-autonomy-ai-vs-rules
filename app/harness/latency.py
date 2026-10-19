"""Latency policies: how a provider's wall-clock latency becomes sim ticks.

The harness owns time. A decision requested at control tick ``t`` is applied
at ``t + ceil(L / dt)``; until then the previous action stays in force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from app.domain.decision import Decision
from app.sim.world import DT


class LatencyMode(StrEnum):
    LOCKSTEP = "lockstep"  # ablation only: decision applied the same tick
    FIXED = "fixed"  # constant injected latency (the sweep cells)
    SAMPLED = "sampled"  # draw from a recorded latency trace (seeded)
    MEASURED = "measured"  # the provider's own measured wall-clock latency


MIN_TICKS = 1
TRACE_SAMPLES = 4096
DEFAULT_SIGMA = 0.4  # lognormal spread of a human-like reaction time


def lognormal_trace(median_ms: float, sigma: float = DEFAULT_SIGMA) -> list[float]:
    """A fixed latency trace with the given median (heavy right tail), same for every flight."""

    draws = np.random.default_rng(20_260_101).lognormal(
        math.log(median_ms), sigma, TRACE_SAMPLES
    )
    return [float(d) for d in draws]


# sense -> act is never faster than one control tick


@dataclass(slots=True)
class LatencyPolicy:
    mode: LatencyMode = LatencyMode.MEASURED
    fixed_ms: float = 0.0
    trace_ms: list[float] = field(default_factory=list)
    seed: int = 0
    _rng: np.random.Generator | None = field(default=None, repr=False)

    def reset(self, flight_seed: int) -> None:
        self._rng = np.random.default_rng([self.seed, flight_seed])

    def resolve(self, decision: Decision) -> tuple[float, int]:
        """Return (latency_ms used, latency in control ticks)."""

        if self.mode == LatencyMode.LOCKSTEP:
            return decision.latency_ms, 0
        if self.mode == LatencyMode.FIXED:
            ms = self.fixed_ms
        elif self.mode == LatencyMode.SAMPLED and self.trace_ms:
            if self._rng is None:
                self._rng = np.random.default_rng(self.seed)
            ms = float(self.trace_ms[int(self._rng.integers(len(self.trace_ms)))])
        else:
            ms = decision.latency_ms
        return ms, max(MIN_TICKS, math.ceil(ms / 1000.0 / DT - 1e-9))


__all__ = [
    "DEFAULT_SIGMA",
    "MIN_TICKS",
    "LatencyMode",
    "LatencyPolicy",
    "lognormal_trace",
]
