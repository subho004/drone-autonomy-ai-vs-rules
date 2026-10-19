from __future__ import annotations

import pytest

from app.domain.actions import Action
from app.domain.decision import Decision
from app.harness.latency import MIN_TICKS, LatencyMode, LatencyPolicy
from app.sim.world import DT


def _d(ms: float) -> Decision:
    return Decision(action=Action.FORWARD, latency_ms=ms)


def test_lockstep_applies_in_the_same_tick() -> None:
    ms, ticks = LatencyPolicy(LatencyMode.LOCKSTEP).resolve(_d(900.0))
    assert ticks == 0 and ms == 900.0


def test_fixed_latency_converts_to_ceil_ticks() -> None:
    pol = LatencyPolicy(LatencyMode.FIXED, fixed_ms=100.0)
    ms, ticks = pol.resolve(_d(1.0))
    assert ms == 100.0 and ticks == 5  # ceil(0.100 / 0.02083)
    assert LatencyPolicy(LatencyMode.FIXED, fixed_ms=DT * 1000).resolve(_d(0))[1] == 1


def test_sub_tick_latency_is_never_faster_than_one_tick() -> None:
    for ms in (0.0, 0.05, 5.0):
        assert (
            LatencyPolicy(LatencyMode.FIXED, fixed_ms=ms).resolve(_d(0))[1] == MIN_TICKS
        )
    assert LatencyPolicy(LatencyMode.MEASURED).resolve(_d(0.2))[1] == MIN_TICKS


def test_measured_uses_the_providers_own_latency() -> None:
    ms, ticks = LatencyPolicy(LatencyMode.MEASURED).resolve(_d(520.0))
    assert ms == 520.0 and ticks == 25


def test_sampled_draws_from_the_trace_and_is_seeded() -> None:
    trace = [100.0, 300.0, 900.0]
    a = LatencyPolicy(LatencyMode.SAMPLED, trace_ms=trace, seed=1)
    b = LatencyPolicy(LatencyMode.SAMPLED, trace_ms=trace, seed=1)
    a.reset(7)
    b.reset(7)
    draws_a = [a.resolve(_d(1.0))[0] for _ in range(20)]
    draws_b = [b.resolve(_d(1.0))[0] for _ in range(20)]
    assert draws_a == draws_b and set(draws_a) <= set(trace) and len(set(draws_a)) > 1


@pytest.mark.parametrize("ms,expected", [(250.0, 12), (1000.0, 48), (3000.0, 144)])
def test_tick_counts_for_sweep_cells(ms: float, expected: int) -> None:
    assert LatencyPolicy(LatencyMode.FIXED, fixed_ms=ms).resolve(_d(0))[1] == expected
