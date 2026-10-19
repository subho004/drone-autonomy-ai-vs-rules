from __future__ import annotations

import time

from test_providers_common import make_obs

from app.domain.actions import Action
from app.domain.decision import DecisionProvider
from app.providers.mock import MockProvider


def test_mock_conforms_and_does_not_sleep() -> None:
    provider = MockProvider(latency_ms=5000.0)
    assert isinstance(provider, DecisionProvider)
    start = time.perf_counter()
    decision = provider.decide(make_obs())
    assert time.perf_counter() - start < 0.5
    assert decision.latency_ms == 5000.0
    assert decision.action is Action.FORWARD and decision.valid


def test_mock_optional_sleep() -> None:
    provider = MockProvider(latency_ms=30.0, sleep=True)
    start = time.perf_counter()
    provider.decide(make_obs())
    assert time.perf_counter() - start >= 0.025


def test_mock_callable_latency_and_policy() -> None:
    provider = MockProvider(
        latency_ms=lambda o: o.tick * 10.0, policy=lambda _o: Action.BRAKE
    )
    decision = provider.decide(make_obs(tick=7))
    assert (decision.latency_ms, decision.action) == (70.0, Action.BRAKE)


def test_oracle_plus_error_is_deterministic_and_calibrated() -> None:
    def run(seed: int) -> list[Action | None]:
        p = MockProvider(policy=lambda _o: Action.HOVER, seed=seed, error_rate=0.3)
        return [p.decide(make_obs()).action for _ in range(2000)]

    first = run(1)
    assert first == run(1)
    assert first != run(2)
    wrong = sum(a is not Action.HOVER for a in first) / len(first)
    assert 0.25 < wrong < 0.35


def test_zero_error_rate_always_matches_policy() -> None:
    p = MockProvider(policy=lambda _o: Action.LEFT, error_rate=0.0)
    assert all(p.decide(make_obs()).action is Action.LEFT for _ in range(50))
