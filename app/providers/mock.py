"""Deterministic offline provider for harness tests and latency x accuracy sweeps.

By default it does NOT sleep: ``Decision.latency_ms`` reports the configured
latency so a simulated-time harness can replay it. With ``error_rate > 0`` and
an oracle ``policy`` it becomes "oracle plus error": each call replaces the
policy's action with a different uniformly drawn action with that probability.
Output depends only on the seed and the call sequence.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from app.domain.actions import Action
from app.domain.decision import Decision
from app.domain.observation import Observation

LatencyMs = float | Callable[[Observation], float]
Policy = Callable[[Observation], Action]


class MockProvider:
    def __init__(
        self,
        latency_ms: LatencyMs = 0.0,
        policy: Policy | None = None,
        seed: int = 0,
        *,
        error_rate: float = 0.0,
        sleep: bool = False,
        provider_id: str = "mock",
    ) -> None:
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be in [0, 1]")
        self._latency_ms = latency_ms
        self._policy: Policy = policy or (lambda _obs: Action.FORWARD)
        self._error_rate = error_rate
        self._sleep = sleep
        self._provider_id = provider_id
        self._rng = random.Random(seed)

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def decide(self, obs: Observation) -> Decision:
        latency = (
            self._latency_ms(obs) if callable(self._latency_ms) else self._latency_ms
        )
        action = self._policy(obs)
        if self._error_rate > 0.0 and self._rng.random() < self._error_rate:
            action = self._rng.choice([a for a in Action if a is not action])
        if self._sleep and latency > 0.0:
            time.sleep(latency / 1000.0)
        return Decision(action=action, latency_ms=float(latency), model_id="mock")

    def close(self) -> None:
        return None


__all__ = ["LatencyMs", "MockProvider", "Policy"]
