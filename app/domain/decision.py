"""Decision records returned by providers and recorded by the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from app.domain.actions import Action
from app.domain.observation import Observation


class DecisionSource(StrEnum):
    MODEL = "model"
    FALLBACK = "fallback"
    SHIELD = "shield"


@dataclass(slots=True)
class Decision:
    """What a provider returned for one observation.

    ``latency_ms`` is the measured wall-clock request-to-parsed-action time
    (``perf_counter_ns``); for local classical arms it is the compute time.
    ``action`` is None when the output was invalid (parse failure, timeout).
    """

    action: Action | None
    latency_ms: float
    probabilities: dict[Action, float] | None = None
    confidence: float | None = None
    valid: bool = True
    error: str | None = None
    raw: str | None = None
    model_id: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float = 0.0
    from_cache: bool = False
    extras: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class DecisionProvider(Protocol):
    """Anything that maps an Observation to a Decision.

    The harness owns latency, staleness, fallback and shielding. Providers
    only decide, and report how long deciding took. Implementations must be
    safe to call from a single worker thread; they hold their own HTTP client.
    """

    @property
    def provider_id(self) -> str: ...

    def decide(self, obs: Observation) -> Decision: ...

    def close(self) -> None: ...


__all__ = ["Decision", "DecisionProvider", "DecisionSource"]
