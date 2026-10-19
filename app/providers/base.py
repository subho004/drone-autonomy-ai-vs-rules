"""Shared plumbing for HTTP-backed providers.

Template method: subclasses build one request and interpret one JSON payload;
this base owns the persistent client, timing, caching, budget accounting and
the "never raise, return an invalid Decision" contract. There are no retries
here: the harness owns fallback.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.logging import get_logger
from app.domain.decision import Decision
from app.domain.observation import Observation
from app.providers.budget import BudgetLedger
from app.providers.cache import ResponseCache
from app.providers.encoding import ObservationEncoder

logger = get_logger(__name__)

DEFAULT_TIMEOUT_S = 3.0
DEFAULT_EST_COST_USD = 0.0002
MAX_ERROR_BODY_CHARS = 160


def invalid_decision(
    error: str,
    *,
    latency_ms: float = 0.0,
    raw: str | None = None,
    model_id: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_usd: float = 0.0,
    probabilities: dict[Any, float] | None = None,
) -> Decision:
    return Decision(
        action=None,
        latency_ms=latency_ms,
        valid=False,
        error=error,
        raw=raw,
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        probabilities=probabilities,
    )


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    url: str
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


class RemoteProvider(ABC):
    """Base for providers that POST one JSON request per decision."""

    def __init__(
        self,
        *,
        family: str,
        model: str,
        api_key: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        encoder: ObservationEncoder | None = None,
        cache: ResponseCache | None = None,
        ledger: BudgetLedger | None = None,
        transport: httpx.BaseTransport | None = None,
        http2: bool = True,
        est_cost_usd: float = DEFAULT_EST_COST_USD,
    ) -> None:
        self._family = family
        self._model = model
        self._api_key = api_key
        self._encoder = encoder or ObservationEncoder()
        self._cache = cache
        self._ledger = ledger
        self._est_cost_usd = est_cost_usd
        self._client = httpx.Client(
            http2=http2, timeout=httpx.Timeout(timeout_s), transport=transport
        )

    @property
    def provider_id(self) -> str:
        return f"{self._family}:{self._model}"

    @abstractmethod
    def _prepare(self, observation_text: str) -> PreparedRequest:
        """Build the request for one encoded observation (no key in ``body``)."""

    @abstractmethod
    def _interpret(self, payload: dict[str, Any]) -> Decision:
        """Turn a 200 JSON payload into a Decision (latency is set by the base)."""

    def decide(self, obs: Observation) -> Decision:
        if not self._api_key:
            return invalid_decision("missing_api_key")
        request = self._prepare(self._encoder.encode(obs))
        key = None
        if self._cache is not None:
            key = self._cache.make_key(self._family, self._model, request.body)
            hit = self._cache.get(key)
            if hit is not None:
                return hit
        if self._ledger is not None:
            self._ledger.check(self._est_cost_usd)
        started = time.perf_counter_ns()
        decision = self._send(request)
        decision.latency_ms = (time.perf_counter_ns() - started) / 1e6
        if self._ledger is not None:
            self._ledger.record(self._family, self._model, decision.cost_usd)
        if self._cache is not None and key is not None:
            self._cache.put(key, self._family, decision)
        return decision

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "***")[:MAX_ERROR_BODY_CHARS]

    def _send(self, request: PreparedRequest) -> Decision:
        try:
            response = self._client.post(
                request.url, json=request.body, headers=request.headers
            )
        except httpx.TimeoutException:
            logger.warning("%s timed out", self.provider_id)
            return invalid_decision("timeout")
        except httpx.HTTPError as exc:
            logger.warning(
                "%s transport error: %s", self.provider_id, type(exc).__name__
            )
            return invalid_decision("transport_error")
        if response.status_code != 200:
            logger.warning("%s HTTP %d", self.provider_id, response.status_code)
            return invalid_decision(
                f"http_{response.status_code}", raw=self._scrub(response.text)
            )
        try:
            payload = response.json()
        except ValueError:
            return invalid_decision("malformed_json", raw=self._scrub(response.text))
        if not isinstance(payload, dict):
            return invalid_decision("malformed_json", raw=self._scrub(response.text))
        try:
            return self._interpret(payload)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            logger.warning(
                "%s unexpected payload: %s", self.provider_id, type(exc).__name__
            )
            return invalid_decision(
                "unexpected_payload", raw=self._scrub(response.text)
            )

    def close(self) -> None:
        self._client.close()


__all__ = [
    "DEFAULT_EST_COST_USD",
    "DEFAULT_TIMEOUT_S",
    "PreparedRequest",
    "RemoteProvider",
    "invalid_decision",
]
