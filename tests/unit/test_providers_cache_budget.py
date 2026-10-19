from __future__ import annotations

import httpx
import pytest
from test_providers_common import make_obs, memory_db

from app.domain.actions import Action
from app.domain.decision import Decision
from app.providers.budget import BudgetExceededError, BudgetLedger
from app.providers.cache import ResponseCache
from app.providers.decisions_endpoint import DecisionsEndpointProvider


def jev_handler(calls: list[int]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13-20260917",
                "answers": {
                    "action": {
                        "choice": "BRAKE",
                        "probabilities": {"BRAKE": 0.9},
                        "confidence": 0.9,
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.01},
            },
        )

    return httpx.MockTransport(handler)


def test_cache_key_is_stable_and_param_sensitive() -> None:
    k1 = ResponseCache.make_key("p", "m", {"a": 1, "b": 2})
    assert k1 == ResponseCache.make_key("p", "m", {"b": 2, "a": 1})
    assert k1 != ResponseCache.make_key("p", "m", {"a": 1, "b": 3})
    assert k1 != ResponseCache.make_key("p", "m2", {"a": 1, "b": 2})
    assert len(k1) == 64


def test_cache_roundtrip_keeps_original_latency() -> None:
    cache = ResponseCache(memory_db())
    key = ResponseCache.make_key("p", "m", {"x": 1})
    assert cache.get(key) is None
    original = Decision(
        action=Action.BRAKE,
        latency_ms=512.5,
        probabilities={Action.BRAKE: 0.9},
        confidence=0.9,
        model_id="snap",
        tokens_in=10,
        cost_usd=0.01,
    )
    cache.put(key, "p", original)
    hit = cache.get(key)
    assert hit is not None and hit.from_cache
    assert hit.action is Action.BRAKE and hit.latency_ms == 512.5
    assert hit.probabilities == {Action.BRAKE: 0.9} and hit.cost_usd == 0.0
    assert (cache.hits, cache.misses) == (1, 1)


def test_invalid_decisions_are_not_cached() -> None:
    cache = ResponseCache(memory_db())
    key = ResponseCache.make_key("p", "m", {})
    cache.put(key, "p", Decision(action=None, latency_ms=1.0, valid=False, error="x"))
    assert cache.get(key) is None


def test_provider_uses_cache_second_call_skips_network() -> None:
    calls: list[int] = []
    conn = memory_db()
    provider = DecisionsEndpointProvider(
        api_key="k", transport=jev_handler(calls), cache=ResponseCache(conn)
    )
    first = provider.decide(make_obs())
    second = provider.decide(make_obs())
    other = provider.decide(make_obs(battery_pct=5.0))
    assert len(calls) == 2
    assert not first.from_cache and second.from_cache and not other.from_cache
    assert second.action is first.action and second.latency_ms == first.latency_ms


def test_ledger_records_and_sums() -> None:
    ledger = BudgetLedger(memory_db(), cap_usd=1.0)
    assert ledger.spent_usd() == 0.0
    ledger.record("decisions", "m", 0.25)
    ledger.record("decisions", "m", 0.25)
    assert ledger.spent_usd() == pytest.approx(0.5)
    assert ledger.remaining_usd() == pytest.approx(0.5)
    ledger.check(0.4)
    with pytest.raises(BudgetExceededError):
        ledger.check(0.6)


def test_ledger_default_cap_from_settings() -> None:
    from app.core.config import settings

    assert BudgetLedger(memory_db()).cap_usd == settings.budget_usd


def test_provider_stops_before_exceeding_cap() -> None:
    calls: list[int] = []
    ledger = BudgetLedger(memory_db(), cap_usd=0.015)
    provider = DecisionsEndpointProvider(
        api_key="k", transport=jev_handler(calls), ledger=ledger, est_cost_usd=0.001
    )
    provider.decide(make_obs())  # spent 0.01, next estimate fits under the cap
    provider.decide(make_obs())  # spent 0.02, over the cap
    with pytest.raises(BudgetExceededError):
        provider.decide(make_obs())
    assert len(calls) == 2
    assert ledger.spent_usd() == pytest.approx(0.02)
