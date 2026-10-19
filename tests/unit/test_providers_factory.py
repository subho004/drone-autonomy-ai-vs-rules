from __future__ import annotations

import pytest
from test_providers_common import make_obs, memory_db

from app.domain.decision import DecisionProvider
from app.harness.store import Store
from app.providers.budget import BudgetLedger
from app.providers.cache import ResponseCache
from app.providers.decisions_endpoint import DecisionsEndpointProvider
from app.providers.factory import build_provider
from app.providers.gemini import GeminiProvider
from app.providers.mock import MockProvider
from app.providers.openai_chat import OpenAIChatProvider


def test_factory_builds_each_kind() -> None:
    cache, ledger = ResponseCache(memory_db()), BudgetLedger(memory_db())
    mock = build_provider({"provider": "mock", "latency_ms": 250, "seed": 4})
    assert isinstance(mock, MockProvider)
    assert mock.decide(make_obs()).latency_ms == 250.0

    jev = build_provider(
        {
            "provider": "decisions_endpoint",
            "model": "typesafe/jev-1.13",
            "timeout_s": 2.0,
        },
        cache=cache,
        ledger=ledger,
    )
    assert isinstance(jev, DecisionsEndpointProvider)
    assert jev.provider_id == "decisions:typesafe/jev-1.13"

    oai = build_provider(
        {"provider": "openai_chat", "model": "gpt-6-luna", "reasoning_effort": "none"}
    )
    orc = build_provider({"provider": "openrouter_chat", "model": "openai/gpt-4o-mini"})
    gem = build_provider({"provider": "gemini", "model": "gemini-3.5-flash-lite"})
    assert (
        isinstance(oai, OpenAIChatProvider)
        and oai.provider_id == "openai_chat:gpt-6-luna"
    )
    assert isinstance(orc, OpenAIChatProvider) and orc.provider_id.startswith(
        "openrouter_chat:"
    )
    assert isinstance(gem, GeminiProvider)
    for provider in (mock, jev, oai, orc, gem):
        assert isinstance(provider, DecisionProvider)
        provider.close()


def test_factory_unknown_kind() -> None:
    with pytest.raises(ValueError):
        build_provider({"provider": "carrier_pigeon"})


def test_factory_accepts_store_arm_config() -> None:
    store = Store(":memory:")
    store.upsert_arm(
        "jev", "model", "decisions_endpoint", "typesafe/jev-1.13", {"encoding": "json"}
    )
    provider = build_provider(store.arm_config("jev"))
    assert provider.provider_id == "decisions:typesafe/jev-1.13"
    provider.close()
