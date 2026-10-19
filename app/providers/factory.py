"""Arm config -> DecisionProvider.

``arm_config`` is the dict returned by ``Store.arm_config()``: the stored
``config_json`` merged with ``kind``, ``provider`` and ``model``. Recognised
``provider`` values: mock, decisions_endpoint, openai_chat, openrouter_chat,
gemini. Optional keys (all JSON-serialisable): timeout_s, encoding, est_cost_usd,
base_url, routing, reasoning_effort, top_logprobs, max_completion_tokens,
thinking_level, structured_enum, price_in, price_out; for mock: latency_ms,
error_rate, seed, sleep.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.domain.decision import DecisionProvider
from app.providers.budget import BudgetLedger
from app.providers.cache import ResponseCache
from app.providers.decisions_endpoint import DEFAULT_MODEL, DecisionsEndpointProvider
from app.providers.encoding import ObservationEncoder
from app.providers.gemini import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from app.providers.gemini import GeminiProvider
from app.providers.mock import MockProvider
from app.providers.openai_chat import OpenAIChatProvider

_COMMON_KEYS = ("timeout_s", "est_cost_usd")
_OPENAI_KEYS = (
    "base_url",
    "routing",
    "reasoning_effort",
    "top_logprobs",
    "max_completion_tokens",
    "price_in",
    "price_out",
)
_GEMINI_KEYS = (
    "base_url",
    "thinking_level",
    "max_output_tokens",
    "structured_enum",
    "price_in",
    "price_out",
)


def _pick(config: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: config[k] for k in keys if k in config}


def _remote_kwargs(
    config: Mapping[str, Any], cache: ResponseCache | None, ledger: BudgetLedger | None
) -> dict[str, Any]:
    kwargs = _pick(config, _COMMON_KEYS)
    kwargs["encoder"] = ObservationEncoder(config.get("encoding", "hybrid"))
    kwargs["cache"] = cache
    kwargs["ledger"] = ledger
    return kwargs


def _mock(config: Mapping[str, Any]) -> DecisionProvider:
    return MockProvider(
        latency_ms=float(config.get("latency_ms", 0.0)),
        seed=int(config.get("seed", 0)),
        error_rate=float(config.get("error_rate", 0.0)),
        sleep=bool(config.get("sleep", False)),
    )


def build_provider(
    arm_config: Mapping[str, Any],
    *,
    cache: ResponseCache | None = None,
    ledger: BudgetLedger | None = None,
) -> DecisionProvider:
    """Build the provider named by ``arm_config['provider']``; ValueError if unknown."""

    kind = str(arm_config.get("provider", ""))
    model = str(arm_config.get("model") or "")
    if kind == "mock":
        return _mock(arm_config)
    remote = _remote_kwargs(arm_config, cache, ledger)
    if kind == "decisions_endpoint":
        return DecisionsEndpointProvider(
            model or DEFAULT_MODEL,
            base_url=arm_config.get("base_url"),
            routing=arm_config.get("routing"),
            **remote,
        )
    if kind in ("openai_chat", "openrouter_chat"):
        return OpenAIChatProvider(
            model or "gpt-4o-mini",
            flavor="openai" if kind == "openai_chat" else "openrouter",
            **_pick(arm_config, _OPENAI_KEYS),
            **remote,
        )
    if kind == "gemini":
        return GeminiProvider(
            model or GEMINI_DEFAULT_MODEL, **_pick(arm_config, _GEMINI_KEYS), **remote
        )
    raise ValueError(f"unknown provider kind: {kind!r}")


PROVIDER_KINDS: tuple[str, ...] = (
    "mock",
    "decisions_endpoint",
    "openai_chat",
    "openrouter_chat",
    "gemini",
)

__all__ = ["PROVIDER_KINDS", "build_provider"]
