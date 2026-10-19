"""Chat Completions provider (OpenAI direct or OpenRouter).

One output letter A..K per decision, ``temperature`` 0, and ``logprobs`` over
the first generated token to recover per-action probabilities. Quirks from
docs/research/16c: gpt-6-luna needs ``reasoning_effort='none'``,
``max_completion_tokens`` > 2 and ``top_logprobs`` <= 5; gpt-4o-mini and
gpt-4.1-nano accept 20. Probabilities are renormalised over the 11 aliases.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.domain.decision import Decision
from app.providers.base import PreparedRequest, RemoteProvider, invalid_decision
from app.providers.parsing import (
    ParseStatus,
    parse_action_ex,
    probabilities_from_logprobs,
)
from app.providers.prompts import ALIAS_ANSWER_REMINDER, SYSTEM_PROMPT_ALIAS

OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_COMPLETION_TOKENS = 16
LUNA_TOP_LOGPROBS = 5
DEFAULT_TOP_LOGPROBS = 20

# (input, output) USD per 1M tokens; override per arm with price_in / price_out.
PRICING_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-6-luna": (0.10, 0.50),
}


def default_top_logprobs(model: str) -> int:
    return LUNA_TOP_LOGPROBS if "gpt-6" in model else DEFAULT_TOP_LOGPROBS


def estimate_cost_usd(
    model: str,
    tokens_in: int | None,
    tokens_out: int | None,
    price_in: float | None = None,
    price_out: float | None = None,
) -> float:
    """USD from token counts; 0.0 when the model's price is unknown."""

    base = model.split("/")[-1]
    default_in, default_out = PRICING_PER_M.get(base, (0.0, 0.0))
    p_in = default_in if price_in is None else price_in
    p_out = default_out if price_out is None else price_out
    return ((tokens_in or 0) * p_in + (tokens_out or 0) * p_out) / 1_000_000


class OpenAIChatProvider(RemoteProvider):
    """``flavor='openai'`` for api.openai.com, ``'openrouter'`` for OpenRouter."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        flavor: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        top_logprobs: int | None = None,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        routing: dict[str, Any] | None = None,
        price_in: float | None = None,
        price_out: float | None = None,
        **kwargs: Any,
    ) -> None:
        if flavor not in ("openai", "openrouter"):
            raise ValueError(f"unknown flavor: {flavor}")
        secret = (
            settings.openai_api_key
            if flavor == "openai"
            else settings.openrouter_api_key
        )
        super().__init__(
            family="openai_chat" if flavor == "openai" else "openrouter_chat",
            model=model,
            api_key=api_key if api_key is not None else secret.get_secret_value(),
            **kwargs,
        )
        default_url = (
            OPENAI_BASE_URL if flavor == "openai" else settings.openrouter_base_url
        )
        self._url = f"{(base_url or default_url).rstrip('/')}/chat/completions"
        self._flavor = flavor
        self._reasoning_effort = reasoning_effort
        self._top_logprobs = top_logprobs or default_top_logprobs(model)
        self._max_tokens = max_completion_tokens
        self._routing = routing
        self._price_in = price_in
        self._price_out = price_out

    def _prepare(self, observation_text: str) -> PreparedRequest:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_ALIAS},
                {
                    "role": "user",
                    "content": f"{observation_text}\n\n{ALIAS_ANSWER_REMINDER}",
                },
            ],
            "max_completion_tokens": self._max_tokens,
            "logprobs": True,
            "top_logprobs": self._top_logprobs,
        }
        effort = self._reasoning_effort
        if effort is not None:
            if self._flavor == "openai":
                body["reasoning_effort"] = effort
            else:
                body["reasoning"] = {"effort": effort}
        if effort in (None, "none"):
            body["temperature"] = 0
        if self._routing:
            body["provider"] = self._routing
        return PreparedRequest(
            self._url, body, {"Authorization": f"Bearer {self._api_key}"}
        )

    def _interpret(self, payload: dict[str, Any]) -> Decision:
        choice = payload["choices"][0]
        content = choice["message"].get("content")
        usage = payload.get("usage") or {}
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        cost = usage.get("cost")
        cost_usd = (
            float(cost)
            if cost is not None
            else estimate_cost_usd(
                self._model, tokens_in, tokens_out, self._price_in, self._price_out
            )
        )
        probs = None
        entries = ((choice.get("logprobs") or {}).get("content")) or []
        if entries:
            parsed = probabilities_from_logprobs(
                (e["token"], float(e["logprob"])) for e in entries[0]["top_logprobs"]
            )
            if parsed is not None:
                probs = parsed[0]
        action, status = parse_action_ex(content)
        if action is None:
            return invalid_decision(
                "ambiguous_output"
                if status is ParseStatus.AMBIGUOUS
                else "invalid_output",
                raw=content,
                model_id=payload.get("model"),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                probabilities=probs,
            )
        return Decision(
            action=action,
            latency_ms=0.0,
            probabilities=probs,
            confidence=None if probs is None else max(probs.values()),
            raw=content,
            model_id=payload.get("model"),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )


__all__ = [
    "DEFAULT_TOP_LOGPROBS",
    "LUNA_TOP_LOGPROBS",
    "OPENAI_BASE_URL",
    "PRICING_PER_M",
    "OpenAIChatProvider",
    "default_top_logprobs",
    "estimate_cost_usd",
]
