"""Gemini generateContent provider (no probabilities).

Auth is the ``x-goog-api-key`` header, never a query string. Probe findings
(docs/research/16c): ``gemini-3.5-flash-lite`` with ``thinkingLevel='minimal'``
works with ``maxOutputTokens`` >= 16 (4 returned an empty reply).
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.domain.actions import ALIAS_TO_ACTION
from app.domain.decision import Decision
from app.providers.base import PreparedRequest, RemoteProvider, invalid_decision
from app.providers.openai_chat import estimate_cost_usd
from app.providers.parsing import ParseStatus, parse_action_ex
from app.providers.prompts import ALIAS_ANSWER_REMINDER, SYSTEM_PROMPT_ALIAS

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_OUTPUT_TOKENS = 16
# Assumed (not fetched) USD per 1M tokens; override with price_in / price_out.
ASSUMED_PRICE_PER_M = (0.10, 0.40)


class GeminiProvider(RemoteProvider):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str = GEMINI_BASE_URL,
        thinking_level: str | None = "minimal",
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        structured_enum: bool = False,
        price_in: float | None = None,
        price_out: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            family="gemini",
            model=model,
            api_key=api_key
            if api_key is not None
            else settings.gemini_api_key.get_secret_value(),
            **kwargs,
        )
        self._url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
        self._thinking_level = thinking_level
        self._max_output_tokens = max_output_tokens
        self._structured_enum = structured_enum
        self._price_in = ASSUMED_PRICE_PER_M[0] if price_in is None else price_in
        self._price_out = ASSUMED_PRICE_PER_M[1] if price_out is None else price_out

    def _prepare(self, observation_text: str) -> PreparedRequest:
        config: dict[str, Any] = {
            "temperature": 0,
            "maxOutputTokens": self._max_output_tokens,
        }
        if self._thinking_level is not None:
            config["thinkingConfig"] = {"thinkingLevel": self._thinking_level}
        if self._structured_enum:
            config["responseMimeType"] = "text/x.enum"
            config["responseSchema"] = {
                "type": "STRING",
                "enum": sorted(ALIAS_TO_ACTION),
            }
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT_ALIAS}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{observation_text}\n\n{ALIAS_ANSWER_REMINDER}"}
                    ],
                }
            ],
            "generationConfig": config,
        }
        return PreparedRequest(self._url, body, {"x-goog-api-key": self._api_key})

    def _interpret(self, payload: dict[str, Any]) -> Decision:
        usage = payload.get("usageMetadata") or {}
        tokens_in = usage.get("promptTokenCount")
        tokens_out = (usage.get("candidatesTokenCount") or 0) + (
            usage.get("thoughtsTokenCount") or 0
        )
        common: dict[str, Any] = {
            "model_id": payload.get("modelVersion"),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": estimate_cost_usd(
                self._model, tokens_in, tokens_out, self._price_in, self._price_out
            ),
        }
        candidates = payload.get("candidates") or []
        parts = (
            ((candidates[0].get("content") or {}).get("parts") or [])
            if candidates
            else []
        )
        text = "".join(str(p.get("text", "")) for p in parts)
        action, status = parse_action_ex(text)
        if action is None:
            error = {
                ParseStatus.EMPTY: "empty_output",
                ParseStatus.AMBIGUOUS: "ambiguous_output",
            }.get(status, "invalid_output")
            return invalid_decision(error, raw=text or None, **common)
        return Decision(action=action, latency_ms=0.0, raw=text, **common)


__all__ = ["DEFAULT_MODEL", "GEMINI_BASE_URL", "GeminiProvider"]
