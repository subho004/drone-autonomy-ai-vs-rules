"""OpenRouter Decisions endpoint provider (Jev, mercury-decide, ...).

``POST {root}/api/alpha/decisions`` with ``model``, ``state`` and one
``choice`` question. Verified live (docs/research/16c): the path has no
``/v1`` segment, auth is Bearer, and the reply carries
``answers.action{choice, probabilities, confidence}``, ``usage``, ``id``,
``provider`` and the dated snapshot id in ``model``.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.domain.actions import Action
from app.domain.decision import Decision
from app.providers.base import (
    PreparedRequest,
    RemoteProvider,
    invalid_decision,
)
from app.providers.parsing import ParseStatus, parse_action_ex
from app.providers.prompts import (
    DECISION_QUESTION,
    SYSTEM_PROMPT_NAME,
    action_criteria,
)

DEFAULT_MODEL = "typesafe/jev-1.13"
DECISIONS_PATH = "/api/alpha/decisions"


def decisions_url(openrouter_base_url: str) -> str:
    """``https://openrouter.ai/api/v1`` -> ``https://openrouter.ai/api/alpha/decisions``."""

    root = openrouter_base_url.rstrip("/")
    if root.endswith("/api/v1"):
        root = root[: -len("/v1")][: -len("/api")]
    elif root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}{DECISIONS_PATH}"


def _parse_probabilities(raw: object) -> dict[Action, float] | None:
    if not isinstance(raw, dict):
        return None
    probs: dict[Action, float] = {}
    for name, value in raw.items():
        action, _ = parse_action_ex(str(name))
        if action is not None:
            probs[action] = float(value)
    return probs or None


class DecisionsEndpointProvider(RemoteProvider):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        routing: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            family="decisions",
            model=model,
            api_key=api_key
            if api_key is not None
            else settings.openrouter_api_key.get_secret_value(),
            **kwargs,
        )
        self._url = decisions_url(base_url or settings.openrouter_base_url)
        self._routing = routing

    def _prepare(self, observation_text: str) -> PreparedRequest:
        body: dict[str, Any] = {
            "model": self._model,
            "state": f"{SYSTEM_PROMPT_NAME}\n\n{observation_text}",
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": DECISION_QUESTION,
                    "criteria": action_criteria(),
                }
            },
        }
        if self._routing:
            body["provider"] = self._routing
        return PreparedRequest(
            self._url, body, {"Authorization": f"Bearer {self._api_key}"}
        )

    def _interpret(self, payload: dict[str, Any]) -> Decision:
        answer = payload["answers"]["action"]
        usage = payload.get("usage") or {}
        common: dict[str, Any] = {
            "model_id": payload.get("model"),
            "tokens_in": usage.get("input_tokens"),
            "tokens_out": usage.get("output_tokens"),
            "cost_usd": float(usage.get("cost") or 0.0),
        }
        choice = answer.get("choice")
        action, status = parse_action_ex(None if choice is None else str(choice))
        if action is None or status is not ParseStatus.EXACT:
            return invalid_decision(
                "ambiguous_output"
                if status is ParseStatus.AMBIGUOUS
                else "invalid_output",
                raw=None if choice is None else str(choice),
                **common,
            )
        confidence = answer.get("confidence")
        return Decision(
            action=action,
            latency_ms=0.0,
            probabilities=_parse_probabilities(answer.get("probabilities")),
            confidence=None if confidence is None else float(confidence),
            raw=str(choice),
            **common,
        )


__all__ = ["DEFAULT_MODEL", "DecisionsEndpointProvider", "decisions_url"]
