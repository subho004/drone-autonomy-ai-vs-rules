"""A (deliberately small) language model as an orb's brain.

The model sees where the drone is in the orb's own frame and answers one letter: L (turn
left), S (straight) or R (turn right). Spend is capped locally; once the cap is hit the orb
goes straight.
"""

from __future__ import annotations

import time

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.harness.orb_control import LEFT, RIGHT, STRAIGHT, OrbChoice, OrbView
from app.providers.openai_chat import OPENAI_BASE_URL, estimate_cost_usd

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You steer a glowing orb in a video game. Your goal is to touch the player's drone, "
    "which is the target. Each turn you are told where the target is relative to your "
    "heading. Answer with exactly one letter: L to turn left, R to turn right, S to go "
    "straight. Turn toward the target. Answer only the letter."
)
LETTER_TO_CMD = {"L": LEFT, "S": STRAIGHT, "R": RIGHT}
GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
SPEND_CAP_USD = 1.0
TIMEOUT_S = 8.0


def describe(view: OrbView) -> str:
    bearing = view.bearing_deg
    side = "AHEAD" if abs(bearing) < 12.0 else ("LEFT" if bearing > 0 else "RIGHT")
    if view.forward < 0:
        side = "BEHIND " + ("LEFT" if view.left > 0 else "RIGHT")
    trend = "CLOSING" if view.closing > 0.1 else "OPENING"
    return f"target: {side}, bearing {bearing:.0f} deg, distance {view.distance:.1f} m, {trend}"


class OrbLLMSteerer:
    """OpenAI chat model as an orb brain (default: the small, cheap gpt-4o-mini)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        cap_usd: float = SPEND_CAP_USD,
        flavor: str = "openai",
    ) -> None:
        self.model = model
        self.name = f"orb-llm:{model}"
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self._extra: dict[str, object] = {}
        if flavor == "gemini":  # Gemini's OpenAI-compatible endpoint
            key = settings.gemini_api_key.get_secret_value()
            base = GEMINI_OPENAI_URL
            self._extra = {"reasoning_effort": "minimal"}
        else:
            key = settings.openai_api_key.get_secret_value()
            base = OPENAI_BASE_URL
        self._client = httpx.Client(
            base_url=base,
            headers={"Authorization": f"Bearer {key}"},
            timeout=TIMEOUT_S,
        )

    def steer(self, view: OrbView) -> OrbChoice:
        if self.spent_usd >= self.cap_usd:
            return OrbChoice(STRAIGHT, 0.0)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": describe(view)},
            ],
            "temperature": 0,
            "max_tokens": 8,
            **self._extra,
        }
        t0 = time.perf_counter()
        try:
            resp = self._client.post("/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError:
            logger.warning("orb model call failed; going straight")
            return OrbChoice(STRAIGHT, (time.perf_counter() - t0) * 1000.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage = data.get("usage", {})
        cost = estimate_cost_usd(
            self.model, usage.get("prompt_tokens"), usage.get("completion_tokens")
        )
        self.spent_usd += cost
        text = str(data["choices"][0]["message"].get("content") or "").strip().upper()
        cmd = LETTER_TO_CMD.get(text[:1], STRAIGHT)
        return OrbChoice(cmd, latency_ms, cost)

    def close(self) -> None:
        self._client.close()


__all__ = ["OrbLLMSteerer", "describe"]
