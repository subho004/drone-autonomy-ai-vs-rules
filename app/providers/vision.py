"""Yes/no vision questions to chat models, with a hard USD cap and no key leakage.

Used by the inspection resolution study and the inspection flight demo. One persistent
HTTP/2 client, strictly sequential calls with a minimum pause, header auth only, and a
persisted running cost that aborts before the cap is passed. The API keys live only in
private attributes and are scrubbed from every error string that leaves this module.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
MAX_OUTPUT_TOKENS = (
    16  # gemini-3.5-flash-lite returns nothing at 4; gpt-6-luna needs > 2
)
TIMEOUT_S = 30.0
MIN_PAUSE_S = 0.3
EST_CALL_COST_USD = 0.002  # pre-call allowance checked against the cap
MAX_CONSECUTIVE_429 = 3
BACKOFF_S = 2.0
FATAL_STATUSES = frozenset({401, 402, 403})
BILLING_MARKERS = ("billing", "insufficient_quota", "quota exceeded")
MAX_ERROR_CHARS = 160
CONTEXT = (
    "This is a camera image of one concrete wall panel taken by an inspection drone."
)
SYSTEM = (
    "You are a vision inspection assistant. You can see and analyse any image you are "
    "given. Reply with a single word."
)  # without it gpt-4o-mini sometimes answers "I can't analyze images"
QUESTION = "Does this panel show a defect (crack or corrosion)? Answer YES or NO."

Family = Literal["openai", "gemini"]
# (input, output) USD per 1M tokens; gemini's is the project's assumed price (not fetched).
PRICING_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-6-luna": (0.10, 0.50),
    "gemini-3.5-flash-lite": (0.10, 0.40),
}
FAMILY_OF: dict[str, Family] = {
    "gpt-4o-mini": "openai",
    "gpt-6-luna": "openai",
    "gemini-3.5-flash-lite": "gemini",
}


class CostCapReachedError(RuntimeError):
    """Raised before a call that could push total spend past the hard cap."""


class FatalApiError(RuntimeError):
    """Auth, billing or repeated rate-limit failure: stop and report, do not retry."""


class CostCap:
    """Running USD total with a hard limit, persisted so it spans several runs."""

    def __init__(self, limit_usd: float, path: Path | None = None) -> None:
        self.limit_usd = limit_usd
        self._path = path
        self.spent_usd = 0.0
        self.calls = 0
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            self.spent_usd = float(data.get("spent_usd", 0.0))
            self.calls = int(data.get("calls", 0))

    def check(self, estimate_usd: float = EST_CALL_COST_USD) -> None:
        if self.spent_usd + estimate_usd > self.limit_usd:
            raise CostCapReachedError(
                f"cost cap reached: spent ${self.spent_usd:.4f} of ${self.limit_usd:.2f}"
            )

    def add(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        self.calls += 1
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(
                    {
                        "spent_usd": round(self.spent_usd, 6),
                        "calls": self.calls,
                        "limit_usd": self.limit_usd,
                    },
                    indent=2,
                )
            )


@dataclass(slots=True)
class VisionResult:
    model: str
    answer: str
    verdict: bool | None  # True = YES (defect), False = NO, None = unparseable
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    status: int
    error: str | None = None


def parse_yes_no(text: str) -> bool | None:
    """True for YES, False for NO, None when absent or contradictory.

    A leading YES/NO (ignoring quotes, asterisks, brackets) wins; otherwise exactly one of
    the two words must appear.
    """

    lead = re.match(r"[\s*\"'`(\[]*(yes|no)\b", text, re.IGNORECASE)
    if lead:
        return lead.group(1).lower() == "yes"
    has_yes = re.search(r"\byes\b", text, re.IGNORECASE) is not None
    has_no = re.search(r"\bno\b", text, re.IGNORECASE) is not None
    if has_yes == has_no:
        return None
    return has_yes


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICING_PER_M.get(model, (0.0, 0.0))
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


def png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def build_openai_body(
    model: str, prompt: str, png: bytes | None, detail: str = "low"
) -> dict[str, Any]:
    content: Any = prompt
    if png is not None:
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": png_data_url(png), "detail": detail},
            },
        ]
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
    }
    if "gpt-6" in model:
        body["reasoning_effort"] = "none"
    return body


def build_gemini_body(prompt: str, png: bytes | None) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if png is not None:
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(png).decode(),
                }
            }
        )
    parts.append({"text": prompt})
    return {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }


class VisionClient:
    """Sequential yes/no asker over OpenAI chat completions and Gemini generateContent."""

    def __init__(
        self,
        cap: CostCap,
        *,
        transport: httpx.BaseTransport | None = None,
        openai_key: str | None = None,
        gemini_key: str | None = None,
        min_pause_s: float = MIN_PAUSE_S,
    ) -> None:
        self._cap = cap
        self._keys = {
            "openai": settings.openai_api_key.get_secret_value()
            if openai_key is None
            else openai_key,
            "gemini": settings.gemini_api_key.get_secret_value()
            if gemini_key is None
            else gemini_key,
        }
        self._client = httpx.Client(
            http2=transport is None, transport=transport, timeout=TIMEOUT_S
        )
        self._min_pause_s = min_pause_s
        self._last_end = 0.0
        self._consecutive_429 = 0

    def close(self) -> None:
        self._client.close()

    # ----------------------------------------------------------------------- public
    def ask(
        self, model: str, png: bytes | None, text: str | None = None
    ) -> VisionResult:
        """Ask the standard question about ``png`` (image) or ``text`` (description)."""

        prompt = f"{CONTEXT} {QUESTION}" if text is None else f"{text} {QUESTION}"
        family = FAMILY_OF[model]
        url, body, headers = self._request(family, model, prompt, png)
        while True:
            self._cap.check()
            self._pace()
            started = time.perf_counter_ns()
            response = self._post(url, body, headers)
            latency_ms = (time.perf_counter_ns() - started) / 1e6
            self._last_end = time.perf_counter()
            if response.status_code != 429:
                self._consecutive_429 = 0
                break
            self._consecutive_429 += 1
            logger.warning("%s HTTP 429 (%d in a row)", model, self._consecutive_429)
            if self._consecutive_429 >= MAX_CONSECUTIVE_429:
                raise FatalApiError("repeated HTTP 429: stopping")
            time.sleep(BACKOFF_S * self._consecutive_429)
        return self._result(family, model, response, latency_ms)

    # ---------------------------------------------------------------------- internals
    def _request(
        self, family: Family, model: str, prompt: str, png: bytes | None
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        if family == "openai":
            return (
                OPENAI_URL,
                build_openai_body(model, prompt, png),
                {"Authorization": f"Bearer {self._keys['openai']}"},
            )
        return (
            GEMINI_URL.format(model=model),
            build_gemini_body(prompt, png),
            {"x-goog-api-key": self._keys["gemini"]},
        )

    def _pace(self) -> None:
        wait = self._min_pause_s - (time.perf_counter() - self._last_end)
        if wait > 0:
            time.sleep(wait)

    def _post(
        self, url: str, body: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        try:
            return self._client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            # The exception text can echo the request; keep only its type.
            raise FatalApiError(f"transport failure: {type(exc).__name__}") from None

    def _scrub(self, text: str) -> str:
        for key in self._keys.values():
            if key:
                text = text.replace(key, "***")
        return text[:MAX_ERROR_CHARS]

    def _result(
        self, family: Family, model: str, response: httpx.Response, latency_ms: float
    ) -> VisionResult:
        status = response.status_code
        if status != 200:
            snippet = self._scrub(response.text)
            if status in FATAL_STATUSES or any(
                m in response.text.lower() for m in BILLING_MARKERS
            ):
                raise FatalApiError(f"HTTP {status}: {snippet}")
            return VisionResult(
                model, "", None, latency_ms, 0, 0, 0.0, status, f"http_{status}"
            )
        try:
            payload = response.json()
            text, tokens_in, tokens_out = (
                _read_openai(payload) if family == "openai" else _read_gemini(payload)
            )
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return VisionResult(
                model, "", None, latency_ms, 0, 0, 0.0, status, "bad_payload"
            )
        cost = estimate_cost_usd(model, tokens_in, tokens_out)
        self._cap.add(cost)
        verdict = parse_yes_no(text)
        error = None if verdict is not None else "unparseable"
        return VisionResult(
            model, text, verdict, latency_ms, tokens_in, tokens_out, cost, status, error
        )


def _read_openai(payload: dict[str, Any]) -> tuple[str, int, int]:
    text = payload["choices"][0]["message"].get("content") or ""
    usage = payload.get("usage") or {}
    return (
        str(text),
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def _read_gemini(payload: dict[str, Any]) -> tuple[str, int, int]:
    candidates = payload.get("candidates") or []
    parts = (
        ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    )
    text = "".join(str(part.get("text", "")) for part in parts)
    usage = payload.get("usageMetadata") or {}
    out = int(usage.get("candidatesTokenCount") or 0) + int(
        usage.get("thoughtsTokenCount") or 0
    )
    return text, int(usage.get("promptTokenCount") or 0), out


__all__ = [
    "CONTEXT",
    "FAMILY_OF",
    "PRICING_PER_M",
    "QUESTION",
    "CostCap",
    "CostCapReachedError",
    "FatalApiError",
    "VisionClient",
    "VisionResult",
    "estimate_cost_usd",
    "parse_yes_no",
]
