from __future__ import annotations

import json
import math
from collections.abc import Callable

import httpx
import pytest
from test_providers_common import make_obs

from app.domain.actions import Action
from app.providers.decisions_endpoint import DecisionsEndpointProvider, decisions_url
from app.providers.gemini import GeminiProvider
from app.providers.openai_chat import OpenAIChatProvider

KEY = "sk-test-secret-123"
Handler = Callable[[httpx.Request], httpx.Response]


def decisions(handler: Handler, **kw: object) -> DecisionsEndpointProvider:
    return DecisionsEndpointProvider(
        "typesafe/jev-1.13",
        api_key=KEY,
        transport=httpx.MockTransport(handler),
        **kw,  # type: ignore[arg-type]
    )


def jev_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "gen-1",
            "model": "typesafe/jev-1.13-20260917",
            "provider": "TypeSafe",
            "answers": {
                "action": {
                    "type": "choice",
                    "choice": "AVOID_LEFT",
                    "probabilities": {"AVOID_LEFT": 0.97, "BRAKE": 0.03},
                    "confidence": 0.97,
                }
            },
            "usage": {"input_tokens": 849, "output_tokens": 110, "cost": 0.0000357},
        },
    )


def test_decisions_url_strips_v1() -> None:
    assert decisions_url("https://openrouter.ai/api/v1") == (
        "https://openrouter.ai/api/alpha/decisions"
    )
    assert (
        decisions_url("http://localhost:9/v1/")
        == "http://localhost:9/api/alpha/decisions"
    )


def test_decisions_happy_path_and_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return jev_ok(request)

    provider = decisions(handler)
    decision = provider.decide(make_obs())
    assert decision.valid and decision.action is Action.AVOID_LEFT
    assert decision.probabilities == {Action.AVOID_LEFT: 0.97, Action.BRAKE: 0.03}
    assert decision.confidence == 0.97
    assert decision.model_id == "typesafe/jev-1.13-20260917"
    assert (decision.tokens_in, decision.tokens_out) == (849, 110)
    assert decision.cost_usd == pytest.approx(3.57e-5)
    assert decision.latency_ms > 0
    request = seen[0]
    assert request.url.path == "/api/alpha/decisions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["model"] == "typesafe/jev-1.13"
    question = body["questions"]["action"]
    assert question["type"] == "choice" and len(question["criteria"]) == 11
    assert "batt_state=OK" in body["state"]
    assert KEY not in request.content.decode()
    assert KEY not in repr(provider)


def test_decisions_timeout_returns_invalid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    decision = decisions(handler).decide(make_obs())
    assert not decision.valid and decision.action is None
    assert decision.error == "timeout" and decision.latency_ms >= 0


def test_decisions_429_and_scrubbed_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text=f"rate limited for {KEY}")

    decision = decisions(handler).decide(make_obs())
    assert decision.error == "http_429" and not decision.valid
    assert decision.raw is not None and KEY not in decision.raw


def test_decisions_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    assert decisions(handler).decide(make_obs()).error == "transport_error"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json {"),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(200, json={"answers": {}}),
        httpx.Response(200, json={"answers": {"action": {"choice": "LEFT or RIGHT"}}}),
        httpx.Response(200, json={"answers": {"action": {"choice": "sideways"}}}),
    ],
)
def test_decisions_bad_payloads_never_raise(response: httpx.Response) -> None:
    decision = decisions(lambda _r: response).decide(make_obs())
    assert not decision.valid and decision.action is None and decision.error


def test_decisions_missing_key_makes_no_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called")

    provider = DecisionsEndpointProvider(
        api_key="", transport=httpx.MockTransport(handler)
    )
    assert provider.decide(make_obs()).error == "missing_api_key"


def chat_response(
    content: str, entries: list[tuple[str, float]] | None = None
) -> httpx.Response:
    logprobs = None
    if entries is not None:
        logprobs = {
            "content": [
                {
                    "token": content,
                    "logprob": 0.0,
                    "top_logprobs": [{"token": t, "logprob": lp} for t, lp in entries],
                }
            ]
        }
    return httpx.Response(
        200,
        json={
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [
                {
                    "message": {"content": content},
                    "logprobs": logprobs,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1},
        },
    )


def chat(
    handler: Handler, model: str = "gpt-4o-mini", **kw: object
) -> OpenAIChatProvider:
    return OpenAIChatProvider(
        model,
        api_key=KEY,
        transport=httpx.MockTransport(handler),
        **kw,  # type: ignore[arg-type]
    )


def test_openai_logprobs_renormalised_and_cost() -> None:
    entries = [("H", math.log(0.6)), ("F", math.log(0.2)), ("The", math.log(0.2))]
    provider = chat(lambda _r: chat_response("H", entries))
    decision = provider.decide(make_obs())
    assert decision.action is Action.AVOID_LEFT
    assert decision.probabilities is not None
    assert decision.probabilities[Action.AVOID_LEFT] == pytest.approx(0.75)
    assert decision.probabilities[Action.BRAKE] == pytest.approx(0.25)
    assert decision.confidence == pytest.approx(0.75)
    assert decision.cost_usd == pytest.approx((1000 * 0.15 + 1 * 0.60) / 1e6)
    assert decision.model_id == "gpt-4o-mini-2024-07-18"


def test_openai_request_shape_gpt4o_vs_luna() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        assert request.headers["authorization"] == f"Bearer {KEY}"
        return chat_response("A")

    chat(handler).decide(make_obs())
    chat(handler, model="gpt-6-luna", reasoning_effort="none").decide(make_obs())
    chat(
        handler,
        model="openai/gpt-6-luna",
        flavor="openrouter",
        reasoning_effort="none",
        routing={"order": ["OpenAI"]},
    ).decide(make_obs())
    mini, luna, router = bodies
    assert mini["top_logprobs"] == 20 and "reasoning_effort" not in mini
    assert mini["temperature"] == 0
    assert luna["top_logprobs"] == 5 and luna["reasoning_effort"] == "none"
    assert luna["max_completion_tokens"] >= 3  # type: ignore[operator]
    assert router["reasoning"] == {"effort": "none"} and router["provider"] == {
        "order": ["OpenAI"]
    }
    assert mini["messages"][0]["role"] == "system"  # type: ignore[index]
    assert "batt_state" in mini["messages"][1]["content"]  # type: ignore[index]


def test_openai_ambiguous_output_is_invalid() -> None:
    decision = chat(lambda _r: chat_response("LEFT or RIGHT")).decide(make_obs())
    assert not decision.valid and decision.error == "ambiguous_output"


def test_openai_empty_content_and_http_error() -> None:
    assert (
        chat(lambda _r: chat_response("")).decide(make_obs()).error == "invalid_output"
    )
    err = chat(lambda _r: httpx.Response(400, text="limit reached")).decide(make_obs())
    assert err.error == "http_400" and err.raw == "limit reached"


def test_openrouter_uses_reported_cost_and_url() -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        resp = chat_response("B")
        payload = resp.json()
        payload["usage"]["cost"] = 0.00123
        return httpx.Response(200, json=payload)

    provider = chat(
        handler,
        model="openai/gpt-4o-mini",
        flavor="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )
    decision = provider.decide(make_obs())
    assert urls == ["https://openrouter.ai/api/v1/chat/completions"]
    assert decision.cost_usd == pytest.approx(0.00123)
    assert provider.provider_id == "openrouter_chat:openai/gpt-4o-mini"


def gemini_response(text: str | None) -> httpx.Response:
    parts = [] if text is None else [{"text": text}]
    candidate = {"content": {"parts": parts}, "finishReason": "MAX_TOKENS"}
    return httpx.Response(
        200,
        json={
            "candidates": [
                candidate if text is None else {"content": {"parts": parts}}
            ],
            "usageMetadata": {"promptTokenCount": 700, "candidatesTokenCount": 1},
            "modelVersion": "gemini-3.5-flash-lite",
        },
    )


def gemini(handler: Handler, **kw: object) -> GeminiProvider:
    return GeminiProvider(api_key=KEY, transport=httpx.MockTransport(handler), **kw)  # type: ignore[arg-type]


def test_gemini_happy_path_and_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return gemini_response("G\n")

    decision = gemini(handler).decide(make_obs())
    assert decision.action is Action.HOVER and decision.probabilities is None
    assert decision.model_id == "gemini-3.5-flash-lite" and decision.tokens_in == 700
    request = seen[0]
    assert request.url.path.endswith("/models/gemini-3.5-flash-lite:generateContent")
    assert request.headers["x-goog-api-key"] == KEY
    assert "key=" not in str(request.url)
    config = json.loads(request.content)["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert config["maxOutputTokens"] >= 16 and config["temperature"] == 0


def test_gemini_empty_reply_is_invalid() -> None:
    decision = gemini(lambda _r: gemini_response(None)).decide(make_obs())
    assert decision.error == "empty_output" and not decision.valid


def test_gemini_structured_enum_option() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return gemini_response("A")

    gemini(handler, structured_enum=True).decide(make_obs())
    config = bodies[0]["generationConfig"]
    assert config["responseMimeType"] == "text/x.enum"  # type: ignore[index]


def test_close_can_be_called_twice() -> None:
    provider = decisions(jev_ok)
    provider.close()
    provider.close()
