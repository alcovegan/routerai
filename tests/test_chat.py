from __future__ import annotations

from decimal import Decimal

import pytest

from routerai import RouterAI
from routerai.errors import (
    APIStatusError,
    AuthenticationError,
    InsufficientFundsError,
    NoProviderError,
    RateLimitError,
)

from .conftest import httpx_response

CHAT_PAYLOAD = {
    "id": "rai-abc123",
    "model": "deepseek/deepseek-v4-pro",
    "service_tier": "flex",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Привет!",
                "reasoning_content": "Пользователь поздоровался",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Москва"}'},
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46, "cost": 0.0053},
}


@pytest.fixture()
def chat_route(respx_mock):
    return respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(CHAT_PAYLOAD, headers={"X-Generation-Id": "gen-42"})
    )


def test_complete_parses_everything(chat_route):
    client = RouterAI(api_key="sk-test")
    result = client.chat.complete("deepseek/deepseek-v4-pro", "Привет")

    assert result.content == "Привет!"
    assert result.reasoning == "Пользователь поздоровался"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == '{"city": "Москва"}'
    assert result.finish_reason == "stop"
    assert result.usage.tokens() == 46
    assert result.usage.cost_rub == Decimal("0.0053")
    assert result.cost_rub == Decimal("0.0053")
    assert result.service_tier == "flex"
    assert result.generation_id == "gen-42"
    client.close()


def test_complete_sends_routerai_extensions(chat_route, respx_mock):
    client = RouterAI(api_key="sk-test")
    client.chat.complete(
        "deepseek/deepseek-v4-pro",
        "Привет",
        service_tier="flex",
        provider={"country": "ru", "allow_fallbacks": False},
    )
    body = chat_route.calls.last.request.content
    payload = __import__("json").loads(body)
    assert payload["service_tier"] == "flex"
    assert payload["provider"] == {"country": "ru", "allow_fallbacks": False}
    assert chat_route.calls.last.request.headers["Authorization"] == "Bearer sk-test"
    client.close()


def test_complete_system_prompt(chat_route, respx_mock):
    client = RouterAI(api_key="sk-test")
    client.chat.complete("m", "Привет", system="Ты ассистент")
    payload = __import__("json").loads(chat_route.calls.last.request.content)
    assert payload["messages"][0] == {"role": "system", "content": "Ты ассистент"}
    client.close()


def test_stream_parses_chunks(respx_mock):
    sse = [
        'data: {"choices":[{"delta":{"content":"При"}}]}',
        'data: {"choices":[{"delta":{"content":"вет"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":10,"cost":0.001}}',
        "data: [DONE]",
    ]
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response("\n".join(sse).encode())
    )
    client = RouterAI(api_key="sk-test")
    chunks = list(client.chat.stream("m", "Привет"))
    text = "".join(c.content for c in chunks)
    assert text == "Привет"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].cost_rub == Decimal("0.001")
    client.close()


def test_stream_request_has_stream_true(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response("data: [DONE]")
    )
    client = RouterAI(api_key="sk-test")
    list(client.chat.stream("m", "Привет"))
    body = __import__("json").loads(respx_mock.calls.last.request.content)
    assert body["stream"] is True
    client.close()


@pytest.mark.parametrize(
    ("status", "payload", "error_cls"),
    [
        (401, {"error": {"message": "Invalid API key"}}, AuthenticationError),
        (402, {"error": {"message": "Insufficient balance"}}, InsufficientFundsError),
        (429, {"error": {"message": "Too many requests"}}, RateLimitError),
        (503, {"error": {"message": "No provider available for model"}}, NoProviderError),
    ],
)
def test_error_mapping(respx_mock, status, payload, error_cls):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(payload, status_code=status)
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(error_cls):
        client.chat.complete("m", "x")
    client.close()


def test_retry_on_5xx_then_success(respx_mock):
    route = respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        side_effect=[
            httpx_response({"error": {"message": "boom"}}, status_code=503),
            httpx_response(CHAT_PAYLOAD),
        ]
    )
    client = RouterAI(
        api_key="sk-test", max_retries=2, retry_backoff=0.01, retry_unsafe_methods=True
    )
    result = client.chat.complete("m", "x")
    assert result.content == "Привет!"
    assert route.call_count == 2
    client.close()


def test_post_5xx_not_retried_by_default(respx_mock):
    route = respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response({"error": {"message": "boom"}}, status_code=503)
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    with pytest.raises(APIStatusError):
        client.chat.complete("m", "x")
    assert route.call_count == 1
    client.close()


def test_get_5xx_retried_by_default(respx_mock):
    route = respx_mock.get("https://routerai.ru/api/v1/models").mock(
        side_effect=[
            httpx_response({"error": {"message": "boom"}}, status_code=503),
            httpx_response({"data": []}),
        ]
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    assert client.models.all() == []
    assert route.call_count == 2
    client.close()


async def test_acomplete(chat_route):
    client = RouterAI(api_key="sk-test")
    result = await client.chat.acomplete("deepseek/deepseek-v4-pro", "Привет")
    assert result.content == "Привет!"
    assert result.cost_rub == Decimal("0.0053")
    await client.aclose()


async def test_astream(respx_mock):
    sse = [
        'data: {"choices":[{"delta":{"content":"ок"}}]}',
        "data: [DONE]",
    ]
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response("\n".join(sse).encode())
    )
    client = RouterAI(api_key="sk-test")
    chunks = [c async for c in client.chat.astream("m", "x")]
    assert "".join(c.content for c in chunks) == "ок"
    await client.aclose()
