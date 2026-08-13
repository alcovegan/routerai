"""Regression tests for defects found in the code audit (codex-review.md)."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from routerai import Registry, RouterAI
from routerai.errors import (
    AuthenticationError,
    StreamInterruptedError,
)

from .conftest import CATALOG, httpx_response

CHAT_OK = {
    "id": "rai-1",
    "model": "m",
    "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
    ],
}


# --- HTTP-01: sync and async share the same instance ---


def test_mixed_sync_async_on_same_client(respx_mock):
    respx_mock.get("https://routerai.ru/api/v1/models").mock(
        return_value=httpx_response({"data": CATALOG})
    )
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(CHAT_OK)
    )
    client = RouterAI(api_key="sk-test")
    assert len(client.models.all()) == 3
    assert client.chat.complete("m", "x").content == "ok"

    async def run():
        assert len(await client.models.aall()) == 3
        result = await client.chat.acomplete("m", "x")
        assert result.content == "ok"
        await client.aclose()

    asyncio.run(run())
    client.close()


def test_injected_clients_not_closed(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(CHAT_OK)
    )
    external_sync = httpx.Client()
    external_async = httpx.AsyncClient()
    client = RouterAI(
        api_key="sk-test", http_client=external_sync, async_http_client=external_async
    )
    client.chat.complete("m", "x")
    asyncio.run(client.chat.acomplete("m", "x"))
    client.close()
    asyncio.run(client.aclose())
    assert not external_sync.is_closed
    assert not external_async.is_closed
    external_sync.close()
    asyncio.run(external_async.aclose())


def test_owned_clients_closed():
    client = RouterAI(api_key="sk-test")
    client._http.request("GET", "models") if False else None
    client.close()
    assert client._http._sync_client is None
    asyncio.run(client.aclose())
    assert client._http._async_client is None


# --- STREAM-01: streaming lifecycle ---


def test_streaming_401_raises_typed_error(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response({"error": {"message": "Invalid API key"}}, status_code=401)
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(AuthenticationError):
        list(client.chat.stream("m", "x"))
    client.close()


def test_midstream_disconnect_no_retry(respx_mock):
    """After the first chunk is delivered there must be no automatic retry."""

    class BreakingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            raise httpx.ReadError("connection lost")

    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, stream=BreakingStream())
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    with pytest.raises(StreamInterruptedError) as exc_info:
        list(client.chat.stream("m", "x"))
    assert exc_info.value.chunks_received == 1
    assert respx_mock.calls.call_count == 1
    client.close()


def test_consumer_exception_no_retry(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(
            "\n".join(
                [
                    'data: {"choices":[{"delta":{"content":"a"}}]}',
                    'data: {"choices":[{"delta":{"content":"b"}}]}',
                    "data: [DONE]",
                ]
            ).encode()
        )
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=0.01)

    def consume():
        for chunk in client.chat.stream("m", "x"):
            _ = chunk.content
            raise ValueError("consumer aborted")

    with pytest.raises(ValueError, match="consumer aborted"):
        consume()
    assert respx_mock.calls.call_count == 1
    client.close()


# --- REGISTRY-01: registry isolation ---


def test_registries_do_not_leak_active_clients():
    a = RouterAI(api_key="sk-a")
    b = RouterAI(api_key="sk-b")
    r1 = Registry(one=a)
    r2 = Registry(two=b)
    assert r1.current() is a
    assert r2.current() is b
    with r2.using("two"):
        assert r2.current() is b
        assert r1.current() is a
    assert r1.current() is a
    r1.close_all()
    r2.close_all()


def test_registry_remove_updates_default():
    a = RouterAI(api_key="sk-a")
    b = RouterAI(api_key="sk-b")
    reg = Registry(a=a, b=b)
    assert reg.default == "a"
    reg.remove("a")
    assert reg.default == "b"
    assert reg.current() is b
    reg.close_all()


def test_registry_thread_isolation():
    import threading

    a = RouterAI(api_key="sk-a")
    b = RouterAI(api_key="sk-b")
    reg = Registry(a=a, b=b)
    seen = {}

    def worker():
        with reg.using("b"):
            seen["inner"] = reg.current()
        seen["outer"] = reg.current()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen["inner"] is b
    assert seen["outer"] is a  # fresh thread falls back to the registry default
    assert reg.current() is a
    reg.close_all()


# --- MODELS-01: async cache ttl + single-flight ---


async def test_async_cache_honors_ttl(respx_mock):
    respx_mock.get("https://routerai.ru/api/v1/models").mock(
        return_value=httpx_response({"data": CATALOG})
    )
    client = RouterAI(api_key="sk-test", models_ttl=-1)
    await client.models.aall()
    await client.models.aall()
    assert respx_mock.calls.call_count == 2
    await client.aclose()


async def test_async_cache_single_flight(respx_mock):
    route = respx_mock.get("https://routerai.ru/api/v1/models").mock(
        return_value=httpx_response({"data": CATALOG})
    )
    client = RouterAI(api_key="sk-test")

    async def delayed():
        import time

        time.sleep(0.05)
        return route

    await asyncio.gather(*(client.models.aall() for _ in range(10)))
    assert route.call_count == 1
    await client.aclose()


# --- CONFIG-01: base url precedence ---


def test_base_url_env_var_is_honored(monkeypatch):
    monkeypatch.setenv("ROUTERAI_BASE_URL", "https://staging.example/v1")
    client = RouterAI(api_key="sk-test")
    assert client._http._base_url == "https://staging.example/v1"
    client.close()


def test_base_url_explicit_wins_over_env(monkeypatch):
    monkeypatch.setenv("ROUTERAI_BASE_URL", "https://staging.example/v1")
    client = RouterAI(api_key="sk-test", base_url="https://explicit.example/v1")
    assert client._http._base_url == "https://explicit.example/v1"
    client.close()


def test_base_url_default(monkeypatch):
    monkeypatch.delenv("ROUTERAI_BASE_URL", raising=False)
    client = RouterAI(api_key="sk-test")
    assert client._http._base_url == "https://routerai.ru/api/v1"
    client.close()


# --- CHAT-01: lossless chat ---


def test_messages_rejects_string_entries():
    from routerai.resources.chat import _messages

    with pytest.raises(ValueError, match="index 0"):
        _messages(["hello"])
    assert _messages("hello") == [{"role": "user", "content": "hello"}]


def test_multiple_choices_are_preserved(respx_mock):
    payload = {
        "id": "rai-2",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A"},
            },
            {
                "index": 1,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "B"},
            },
        ],
        "usage": {"total_tokens": 10, "cost": 0.01},
    }
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(payload)
    )
    client = RouterAI(api_key="sk-test")
    result = client.chat.complete("m", "x")
    assert len(result.choices) == 2
    assert result.content == "A"  # convenience field = choices[0] only
    assert [c.message.content for c in result.choices] == ["A", "B"]
    assert result.cost_rub == Decimal("0.01")
    client.close()


def test_reasoning_key_parsed(respx_mock):
    payload = {
        "id": "rai-3",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "answer", "reasoning": "thought"},
            }
        ],
    }
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(payload)
    )
    client = RouterAI(api_key="sk-test")
    result = client.chat.complete("m", "x")
    assert result.reasoning == "thought"
    client.close()


# --- RETRY-01: retry-after header ---


def test_retry_after_header_controls_delay(respx_mock):
    route = respx_mock.get("https://routerai.ru/api/v1/models").mock(
        side_effect=[
            httpx_response(
                {"error": "slow down"}, status_code=429, headers={"Retry-After": "0.01"}
            ),
            httpx_response({"data": []}),
        ]
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=10)
    started = __import__("time").monotonic()
    client.models.all()
    elapsed = __import__("time").monotonic() - started
    assert route.call_count == 2
    assert elapsed < 3  # Retry-After (0.01) honored instead of backoff (10s)
    client.close()


def test_request_body_is_valid_json():
    """Every request body must round-trip through the documented contract."""
    from routerai.resources.chat import _messages

    body = {"model": "m", "messages": _messages("x")}
    assert json.loads(json.dumps(body)) == body
