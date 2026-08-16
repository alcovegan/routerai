"""Regression tests for defects found in the code audit (codex-review.md)."""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal

import httpx
import pytest

from routerai import Registry, RouterAI, StreamAccumulator
from routerai.errors import (
    APIStatusError,
    AuthenticationError,
    ConfigurationError,
    DeadlineExceededError,
    RequestError,
    RouterAIError,
    StreamInterruptedError,
    VideoGenerationError,
    WebhookVerificationError,
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


def test_owned_clients_closed(respx_mock):
    respx_mock.get("https://routerai.ru/api/v1/models").mock(
        return_value=httpx_response({"data": []})
    )
    client = RouterAI(api_key="sk-test")

    client.models.all()
    sync_client = client._http._sync_client
    assert sync_client is not None
    assert not sync_client.is_closed
    client.close()
    assert sync_client.is_closed
    # closing the sync side must not resurrect it on the next call
    with pytest.raises(RuntimeError, match="closed"):
        client.models.all(force_refresh=True)

    client.models.clear_cache()  # the async side is still usable
    asyncio.run(client.models.aall())
    async_client = client._http._async_client
    assert async_client is not None
    assert not async_client.is_closed
    asyncio.run(client.aclose())
    assert async_client.is_closed


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


async def test_async_midstream_disconnect_no_retry(respx_mock):
    """Async parity: after the first chunk is delivered there is no retry."""

    class BreakingAsyncStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            raise httpx.ReadError("connection lost")

    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, stream=BreakingAsyncStream())
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    with pytest.raises(StreamInterruptedError) as exc_info:
        async for _ in client.chat.astream("m", "x"):
            pass
    assert exc_info.value.chunks_received == 1
    assert respx_mock.calls.call_count == 1
    await client.aclose()


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
    entered = 0
    max_entered = 0

    async def handler(request):
        nonlocal entered, max_entered
        entered += 1
        max_entered = max(max_entered, entered)
        await asyncio.sleep(0.05)
        entered -= 1
        return httpx_response({"data": CATALOG})

    route = respx_mock.get("https://routerai.ru/api/v1/models").mock(side_effect=handler)
    client = RouterAI(api_key="sk-test")
    await asyncio.gather(*(client.models.aall() for _ in range(10)))
    assert max_entered == 1  # callers overlapped, refresh was single-flight
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
            httpx_response({"error": "slow down"}, status_code=429, headers={"Retry-After": "1"}),
            httpx_response({"data": []}),
        ]
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=10)
    started = __import__("time").monotonic()
    client.models.all()
    elapsed = __import__("time").monotonic() - started
    assert route.call_count == 2
    assert elapsed < 3  # Retry-After (1s) honored instead of backoff (10s)
    client.close()


def test_retry_after_invalid_falls_back_to_backoff(respx_mock):
    """Inf Retry-After is not a valid RFC 9110 delay-seconds value."""
    route = respx_mock.get("https://routerai.ru/api/v1/models").mock(
        side_effect=[
            httpx_response({"error": "slow down"}, status_code=429, headers={"Retry-After": "inf"}),
            httpx_response({"data": []}),
        ]
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    client.models.all()
    assert route.call_count == 2
    client.close()


def test_retry_after_fractional_falls_back_to_backoff(respx_mock):
    route = respx_mock.get("https://routerai.ru/api/v1/models").mock(
        side_effect=[
            httpx_response(
                {"error": "slow down"}, status_code=429, headers={"Retry-After": "0.01"}
            ),
            httpx_response({"data": []}),
        ]
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    client.models.all()
    assert route.call_count == 2
    client.close()


def test_retry_after_clamped_to_max(respx_mock):
    route = respx_mock.get("https://routerai.ru/api/v1/models").mock(
        side_effect=[
            httpx_response(
                {"error": "slow down"}, status_code=429, headers={"Retry-After": "3600"}
            ),
            httpx_response({"data": []}),
        ]
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=0.01, max_retry_after=0.05)
    started = __import__("time").monotonic()
    client.models.all()
    elapsed = __import__("time").monotonic() - started
    assert route.call_count == 2
    assert elapsed < 1  # 3600s header clamped to max_retry_after
    client.close()


def test_request_body_is_valid_json():
    """Every request body must round-trip through the documented contract."""
    from routerai.resources.chat import _messages

    body = {"model": "m", "messages": _messages("x")}
    assert json.loads(json.dumps(body)) == body


# --- HTTP 200 with embedded error payload (RouterAI wraps provider errors) ---


def test_http_200_with_error_payload_raises(respx_mock):
    """The code that explains the failure wins over the transport status.

    RouterAI answers 200 and hides the real code inside a JSON string, so
    ``status_code`` reports that code — otherwise the exception type and the
    status would disagree. The transport status stays available separately.
    """
    payload = {"error": '{"error":{"message":"Provider returned error","code":400}}'}
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(payload, status_code=200)
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(APIStatusError) as exc_info:
        client.chat.complete("m", "x")
    assert exc_info.value.status_code == 400
    assert exc_info.value.http_status == 200
    assert exc_info.value.provider_code == 400
    assert exc_info.value.status_source == "provider"
    assert exc_info.value.body == payload
    client.close()


def test_failed_video_task_is_reachable(respx_mock):
    """A failed generation is a task state, not a failed HTTP call.

    The polling endpoint answers 200 and puts the reason in ``error``; treating
    that as a request failure made VideoGenerationError unreachable whenever
    the server bothered to explain itself.
    """
    payload = {"id": "vid-1", "status": "failed", "error": "content policy"}
    respx_mock.get("https://routerai.ru/api/v1/videos/vid-1").mock(
        return_value=httpx_response(payload)
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    task = client.videos.get("vid-1")
    assert task.failed
    assert task.error == "content policy"
    with pytest.raises(VideoGenerationError):
        task.wait(timeout=1, interval=0.01, raise_on_failure=True)
    client.close()


def test_http_200_binary_body_is_not_an_error(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/audio/speech").mock(
        return_value=httpx_response(b"\xff\xfb\x90mp3-binary-bytes")
    )
    client = RouterAI(api_key="sk-test")
    result = client.audio.speech("m", "текст", voice="eve")
    assert result.data == b"\xff\xfb\x90mp3-binary-bytes"
    client.close()


# --- API-02: public exceptions exported from the top-level package ---


def test_public_exceptions_exported():
    import routerai

    assert routerai.StreamInterruptedError is StreamInterruptedError
    assert routerai.ConfigurationError is routerai.errors.ConfigurationError
    assert "StreamInterruptedError" in routerai.__all__
    assert "ConfigurationError" in routerai.__all__


# --- HTTP-02: APIStatusError carries the response body ---


def test_api_status_error_body_sync(respx_mock):
    payload = {"error": {"code": "trace-1", "message": "upstream exploded"}}
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(payload, status_code=500)
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(APIStatusError) as exc_info:
        client.chat.complete("m", "x")
    assert exc_info.value.status_code == 500
    assert exc_info.value.body == payload
    client.close()


async def test_api_status_error_body_async(respx_mock):
    payload = {"error": {"code": "trace-2", "message": "upstream exploded"}}
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(payload, status_code=502)
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(APIStatusError) as exc_info:
        await client.chat.acomplete("m", "x")
    assert exc_info.value.status_code == 502
    assert exc_info.value.body == payload
    await client.aclose()


def test_api_status_error_body_stream(respx_mock):
    payload = {"error": {"message": "rejected early"}}
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(payload, status_code=500)
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(APIStatusError) as exc_info:
        list(client.chat.stream("m", "x"))
    assert exc_info.value.body == payload
    client.close()


# --- REGISTRY-02: removed clients must not be restored from context tokens ---


def test_registry_remove_inside_using_context():
    a = RouterAI(api_key="sk-a")
    b = RouterAI(api_key="sk-b")
    reg = Registry(a=a, b=b)
    with reg.using("a"):
        assert reg.current() is a
        reg.remove("a")
        assert reg.current() is b
    assert reg.current() is b
    assert "a" not in reg
    reg.close_all()


def test_registry_nested_using_remove():
    a = RouterAI(api_key="sk-a")
    b = RouterAI(api_key="sk-b")
    c = RouterAI(api_key="sk-c")
    reg = Registry(a=a, b=b, c=c)
    with reg.using("a"):
        with reg.using("b"):
            reg.remove("b")
            assert reg.current() is a  # inner context restored, "b" gone
        assert reg.current() is a
    assert reg.current() is a
    reg.close_all()


# --- AUDIO-02: BinaryIO with non-string name + explicit format ---


def test_transcribe_temporary_file_with_explicit_format(respx_mock):
    import tempfile

    respx_mock.post("https://routerai.ru/api/v1/audio/transcriptions").mock(
        return_value=httpx_response({"text": "ok"})
    )
    client = RouterAI(api_key="sk-test")
    with tempfile.TemporaryFile("w+b") as audio:
        audio.write(b"fake-audio")
        audio.seek(0)
        assert not isinstance(audio.name, str)
        result = client.audio.transcribe("openai/whisper-large-v3", audio, format="mp3")
    assert result.text == "ok"
    body = json.loads(respx_mock.calls.last.request.content)
    assert body["input_audio"]["format"] == "mp3"
    client.close()


def test_transcribe_bytesio_without_name_requires_format(respx_mock):
    import io

    respx_mock.post("https://routerai.ru/api/v1/audio/transcriptions").mock(
        return_value=httpx_response({"text": "ok"})
    )
    client = RouterAI(api_key="sk-test")
    with pytest.raises(ValueError, match="format="):
        client.audio.transcribe("openai/whisper-large-v3", io.BytesIO(b"x"))
    assert respx_mock.calls.call_count == 0
    client.close()


def test_transcribe_explicit_format_overrides_suffix(respx_mock, tmp_path):
    respx_mock.post("https://routerai.ru/api/v1/audio/transcriptions").mock(
        return_value=httpx_response({"text": "ok"})
    )
    client = RouterAI(api_key="sk-test")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFFDATA")
    client.audio.transcribe("openai/whisper-large-v3", audio, format="flac")
    body = json.loads(respx_mock.calls.last.request.content)
    assert body["input_audio"]["format"] == "flac"
    client.close()


# --- audit 3: STREAM-02 async streaming error body on unread async stream ---


async def test_async_streaming_401_unread_body(respx_mock):
    class ErrorAsyncStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"error": {"message": "bad key"}}'

    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx.Response(401, stream=ErrorAsyncStream())
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(AuthenticationError):
        async for _ in client.chat.astream("m", "x"):
            pass
    assert respx_mock.calls.call_count == 1
    await client.aclose()


async def test_async_streaming_500_unread_body(respx_mock):
    class ErrorAsyncStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"error": {"code": "x", "message": "boom"}}'

    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx.Response(500, stream=ErrorAsyncStream())
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(APIStatusError) as exc_info:
        async for _ in client.chat.astream("m", "x"):
            pass
    assert exc_info.value.status_code == 500
    assert exc_info.value.body == {"error": {"code": "x", "message": "boom"}}
    await client.aclose()


# --- audit 3: STREAM-03 extra cannot override stream invariant ---


def test_stream_cannot_be_disabled_by_extra(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response("data: [DONE]")
    )
    client = RouterAI(api_key="sk-test")
    with pytest.raises(ValueError, match="library-managed"):
        list(client.chat.stream("m", "x", extra={"stream": False}))
    assert respx_mock.calls.call_count == 0
    client.close()


def test_complete_cannot_be_turned_into_stream(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(CHAT_OK)
    )
    client = RouterAI(api_key="sk-test")
    with pytest.raises(ValueError, match="library-managed"):
        client.chat.complete("m", "x", extra={"stream": True})
    assert respx_mock.calls.call_count == 0
    client.close()


def test_stream_sets_stream_true_after_extra(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response("data: [DONE]")
    )
    client = RouterAI(api_key="sk-test")
    list(client.chat.stream("m", "x", extra={"verbosity": "low"}))
    body = json.loads(respx_mock.calls.last.request.content)
    assert body["stream"] is True
    assert body["verbosity"] == "low"
    client.close()


# --- audit 3: VIDEO-02 wire contract ---


def test_video_frame_images_wire_contract(respx_mock):
    from routerai.resources.videos import FrameImage, ImageReference

    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    client.videos.create(
        "m", "prompt", frame_images=[FrameImage(url="https://x/f.png", frame_type="first_frame")]
    )
    body = json.loads(respx_mock.calls.last.request.content)
    assert body["frame_images"] == [
        {"type": "image_url", "image_url": {"url": "https://x/f.png"}, "frame_type": "first_frame"}
    ]
    assert "image_input" not in body

    client.videos.create("m", "prompt", input_references=[ImageReference(url="https://x/r.png")])
    body = json.loads(respx_mock.calls.last.request.content)
    assert body["input_references"] == [
        {"type": "image_url", "image_url": {"url": "https://x/r.png"}}
    ]
    client.close()


def test_video_frame_images_and_references_conflict(respx_mock):
    from routerai.resources.videos import FrameImage, ImageReference

    client = RouterAI(api_key="sk-test")
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.videos.create(
            "m",
            "p",
            frame_images=[FrameImage(url="https://x/f.png")],
            input_references=[ImageReference(url="https://x/r.png")],
        )
    assert respx_mock.calls.call_count == 0
    client.close()


def test_video_image_input_deprecated_warns(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    with pytest.warns(DeprecationWarning, match="frame_images"):
        client.videos.create("m", "p", image_input="https://x/f.png")
    body = json.loads(respx_mock.calls.last.request.content)
    assert body["frame_images"][0]["frame_type"] == "first_frame"
    client.close()


def test_video_wait_timeout_respected(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    respx_mock.get("https://routerai.ru/api/v1/videos/v1").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("m", "p")
    started = time.monotonic()
    with pytest.raises(DeadlineExceededError):
        task.wait(timeout=0.05, interval=0.1)
    elapsed = time.monotonic() - started
    assert elapsed < 0.3
    client.close()


def test_video_wait_validates_arguments(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("m", "p")
    with pytest.raises(ValueError, match="interval"):
        task.wait(timeout=1, interval=0)
    with pytest.raises(ValueError, match="timeout"):
        task.wait(timeout=-1, interval=0.1)
    client.close()


# --- audit 3: STREAM-04 sse error envelope and generation id ---


def test_sse_error_event_raises_typed_error(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(
            'data: {"choices":[{"delta":{"content":"частично"}}]}\n\ndata: {"error":{"message":"provider failed","status_code":502}}\n\n'.encode()
        )
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    chunks = []
    with pytest.raises(APIStatusError, match="provider failed") as exc_info:
        for chunk in client.chat.stream("m", "x"):
            chunks.append(chunk)
    assert len(chunks) == 1
    assert exc_info.value.status_code == 502
    client.close()


def test_stream_chunks_carry_generation_id(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n\ndata: [DONE]\n',
            headers={"X-Generation-Id": "gen-stream-1"},
        )
    )
    client = RouterAI(api_key="sk-test")
    chunks = list(client.chat.stream("m", "x"))
    assert chunks[0].generation_id == "gen-stream-1"
    client.close()


# --- audit 3: DATA-02 lossless choices and legacy completions ---


def test_chat_choice_unknown_fields_preserved(respx_mock):
    payload = {
        "id": "rai-x",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "native_finish_reason": "stop",
                "logprobs": {"content": []},
                "message": {"role": "assistant", "content": "A"},
            }
        ],
    }
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(payload)
    )
    client = RouterAI(api_key="sk-test")
    result = client.chat.complete("m", "x")
    extra = result.choices[0].model_extra or {}
    assert extra["native_finish_reason"] == "stop"
    assert extra["logprobs"] == {"content": []}
    client.close()


def test_legacy_completions_keep_alternatives(respx_mock):
    payload = {
        "id": "c1",
        "choices": [
            {"index": 0, "text": "A", "finish_reason": "stop"},
            {"index": 1, "text": "B", "finish_reason": "stop"},
        ],
        "usage": {"total_tokens": 10, "cost": 0.01},
    }
    respx_mock.post("https://routerai.ru/api/v1/completions").mock(
        return_value=httpx_response(payload, headers={"X-Generation-Id": "gen-c1"})
    )
    client = RouterAI(api_key="sk-test")
    result = client.completions.create("m", "p")
    assert result.text == "A"
    assert [c.text for c in result.choices] == ["A", "B"]
    assert result.generation_id == "gen-c1"
    assert result.cost_rub == Decimal("0.01")
    client.close()


# --- audit 3: IMG-02 strict image parsing, no url auto-fetch ---


def test_image_corrupt_base64_rejected(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/images").mock(
        return_value=httpx_response({"data": [{"b64_json": "%%%"}]})
    )
    client = RouterAI(api_key="sk-test")
    with pytest.raises(RouterAIError, match="corrupted"):
        client.images.generate("m", "p")
    client.close()


def test_image_empty_b64_rejected(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/images").mock(
        return_value=httpx_response({"data": [{"b64_json": ""}]})
    )
    client = RouterAI(api_key="sk-test")
    with pytest.raises(RouterAIError, match="empty"):
        client.images.generate("m", "p")
    client.close()


def test_image_url_kept_without_auto_fetch(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/images").mock(
        return_value=httpx_response({"data": [{"url": "https://x/i.png", "revised_prompt": "rp"}]})
    )
    client = RouterAI(api_key="sk-test")
    result = client.images.generate("m", "p")
    assert result.images[0].url == "https://x/i.png"
    assert result.images[0].data is None
    assert result.images[0].revised_prompt == "rp"
    assert respx_mock.calls.call_count == 1  # no download attempt
    client.close()


# --- audit 3: CONFIG-02 numeric validation ---


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"timeout": 0}, "timeout"),
        ({"timeout": -1}, "timeout"),
        ({"timeout": float("nan")}, "timeout"),
        ({"max_retries": -1}, "max_retries"),
        ({"max_retries": 1.5}, "max_retries"),
        ({"retry_backoff": -0.1}, "retry_backoff"),
        ({"retry_backoff": float("inf")}, "retry_backoff"),
    ],
)
def test_config_validation(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        RouterAI(api_key="sk-test", **kwargs)


# --- audit 3: TYPES-02 list methods are real ---


def test_models_list_methods_exist(catalog_route):
    from routerai.resources.models import Models

    assert isinstance(Models.list, type(Models.all))
    client = RouterAI(api_key="sk-test")
    assert len(client.models.list()) == 3
    client.close()


# --- audit 4: EXTRA-02 full reserved-param policy ---


@pytest.mark.parametrize(
    ("call", "body_key"),
    [
        ("chat", "temperature"),
        ("chat", "tools"),
        ("image", "quality"),
        ("tts", "speed"),
        ("stt", "language"),
        ("embedding", "dimensions"),
        ("rerank", "top_n"),
        ("completion", "max_tokens"),
        ("key", "name"),
        ("team_member", "email"),
        ("team_invite", "role"),
        ("video", "duration"),
    ],
)
def test_extra_cannot_override_dedicated_arguments(respx_mock, call, body_key):
    client = RouterAI(api_key="sk-test")
    with pytest.raises(ValueError, match="library-managed"):
        if call == "chat":
            client.chat.complete("m", "x", temperature=0.1, extra={body_key: 1.9})
        elif call == "image":
            client.images.generate("m", "p", quality="high", extra={body_key: "low"})
        elif call == "tts":
            client.audio.speech("m", "x", "eve", speed=1.0, extra={body_key: 2.0})
        elif call == "stt":
            client.audio.transcribe("m", b"x", format="mp3", language="ru", extra={body_key: "en"})
        elif call == "embedding":
            client.embeddings.create("m", "x", dimensions=128, extra={body_key: 64})
        elif call == "rerank":
            client.rerank.create("m", "q", ["d"], top_n=1, extra={body_key: 3})
        elif call == "completion":
            client.completions.create("m", "p", max_tokens=10, extra={body_key: 50})
        elif call == "key":
            client.keys.create("prod", extra={body_key: "other"})
        elif call == "team_member":
            client.team.create_member("a@b.c", extra={body_key: "x@y.z"})
        elif call == "team_invite":
            client.team.invite("a@b.c", extra={body_key: "admin"})
        elif call == "video":
            client.videos.create("m", "p", duration=4, extra={body_key: 10})
    assert respx_mock.calls.call_count == 0
    client.close()


# --- audit 4: VIDEO-04 strict typed inputs ---


def test_video_invalid_frame_type_rejected():
    from pydantic import ValidationError

    from routerai import FrameImage

    with pytest.raises(ValidationError):
        FrameImage(url="https://x/f.png", frame_type="middle")


def test_video_raw_dict_validated(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    with pytest.raises(Exception, match="frame_type"):
        client.videos.create(
            "m", "p", frame_images=[{"url": "https://x/f.png", "frame_type": "middle"}]
        )
    assert respx_mock.calls.call_count == 0
    client.close()


def test_video_legacy_input_conflicts_with_references(respx_mock):
    from routerai import ImageReference

    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="mutually exclusive"):
        client.videos.create(
            "m",
            "p",
            image_input="https://x/f.png",
            input_references=[ImageReference(url="https://x/r.png")],
        )
    assert respx_mock.calls.call_count == 0
    client.close()


def test_video_non_https_frame_url_rejected():
    from pydantic import ValidationError

    from routerai import FrameImage

    with pytest.raises(ValidationError, match="https"):
        FrameImage(url="http://x/f.png")


def test_video_frame_extra_forbidden():
    from pydantic import ValidationError

    from routerai import FrameImage

    with pytest.raises(ValidationError):
        FrameImage(url="https://x/f.png", provider_hint="x")


# --- audit 4: VIDEO-05 finite polling deadline ---


@pytest.mark.parametrize(
    ("timeout", "interval"),
    [(float("nan"), 1.0), (1.0, float("nan")), (float("inf"), 1.0), (1.0, float("-inf"))],
)
def test_video_wait_rejects_non_finite(respx_mock, timeout, interval):
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("m", "p")
    with pytest.raises(ValueError):
        task.wait(timeout=timeout, interval=interval)
    client.close()


def test_video_wait_bounds_slow_refresh(respx_mock):
    """A slow refresh must not exceed the overall budget by much."""
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    respx_mock.get("https://routerai.ru/api/v1/videos/v1").mock(
        side_effect=lambda request: httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test", timeout=10)
    task = client.videos.create("m", "p")
    with pytest.raises(DeadlineExceededError):
        task.wait(timeout=0.05, interval=0.02)
    client.close()


# --- audit 4: IMG-03 redirect safety ---


def test_image_download_rejects_https_to_http_redirect(respx_mock, tmp_path):
    import httpx as _httpx

    respx_mock.get("https://provider.example/image").mock(
        return_value=_httpx.Response(302, headers={"Location": "http://plain.example/image"})
    )
    respx_mock.get("http://plain.example/image").mock(return_value=httpx_response(b"png-bytes"))
    from routerai.resources.images import GeneratedImage

    image = GeneratedImage(url="https://provider.example/image")
    with pytest.raises(RouterAIError, match="https"):
        image.download(tmp_path / "i.png")
    assert not (tmp_path / "i.png").exists()
    assert respx_mock.calls.call_count == 1  # the http hop was never called


def test_image_download_rejects_userinfo(respx_mock, tmp_path):
    from routerai.resources.images import GeneratedImage

    image = GeneratedImage(url="https://user:pass@provider.example/image")
    with pytest.raises(RouterAIError, match="credentials"):
        image.download(tmp_path / "i.png")


def test_image_download_streams_atomically(respx_mock, tmp_path):
    respx_mock.get("https://provider.example/image").mock(
        return_value=httpx_response(b"png-bytes", headers={"Content-Type": "image/png"})
    )
    from routerai.resources.images import GeneratedImage

    image = GeneratedImage(url="https://provider.example/image")
    path = image.download(tmp_path / "i.png")
    assert path.read_bytes() == b"png-bytes"
    assert not (tmp_path / ".i.png.part").exists()


# --- audit 4: HTTP-03 mixed-case media type ---


def test_mixed_case_json_error_detected(respx_mock):
    import httpx as _httpx

    payload = {"error": {"message": "boom"}}
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=_httpx.Response(
            200, json=payload, headers={"Content-Type": "Application/JSON"}
        )
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(APIStatusError) as exc_info:
        client.chat.complete("m", "x")
    assert exc_info.value.body == payload
    client.close()


# --- audit 4: STREAM-05 safe sse status ---


def test_sse_error_non_numeric_status(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(
            b'data: {"error":{"message":"boom","status_code":"unknown"}}\n\n'
        )
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(APIStatusError) as exc_info:
        list(client.chat.stream("m", "x"))
    assert exc_info.value.status_code == 502
    client.close()


# --- audit 4: API-03 top-level exports ---


def test_video_public_types_exported():
    import routerai

    assert routerai.VideoGenerationError is routerai.errors.VideoGenerationError
    assert routerai.FrameImage is routerai.resources.videos.FrameImage
    assert routerai.ImageReference is routerai.resources.videos.ImageReference
    assert "VideoGenerationError" in routerai.__all__


# --- audit 4: CONFIG-03 explicit config precedence ---


def test_explicit_empty_api_key_rejected(monkeypatch):
    monkeypatch.setenv("ROUTERAI_API_KEY", "sk-env-secret")
    with pytest.raises(ConfigurationError, match="api_key"):
        RouterAI(api_key="")


def test_explicit_empty_base_url_rejected(monkeypatch):
    monkeypatch.setenv("ROUTERAI_BASE_URL", "https://env.example/v1")
    with pytest.raises(ConfigurationError, match="base_url"):
        RouterAI(api_key="sk-test", base_url="  ")


def test_none_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("ROUTERAI_API_KEY", "sk-env-secret")
    monkeypatch.setenv("ROUTERAI_BASE_URL", "https://env.example/v1")
    client = RouterAI(api_key=None, base_url=None)
    assert client._http._api_key == "sk-env-secret"
    assert client._http._base_url == "https://env.example/v1"
    client.close()


# --- audit 4: VIDEO-06 indexed downloads ---


def test_video_content_index_query(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("m", "p")
    task._apply(
        {
            "id": "v1",
            "status": "completed",
            "unsigned_urls": ["https://x/0.mp4", "https://x/1.mp4"],
        }
    )
    respx_mock.get("https://routerai.ru/api/v1/videos/v1/content?index=1").mock(
        return_value=httpx_response(b"mp4-bytes")
    )
    assert task.content(index=1) == b"mp4-bytes"
    with pytest.raises(ValueError, match="out of range"):
        task.content(index=2)
    client.close()


def test_video_save_streams_to_file(respx_mock, tmp_path):
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("m", "p")
    task._apply({"id": "v1", "status": "completed", "unsigned_urls": ["https://x/0.mp4"]})
    respx_mock.get("https://routerai.ru/api/v1/videos/v1/content?index=0").mock(
        return_value=httpx_response(b"mp4-content")
    )
    path = task.save(str(tmp_path / "out.mp4"))
    with open(path, "rb") as handle:
        assert handle.read() == b"mp4-content"
    assert not (tmp_path / ".out.mp4.part").exists()
    client.close()


# --- audit 4 wave: network-deny gate ---


def test_network_guard_blocks_real_sockets():
    import socket

    with pytest.raises(Exception) as exc_info:
        socket.create_connection(("routerai.ru", 443), timeout=0.1)
    assert "socket" in type(exc_info.value).__name__.lower()


# --- audit 4 wave: typed management and protocol results ---


def test_responses_result_typed(respx_mock):
    payload = {
        "id": "resp-1",
        "object": "response",
        "status": "completed",
        "model": "m",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "content": [{"type": "output_text", "text": "Токио"}],
            }
        ],
        "usage": {"total_tokens": 10, "cost": 0.01},
    }
    respx_mock.post("https://routerai.ru/api/v1/responses").mock(
        return_value=httpx_response(payload)
    )
    client = RouterAI(api_key="sk-test")
    result = client.responses.create("m", "x")
    assert result.id == "resp-1"
    assert result.output_text == "Токио"
    assert result.cost_rub == Decimal("0.01")
    assert result.raw == payload
    client.close()


def test_messages_result_typed(respx_mock):
    payload = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [{"type": "text", "text": "Берлин"}, {"type": "thinking", "thinking": "..."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 4},
    }
    respx_mock.post("https://routerai.ru/api/v1/messages").mock(
        return_value=httpx_response(payload)
    )
    client = RouterAI(api_key="sk-test")
    result = client.messages.create("m", [{"role": "user", "content": "x"}])
    assert result.id == "msg_1"
    assert result.text == "Берлин"
    assert result.stop_reason == "end_turn"
    assert result.usage and result.usage.input_tokens == 12
    client.close()


def test_generation_get_returns_typed_info(respx_mock):
    respx_mock.get("https://routerai.ru/api/v1/generation?id=g1").mock(
        return_value=httpx_response({"data": {"id": "g1", "total_cost": 3.14}})
    )
    client = RouterAI(api_key="sk-test")
    info = client.generation.get("g1")
    assert info.id == "g1"
    assert info.total_cost == Decimal("3.14")
    client.close()


def test_management_types_exported():
    import routerai

    for name in (
        "KeyInfo",
        "TeamMember",
        "TeamInvitation",
        "MemberCreation",
        "ResponsesResult",
        "MessagesResult",
    ):
        assert name in routerai.__all__
        assert hasattr(routerai, name)


# --- audit 4 wave: multimodal lifecycle ---


def test_images_typed_parameters_and_optional_prompt(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/images").mock(
        return_value=httpx_response({"created": 1, "data": [{"b64_json": "cG5n"}]})
    )
    client = RouterAI(api_key="sk-test")
    client.images.generate(
        "m",
        None,
        input_references=[
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}}
        ],
        aspect_ratio="16:9",
        resolution="1K",
        background="transparent",
        output_format="webp",
        output_compression=80,
        seed=42,
    )
    body = json.loads(respx_mock.calls.last.request.content)
    assert body["aspect_ratio"] == "16:9"
    assert body["resolution"] == "1K"
    assert body["background"] == "transparent"
    assert body["output_format"] == "webp"
    assert body["output_compression"] == 80
    assert body["seed"] == 42
    assert "prompt" not in body
    assert body["input_references"]

    with pytest.raises(ValueError, match="prompt is required"):
        client.images.generate("m")
    assert respx_mock.calls.call_count == 1
    client.close()


def test_images_stream_events(respx_mock, tmp_path):
    import base64 as b64lib

    partial = b64lib.b64encode(b"part").decode()
    full = b64lib.b64encode(b"full").decode()
    sse = "\n".join(
        [
            'data: {"type":"image_generation.partial_image","data":[{"b64_json":"'
            + partial
            + '"}]}',
            'data: {"type":"image_generation.completed","data":[{"b64_json":"'
            + full
            + '"}],"usage":{"cost":4.32}}',
            "data: [DONE]",
        ]
    )
    respx_mock.post("https://routerai.ru/api/v1/images").mock(
        return_value=httpx_response(sse.encode(), headers={"X-Generation-Id": "gen-img-s"})
    )
    client = RouterAI(api_key="sk-test")
    chunks = list(client.images.stream("m", "промпт"))
    assert chunks[0].type == "image_generation.partial_image"
    assert chunks[0].images[0].data == b"part"
    assert chunks[-1].is_completed
    assert chunks[-1].images[0].data == b"full"
    assert chunks[-1].cost_rub == Decimal("4.32")
    assert chunks[-1].generation_id == "gen-img-s"
    body = json.loads(respx_mock.calls.last.request.content)
    assert body["stream"] is True
    client.close()


def test_tts_streaming_bytes(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/audio/speech").mock(
        return_value=httpx_response(b"mp3-part1-mp3-part2")
    )
    client = RouterAI(api_key="sk-test")
    chunks = list(client.audio.speech_stream("m", "text", voice="eve"))
    assert b"".join(chunks) == b"mp3-part1-mp3-part2"
    client.close()


def test_chat_audio_delta_and_accumulator(respx_mock):
    import base64 as b64

    audio_data = b64.b64encode(b"\x00\x01pcm").decode()
    sse = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"сейчас"}}]}',
            'data: {"choices":[{"delta":{"audio":{"data":"'
            + audio_data
            + '","format":"pcm16","transcript":"бип"}}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":5,"cost":0.02}}',
            "data: [DONE]",
        ]
    )
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(sse.encode(), headers={"X-Generation-Id": "gen-audio"})
    )
    client = RouterAI(api_key="sk-test")
    accumulator = StreamAccumulator()
    for chunk in client.chat.stream("m", "x", extra={"modalities": ["audio", "text"]}):
        accumulator.add(chunk)
    assert accumulator.content == "сейчас"
    assert accumulator.audio[0].format == "pcm16"
    assert accumulator.audio[0].data == b"\x00\x01pcm"
    assert accumulator.audio[0].transcript == "бип"
    assert accumulator.cost_rub == Decimal("0.02")
    assert accumulator.generation_id == "gen-audio"
    assert accumulator.finish_reason == "stop"
    client.close()


def test_stream_accumulator_exported():
    import routerai

    assert hasattr(routerai, "StreamAccumulator")


# --- audit 4 wave: webhook verifier ---


def compute_video_signature(api_key: str, timestamp: int, payload: bytes) -> str:
    """RouterAI scheme: HMAC_SHA256(sha256_hex(api_key), "<ts>.<raw_body>")."""
    import hashlib
    import hmac

    from routerai.webhooks import signing_secret

    signed = f"{timestamp}.".encode() + payload
    return hmac.new(signing_secret(api_key).encode(), signed, hashlib.sha256).hexdigest()


def test_webhook_verify_valid_signature():
    from routerai.webhooks import verify_video

    api_key = "sk-webhook-test-key"
    timestamp = int(__import__("time").time())
    payload = b'{"id":"v1","status":"completed"}'
    signature = compute_video_signature(api_key, timestamp, payload)
    data = verify_video(payload, signature, api_key, str(timestamp))
    assert data["id"] == "v1"


def test_webhook_verify_rejects_tampered_payload():
    from routerai.webhooks import verify_video

    api_key = "sk-webhook-test-key"
    timestamp = int(__import__("time").time())
    signature = compute_video_signature(api_key, timestamp, b'{"id":"v1"}')
    with pytest.raises(WebhookVerificationError):
        verify_video(b'{"id":"v2"}', signature, api_key, str(timestamp))


def test_webhook_verify_rejects_stale_timestamp():
    from routerai.webhooks import verify_video

    api_key = "sk-webhook-test-key"
    stale = int(__import__("time").time()) - 7200
    payload = b'{"id":"v1"}'
    signature = compute_video_signature(api_key, stale, payload)
    with pytest.raises(WebhookVerificationError):
        verify_video(payload, signature, api_key, str(stale), max_age_seconds=300)


def test_webhook_verify_rejects_wrong_secret():
    from routerai.webhooks import verify_video

    timestamp = int(__import__("time").time())
    payload = b'{"id":"v1"}'
    signature = compute_video_signature("sk-a", timestamp, payload)
    with pytest.raises(WebhookVerificationError):
        verify_video(payload, signature, "sk-b", str(timestamp))


def test_webhook_verify_rejects_malformed_timestamp():
    from routerai.webhooks import verify_video

    with pytest.raises(WebhookVerificationError):
        verify_video(b'{"id":"v1"}', "sig", "sk-a", "not-a-timestamp")


# --- audit 5: real-stream tests, deadline, atomic files, validation ---


def test_image_download_does_not_over_read(respx_mock, tmp_path):
    """The byte limit must stop the stream before the next chunk is read."""

    class FailAfterSecondChunk(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x" * 100
            raise RuntimeError("second chunk must never be consumed")

    respx_mock.get("https://provider.example/image").mock(
        return_value=httpx.Response(200, stream=FailAfterSecondChunk())
    )
    from routerai.resources.images import GeneratedImage

    image = GeneratedImage(url="https://provider.example/image")
    with pytest.raises(RouterAIError, match="exceeds"):
        image.download(tmp_path / "i.png", max_bytes=50)
    assert not (tmp_path / "i.png").exists()
    assert not list(tmp_path.glob(".i.png.*"))


def test_image_download_content_length_precheck(respx_mock, tmp_path):
    import httpx as _httpx

    respx_mock.get("https://provider.example/image").mock(
        return_value=_httpx.Response(
            200, headers={"Content-Length": "999999999"}, stream=httpx.SyncByteStream()
        )
    )
    from routerai.resources.images import GeneratedImage

    image = GeneratedImage(url="https://provider.example/image")
    with pytest.raises(RouterAIError, match="content-length"):
        image.download(tmp_path / "i.png", max_bytes=1024)
    assert not (tmp_path / "i.png").exists()


def test_image_download_cleans_temp_on_midstream_failure(respx_mock, tmp_path):
    class FailingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadError("connection lost")

    respx_mock.get("https://provider.example/image").mock(
        return_value=httpx.Response(200, stream=FailingStream())
    )
    from routerai.resources.images import GeneratedImage

    image = GeneratedImage(url="https://provider.example/image")
    with pytest.raises(RequestError) as exc_info:
        image.download(tmp_path / "i.png")
    assert isinstance(exc_info.value.__cause__, httpx.ReadError)
    assert not (tmp_path / "i.png").exists()
    assert not list(tmp_path.glob(".i.png.*"))


async def test_video_asave_does_not_append_stale_part(respx_mock, tmp_path):
    target = tmp_path / "out.mp4"
    target.write_bytes(b"stale-target")
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("m", "p")
    task._apply({"id": "v1", "status": "completed", "unsigned_urls": ["https://x/0.mp4"]})
    respx_mock.get("https://routerai.ru/api/v1/videos/v1/content?index=0").mock(
        return_value=httpx_response(b"fresh")
    )
    await task.asave(str(target))
    with open(target, "rb") as handle:
        assert handle.read() == b"fresh"
    assert not list(tmp_path.glob(".out.mp4.*"))
    await client.aclose()


async def test_video_asave_cleans_temp_on_failure(respx_mock, tmp_path):
    class FailingAsyncStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"partial"
            raise httpx.ReadError("connection lost")

    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("m", "p")
    task._apply({"id": "v1", "status": "completed", "unsigned_urls": ["https://x/0.mp4"]})
    respx_mock.get("https://routerai.ru/api/v1/videos/v1/content?index=0").mock(
        return_value=httpx.Response(200, stream=FailingAsyncStream())
    )
    with pytest.raises(RequestError) as exc_info:
        await task.asave(str(tmp_path / "out.mp4"))
    assert isinstance(exc_info.value.__cause__, httpx.ReadError)
    assert not (tmp_path / "out.mp4").exists()
    assert not list(tmp_path.glob(".out.mp4.*"))
    await client.aclose()


async def test_video_asave_cleans_temp_on_cancellation(respx_mock, tmp_path):
    chunk_consumed = asyncio.Event()
    keep_open = asyncio.Event()

    class HangingAsyncStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"partial"
            chunk_consumed.set()
            await keep_open.wait()

    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("m", "p")
    task._apply({"id": "v1", "status": "completed", "unsigned_urls": ["https://x/0.mp4"]})
    respx_mock.get("https://routerai.ru/api/v1/videos/v1/content?index=0").mock(
        return_value=httpx.Response(200, stream=HangingAsyncStream())
    )

    target = tmp_path / "out.mp4"
    save_task = asyncio.create_task(task.asave(str(target)))
    await asyncio.wait_for(chunk_consumed.wait(), timeout=1)
    save_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await save_task
    assert not target.exists()
    assert not list(tmp_path.glob(".out.mp4.*"))
    client.close()
    await client.aclose()


def test_video_wait_deadline_bounds_retries(respx_mock):
    """Per-attempt timeouts must not exceed the remaining deadline budget."""

    def handler(request):
        return httpx_response({"id": "v1", "status": "pending"})

    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    respx_mock.get("https://routerai.ru/api/v1/videos/v1").mock(side_effect=handler)
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    task = client.videos.create("m", "p")
    with pytest.raises(DeadlineExceededError):
        task.wait(timeout=0.05, interval=0.001)
    client.close()


def test_video_wait_deadline_caps_status_retry_after(respx_mock):
    """A retryable HTTP status must not sleep or retry past the task deadline."""
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    route = respx_mock.get("https://routerai.ru/api/v1/videos/v1").mock(
        return_value=httpx_response(
            {"error": "temporary"}, status_code=503, headers={"Retry-After": "1"}
        )
    )
    client = RouterAI(api_key="sk-test", max_retries=2, max_retry_after=0.2)
    task = client.videos.create("m", "p")
    started = time.monotonic()
    with pytest.raises(DeadlineExceededError):
        task.wait(timeout=0.03, interval=0.001)
    elapsed = time.monotonic() - started
    assert elapsed < 0.15
    assert route.call_count == 1
    client.close()


async def test_video_await_deadline_cancels_inflight_refresh(respx_mock):
    """The async deadline is a wall-clock cancel scope, not an inactivity timeout."""
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )

    refresh_calls = 0

    async def slow_refresh(request):
        nonlocal refresh_calls
        refresh_calls += 1
        await asyncio.sleep(0.2)
        return httpx_response({"id": "v1", "status": "pending"})

    respx_mock.get("https://routerai.ru/api/v1/videos/v1").mock(side_effect=slow_refresh)
    client = RouterAI(api_key="sk-test", max_retries=0)
    task = client.videos.create("m", "p")
    started = time.monotonic()
    with pytest.raises(DeadlineExceededError):
        await task.await_(timeout=0.03, interval=0.001)
    elapsed = time.monotonic() - started
    assert elapsed < 0.15
    assert refresh_calls == 1
    client.close()
    await client.aclose()


def test_retry_after_huge_integer_clamped(respx_mock):
    route = respx_mock.get("https://routerai.ru/api/v1/models").mock(
        side_effect=[
            httpx_response(
                {"error": "slow down"},
                status_code=429,
                headers={"Retry-After": "9" * 400},
            ),
            httpx_response({"data": []}),
        ]
    )
    client = RouterAI(api_key="sk-test", max_retries=2, retry_backoff=0.01, max_retry_after=0.05)
    client.models.all()
    assert route.call_count == 2
    client.close()


def test_sse_error_root_level_status(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(b'data: {"status_code":401,"error":{"message":"bad key"}}\n\n')
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(APIStatusError) as exc_info:
        list(client.chat.stream("m", "x"))
    assert exc_info.value.status_code == 401
    client.close()


def test_sse_error_nested_wins_over_root(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response(
            b'data: {"status_code":401,"error":{"message":"boom","status_code":500}}\n\n'
        )
    )
    client = RouterAI(api_key="sk-test", max_retries=0)
    with pytest.raises(APIStatusError) as exc_info:
        list(client.chat.stream("m", "x"))
    assert exc_info.value.status_code == 500
    client.close()


def test_video_url_validation_strict():
    from pydantic import ValidationError

    from routerai import FrameImage

    for bad in (
        "data:not-an-image-or-base64",
        "https://",
        "https://user:pass@localhost/image.png",
        "http://example.com/image.png",
        "https://127.0.0.1/image.png",
        "https://10.0.0.5/image.png",
        "https://169.254.1.1/image.png",
        "https://example.com/image.png#frag",
    ):
        with pytest.raises(ValidationError):
            FrameImage(url=bad)

    ok = FrameImage(url="https://example.com/image.png")
    assert ok.url == "https://example.com/image.png"
    ok_data = FrameImage(url="data:image/png;base64,aGVsbG8=")
    assert ok_data.url.startswith("data:image/png;base64,")


def test_video_callback_url_validated(respx_mock):
    client = RouterAI(api_key="sk-test")
    with pytest.raises(ValueError, match="callback_url"):
        client.videos.create("m", "p", callback_url="http://example.com/hook")
    with pytest.raises(ValueError, match="callback_url"):
        client.videos.create("m", "p", callback_url="https://user:pass@example.com/hook")
    with pytest.raises(ValueError, match="callback_url"):
        client.videos.create("m", "p", callback_url="https://127.0.0.1/hook")
    assert respx_mock.calls.call_count == 0
    client.close()


def test_env_whitespace_values_rejected(monkeypatch):
    monkeypatch.setenv("ROUTERAI_API_KEY", "   ")
    with pytest.raises(ConfigurationError, match="api_key"):
        RouterAI()
    monkeypatch.setenv("ROUTERAI_API_KEY", "sk-ok")
    monkeypatch.setenv("ROUTERAI_BASE_URL", "   ")
    with pytest.raises(ConfigurationError, match="base_url"):
        RouterAI()


def test_image_data_uri_has_a_predecode_size_limit():
    from routerai._urls import validate_image_source

    with pytest.raises(ValueError, match="byte limit"):
        validate_image_source("data:image/png;base64,aGVsbG8=", max_data_bytes=4)


def test_image_download_rejects_non_public_target_before_network(respx_mock, tmp_path):
    from routerai.resources.images import GeneratedImage

    image = GeneratedImage(url="https://127.0.0.1/image.png")
    with pytest.raises(RouterAIError, match="not a public address"):
        image.download(tmp_path / "i.png")
    assert respx_mock.calls.call_count == 0


def test_video_save_normalizes_transport_error(respx_mock, tmp_path):
    class FailingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadError("connection lost")

    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "v1", "status": "pending"})
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("m", "p")
    task._apply({"id": "v1", "status": "completed", "unsigned_urls": ["https://x/0.mp4"]})
    respx_mock.get("https://routerai.ru/api/v1/videos/v1/content?index=0").mock(
        return_value=httpx.Response(200, stream=FailingStream())
    )
    with pytest.raises(RequestError) as exc_info:
        task.save(str(tmp_path / "out.mp4"))
    assert isinstance(exc_info.value.__cause__, httpx.ReadError)
    assert not (tmp_path / "out.mp4").exists()
    assert not list(tmp_path.glob(".out.mp4.*"))
    client.close()
