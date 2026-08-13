from __future__ import annotations

import base64
import logging
from contextlib import suppress
from decimal import Decimal

import pytest

from routerai import RouterAI

from .conftest import httpx_response


def test_images_generate_saves_b64(respx_mock, tmp_path):
    respx_mock.post("https://routerai.ru/api/v1/images").mock(
        return_value=httpx_response(
            {
                "created": 123,
                "data": [{"b64_json": base64.b64encode(b"PNGDATA").decode()}],
                "usage": {"cost": 4.32},
            },
            headers={"X-Generation-Id": "gen-img"},
        )
    )
    client = RouterAI(api_key="sk-test")
    result = client.images.generate("openai/gpt-image-1", "кот")
    assert result.cost_rub == Decimal("4.32")
    path = result.images[0].save(tmp_path / "cat.png")
    assert path.read_bytes() == b"PNGDATA"
    client.close()


def test_audio_speech_returns_bytes(respx_mock, tmp_path):
    respx_mock.post("https://routerai.ru/api/v1/audio/speech").mock(
        return_value=httpx_response(b"mp3-bytes")
    )
    client = RouterAI(api_key="sk-test")
    result = client.audio.speech("x-ai/grok-voice-tts-1.0", "Привет", voice="eve")
    assert result.data == b"mp3-bytes"
    assert result.save(tmp_path / "hi.mp3").read_bytes() == b"mp3-bytes"
    body = __import__("json").loads(respx_mock.calls.last.request.content)
    assert body["voice"] == "eve"
    assert body["response_format"] == "mp3"
    client.close()


def test_audio_transcribe_contract(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/audio/transcriptions").mock(
        return_value=httpx_response(
            {"text": "привет мир", "usage": {"seconds": 2.1, "cost": 0.01}},
            headers={"X-Generation-Id": "gen-stt"},
        )
    )
    client = RouterAI(api_key="sk-test")
    result = client.audio.transcribe("openai/whisper-large-v3", b"fakeaudio", format="mp3")
    assert result.text == "привет мир"
    assert result.cost_rub == Decimal("0.01")
    assert result.generation_id == "gen-stt"

    body = __import__("json").loads(respx_mock.calls.last.request.content)
    assert body["model"] == "openai/whisper-large-v3"
    assert body["input_audio"]["format"] == "mp3"
    assert body["input_audio"]["data"] == base64.b64encode(b"fakeaudio").decode()
    client.close()


def test_audio_transcribe_infers_format_from_path(respx_mock, tmp_path):
    respx_mock.post("https://routerai.ru/api/v1/audio/transcriptions").mock(
        return_value=httpx_response({"text": "ok"})
    )
    client = RouterAI(api_key="sk-test")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFFDATA")
    client.audio.transcribe("openai/whisper-large-v3", audio)
    body = __import__("json").loads(respx_mock.calls.last.request.content)
    assert body["input_audio"]["format"] == "wav"
    client.close()


def test_audio_transcribe_requires_format_for_raw_bytes(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/audio/transcriptions").mock(
        return_value=httpx_response({"text": "ok"})
    )
    client = RouterAI(api_key="sk-test")
    with pytest.raises(ValueError, match="format="):
        client.audio.transcribe("openai/whisper-large-v3", b"raw-bytes")
    assert respx_mock.calls.call_count == 0
    client.close()


def test_embeddings_and_rerank(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/embeddings").mock(
        return_value=httpx_response(
            {"data": [{"embedding": [0.1, 0.2]}], "usage": {"total_tokens": 3}}
        )
    )
    respx_mock.post("https://routerai.ru/api/v1/rerank").mock(
        return_value=httpx_response(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                ]
            }
        )
    )
    client = RouterAI(api_key="sk-test")
    emb = client.embeddings.create("m", "текст")
    assert emb.embeddings == [[0.1, 0.2]]

    rerank = client.rerank.create("m", "запрос", ["a", "b"])
    assert rerank.top_documents(["a", "b"]) == ["b", "a"]
    client.close()


def test_video_task_polling(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/videos").mock(
        return_value=httpx_response({"id": "vid1", "status": "pending", "polling_url": "u"})
    )
    respx_mock.get("https://routerai.ru/api/v1/videos/vid1").mock(
        return_value=httpx_response(
            {
                "id": "vid1",
                "status": "completed",
                "unsigned_urls": ["https://x/v.mp4"],
                "usage": {"cost": 18.2},
            }
        )
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("x-ai/grok-imagine-video", "кот")
    assert not task.done
    task.wait(timeout=5, interval=0.01)
    assert task.done
    assert task.status == "completed"
    assert task.cost_rub == Decimal("18.2")
    assert task.urls == ["https://x/v.mp4"]
    client.close()


def test_account_credits(respx_mock):
    respx_mock.get("https://routerai.ru/api/v1/credits").mock(
        return_value=httpx_response({"data": {"credits": 531.6199}})
    )
    client = RouterAI(api_key="sk-test")
    assert client.account.credits() == Decimal("531.6199")
    client.close()


def test_keys_and_team(respx_mock):
    respx_mock.get("https://routerai.ru/api/v1/keys").mock(
        return_value=httpx_response({"data": [{"id": "k1", "name": "prod"}]})
    )
    respx_mock.post("https://routerai.ru/api/v1/keys").mock(
        return_value=httpx_response({"data": {"id": "k2", "key": "sk-new"}})
    )
    respx_mock.post("https://routerai.ru/api/v1/team/members").mock(
        return_value=httpx_response({"data": {"id": 42, "email": "user@example.com"}})
    )
    respx_mock.post("https://routerai.ru/api/v1/team/invitations").mock(
        return_value=httpx_response(
            {"data": {"id": 7, "email": "inv@example.com", "invite_url": "https://x/i"}}
        )
    )
    client = RouterAI(api_key="sk-master")
    assert client.keys.list()[0].id == "k1"
    assert client.keys.create("dev").key == "sk-new"
    member = client.team.create_member("user@example.com")
    assert member.data is not None and member.data.id == 42
    invite = client.team.invite("inv@example.com")
    assert invite.id == 7 and invite.invite_url == "https://x/i"
    client.close()


def test_logging_masks_api_key(respx_mock, caplog):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        side_effect=Exception("boom")
    )
    client = RouterAI(api_key="sk-supersecretkey", logger="test")
    client._http._logger.setLevel(logging.INFO)
    with caplog.at_level(logging.WARNING, logger="routerai.test"), suppress(Exception):
        client.chat.complete("m", "x")
    logs = caplog.text
    assert "sk-supersecretkey" not in logs
    client.close()
