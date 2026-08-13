from __future__ import annotations

import base64
import logging
from contextlib import suppress
from decimal import Decimal

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
    result = client.audio.speech("openai/tts-1", "Привет")
    assert result.data == b"mp3-bytes"
    assert result.save(tmp_path / "hi.mp3").read_bytes() == b"mp3-bytes"
    client.close()


def test_audio_transcribe(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/audio/transcriptions").mock(
        return_value=httpx_response(
            {"text": "привет мир", "usage": {"seconds": 2.1, "cost": 0.01}},
            headers={"X-Generation-Id": "gen-stt"},
        )
    )
    client = RouterAI(api_key="sk-test")
    result = client.audio.transcribe("openai/whisper-1", b"fakeaudio")
    assert result.text == "привет мир"
    assert result.cost_rub == Decimal("0.01")
    assert result.generation_id == "gen-stt"
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
        return_value=httpx_response({"id": "vid1", "status": "completed", "url": "https://x/v.mp4"})
    )
    client = RouterAI(api_key="sk-test")
    task = client.videos.create("x-ai/grok-imagine-video", "кот")
    assert not task.done
    task.wait(timeout=5, interval=0.01)
    assert task.done
    assert task.status == "completed"
    client.close()


def test_keys_and_team(respx_mock):
    respx_mock.get("https://routerai.ru/api/v1/keys").mock(
        return_value=httpx_response({"data": [{"id": "k1", "name": "prod"}]})
    )
    respx_mock.post("https://routerai.ru/api/v1/keys").mock(
        return_value=httpx_response({"data": {"id": "k2", "key": "sk-new"}})
    )
    respx_mock.post("https://routerai.ru/api/v1/team/members").mock(
        return_value=httpx_response({"data": {"id": 42}})
    )
    client = RouterAI(api_key="sk-master")
    assert client.keys.list()["data"][0]["id"] == "k1"
    assert client.keys.create("dev")["data"]["key"] == "sk-new"
    assert client.team.create_member("user@example.com")["data"]["id"] == 42
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
