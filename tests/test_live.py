"""Live integration matrix against the real RouterAI API.

These tests spend real rubles and never run in the default suite: they are
gated by ``pytest.mark.live`` plus an explicit opt-in (``--run-live`` flag or
``ROUTERAI_RUN_LIVE=1`` env var) and skip unless ``ROUTERAI_API_KEY`` is set.

Run:
    ROUTERAI_API_KEY=sk-... uv run pytest tests/test_live.py --run-live -v -s
Skip the most expensive test:
    ROUTERAI_API_KEY=sk-... uv run pytest tests/test_live.py --run-live -v -s -k "not video"
"""

from __future__ import annotations

import json
import os
import time
from decimal import Decimal

import pytest

from routerai import RouterAI
from routerai.errors import APIStatusError, AuthenticationError, InsufficientFundsError

pytestmark = pytest.mark.live

ENV_KEY = "ROUTERAI_API_KEY"
CHEAP_CHAT = "inclusionai/ling-2.6-flash"
CHEAP_CHAT_BACKUP = "meta-llama/llama-3.1-8b-instruct"
REASONING_MODEL = "deepseek/deepseek-v4-pro"
IMAGE_MODEL = "google/gemini-2.5-flash-image"
IMAGE_MODEL_MID = "bytedance-seed/seedream-4.5"
TTS_MODEL = "x-ai/grok-voice-tts-1.0"
STT_MODEL = "openai/whisper-large-v3"
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-minilm-l6-v2"
RERANK_MODEL = "voyageai/rerank-2.5-lite"
AUDIO_MODEL = "openai/gpt-audio-mini"
VIDEO_MODEL = "bytedance/seedance-2.0-mini"


def robust_chat(client, model: str, prompt, **kwargs):
    """Try the primary cheap model, fall back to a backup on upstream rate limits."""
    for candidate in (model, CHEAP_CHAT_BACKUP):
        try:
            return client.chat.complete(candidate, prompt, **kwargs)
        except APIStatusError as exc:
            if "rate" in str(exc).lower() and candidate != CHEAP_CHAT_BACKUP:
                print(f"  [upstream rate limit on {candidate}, retrying with {CHEAP_CHAT_BACKUP}]")
                continue
            raise


@pytest.fixture(scope="session")
def live_client():
    key = os.getenv(ENV_KEY)
    if not key:
        pytest.skip(f"set {ENV_KEY} to run live tests")
    client = RouterAI(api_key=key, max_retries=3, retry_backoff=1.0)
    yield client
    client.close()


@pytest.fixture(autouse=True)
def _pace():
    """Respect RouterAI per-minute rate limits between live calls."""
    yield
    time.sleep(6.0)


@pytest.fixture(scope="session")
def spend():
    total = {"rub": Decimal("0")}
    yield total
    print(f"\n\n===== LIVE TESTS TOTAL SPEND: {total['rub']} ₽ =====\n")


def track(spend, cost, *, required: bool = True):
    if cost is None:
        if required:
            raise AssertionError("response has no usage.cost")
        print("  [no usage.cost reported]")
        return None
    assert isinstance(cost, Decimal)
    spend["rub"] += cost
    print(f"  [cost {cost} ₽, running total {spend['rub']} ₽]")
    return cost


def pick_model_with_api(client, api: str, candidates: list[str]) -> str:
    for model_id in candidates:
        detail = client.models.endpoints(model_id)
        if any(api in (ep.supported_apis or []) for ep in detail.endpoints):
            return model_id
    pytest.skip(f"no candidate model supports the '{api}' api")


def cheapest_vision_model(client) -> str:
    models = [
        m
        for m in client.models.all()
        if "image" in m.architecture.input_modalities
        and "text" in m.architecture.output_modalities
        and m.pricing.prompt is not None
        and m.pricing.prompt > 0
        and (m.context_length or 0) >= 8_000
    ]
    models.sort(key=lambda m: m.pricing.prompt)
    assert models, "no vision model found"
    return models[0].id


# --- 1. text: plain completion ---


def test_chat_complete_cheap(live_client, spend):
    result = robust_chat(live_client, CHEAP_CHAT, "Ответь одним словом: столица России?")
    assert result.content and "москва" in result.content.lower()
    assert result.usage and result.usage.total_tokens > 0
    assert result.generation_id
    track(spend, result.cost_rub)


# --- 2. text: streaming ---


def test_chat_stream_cheap(live_client, spend):
    chunks = list(live_client.chat.stream(CHEAP_CHAT, "Посчитай от 1 до 3."))
    assert chunks, "no SSE chunks received"
    text = "".join(chunk.content for chunk in chunks)
    assert "3" in text
    assert chunks[-1].finish_reason in (None, "stop")
    cost = chunks[-1].cost_rub
    if cost is None:
        pytest.skip("provider did not report usage.cost on the last stream chunk")
    track(spend, cost)


# --- 3. text: service_tier flex ---


def test_service_tier_flex(live_client, spend):
    result = robust_chat(live_client, CHEAP_CHAT, "Одно слово: да или нет?", service_tier="flex")
    assert result.content
    # the model may not support flex: the provider either reports the tier or null
    assert result.service_tier in (None, "flex", "default")
    track(spend, result.cost_rub)


# --- 4. reasoning ---


def test_reasoning_deepseek(live_client, spend):
    result = live_client.chat.complete(REASONING_MODEL, "Подумай шаг за шагом: 17*23=?")
    assert result.content
    assert result.reasoning, "no reasoning content returned by a reasoning model"
    track(spend, result.cost_rub)


# --- 5. tools + json ---


def test_tools_and_json_mode(live_client, spend):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    result = robust_chat(
        live_client,
        CHEAP_CHAT,
        "Какая погода в Москве? Вызови инструмент.",
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert result.tool_calls, "expected a tool call"
    assert result.tool_calls[0].name == "get_weather"
    track(spend, result.cost_rub)

    result_json = live_client.chat.complete(
        CHEAP_CHAT,
        'Верни JSON {"ok": true}, только JSON.',
        response_format={"type": "json_object"},
    )
    parsed = json.loads(result_json.content or "")
    assert parsed["ok"] is True
    track(spend, result_json.cost_rub)


# --- 6-7. image generation ---


def test_image_generation_cheap(live_client, spend, tmp_path):
    result = live_client.images.generate(IMAGE_MODEL, "красный квадрат", n=1)
    assert result.images
    path = result.images[0].save(tmp_path / "cheap.png")
    assert path.stat().st_size > 100
    assert result.generation_id
    cost = track(spend, result.cost_rub)
    # guard against silent pricing surprises (catalog price is ~0.01 ₽/image)
    assert cost < 3, f"unexpectedly expensive image: {cost} ₽ for {IMAGE_MODEL}"


def test_image_generation_mid(live_client, spend, tmp_path):
    result = live_client.images.generate(IMAGE_MODEL_MID, "синий круг", n=1)
    assert result.images
    path = result.images[0].save(tmp_path / "mid.png")
    assert path.stat().st_size > 100
    track(spend, result.cost_rub)


# --- 8. vision ---


def test_vision_caption(live_client, spend):
    generated = live_client.images.generate(IMAGE_MODEL, "жёлтый треугольник", n=1)
    track(spend, generated.cost_rub)
    b64 = generated.images[0].b64
    model = cheapest_vision_model(live_client)
    print(f"  [vision model: {model}]")
    result = live_client.chat.complete(
        model,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что на картинке? Одним словом."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    # reasoning models may burn tokens on thinking; require either text or reasoning
    assert result.content or result.reasoning
    track(spend, result.cost_rub)


# --- 9-10. tts -> stt roundtrip ---


def test_tts_then_stt_roundtrip(live_client, spend, tmp_path):
    speech = live_client.audio.speech(
        TTS_MODEL, "Привет, это тест синтеза речи.", voice="eve", response_format="mp3"
    )
    assert speech.data
    assert speech.generation_id
    mp3 = speech.save(tmp_path / "speech.mp3")
    assert mp3.stat().st_size > 100

    transcription = live_client.audio.transcribe(STT_MODEL, mp3)
    assert transcription.text
    assert transcription.usage and transcription.usage.seconds is not None
    track(spend, transcription.cost_rub)


# --- 11. embeddings ---


def test_embeddings(live_client, spend):
    result = live_client.embeddings.create(EMBEDDINGS_MODEL, "привет мир")
    assert result.embeddings and len(result.embeddings[0]) > 0
    assert result.usage and result.usage.total_tokens
    track(spend, result.cost_rub, required=False)


# --- 12. rerank ---


def test_rerank(live_client, spend):
    result = live_client.rerank.create(
        RERANK_MODEL, "столица России", ["Москва", "Париж", "Лондон"]
    )
    assert result.results
    top = result.top_documents(["Москва", "Париж", "Лондон"])
    assert top[0] == "Москва"
    assert result.usage and result.usage.total_tokens
    track(spend, result.cost_rub, required=False)


# --- 13-14. responses + messages api ---


def test_responses_api(live_client, spend):
    model = pick_model_with_api(
        live_client, "responses", [CHEAP_CHAT, REASONING_MODEL, "openai/gpt-5.6-mini"]
    )
    print(f"  [responses model: {model}]")
    payload = live_client.responses.create(model, "Ответь одним словом: столица Японии?")
    assert payload.get("id")
    assert payload.get("status") == "completed"
    output_text = ""
    for item in payload.get("output") or []:
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                output_text += part.get("text", "")
    assert "Токио" in output_text
    usage = payload.get("usage") or {}
    assert usage.get("total_tokens") or usage.get("input_tokens")
    cost = Decimal(str(usage["cost"])) if usage.get("cost") is not None else None
    track(spend, cost, required=False)


def test_messages_api(live_client, spend):
    model = pick_model_with_api(
        live_client, "messages", ["anthropic/claude-sonnet-4.5", REASONING_MODEL]
    )
    print(f"  [messages model: {model}]")
    payload = live_client.messages.create(
        model,
        [{"role": "user", "content": "Ответь одним словом: столица Германии?"}],
        max_tokens=64,
    )
    assert payload.get("id")
    assert payload.get("content")
    usage = payload.get("usage") or {}
    assert usage.get("input_tokens")
    cost = Decimal(str(usage["cost"])) if usage.get("cost") is not None else None
    track(spend, cost, required=False)


# --- 15. audio ---


def test_audio_input_and_output(live_client, spend, tmp_path):
    # audio input: feed a generated mp3 into the audio model
    speech = live_client.audio.speech(
        TTS_MODEL, "Здравствуйте, это тест.", voice="eve", response_format="mp3"
    )
    mp3 = speech.save(tmp_path / "input.mp3")
    import base64

    b64 = base64.b64encode(mp3.read_bytes()).decode()
    result = live_client.chat.complete(
        AUDIO_MODEL,
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}},
                    {"type": "text", "text": "Кратко опиши, что слышишь."},
                ],
            }
        ],
        max_tokens=200,
    )
    assert result.usage and result.usage.total_tokens
    details = ((result.raw or {}).get("usage") or {}).get("prompt_tokens_details") or {}
    assert details.get("audio_tokens", 0) > 0, "audio tokens were not billed as audio"
    track(spend, result.cost_rub)

    # audio output requires streaming; just verify the stream opens and yields chunks
    chunks = list(
        live_client.chat.stream(
            AUDIO_MODEL,
            "Сгенерируй короткий звуковой сигнал, бип.",
            extra={"modalities": ["audio", "text"]},
        )
    )
    assert chunks, "no chunks from audio-output stream"
    assert any(chunk.raw for chunk in chunks)


# --- 16. video (most expensive: ~13 ₽) ---


def test_video_short(live_client, spend):
    try:
        task = live_client.videos.create(
            VIDEO_MODEL, "Кот сидит на подоконнике", duration=4, resolution="480p"
        )
    except InsufficientFundsError as exc:
        # RouterAI holds a minimum balance (~500 ₽) to start a video generation;
        # this is an account-state precondition, not a contract failure.
        if "balance" in str(exc).lower() or "500" in str(exc):
            pytest.skip(f"video requires a higher available balance: {exc}")
        raise
    assert task.id
    task.wait(timeout=300, interval=5)
    assert task.status == "completed", f"video finished with status {task.status!r}"
    assert task.urls, "no video urls returned"
    track(spend, task.cost_rub)


# --- 17. post-hoc cost lookup ---


def test_generation_cost_lookup(live_client, spend):
    result = robust_chat(live_client, CHEAP_CHAT, "Одно слово: привет?")
    assert result.generation_id
    cost = live_client.generation.cost(result.generation_id)
    # cost may lag accounting right after the request; require it to be present
    assert cost is not None and cost >= 0
    track(spend, result.cost_rub)


# --- 18. error paths (free) ---


def test_invalid_key_raises_auth_error():
    client = RouterAI(api_key="sk-definitely-invalid-key", max_retries=0)
    with pytest.raises(AuthenticationError):
        client.chat.complete(CHEAP_CHAT, "тест")
    client.close()


def test_invalid_model_raises_typed_error(live_client):
    with pytest.raises((APIStatusError,)) as exc_info:
        live_client.chat.complete("nonexistent/nonexistent-model", "тест", max_tokens=1)
    assert exc_info.value.status_code in (400, 404)
