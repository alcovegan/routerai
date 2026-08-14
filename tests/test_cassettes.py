"""Поведение SDK на настоящих ответах RouterAI, записанных в кассеты.

Две части. Первая закрепляет то, что уже работает правильно, — чтобы это не
сломали. Вторая описывает дефекты, найденные аудитом: такие тесты помечены
``xfail(strict=True)``, поэтому сейчас они «ожидаемо падают», а как только
дефект починят, тест начнёт проходить и pytest сообщит об этом ошибкой
XPASS — сигнал снять маркер.

Происхождение данных и правила обновления — в ``tests/cassettes/README.md``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from decimal import Decimal

import pytest

from routerai import (
    APIStatusError,
    AuthenticationError,
    RateLimitError,
    StreamAccumulator,
)
from routerai.schemas import ModelPricing, Usage

from .cassettes import cassette_client, load

CATALOG = "catalog_models"


# --------------------------------------------------------------------- работает


def test_catalog_parses_every_pricing_shape():
    client = cassette_client(CATALOG)
    models = client.models.all()
    assert {m.id for m in models} >= {
        "inclusionai/ling-2.6-flash",
        "black-forest-labs/flux.2-pro",
        "cohere/rerank-4-pro",
    }
    text_model = client.models.get("inclusionai/ling-2.6-flash")
    assert text_model.pricing.prompt == Decimal("1.0894754e-06")
    assert "tools" in {c.value for c in client.models.get("mistralai/mistral-nemo").capabilities}


def test_chat_completion_reads_content_cost_and_generation_id():
    client = cassette_client("chat_completion_ok")
    result = client.chat.complete("inclusionai/ling-2.6-flash", "столица Франции?")
    assert result.content == "Париж"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 33
    assert result.usage.tokens() == 38
    assert result.cost_rub == Decimal("5.22948192e-05")
    assert result.finish_reason == "stop"


def test_leading_whitespace_before_json_does_not_break_parsing():
    """RouterAI шлёт пробельный «прогрев» перед телом — это не должно мешать."""
    cassette = load("chat_completion_ok")
    assert cassette["response"]["body"].startswith((" ", "\n")), "кассета должна хранить прогрев"
    client = cassette_client("chat_completion_ok")
    assert client.chat.complete("m", "hi").content == "Париж"


def test_stream_keepalive_comment_is_skipped():
    """Keep-alive приходит SSE-комментарием `: PROCESSING`."""
    cassette = load("chat_stream_text")
    assert ": PROCESSING" in cassette["response"]["body"]
    client = cassette_client("chat_stream_text")
    chunks = list(client.chat.stream("mistralai/mistral-nemo", "посчитай до пяти"))
    assert "".join(c.content for c in chunks) == "1, 2, 3, 4, 5"


def test_stream_reports_usage_in_final_chunk():
    client = cassette_client("chat_stream_text")
    usages = [c.usage for c in client.chat.stream("m", "hi") if c.usage is not None]
    assert usages, "usage должен приходить в последнем чанке"
    assert usages[-1].tokens() == 27


def test_unknown_model_maps_to_api_status_error():
    client = cassette_client("chat_model_not_found")
    with pytest.raises(APIStatusError) as excinfo:
        client.chat.complete("no-such/model-xyz", "hi")
    assert excinfo.value.status_code == 400
    assert "not found" in str(excinfo.value)


def test_unauthorized_maps_to_authentication_error():
    client = cassette_client("auth_401")
    with pytest.raises(AuthenticationError):
        client.keys.list()


def test_embeddings_return_numeric_vectors():
    client = cassette_client("embeddings_ok")
    result = client.embeddings.create("perplexity/pplx-embed-v1-0.6B", ["привет", "hello"])
    assert result.embeddings
    assert all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in result.embeddings[0]
    )


def test_rerank_orders_documents_by_relevance():
    client = cassette_client("rerank_ok")
    result = client.rerank.create(
        "cohere/rerank-4-pro", "погода в москве", ["Прогноз погоды в Москве", "Рецепт борща"]
    )
    assert [r["index"] for r in result.results] == [0, 1]
    assert result.results[0]["relevance_score"] > result.results[1]["relevance_score"]


def test_image_generation_returns_decoded_bytes():
    client = cassette_client("images_generate_ok")
    result = client.images.generate("black-forest-labs/flux.2-pro", "синий круг")
    assert result.images and result.images[0].data
    assert result.cost_rub == Decimal("3.2684262")


def test_generation_info_unwraps_data_envelope():
    client = cassette_client("generation_info")
    info = client.generation.get("rai-1786733732-Nh0dEtoMPn40gTZTk8FA")
    assert info.total_cost == Decimal("5.22948192e-05")


# ------------------------------------------------------------------ дефекты


def test_stream_accumulator_merges_tool_call_deltas():
    """Сервер шлёт вызов инструмента по частям — собрать его должен аккумулятор."""
    client = cassette_client("chat_stream_tool_calls")
    accumulator = StreamAccumulator()
    for chunk in client.chat.stream("mistralai/mistral-nemo", "погода в Москве?"):
        accumulator.add(chunk)

    assert len(accumulator.tool_calls) == 1, "фрагменты одного вызова должны склеиться"
    call = accumulator.tool_calls[0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Moscow"}


def test_provider_rate_limit_maps_to_rate_limit_error():
    """Лимит провайдера приходит внутри HTTP 200 и должен стать RateLimitError."""
    client = cassette_client("chat_stream_provider_error")
    with pytest.raises(RateLimitError):
        list(client.chat.stream("inclusionai/ling-2.6-flash", "привет"))


def test_validation_error_inside_200_keeps_real_status_code():
    client = cassette_client("chat_error_in_200")
    with pytest.raises(APIStatusError) as excinfo:
        client.chat.complete("inclusionai/ling-2.6-flash", "hi", max_tokens=-5)
    assert excinfo.value.status_code == 400
    assert excinfo.value.http_status == 200
    assert excinfo.value.status_source == "provider"


def test_embeddings_rate_limit_maps_to_rate_limit_error():
    client = cassette_client("embeddings_rate_limited")
    with pytest.raises(RateLimitError):
        client.embeddings.create("perplexity/pplx-embed-v1-0.6B", "привет")


def test_transcription_usage_counts_tokens():
    """Транскрипция отдаёт usage в другом именовании полей."""
    client = cassette_client("transcription_ok")
    result = client.audio.transcribe("openai/gpt-4o-mini-transcribe", b"\x00\x01", format="wav")
    assert result.usage is not None
    assert result.usage.tokens() == 8


def test_generation_usage_counts_tokens():
    usage = Usage.model_validate({"input_tokens": 33, "output_tokens": 5})
    assert usage.tokens() == 38


def test_base64_embeddings_are_decoded_to_numbers():
    client = cassette_client("embeddings_base64")
    result = client.embeddings.create(
        "perplexity/pplx-embed-v1-0.6B", "привет", extra={"encoding_format": "base64"}
    )
    assert result.embeddings
    assert all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in result.embeddings[0]
    )


def test_image_model_price_is_reachable():
    client = cassette_client(CATALOG)
    flux = client.models.get("black-forest-labs/flux.2-pro")
    assert flux.pricing.per_million("image_output") is not None


def test_price_filter_excludes_models_priced_in_other_units():
    client = cassette_client(CATALOG)
    cheap = client.models.search(max_price_prompt=1.0)
    assert "black-forest-labs/flux.2-pro" not in {m.id for m in cheap}


def test_authentication_error_is_an_api_status_error():
    client = cassette_client("auth_401")
    with pytest.raises(APIStatusError):
        client.keys.list()


def test_typed_errors_carry_status_and_body():
    client = cassette_client("auth_401")
    with pytest.raises(AuthenticationError) as excinfo:
        client.account.credits()
    assert getattr(excinfo.value, "status_code", None) == 401
    assert getattr(excinfo.value, "body", None) is not None


def test_cost_survives_a_serialization_round_trip():
    """Caching a result must not lose the price: dump then validate keeps it."""
    usage = Usage.model_validate({"total_tokens": 10, "cost": "1.2345"})
    assert usage.cost_rub == Decimal("1.2345")
    assert Usage.model_validate(usage.model_dump()).cost_rub == Decimal("1.2345")
    assert Usage.model_validate(json.loads(usage.model_dump_json())).cost_rub == Decimal("1.2345")
    assert Usage(cost_rub=Decimal("9.99")).cost_rub == Decimal("9.99")
    assert usage.model_dump(by_alias=True)["cost"] == Decimal("1.2345")


def test_pricing_exposes_units_other_than_tokens():
    """Image and rerank models charge per image and per search unit, not per token."""
    client = cassette_client(CATALOG)
    flux = client.models.get("black-forest-labs/flux.2-pro")
    rerank = client.models.get("cohere/rerank-4-pro")
    video = client.models.get("kwaivgi/kling-v3.0-std")

    assert flux.pricing.priced_units() == {"image_output"}
    assert rerank.pricing.priced_units() == {"search_units"}
    assert video.pricing.priced_units() == {"seconds"}
    assert flux.pricing.price("image_output") == Decimal("0.000797955615234375")
    assert not flux.pricing.is_free()
    # a unit the SDK has never heard of still reads back as a Decimal
    unknown = ModelPricing.model_validate({"prompt": 0.0, "brand_new_unit": 0.25})
    assert unknown.price("brand_new_unit") == Decimal("0.25")
    assert unknown.per_million("brand_new_unit") == Decimal("250000")


def test_price_filter_keeps_models_priced_in_tokens():
    """The stricter price filter must not throw out ordinary text models."""
    client = cassette_client(CATALOG)
    cheap = {m.id for m in client.models.search(max_price_prompt=100.0)}
    assert "inclusionai/ling-2.6-flash" in cheap
    assert "hexgrad/kokoro-82m" in cheap
    assert "black-forest-labs/flux.2-pro" not in cheap


def test_image_stream_raises_on_provider_error():
    """The image stream used to hand the error back as an ordinary chunk."""
    import httpx

    from routerai import RouterAI

    body = b'data: {"error":{"message":"provider refused","code":429}}\n\ndata: [DONE]\n\n'
    client = RouterAI(
        api_key="sk-cassette",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, content=body, headers={"content-type": "text/event-stream"}
                )
            )
        ),
    )
    with pytest.raises(RateLimitError) as excinfo:
        list(client.images.stream("black-forest-labs/flux.2-pro", "кот"))
    assert excinfo.value.status_code == 429
    client.close()


def test_stream_keepalive_with_empty_data_field_is_skipped():
    """An empty `data:` line is a keep-alive, not a broken event."""
    import httpx

    from routerai import RouterAI

    body = (
        b"data: \n\n"
        b'data: {"choices":[{"delta":{"content":"\xd0\xbe\xd0\xba"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    client = RouterAI(
        api_key="sk-cassette",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, content=body, headers={"content-type": "text/event-stream"}
                )
            )
        ),
    )
    assert "".join(c.content for c in client.chat.stream("m", "hi")) == "ок"
    client.close()


def test_path_segments_are_escaped():
    """An id from an outside source must not walk out of /api/v1."""
    import httpx

    from routerai import RouterAI

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": {}}, headers={"content-type": "application/json"})

    client = RouterAI(
        api_key="sk-cassette",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.keys.delete("../../v1/team/members/999")
    # the request stays under /api/v1/keys/ instead of walking to another endpoint
    assert seen[0].startswith("https://routerai.ru/api/v1/keys/")
    assert "%2E%2E" in seen[0]
    client.close()


def test_webhook_rejects_a_signature_it_cannot_compare():
    """The signature header is unauthenticated input: it must not crash the handler."""
    from routerai.webhooks import WebhookVerificationError, signing_secret, verify_video

    body = b'{"task_id":"t1","status":"completed"}'
    stamp = str(int(time.time()))
    with pytest.raises(WebhookVerificationError):
        verify_video(body, "подпись-кириллицей", "sk-test", stamp)

    good = hmac.new(
        signing_secret("sk-test").encode("ascii"),
        stamp.encode() + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    assert verify_video(body, good, "sk-test", stamp)["task_id"] == "t1"


def test_transcription_reads_subtitle_formats_as_text():
    """srt/vtt/text are plain text; parsing them as JSON is a parse error."""
    import httpx

    from routerai import RouterAI

    srt = b"1\n00:00:00,000 --> 00:00:02,000\n\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\n"
    client = RouterAI(
        api_key="sk-cassette",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, content=srt, headers={"content-type": "text/plain"})
            )
        ),
    )
    result = client.audio.transcribe(
        "openai/gpt-4o-mini-transcribe", b"\x00\x01", format="wav", response_format="srt"
    )
    assert "Привет" in result.text
    client.close()


def test_cost_formatting_never_loses_a_paid_response():
    """A malformed cost must not turn a delivered answer into an exception."""
    import logging as std_logging

    import httpx

    from routerai import RouterAI

    payload = {
        "choices": [{"message": {"content": "ответ"}}],
        "usage": {"total_tokens": 10, "cost": "н/д"},
    }
    std_logging.getLogger("routerai").setLevel(std_logging.INFO)
    client = RouterAI(
        api_key="sk-cassette",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, json=payload, headers={"content-type": "application/json"}
                )
            )
        ),
    )
    assert client.chat.complete("m", "hi").content == "ответ"
    client.close()
