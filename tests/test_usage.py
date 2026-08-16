"""Учёт расходов: счётчики клиента, блок track() и колбэк on_usage.

Стоимость приходит в каждом ответе RouterAI, но раньше она попадала только
в строчку лога и терялась. Эти тесты закрепляют, что она попадает в счётчики —
в том числе когда логирование выключено и когда ответ пришёл потоком.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import threading
from decimal import Decimal

import httpx
import pytest

from routerai import RouterAI

CHAT_BODY = {
    "model": "m1",
    "choices": [{"message": {"content": "ок"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": "0.5"},
}

STREAM_BODY = (
    b'data: {"model":"m2","choices":[{"delta":{"content":"a"}}]}\n\n'
    b'data: {"model":"m2","choices":[{"delta":{}}],"usage":{"total_tokens":7,"cost":"0.25"}}\n\n'
    b"data: [DONE]\n\n"
)


def _client(**kwargs) -> RouterAI:
    ids = itertools.count(1)

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"X-Generation-Id": f"gen-{next(ids)}"}
        if request.content and json.loads(request.content).get("stream"):
            return httpx.Response(
                200,
                content=STREAM_BODY,
                headers={**headers, "content-type": "text/event-stream"},
            )
        return httpx.Response(
            200, json=CHAT_BODY, headers={**headers, "content-type": "application/json"}
        )

    return RouterAI(
        api_key="sk-usage",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_client_counts_tokens_and_rubles():
    client = _client()
    client.chat.complete("m1", "привет")
    stats = client.usage.snapshot()
    assert stats.requests == 1
    assert stats.total_tokens == 15
    assert stats.cost_rub == Decimal("0.5")
    assert stats.by_model["m1"].cost_rub == Decimal("0.5")
    client.close()


def test_streamed_cost_is_counted_from_the_final_chunk():
    client = _client()
    list(client.chat.stream("m2", "привет"))
    stats = client.usage.snapshot()
    assert stats.streamed == 1
    assert stats.total_tokens == 7
    assert stats.cost_rub == Decimal("0.25")
    client.close()


def test_abandoned_stream_is_still_counted_as_a_request():
    """Breaking out early does not make the tokens free."""
    client = _client()
    for _ in client.chat.stream("m2", "привет"):
        break
    assert client.usage.snapshot().requests == 1
    client.close()


def test_accounting_survives_disabled_logging():
    """Counters that only work with INFO logging on are counters nobody trusts."""
    client = _client()
    logging.disable(logging.CRITICAL)
    try:
        client.chat.complete("m1", "привет")
    finally:
        logging.disable(logging.NOTSET)
    assert client.usage.snapshot().cost_rub == Decimal("0.5")
    client.close()


def test_track_counts_only_its_own_block():
    client = _client()
    client.chat.complete("m1", "до блока")
    with client.track("ingest") as spent:
        client.chat.complete("m1", "внутри")
        client.chat.complete("m1", "внутри")
    client.chat.complete("m1", "после блока")

    assert spent.requests == 2
    assert spent.cost_rub == Decimal("1.0")
    assert client.usage.snapshot().requests == 4
    assert client.usage.snapshot().by_label["ingest"].cost_rub == Decimal("1.0")
    client.close()


def test_track_blocks_nest():
    client = _client()
    with client.track("outer") as outer:
        client.chat.complete("m1", "раз")
        with client.track("inner") as inner:
            client.chat.complete("m1", "два")
    assert inner.requests == 1
    assert outer.requests == 2
    client.close()


def test_on_usage_receives_records_until_unsubscribed():
    client = _client()
    seen = []
    unsubscribe = client.on_usage(seen.append)
    client.chat.complete("m1", "привет")
    unsubscribe()
    client.chat.complete("m1", "привет")

    assert len(seen) == 1
    assert seen[0].model == "m1"
    assert seen[0].cost_rub == Decimal("0.5")
    assert seen[0].generation_id == "gen-1"
    client.close()


def test_a_failing_callback_does_not_break_the_request():
    client = _client()

    def explode(record):
        raise RuntimeError("callback is broken")

    client.on_usage(explode)
    assert client.chat.complete("m1", "привет").content == "ок"
    assert client.usage.snapshot().requests == 1
    client.close()


def test_polling_a_video_counts_the_generation_once():
    """Every refresh repeats the same usage; charging per poll would multiply it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": "v1", "status": "processing", "usage": {"total_tokens": 0, "cost": "10.0"}},
            headers={"content-type": "application/json", "X-Generation-Id": "vid-gen"},
        )

    client = RouterAI(
        api_key="sk-usage",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    task = client.videos.get("v1")
    for _ in range(3):
        task.refresh()

    stats = client.usage.snapshot()
    assert stats.requests == 4
    assert stats.cost_rub == Decimal("10.0")
    client.close()


def test_counters_hold_up_under_threads():
    client = _client()
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(25):
                client.chat.complete("m1", "привет")
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    stats = client.usage.snapshot()
    assert stats.requests == 100
    assert stats.cost_rub == Decimal("50.0")
    client.close()


@pytest.mark.parametrize("concurrency", [4])
def test_tasks_started_inside_track_are_counted(concurrency: int):
    ids = itertools.count(1)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=CHAT_BODY,
            headers={"content-type": "application/json", "X-Generation-Id": f"agen-{next(ids)}"},
        )

    client = RouterAI(
        api_key="sk-usage",
        max_retries=0,
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def main() -> None:
        with client.track("batch") as spent:
            await asyncio.gather(
                *(client.chat.acomplete("m1", "привет") for _ in range(concurrency))
            )
        assert spent.requests == concurrency
        assert spent.cost_rub == Decimal("0.5") * concurrency

    asyncio.run(main())
    asyncio.run(client.aclose())
