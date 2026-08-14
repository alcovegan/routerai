"""Загрузка кассет с записанными ответами RouterAI.

Кассета — это один реальный обмен с API, сохранённый во время аудита. Здесь
только чтение файла и сборка клиента поверх ``httpx.MockTransport``; сами
данные лежат в JSON рядом, происхождение описано в README.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

CASSETTES_DIR = Path(__file__).parent


def load(name: str) -> dict[str, Any]:
    """Прочитать кассету по имени файла без расширения."""
    path = CASSETTES_DIR / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in CASSETTES_DIR.glob("*.json"))
        raise FileNotFoundError(f"нет кассеты {name!r}; доступны: {available}")
    return json.loads(path.read_text(encoding="utf-8"))


def response_of(cassette: dict[str, Any]) -> httpx.Response:
    """Собрать httpx-ответ из кассеты, сохраняя статус и content-type."""
    recorded = cassette["response"]
    body = recorded["body"]
    content = body.encode("utf-8") if isinstance(body, str) else b""
    return httpx.Response(
        recorded["status"],
        content=content,
        headers={"content-type": recorded.get("content_type", "application/json")},
    )


def transport(*names: str) -> httpx.MockTransport:
    """Транспорт, отвечающий кассетами по очереди (последняя повторяется)."""
    cassettes = [load(name) for name in names]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        index = min(calls["n"], len(cassettes) - 1)
        calls["n"] += 1
        return response_of(cassettes[index])

    return httpx.MockTransport(handler)


def cassette_client(*names: str, **kwargs: Any):
    """Клиент RouterAI, отвечающий записанными телами вместо сети."""
    from routerai import RouterAI

    kwargs.setdefault("max_retries", 0)
    return RouterAI(
        api_key="sk-cassette",
        http_client=httpx.Client(transport=transport(*names)),
        **kwargs,
    )


def acassette_client(*names: str, **kwargs: Any):
    """Асинхронный вариант :func:`cassette_client`."""
    from routerai import RouterAI

    kwargs.setdefault("max_retries", 0)
    cassettes = [load(name) for name in names]
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        index = min(calls["n"], len(cassettes) - 1)
        calls["n"] += 1
        return response_of(cassettes[index])

    return RouterAI(
        api_key="sk-cassette",
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,
    )
