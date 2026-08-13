from __future__ import annotations

from decimal import Decimal

from routerai import Registry, RouterAI

from .conftest import httpx_response


def test_registry_switching(respx_mock):
    route = respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response({"choices": [{"message": {"content": "ok"}}]})
    )
    reg = Registry(main=RouterAI(api_key="sk-main"), personal=RouterAI(api_key="sk-personal"))

    with reg.using("personal") as client:
        client.chat.complete("m", "x")
        assert reg.current() is client

    reg["main"].chat.complete("m", "x")
    assert reg.current() is reg["main"]

    auths = [c.request.headers["Authorization"] for c in route.calls]
    assert auths == ["Bearer sk-personal", "Bearer sk-main"]

    reg.add("third", RouterAI(api_key="sk-third"))
    assert "third" in reg
    reg.close_all()


def test_registry_get_and_remove(respx_mock):
    respx_mock.post("https://routerai.ru/api/v1/chat/completions").mock(
        return_value=httpx_response({"choices": []})
    )
    reg = Registry(a=RouterAI(api_key="sk-a"), b=RouterAI(api_key="sk-b"))
    assert reg.get("missing") is None
    reg.remove("a")
    assert "a" not in reg
    reg.close_all()


def test_generation_cost(respx_mock):
    respx_mock.get("https://routerai.ru/api/v1/generation?id=gen-42").mock(
        return_value=httpx_response({"id": "gen-42", "total_cost": 12.34})
    )
    client = RouterAI(api_key="sk-test")
    assert client.generation.cost("gen-42") == Decimal("12.34")
    client.close()
