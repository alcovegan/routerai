"""Цикл выполнения инструментов: модель просит функцию — SDK её вызывает.

Без этого пользователь каждый раз пишет одно и то же: описать функцию схемой,
заметить запрос, разобрать аргументы, приложить результат и спросить снова.
"""

from __future__ import annotations

import asyncio
import itertools
import json

import httpx
import pytest

from routerai import RouterAI
from routerai.tools import parse_arguments, tool_schema


def get_weather(city: str, units: str = "celsius") -> str:
    """Узнать погоду в городе."""
    return f"в городе {city}: +17 ({units})"


def _client(responses: list[dict]) -> RouterAI:
    turns = itertools.count()

    def handler(request: httpx.Request) -> httpx.Response:
        index = next(turns)
        payload = responses[min(index, len(responses) - 1)]
        return httpx.Response(
            200,
            json=payload,
            headers={"content-type": "application/json", "X-Generation-Id": f"gen-{index}"},
        )

    return RouterAI(
        api_key="sk-tools",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _asks_for_weather(arguments: str = '{"city": "Москва"}') -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": arguments},
                        }
                    ],
                }
            }
        ],
        "usage": {"total_tokens": 20, "cost": "0.1"},
    }


FINAL = {
    "choices": [{"message": {"content": "В Москве +17."}}],
    "usage": {"total_tokens": 30, "cost": "0.2"},
}


def test_schema_comes_from_the_signature():
    schema = tool_schema(get_weather)
    assert schema["function"]["name"] == "get_weather"
    assert schema["function"]["description"] == "Узнать погоду в городе."
    params = schema["function"]["parameters"]
    assert params["properties"]["city"]["type"] == "string"
    assert params["required"] == ["city"]  # units has a default
    assert params["properties"]["units"]["default"] == "celsius"


def test_the_loop_calls_the_function_and_answers():
    client = _client([_asks_for_weather(), FINAL])
    answer = client.chat.run_tools("m", "Погода в Москве?", tools=[get_weather])

    assert answer.content == "В Москве +17."
    assert answer.turns == 2
    assert [run.name for run in answer.runs] == ["get_weather"]
    assert answer.runs[0].arguments == {"city": "Москва"}
    assert "+17" in answer.runs[0].result
    # the tool result went back to the model as its own turn
    assert answer.messages[-1]["role"] == "tool"
    assert answer.messages[-1]["tool_call_id"] == "call_1"
    client.close()


def test_a_failing_tool_is_reported_to_the_model_not_raised():
    def broken(city: str) -> str:
        raise ValueError("сервис погоды недоступен")

    client = _client([_asks_for_weather(), FINAL])
    answer = client.chat.run_tools("m", "Погода?", tools={"get_weather": broken})
    assert answer.runs[0].error is not None
    assert "сервис погоды недоступен" in answer.runs[0].error
    assert answer.content == "В Москве +17."
    client.close()


def test_a_model_stuck_asking_stops_at_max_turns():
    """A loop that never ends must cost a bounded amount of money."""
    client = _client([_asks_for_weather()])
    answer = client.chat.run_tools("m", "Погода?", tools=[get_weather], max_turns=3)
    assert answer.turns == 3
    assert len(answer.runs) == 3
    assert client.usage.snapshot().requests == 3
    client.close()


def test_unknown_tool_is_answered_instead_of_crashing():
    client = _client(
        [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "unknown", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            },
            FINAL,
        ]
    )
    answer = client.chat.run_tools("m", "hi", tools=[get_weather])
    assert answer.runs[0].error is not None
    assert "unknown" in answer.runs[0].error
    client.close()


def test_malformed_arguments_are_reported_as_a_tool_error():
    client = _client([_asks_for_weather(arguments="{не json"), FINAL])
    answer = client.chat.run_tools("m", "hi", tools=[get_weather])
    assert answer.runs[0].error is not None
    assert answer.content == "В Москве +17."
    client.close()


def test_async_tools_are_awaited():
    async def aweather(city: str) -> str:
        await asyncio.sleep(0)
        return f"{city}: +18"

    turns = itertools.count()

    async def handler(request: httpx.Request) -> httpx.Response:
        index = next(turns)
        payload = _asks_for_weather() if index == 0 else FINAL
        return httpx.Response(
            200,
            json=payload,
            headers={"content-type": "application/json", "X-Generation-Id": f"agen-{index}"},
        )

    client = RouterAI(
        api_key="sk-tools",
        max_retries=0,
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def main() -> None:
        answer = await client.chat.arun_tools("m", "Погода?", tools={"get_weather": aweather})
        assert answer.runs[0].result == "Москва: +18"
        assert answer.content == "В Москве +17."
        await client.aclose()

    asyncio.run(main())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [('{"a": 1}', {"a": 1}), ("", {}), ({"a": 2}, {"a": 2})],
)
def test_arguments_parse_from_either_wire_shape(raw, expected):
    assert parse_arguments(raw) == expected


def test_tool_calls_are_sent_back_in_the_shape_the_api_expects():
    client = _client([_asks_for_weather(), FINAL])
    answer = client.chat.run_tools("m", "hi", tools=[get_weather])
    assistant = next(m for m in answer.messages if m["role"] == "assistant")
    call = assistant["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Москва"}
    client.close()
