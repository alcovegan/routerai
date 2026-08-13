from __future__ import annotations

import os

import httpx
import pytest

CATALOG = [
    {
        "id": "deepseek/deepseek-v4-pro",
        "name": "DeepSeek: DeepSeek V4 Pro",
        "created": 1777000679,
        "description": "DeepSeek V4 Pro MoE reasoning model",
        "context_length": 1000000,
        "architecture": {
            "modality": "text->text",
            "tokenizer": "DeepSeek",
            "instruct_type": None,
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        },
        "pricing": {"prompt": 0.000065, "completion": 0.000131},
        "per_request_limits": None,
        "supported_parameters": ["temperature", "tools", "include_reasoning"],
        "default_parameters": None,
    },
    {
        "id": "openai/gpt-image-1",
        "name": "OpenAI: GPT Image 1",
        "created": 1720000000,
        "description": "Image generation model",
        "context_length": None,
        "architecture": {
            "modality": "text->image",
            "tokenizer": None,
            "instruct_type": None,
            "input_modalities": ["text"],
            "output_modalities": ["image"],
        },
        "pricing": {"prompt": None, "completion": 0.04},
        "per_request_limits": None,
        "supported_parameters": None,
        "default_parameters": None,
    },
    {
        "id": "anthropic/claude-sonnet-5",
        "name": "Anthropic: Claude Sonnet 5",
        "created": 1780000000,
        "description": "Vision model with tools",
        "context_length": 1000000,
        "architecture": {
            "modality": "text+image->text",
            "tokenizer": "Claude",
            "instruct_type": None,
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
        "pricing": {"prompt": 0.000215, "completion": 0.001078},
        "per_request_limits": None,
        "supported_parameters": ["temperature", "tools"],
        "default_parameters": {},
    },
]


def httpx_response(payload, status_code: int = 200, headers: dict | None = None) -> httpx.Response:
    if isinstance(payload, bytes):
        return httpx.Response(status_code, content=payload, headers=headers)
    return httpx.Response(status_code, json=payload, headers=headers)


@pytest.fixture()
def catalog_route(respx_mock):
    return respx_mock.get("https://routerai.ru/api/v1/models").mock(
        return_value=httpx_response({"data": CATALOG})
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run live tests against the real RouterAI API",
    )


def pytest_collection_modifyitems(config, items):
    """Keep the default suite hermetic: live tests need an explicit opt-in."""
    run_live = config.getoption("--run-live") or os.getenv("ROUTERAI_RUN_LIVE") == "1"
    if run_live:
        return
    skip = pytest.mark.skip(reason="live tests require --run-live or ROUTERAI_RUN_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _network_guard(request):
    """Block real sockets for every non-live test (second hermeticity layer).

    Unix sockets stay enabled: asyncio's selector event loop uses
    ``socket.socketpair()`` for its self-pipe on Python 3.14.
    """
    import pytest_socket

    if "live" in request.node.keywords:
        yield
        return
    pytest_socket.disable_socket(allow_unix_socket=True)
    try:
        yield
    finally:
        pytest_socket.enable_socket()
