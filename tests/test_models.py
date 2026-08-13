from __future__ import annotations

from decimal import Decimal

import pytest

from routerai import Capability, Model, RouterAI
from routerai.errors import ModelNotFoundError

from .conftest import CATALOG, httpx_response


def test_capabilities_derivation():
    models = {m["id"]: Model.model_validate(m) for m in CATALOG}
    deepseek = models["deepseek/deepseek-v4-pro"]
    assert {Capability.TEXT, Capability.REASONING, Capability.TOOLS} <= deepseek.capabilities

    image = models["openai/gpt-image-1"]
    assert Capability.IMAGE_GENERATION in image.capabilities
    assert Capability.VISION not in image.capabilities

    claude = models["anthropic/claude-sonnet-5"]
    assert Capability.VISION in claude.capabilities
    assert Capability.TOOLS in claude.capabilities
    assert Capability.REASONING not in claude.capabilities


def test_pricing_per_million():
    model = Model.model_validate(CATALOG[0])
    assert model.pricing.per_million("prompt") == Decimal("65")
    assert model.author == "deepseek"
    assert model.slug == "deepseek-v4-pro"


def test_models_list_and_get(catalog_route):
    client = RouterAI(api_key="sk-test")
    models = client.models.all()
    assert len(models) == 3
    assert client.models.get("deepseek/deepseek-v4-pro").id == "deepseek/deepseek-v4-pro"
    with pytest.raises(ModelNotFoundError):
        client.models.get("nope/nope")
    client.close()


def test_models_cache_single_request(catalog_route):
    client = RouterAI(api_key="sk-test")
    client.models.all()
    client.models.all()
    client.models.all()
    assert catalog_route.call_count == 1
    client.close()


def test_models_search(catalog_route):
    client = RouterAI(api_key="sk-test")
    hits = client.models.search("deepseek", capabilities=["reasoning"], min_context=1_000_000)
    assert [m.id for m in hits] == ["deepseek/deepseek-v4-pro"]

    assert client.models.search(developer="anthropic")[0].id == "anthropic/claude-sonnet-5"
    assert client.models.search(q="nonexistent") == []
    assert [m.id for m in client.models.search("claude")] == ["anthropic/claude-sonnet-5"]
    client.close()


def test_models_by_capability_and_grouped(catalog_route):
    client = RouterAI(api_key="sk-test")
    assert [m.id for m in client.models.by_capability("image")] == ["openai/gpt-image-1"]
    grouped = client.models.grouped()
    assert grouped[Capability.REASONING][0].id == "deepseek/deepseek-v4-pro"
    assert len(client.models.reasoning()) == 1
    assert len(client.models.vision()) == 1
    client.close()


def test_models_endpoints(respx_mock):
    respx_mock.get("https://routerai.ru/api/v1/models/deepseek/deepseek-v4-pro/endpoints").mock(
        return_value=httpx_response(
            {
                "data": {
                    "id": "deepseek/deepseek-v4-pro",
                    "name": "DeepSeek: DeepSeek V4 Pro",
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    "endpoints": [
                        {
                            "name": "DeepSeek | deepseek/deepseek-v4-pro",
                            "provider_name": "DeepSeek",
                            "tag": "deepseek",
                            "country": "cn",
                            "context_length": 1000000,
                            "status": 0,
                            "pricing": {"prompt": 0.000065, "completion": 0.000131},
                        }
                    ],
                }
            }
        )
    )
    client = RouterAI(api_key="sk-test")
    detail = client.models.endpoints("deepseek/deepseek-v4-pro")
    assert detail.id == "deepseek/deepseek-v4-pro"
    assert detail.endpoints[0].tag == "deepseek"
    assert detail.endpoints[0].pricing.prompt == Decimal("0.000065")
    client.close()
