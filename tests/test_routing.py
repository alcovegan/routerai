"""Роутинг через строку модели и алиасы моделей.

Тесты идут поверх кассет — записей настоящих обменов с RouterAI, снятых
3 сентября 2026 года. Это важно именно здесь: и @-синтаксис, и алиасы
описаны скупо (алиасы — вообще только анонсом в телеграме), а поведение
сервера оказалось не тем, что подсказывает интуиция.

Что показал живой прогон и что закреплено ниже:

* алиас в поле ``model`` ответа не сохраняется — сервер отвечает каноническим
  id той модели, на которую алиас сейчас указывает;
* но совпадение id запроса и ответа не гарантировано и без алиасов: deepseek
  отвечает коротким ``deepseek-v4-flash``, которого в каталоге нет вовсе;
* @-суффикс в ответе не отражается;
* конфликт ``provider`` в строке и в теле — это 400, и ловить его до отправки
  дешевле, чем платить круглый рейс.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from routerai import (
    ConfigurationError,
    ModelNotFoundError,
    NotFoundError,
    is_alias,
    route,
    split_model,
)
from routerai._routing import build_model, conflicting_keys
from routerai.resources.models import filter_models
from routerai.schemas import Model

from .cassettes import cassette_client, load, response_of

# --- разбор и сборка строки модели ---


def test_alias_is_recognised_by_the_tilde_not_by_the_suffix():
    """`openai/gpt-chat-latest` — обычная модель, а не алиас."""
    assert is_alias("~anthropic/claude-opus-latest")
    assert not is_alias("openai/gpt-chat-latest")
    assert not is_alias("anthropic/claude-opus-5")


def test_split_model_separates_the_id_from_the_routing():
    parsed = split_model("anthropic/claude-opus-5@provider=amazon-bedrock&allow_fallbacks=false")
    assert parsed.model == "anthropic/claude-opus-5"
    assert parsed.provider == "amazon-bedrock"
    assert parsed.allow_fallbacks is False
    assert parsed.has_routing


def test_split_model_keeps_an_alias_marker():
    parsed = split_model("~z-ai/glm-flash-latest@provider=deepinfra")
    assert parsed.model == "~z-ai/glm-flash-latest"
    assert parsed.is_alias


def test_a_plain_id_parses_to_itself():
    parsed = split_model("mistralai/mistral-nemo")
    assert parsed.model == "mistralai/mistral-nemo"
    assert not parsed.has_routing
    assert str(parsed) == "mistralai/mistral-nemo"


def test_unknown_routing_keys_survive_the_round_trip():
    """Параметр, которого SDK не знает, должен дойти до сервера как есть."""
    parsed = split_model("m/x@provider=a&future_key=42")
    assert parsed.unknown == {"future_key": "42"}
    assert str(parsed) == "m/x@provider=a&future_key=42"


def test_route_builds_the_documented_syntax():
    assert (
        route("anthropic/claude-opus-5", provider="amazon-bedrock", allow_fallbacks=False)
        == "anthropic/claude-opus-5@provider=amazon-bedrock&allow_fallbacks=false"
    )


def test_routing_twice_does_not_stack_suffixes():
    once = route("m/x", provider="a")
    assert build_model(once, provider="b") == "m/x@provider=b"


def test_route_without_parameters_returns_the_plain_id():
    assert route("m/x") == "m/x"


# --- конфликт строки и тела ---


@pytest.mark.parametrize(
    ("model", "body", "expected"),
    [
        ("m/x@provider=a", {"only": ["a"]}, ["provider"]),
        ("m/x@provider=a", {"order": ["a"]}, ["provider"]),
        ("m/x@allow_fallbacks=false", {"allow_fallbacks": False}, ["allow_fallbacks"]),
        ("m/x@provider=a", {"country": "ru"}, []),
        ("m/x", {"only": ["a"]}, []),
        ("m/x@provider=a", None, []),
    ],
)
def test_conflicting_keys_matches_what_the_server_rejects(model, body, expected):
    assert conflicting_keys(model, body) == expected


def test_the_server_really_answers_400_on_that_conflict():
    """Кассета: провайдер задан и в строке модели, и в теле."""
    cassette = load("chat_route_conflict_400")
    assert cassette["response"]["status"] == 400
    assert "both in model string and request body" in cassette["response"]["body"]


def test_conflict_is_caught_before_a_request_goes_out():
    client = cassette_client("chat_completion_ok")
    with pytest.raises(ConfigurationError, match="provider"):
        client.chat.complete(
            "mistralai/mistral-nemo@provider=deepinfra",
            "привет",
            provider={"only": ["deepinfra"]},
        )
    client.close()


def test_routing_without_a_body_conflict_is_allowed_through():
    client = cassette_client("chat_route_suffix")
    result = client.chat.complete(
        route("mistralai/mistral-nemo", provider="deepinfra", allow_fallbacks=False), "1+1?"
    )
    assert result.model == "mistralai/mistral-nemo"
    client.close()


# --- что сервер кладёт в model ответа ---


def test_alias_comes_back_resolved_to_its_target():
    """Запрошен ~z-ai/glm-flash-latest, в ответе z-ai/glm-5.3-flash."""
    cassette = load("chat_alias_resolved")
    assert cassette["request"]["body"]["model"] == "~z-ai/glm-flash-latest"
    assert json.loads(cassette["response"]["body"])["model"] == "z-ai/glm-5.3-flash"


def test_routing_suffix_is_not_echoed_back():
    cassette = load("chat_route_suffix")
    assert "@" in cassette["request"]["body"]["model"]
    assert json.loads(cassette["response"]["body"])["model"] == "mistralai/mistral-nemo"


def test_some_models_answer_with_a_name_that_is_not_in_the_catalog():
    """Не свойство алиаса: канонический id отвечает тем же коротким именем."""
    cassette = load("chat_short_model_name")
    assert cassette["request"]["body"]["model"] == "deepseek/deepseek-v4-flash-0731"
    assert json.loads(cassette["response"]["body"])["model"] == "deepseek-v4-flash"


# --- учёт расходов при таком расхождении ---


def test_spend_is_filed_under_the_model_the_caller_asked_for():
    """Иначе расход по deepseek уезжает в строку, которой нет в каталоге."""
    client = cassette_client("chat_short_model_name")
    client.chat.complete("deepseek/deepseek-v4-flash-0731", "1+1?")
    stats = client.usage.snapshot()
    assert list(stats.by_model) == ["deepseek/deepseek-v4-flash-0731"]
    client.close()


def test_the_served_name_is_still_kept_on_the_record():
    client = cassette_client("chat_short_model_name")
    seen = []
    client.on_usage(seen.append)
    client.chat.complete("deepseek/deepseek-v4-flash-0731", "1+1?")
    assert seen[0].requested_model == "deepseek/deepseek-v4-flash-0731"
    assert seen[0].model == "deepseek-v4-flash"
    client.close()


def test_routing_variants_of_one_model_share_a_group():
    """Пиннинг провайдера не должен размножать модель в отчёте о расходах."""
    client = cassette_client("chat_route_suffix")
    client.chat.complete("mistralai/mistral-nemo", "1+1?")
    client.chat.complete(route("mistralai/mistral-nemo", provider="deepinfra"), "1+1?")
    client.chat.complete(
        route("mistralai/mistral-nemo", provider="io-net", allow_fallbacks=False), "1+1?"
    )
    stats = client.usage.snapshot()
    assert list(stats.by_model) == ["mistralai/mistral-nemo"]
    assert stats.by_model["mistralai/mistral-nemo"].requests == 3
    client.close()


# --- каталог ---


def _catalog() -> list[Model]:
    body = json.loads(load("catalog_aliases")["response"]["body"])
    return [Model.model_validate(item) for item in body["data"]]


def test_an_alias_entry_looks_exactly_like_an_ordinary_one():
    """Ни alias_of, ни canonical_id: отличить можно только по префиксу."""
    catalog = {m.id: m for m in _catalog()}
    alias = catalog["~z-ai/glm-flash-latest"]
    target = catalog["z-ai/glm-5.3-flash"]
    assert alias.is_alias and not target.is_alias
    assert set(alias.model_dump()) == set(target.model_dump())
    assert alias.name == target.name
    assert alias.pricing.prompt == target.pricing.prompt


def test_the_latest_suffix_alone_does_not_make_an_alias():
    catalog = {m.id: m for m in _catalog()}
    assert not catalog["openai/gpt-chat-latest"].is_alias


def test_author_ignores_the_alias_marker():
    """Иначе developer="z-ai" теряет алиасы этого же разработчика."""
    catalog = {m.id: m for m in _catalog()}
    assert catalog["~z-ai/glm-flash-latest"].author == "z-ai"
    assert catalog["~z-ai/glm-flash-latest"].slug == "glm-flash-latest"


def test_search_by_developer_finds_both_the_alias_and_the_release():
    found = {m.id for m in filter_models(_catalog(), developer="z-ai")}
    assert found == {"~z-ai/glm-flash-latest", "z-ai/glm-5.3-flash"}


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("include", {"~z-ai/glm-flash-latest", "z-ai/glm-5.3-flash"}),
        ("exclude", {"z-ai/glm-5.3-flash"}),
        ("only", {"~z-ai/glm-flash-latest"}),
    ],
)
def test_alias_filter_modes(mode, expected):
    found = {m.id for m in filter_models(_catalog(), developer="z-ai", aliases=mode)}
    assert found == expected


def test_cheapest_leaves_aliases_out_by_default():
    """Алиас несёт цену цели, и при равенстве выбор стал бы делом сортировки."""
    client = cassette_client("catalog_aliases")
    picked = client.models.cheapest(capabilities=["text"])
    assert not picked.is_alias
    client.close()


def test_cheapest_can_be_asked_for_an_alias():
    client = cassette_client("catalog_aliases")
    picked = client.models.cheapest(capabilities=["text"], aliases="only")
    assert picked.is_alias
    client.close()


def test_resolve_returns_the_release_an_alias_points_at():
    client = cassette_client("catalog_aliases")
    assert client.models.resolve("~z-ai/glm-flash-latest").id == "z-ai/glm-5.3-flash"
    assert client.models.resolve("~anthropic/claude-opus-latest").id == "anthropic/claude-opus-5"
    client.close()


def test_resolve_ignores_a_routing_suffix():
    client = cassette_client("catalog_aliases")
    resolved = client.models.resolve("~z-ai/glm-flash-latest@provider=deepinfra")
    assert resolved.id == "z-ai/glm-5.3-flash"
    client.close()


def test_resolve_of_a_plain_id_is_the_model_itself():
    client = cassette_client("catalog_aliases")
    assert client.models.resolve("mistralai/mistral-nemo").id == "mistralai/mistral-nemo"
    client.close()


def test_resolve_of_an_unknown_id_raises():
    client = cassette_client("catalog_aliases")
    with pytest.raises(ModelNotFoundError, match="not found in catalog"):
        client.models.resolve("nobody/nothing")
    client.close()


def test_aliases_lists_only_alias_entries():
    client = cassette_client("catalog_aliases")
    found = client.models.aliases()
    assert found and all(m.is_alias for m in found)
    assert "openai/gpt-chat-latest" not in {m.id for m in found}
    client.close()


def test_endpoints_rejects_an_id_without_a_developer_part():
    """Раньше это был ValueError из распаковки — мимо границы ошибок SDK."""
    client = cassette_client("catalog_aliases")
    with pytest.raises(ModelNotFoundError, match="developer"):
        client.models.endpoints("just-a-name")
    client.close()


def test_alias_pricing_matches_its_target_in_the_recorded_catalog():
    """Если это перестанет быть правдой, выбор по цене врёт про алиасы."""
    catalog = {m.id: m for m in _catalog()}
    for alias_id, target_id in [
        ("~z-ai/glm-flash-latest", "z-ai/glm-5.3-flash"),
        ("~deepseek/deepseek-v4-flash-latest", "deepseek/deepseek-v4-flash-0731"),
        ("~anthropic/claude-opus-latest", "anthropic/claude-opus-5"),
    ]:
        alias, target = catalog[alias_id], catalog[target_id]
        assert alias.pricing.per_million("prompt") == target.pricing.per_million("prompt")
        assert isinstance(alias.pricing.per_million("prompt"), Decimal)


# --- эндпоинты, которые @-синтаксис не понимают ---


def test_embeddings_reads_a_routed_string_as_a_model_name():
    """Сервер не разбирает суффикс вне chat: вся строка уходит в имя модели."""
    cassette = load("embeddings_route_suffix_400")
    assert cassette["response"]["status"] == 400
    assert "@provider=deepinfra' not found" in cassette["response"]["body"]


def test_pinning_an_absent_provider_is_a_not_found():
    cassette = load("chat_route_unknown_provider_404")
    assert cassette["response"]["status"] == 404
    response = response_of(cassette)
    assert "provider preferences" in response.text


def test_the_sdk_raises_not_found_for_a_pinned_absent_provider():
    """Ради этого и пиннят: молчаливая подмена провайдера становится ошибкой."""
    client = cassette_client("chat_route_unknown_provider_404")
    with pytest.raises(NotFoundError) as caught:
        client.chat.complete(
            route("mistralai/mistral-nemo", provider="no-such-provider-xyz", allow_fallbacks=False),
            "1+1?",
        )
    assert caught.value.status_code == 404
    assert "provider preferences" in str(caught.value)
    client.close()
