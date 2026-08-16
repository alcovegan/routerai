"""Синхронные и асинхронные близнецы должны иметь одинаковые сигнатуры.

Половина кодовой базы живёт парами вида ``complete``/``acomplete``. Ничто не
мешало им разъехаться — этот тест мешает.
"""

from __future__ import annotations

import inspect

import pytest

from routerai import RouterAI
from routerai.resources.chat import Chat
from routerai.resources.completions import Completions
from routerai.resources.embeddings import Embeddings, Rerank
from routerai.resources.images import Images
from routerai.resources.keys import Keys
from routerai.resources.models import Models
from routerai.resources.team import Team

RESOURCES = [Chat, Models, Images, Embeddings, Rerank, Completions, Keys, Team]

# Осознанные исключения: у этих пар разная семантика, а не разошедшиеся копии.
KNOWN_DIFFERENT = {
    ("Models", "all"),
    ("Models", "list"),
}

# Локальные операции без ввода-вывода: асинхронный близнец им не нужен.
SYNC_ONLY = {("Models", "clear_cache")}


def _pairs(cls: type) -> list[tuple[str, str]]:
    names = {
        name for name, value in vars(cls).items() if callable(value) and not name.startswith("_")
    }
    return [(f"a{name}", name) for name in sorted(names) if f"a{name}" in names]


@pytest.mark.parametrize("cls", RESOURCES, ids=lambda c: c.__name__)
def test_async_twin_has_the_same_signature(cls: type) -> None:
    for async_name, sync_name in _pairs(cls):
        if (cls.__name__, sync_name) in KNOWN_DIFFERENT:
            continue
        sync = inspect.signature(getattr(cls, sync_name))
        asynchronous = inspect.signature(getattr(cls, async_name))
        assert list(sync.parameters) == list(asynchronous.parameters), (
            f"{cls.__name__}.{sync_name} and .{async_name} take different parameters"
        )


def test_every_public_resource_method_has_an_async_twin() -> None:
    """A method without a twin is unusable from async code."""
    missing: list[str] = []
    for cls in RESOURCES:
        for name, value in vars(cls).items():
            if name.startswith("_") or not callable(value) or name.startswith("a"):
                continue
            if (cls.__name__, name) in SYNC_ONLY:
                continue
            if f"a{name}" not in vars(cls):
                missing.append(f"{cls.__name__}.{name}")
    assert missing == [], f"no async twin for: {missing}"


def test_client_closes_both_sides() -> None:
    client = RouterAI(api_key="sk-parity")
    assert hasattr(client, "close") and hasattr(client, "aclose")


def test_annotations_resolve_at_runtime() -> None:
    """get_type_hints must work: FastAPI, sphinx and pydantic resolve annotations.

    Importing Unpack only under TYPE_CHECKING made this raise NameError on
    Python 3.10, where Unpack is not yet in typing.
    """
    import typing

    for cls in RESOURCES:
        for name, value in vars(cls).items():
            if name.startswith("_") or not callable(value):
                continue
            typing.get_type_hints(value)
