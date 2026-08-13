from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import RouterAI


_active_client: ContextVar[RouterAI | None] = ContextVar("routerai_active_client", default=None)


class Registry:
    """Named collection of :class:`RouterAI` clients (e.g. several API keys).

    The active client is tracked with a context variable, so switching it
    inside a ``with`` block is safe for threads and async tasks::

        registry = Registry(main=RouterAI(api_key=A), personal=RouterAI(api_key=B))
        registry["personal"].chat.complete(...)
        with registry.using("personal"):
            client = registry.current()  # -> personal
    """

    def __init__(self, **clients: RouterAI) -> None:
        self._clients: dict[str, RouterAI] = dict(clients)
        self._default = next(iter(self._clients), None)
        if self._default is not None:
            _active_client.set(self._clients[self._default])

    def __getitem__(self, name: str) -> RouterAI:
        return self._clients[name]

    def __contains__(self, name: str) -> bool:
        return name in self._clients

    def __iter__(self) -> Iterator[str]:
        return iter(self._clients)

    def add(self, name: str, client: RouterAI, *, make_default: bool = False) -> None:
        self._clients[name] = client
        if make_default or self._default is None:
            self._default = name
            _active_client.set(client)

    def remove(self, name: str) -> None:
        client = self._clients.pop(name)
        if _active_client.get() is client:
            fallback = next(iter(self._clients.values()), None)
            _active_client.set(fallback)

    def current(self) -> RouterAI | None:
        return _active_client.get()

    def get(self, name: str, default: RouterAI | None = None) -> RouterAI | None:
        return self._clients.get(name, default)

    @contextmanager
    def using(self, name: str) -> Iterator[RouterAI]:
        client = self._clients[name]
        token = _active_client.set(client)
        try:
            yield client
        finally:
            _active_client.reset(token)

    def as_mapping(self) -> Mapping[str, RouterAI]:
        return dict(self._clients)

    def close_all(self) -> None:
        for client in self._clients.values():
            client.close()

    async def aclose_all(self) -> None:
        for client in self._clients.values():
            await client.aclose()

    def __repr__(self) -> str:
        return f"Registry({', '.join(sorted(self._clients))})"


def active_client() -> RouterAI | None:
    """Return the client set by the last ``Registry.using`` context (if any)."""
    return _active_client.get()


def set_active_client(client: RouterAI | None) -> Any:
    token = _active_client.set(client)
    return token
