from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import RouterAI


class Registry:
    """Named collection of :class:`RouterAI` clients (e.g. several API keys).

    The active client is tracked per-registry with a context variable, so
    multiple registries can coexist in one process without leaking clients
    into each other, and switching inside a ``with`` block is safe for
    threads and async tasks::

        registry = Registry(main=RouterAI(api_key=A), personal=RouterAI(api_key=B))
        registry["personal"].chat.complete(...)
        with registry.using("personal"):
            client = registry.current()  # -> personal
    """

    def __init__(self, **clients: RouterAI) -> None:
        self._clients: dict[str, RouterAI] = dict(clients)
        self._default: str | None = next(iter(self._clients), None)
        self._active: ContextVar[RouterAI | None] = ContextVar(
            f"routerai_registry_{id(self)}", default=None
        )
        if self._default is not None:
            self._active.set(self._clients[self._default])

    def __getitem__(self, name: str) -> RouterAI:
        return self._clients[name]

    def __contains__(self, name: str) -> bool:
        return name in self._clients

    def __iter__(self) -> Iterator[str]:
        return iter(self._clients)

    @property
    def default(self) -> str | None:
        return self._default

    def add(self, name: str, client: RouterAI, *, make_default: bool = False) -> None:
        self._clients[name] = client
        if make_default or self._default is None:
            self._default = name
            self._active.set(client)

    def remove(self, name: str) -> None:
        client = self._clients.pop(name)
        if self._active.get() is client:
            fallback = next(iter(self._clients.values()), None)
            self._active.set(fallback)
        if self._default == name:
            self._default = next(iter(self._clients), None)

    def current(self) -> RouterAI | None:
        """Client set by the last ``using`` context in this registry."""
        active = self._active.get()
        if active is not None:
            return active
        if self._default is not None:
            return self._clients[self._default]
        return None

    def get(self, name: str, default: RouterAI | None = None) -> RouterAI | None:
        return self._clients.get(name, default)

    @contextmanager
    def using(self, name: str) -> Iterator[RouterAI]:
        client = self._clients[name]
        token = self._active.set(client)
        try:
            yield client
        finally:
            self._active.reset(token)

    def as_mapping(self) -> Mapping[str, RouterAI]:
        return dict(self._clients)

    def close_all(self) -> None:
        self._active.set(None)
        for client in self._clients.values():
            client.close()

    async def aclose_all(self) -> None:
        self._active.set(None)
        for client in self._clients.values():
            await client.aclose()

    def __repr__(self) -> str:
        return f"Registry({', '.join(sorted(self._clients))})"
