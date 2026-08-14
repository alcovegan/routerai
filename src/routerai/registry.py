from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import RouterAI

# One module-level variable for every registry, keyed by registry id. Creating
# a ContextVar per instance is the pattern CPython's docs warn about: each one
# is kept alive by every context it was ever set in.
_ACTIVE: ContextVar[dict[int, str] | None] = ContextVar("routerai_active_clients", default=None)


def _active() -> dict[int, str]:
    return _ACTIVE.get() or {}


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
        self._lock = threading.Lock()
        self._default: str | None = next(iter(self._clients), None)

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
        """Register a client. The default is plain state, visible everywhere.

        It used to be written to a context variable as well, so making a client
        the default from inside a task changed ``default`` but not
        ``current()`` for the caller.
        """
        with self._lock:
            self._clients[name] = client
            if make_default or self._default is None:
                self._default = name

    def remove(self, name: str) -> None:
        with self._lock:
            self._clients.pop(name)
            if self._default == name:
                self._default = next(iter(self._clients), None)

    def current(self) -> RouterAI | None:
        """Client set by the last ``using`` context in this registry.

        The active value is stored as a name and resolved against the
        current set of clients, so removed clients never leak back from a
        stale context token.
        """
        name = _active().get(id(self))
        if name is not None:
            client = self._clients.get(name)
            if client is not None:
                return client
        if self._default is not None:
            return self._clients[self._default]
        return None

    def get(self, name: str, default: RouterAI | None = None) -> RouterAI | None:
        return self._clients.get(name, default)

    @contextmanager
    def using(self, name: str) -> Iterator[RouterAI]:
        client = self._clients[name]
        token = _ACTIVE.set({**_active(), id(self): name})
        try:
            yield client
        finally:
            _ACTIVE.reset(token)

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
