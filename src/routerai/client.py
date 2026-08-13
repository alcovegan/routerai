from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ._http import DEFAULT_BASE_URL, HTTPClient
from .resources.account import Account
from .resources.audio import Audio
from .resources.chat import Chat
from .resources.completions import Completions, Messages, Responses
from .resources.embeddings import Embeddings, Rerank
from .resources.generation import Generation
from .resources.images import Images
from .resources.keys import Keys
from .resources.models import Models
from .resources.team import Team
from .resources.videos import Videos

ENV_API_KEY = "ROUTERAI_API_KEY"
ENV_BASE_URL = "ROUTERAI_BASE_URL"


class RouterAI:
    """Synchronous and asynchronous client for the RouterAI API.

    Args:
        api_key: API key (defaults to the ``ROUTERAI_API_KEY`` env var).
        base_url: API base URL. Precedence: explicit argument, then
            ``ROUTERAI_BASE_URL`` env var, then the production default.
        timeout: request timeout in seconds.
        max_retries: retry attempts with exponential backoff and jitter.
            Only safe methods (GET/HEAD) are retried on 5xx by default; use
            ``retry_unsafe_methods=True`` to extend retries to POST/PATCH/DELETE.
        retry_backoff: base delay between retries in seconds.
        retry_unsafe_methods: retry non-idempotent methods on 5xx too.
            RouterAI performs its own provider fallback, so a client-side
            retry of POST may start a new (billed) generation.
        logger: a ``logging.Logger``, a child logger name, or None for the
            default ``routerai`` namespace logger.
        models_ttl: cache lifetime for the models catalog in seconds.
        http_client: optional sync httpx transport (not closed by ``close()``
            unless owned). The same instance may be used for sync and async
            calls; transports are kept in separate slots.
        async_http_client: optional async httpx transport.

    Resource namespaces: ``chat``, ``models``, ``completions``, ``responses``,
    ``messages``, ``images``, ``videos``, ``audio``, ``embeddings``,
    ``rerank``, ``generation``, ``keys``, ``team``, ``account``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        retry_unsafe_methods: bool = False,
        logger: logging.Logger | str | None = None,
        models_ttl: float = 600.0,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv(ENV_API_KEY)
        self._http = HTTPClient(
            api_key=api_key,
            base_url=base_url or os.getenv(ENV_BASE_URL) or DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_unsafe_methods=retry_unsafe_methods,
            logger=logger,
            http_client=http_client,
            async_http_client=async_http_client,
        )
        self.chat = Chat(self._http)
        self.models = Models(self._http, ttl=models_ttl)
        self.account = Account(self._http)
        self.completions = Completions(self._http)
        self.responses = Responses(self._http)
        self.messages = Messages(self._http)
        self.images = Images(self._http)
        self.videos = Videos(self._http)
        self.audio = Audio(self._http)
        self.embeddings = Embeddings(self._http)
        self.rerank = Rerank(self._http)
        self.generation = Generation(self._http)
        self.keys = Keys(self._http)
        self.team = Team(self._http)

    @property
    def logger(self) -> logging.Logger:
        return self._http.logger

    def close(self) -> None:
        self._http.close()

    async def aclose(self) -> None:
        await self._http.aclose()

    def __enter__(self) -> RouterAI:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def __aenter__(self) -> RouterAI:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"RouterAI(base_url={self._http._base_url!r})"
