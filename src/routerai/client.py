from __future__ import annotations

import logging
import os
from typing import Any

from ._http import DEFAULT_BASE_URL, HTTPClient
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
        base_url: API base URL (defaults to https://routerai.ru/api/v1).
        timeout: request timeout in seconds.
        max_retries: retries on 429/5xx with exponential backoff.
        logger: a ``logging.Logger``, a child logger name, or None for the
            default ``routerai`` namespace logger.
        models_ttl: cache lifetime for the models catalog in seconds.

    Resource namespaces: ``chat``, ``models``, ``completions``, ``responses``,
    ``messages``, ``images``, ``videos``, ``audio``, ``embeddings``,
    ``rerank``, ``generation``, ``keys``, ``team``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        logger: logging.Logger | str | None = None,
        models_ttl: float = 600.0,
        http_client: Any = None,
    ) -> None:
        api_key = api_key or os.getenv(ENV_API_KEY)
        self._http = HTTPClient(
            api_key=api_key,
            base_url=base_url or os.getenv(ENV_BASE_URL, DEFAULT_BASE_URL),
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            logger=logger,
            http_client=http_client,
        )
        self.chat = Chat(self._http)
        self.models = Models(self._http, ttl=models_ttl)
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
