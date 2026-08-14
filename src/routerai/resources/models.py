from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterable
from decimal import Decimal
from typing import TYPE_CHECKING

from ..errors import ModelNotFoundError
from ..schemas import Capability, Model, ModelDetail

if TYPE_CHECKING:
    from .._http import HTTPClient

_DEFAULT_TTL = 600.0

ModelList = list[Model]


class Models:
    """Catalog of available models: listing, lookup, search, grouping.

    All listing methods hit ``GET /api/v1/models`` (no auth required) and
    cache the result for ``ttl`` seconds. Search is performed client-side
    because the upstream endpoint ignores query filters.

    The cache is shared between sync and async paths; async refreshes are
    single-flight (concurrent callers share one in-flight fetch).
    """

    def __init__(self, http: HTTPClient, *, ttl: float = _DEFAULT_TTL) -> None:
        self._http = http
        self._ttl = ttl
        self._cache: ModelList | None = None
        self._fetched_at: float | None = None
        self._lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None

    # --- cache ---

    def _is_fresh(self) -> bool:
        if self._cache is None or self._fetched_at is None:
            return False
        return time.monotonic() - self._fetched_at < self._ttl

    def _refresh_if_stale(self) -> None:
        if self._is_fresh():
            return
        with self._lock:
            if self._is_fresh():
                return
            self._fetch_sync()

    def _fetch_sync(self) -> None:
        response = self._http.get("models")
        data = response.json()["data"]
        self._cache = [Model.model_validate(item) for item in data]
        self._fetched_at = time.monotonic()

    def _ensure_async_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._async_lock is None or self._async_loop is not loop:
            self._async_lock = asyncio.Lock()
            self._async_loop = loop
        return self._async_lock

    async def _arefresh_if_stale(self) -> None:
        if self._is_fresh():
            return
        lock = self._ensure_async_lock()
        async with lock:
            if self._is_fresh():
                return
            response = await self._http.aget("models")
            data = response.json()["data"]
            self._cache = [Model.model_validate(item) for item in data]
            self._fetched_at = time.monotonic()

    def clear_cache(self) -> None:
        self._cache = None
        self._fetched_at = None

    # --- listing ---

    def all(self, *, force_refresh: bool = False) -> ModelList:
        """Return the full catalog of models."""
        if force_refresh:
            self.clear_cache()
        self._refresh_if_stale()
        assert self._cache is not None
        return list(self._cache)

    async def aall(self, *, force_refresh: bool = False) -> ModelList:
        if force_refresh:
            self.clear_cache()
        await self._arefresh_if_stale()
        assert self._cache is not None
        return list(self._cache)

    def get(self, model_id: str) -> Model:
        """Return a model by its exact id, e.g. ``"deepseek/deepseek-v4-pro"``."""
        for model in self.all():
            if model.id == model_id:
                return model
        raise ModelNotFoundError(f"model '{model_id}' not found in catalog")

    async def aget(self, model_id: str) -> Model:
        for model in await self.aall():
            if model.id == model_id:
                return model
        raise ModelNotFoundError(f"model '{model_id}' not found in catalog")

    # --- search ---

    def search(
        self,
        q: str | None = None,
        *,
        input_modalities: Iterable[str] | None = None,
        output_modalities: Iterable[str] | None = None,
        capabilities: Iterable[str | Capability] | None = None,
        developer: str | None = None,
        min_context: int | None = None,
        max_price_prompt: float | None = None,
        max_price_completion: float | None = None,
        reasoning: bool | None = None,
        tools: bool | None = None,
    ) -> ModelList:
        """Client-side search over the catalog.

        Args:
            q: case-insensitive substring matched against id, name, description.
            input_modalities/output_modalities: filter by architecture modalities.
            capabilities: models must have ALL listed capabilities.
            developer: match the model author (the part of id before "/").
            min_context: minimal context length in tokens.
            max_price_prompt/max_price_completion: max price in rubles per 1M tokens.
            reasoning/tools: shortcuts for capability filters.
        """
        results: list[Model] = []
        caps = _normalize_capabilities(capabilities)
        if reasoning is True:
            caps |= {Capability.REASONING}
        if tools is True:
            caps |= {Capability.TOOLS}

        for model in self.all():
            if q and not _matches_query(model, q.lower()):
                continue
            if developer and model.author.lower() != developer.lower():
                continue
            if input_modalities is not None:
                wanted = set(input_modalities)
                if not wanted.issubset(set(model.architecture.input_modalities)):
                    continue
            if output_modalities is not None:
                wanted = set(output_modalities)
                if not wanted.issubset(set(model.architecture.output_modalities)):
                    continue
            if caps and not caps.issubset(model.capabilities):
                continue
            if min_context and (model.context_length is None or model.context_length < min_context):
                continue
            if max_price_prompt is not None and not _within_price(
                model, "prompt", max_price_prompt
            ):
                continue
            if max_price_completion is not None and not _within_price(
                model, "completion", max_price_completion
            ):
                continue
            results.append(model)
        return results

    # --- capabilities ---

    def by_capability(self, capability: str | Capability) -> ModelList:
        cap = Capability(capability)
        return [m for m in self.all() if cap in m.capabilities]

    def grouped(self) -> dict[Capability, ModelList]:
        """Group all models by capability (a model may appear in several groups)."""
        grouped: dict[Capability, ModelList] = {cap: [] for cap in Capability}
        for model in self.all():
            for cap in model.capabilities:
                grouped[cap].append(model)
        return {cap: models for cap, models in grouped.items() if models}

    def text(self) -> ModelList:
        return self.by_capability(Capability.TEXT)

    def reasoning(self) -> ModelList:
        return self.by_capability(Capability.REASONING)

    def vision(self) -> ModelList:
        return self.by_capability(Capability.VISION)

    def image_generation(self) -> ModelList:
        return self.by_capability(Capability.IMAGE_GENERATION)

    def embeddings(self) -> ModelList:
        return self.by_capability(Capability.EMBEDDINGS)

    def rerank(self) -> ModelList:
        return self.by_capability(Capability.RERANK)

    def speech(self) -> ModelList:
        return self.by_capability(Capability.SPEECH)

    def audio_generation(self) -> ModelList:
        return self.by_capability(Capability.AUDIO_GENERATION)

    def transcription(self) -> ModelList:
        return self.by_capability(Capability.TRANSCRIPTION)

    # --- endpoints ---

    def endpoints(self, model_id: str) -> ModelDetail:
        """Provider endpoints for a model: slugs, prices, limits, status."""
        author, slug = model_id.split("/", 1)
        response = self._http.get(f"models/{author}/{slug}/endpoints")
        return ModelDetail.model_validate(response.json()["data"])

    async def aendpoints(self, model_id: str) -> ModelDetail:
        author, slug = model_id.split("/", 1)
        response = await self._http.aget(f"models/{author}/{slug}/endpoints")
        return ModelDetail.model_validate(response.json()["data"])

    # --- aliases (real methods so type checkers see them) ---

    def list(self, *, force_refresh: bool = False) -> ModelList:
        """Alias for :meth:`all`."""
        return self.all(force_refresh=force_refresh)

    async def alist(self, *, force_refresh: bool = False) -> ModelList:
        """Alias for :meth:`aall`."""
        return await self.aall(force_refresh=force_refresh)


def _within_price(model: Model, unit: str, limit: float) -> bool:
    """Whether the model fits a per-million price limit for ``unit``.

    A model billed in another unit (images, seconds, search units) reports zero
    for tokens. Treating that as "free" is how a price filter ends up returning
    the most expensive models in the catalog, so such models are excluded
    instead: the limit simply does not apply to them.
    """
    price = model.pricing.per_million(unit)
    if price is None:
        return False
    if price == 0 and model.pricing.priced_units() - {unit}:
        return False
    return price <= Decimal(str(limit))


def _normalize_capabilities(values: Iterable[str | Capability] | None) -> set[Capability]:
    if values is None:
        return set()
    return {Capability(v) for v in values}


def _matches_query(model: Model, q: str) -> bool:
    haystack = " ".join(
        part for part in (model.id, model.name or "", model.description or "") if part
    ).lower()
    return q in haystack
