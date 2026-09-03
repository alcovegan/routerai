from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterable
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from .._routing import is_alias, split_model
from ..errors import ModelNotFoundError
from ..schemas import Capability, Model, ModelDetail

if TYPE_CHECKING:
    from .._http import HTTPClient

_DEFAULT_TTL = 600.0

ModelList = list[Model]

AliasFilter = Literal["include", "exclude", "only"]
"""What to do with alias entries: keep them, drop them, or return only them."""


def _split_id(model_id: str) -> tuple[str, str]:
    """Split a catalog id into author and slug, alias marker kept.

    The marker stays because RouterAI serves alias paths as-is:
    ``GET /models/~deepseek/deepseek-v4-flash-latest/endpoints`` answers 200.
    """
    base = split_model(model_id).model
    author, separator, slug = base.partition("/")
    if not separator or not slug:
        raise ModelNotFoundError(f"model id {model_id!r} is not in '<developer>/<model>' form")
    return author, slug


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
        return list(self._cache or [])

    async def aall(self, *, force_refresh: bool = False) -> ModelList:
        if force_refresh:
            self.clear_cache()
        await self._arefresh_if_stale()
        return list(self._cache or [])

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
        aliases: AliasFilter = "include",
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
            aliases: "include" (default) keeps alias entries like
                ``~anthropic/claude-opus-latest`` alongside the concrete
                releases they point at, "exclude" drops them, "only" returns
                just the aliases.
        """
        return filter_models(
            self.all(),
            q,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            capabilities=capabilities,
            developer=developer,
            min_context=min_context,
            max_price_prompt=max_price_prompt,
            max_price_completion=max_price_completion,
            reasoning=reasoning,
            tools=tools,
            aliases=aliases,
        )

    async def asearch(
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
        aliases: AliasFilter = "include",
    ) -> ModelList:
        """Async :meth:`search` — the sync one blocks the event loop on refresh."""
        return filter_models(
            await self.aall(),
            q,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            capabilities=capabilities,
            developer=developer,
            min_context=min_context,
            max_price_prompt=max_price_prompt,
            max_price_completion=max_price_completion,
            reasoning=reasoning,
            tools=tools,
            aliases=aliases,
        )

    # --- capabilities ---

    def by_capability(self, capability: str | Capability) -> ModelList:
        cap = Capability(capability)
        return [m for m in self.all() if cap in m.capabilities]

    def grouped(self) -> dict[Capability, ModelList]:
        """Group all models by capability (a model may appear in several groups)."""
        return group_by_capability(self.all())

    def text(self) -> ModelList:
        return self.by_capability(Capability.TEXT)

    async def atext(self) -> ModelList:
        return await self.aby_capability(Capability.TEXT)

    def reasoning(self) -> ModelList:
        return self.by_capability(Capability.REASONING)

    async def areasoning(self) -> ModelList:
        return await self.aby_capability(Capability.REASONING)

    def vision(self) -> ModelList:
        return self.by_capability(Capability.VISION)

    async def avision(self) -> ModelList:
        return await self.aby_capability(Capability.VISION)

    def image_generation(self) -> ModelList:
        return self.by_capability(Capability.IMAGE_GENERATION)

    async def aimage_generation(self) -> ModelList:
        return await self.aby_capability(Capability.IMAGE_GENERATION)

    def embeddings(self) -> ModelList:
        return self.by_capability(Capability.EMBEDDINGS)

    async def aembeddings(self) -> ModelList:
        return await self.aby_capability(Capability.EMBEDDINGS)

    def rerank(self) -> ModelList:
        return self.by_capability(Capability.RERANK)

    async def arerank(self) -> ModelList:
        return await self.aby_capability(Capability.RERANK)

    def speech(self) -> ModelList:
        return self.by_capability(Capability.SPEECH)

    async def aspeech(self) -> ModelList:
        return await self.aby_capability(Capability.SPEECH)

    def audio_generation(self) -> ModelList:
        return self.by_capability(Capability.AUDIO_GENERATION)

    async def aaudio_generation(self) -> ModelList:
        return await self.aby_capability(Capability.AUDIO_GENERATION)

    def transcription(self) -> ModelList:
        return self.by_capability(Capability.TRANSCRIPTION)

    async def atranscription(self) -> ModelList:
        return await self.aby_capability(Capability.TRANSCRIPTION)

    async def aby_capability(self, capability: str | Capability) -> ModelList:
        cap = Capability(capability)
        return [m for m in await self.aall() if cap in m.capabilities]

    async def agrouped(self) -> dict[Capability, ModelList]:
        return group_by_capability(await self.aall())

    # --- picking a model ---

    def cheapest(
        self,
        *,
        capabilities: Iterable[str | Capability] | None = None,
        min_context: int | None = None,
        developer: str | None = None,
        unit: str = "blended",
        exclude: Iterable[str] = (),
        aliases: AliasFilter = "exclude",
    ) -> Model:
        """The cheapest model matching the filters.

        ``unit`` is "prompt", "completion", "blended" (three parts prompt to one
        part completion, the usual shape of a chat workload) or any other
        priced unit, e.g. "image_output".
        """
        return pick_cheapest(
            self.all(),
            capabilities=capabilities,
            min_context=min_context,
            developer=developer,
            unit=unit,
            exclude=exclude,
            aliases=aliases,
        )

    async def acheapest(
        self,
        *,
        capabilities: Iterable[str | Capability] | None = None,
        min_context: int | None = None,
        developer: str | None = None,
        unit: str = "blended",
        exclude: Iterable[str] = (),
        aliases: AliasFilter = "exclude",
    ) -> Model:
        return pick_cheapest(
            await self.aall(),
            capabilities=capabilities,
            min_context=min_context,
            developer=developer,
            unit=unit,
            exclude=exclude,
            aliases=aliases,
        )

    # --- aliases ---

    def aliases(self) -> ModelList:
        """Alias entries in the catalog: ids that follow the newest release.

        An alias looks like ``~anthropic/claude-opus-latest``; its ``name``
        describes whatever it currently points at.
        """
        return filter_models(self.all(), aliases="only")

    async def aaliases(self) -> ModelList:
        """Async :meth:`aliases`."""
        return filter_models(await self.aall(), aliases="only")

    def resolve(self, model_id: str) -> Model:
        """The concrete model an alias currently points at.

        A non-alias id resolves to itself, so this is safe to call on anything
        a caller might pass. Any ``@`` routing suffix is ignored.

        The catalog gives an alias no pointer back to its target — the entry is
        shaped exactly like an ordinary one — so the match is made on ``name``,
        which an alias inherits from the release it stands for.
        """
        return _resolve_alias(self.all(), model_id)

    async def aresolve(self, model_id: str) -> Model:
        """Async :meth:`resolve`."""
        return _resolve_alias(await self.aall(), model_id)

    # --- endpoints ---

    def endpoints(self, model_id: str) -> ModelDetail:
        """Provider endpoints for a model: slugs, prices, limits, status.

        Accepts an alias too; the server answers for the release it points at
        while echoing the alias id back.
        """
        author, slug = _split_id(model_id)
        response = self._http.get(f"models/{author}/{slug}/endpoints")
        return ModelDetail.model_validate(response.json()["data"])

    async def aendpoints(self, model_id: str) -> ModelDetail:
        author, slug = _split_id(model_id)
        response = await self._http.aget(f"models/{author}/{slug}/endpoints")
        return ModelDetail.model_validate(response.json()["data"])

    # --- method aliases (real methods so type checkers see them) ---

    def list(self, *, force_refresh: bool = False) -> ModelList:
        """Alias for :meth:`all`."""
        return self.all(force_refresh=force_refresh)

    async def alist(self, *, force_refresh: bool = False) -> ModelList:
        """Alias for :meth:`aall`."""
        return await self.aall(force_refresh=force_refresh)


def group_by_capability(models: ModelList) -> dict[Capability, ModelList]:
    grouped: dict[Capability, ModelList] = {cap: [] for cap in Capability}
    for model in models:
        for cap in model.capabilities:
            grouped[cap].append(model)
    return {cap: found for cap, found in grouped.items() if found}


def _resolve_alias(models: ModelList, model_id: str) -> Model:
    """Resolve an alias id against a fetched catalog. Pure function: no I/O."""
    wanted = split_model(model_id).model
    entry = next((m for m in models if m.id == wanted), None)
    if entry is None:
        raise ModelNotFoundError(f"model {wanted!r} not found in catalog")
    if not is_alias(wanted):
        return entry

    targets = [m for m in models if not m.is_alias and m.name == entry.name]
    if len(targets) == 1:
        return targets[0]
    if not targets:
        raise ModelNotFoundError(
            f"alias {wanted!r} points at {entry.name!r}, which is not in the catalog"
        )
    raise ModelNotFoundError(
        f"alias {wanted!r} points at {entry.name!r}, which matches several "
        f"catalog entries: {sorted(m.id for m in targets)}"
    )


def pick_cheapest(
    models: ModelList,
    *,
    capabilities: Iterable[str | Capability] | None = None,
    min_context: int | None = None,
    developer: str | None = None,
    unit: str = "blended",
    exclude: Iterable[str] = (),
    aliases: AliasFilter = "exclude",
) -> Model:
    """Cheapest model matching the filters, with deterministic tie-breaking.

    Aliases are left out by default. An alias carries the same price as the
    release it points at, so including them adds a duplicate of some model at
    exactly the tying price — and which of the two wins would then depend on
    id ordering rather than on anything the caller asked for. Pass
    ``aliases="include"`` to accept an id that follows the newest release.
    """
    skip = set(exclude)
    candidates = filter_models(
        models,
        capabilities=capabilities,
        min_context=min_context,
        developer=developer,
        aliases=aliases,
    )
    scored: list[tuple[Decimal, int, str, Model]] = []
    for model in candidates:
        if model.id in skip:
            continue
        price = _score(model, unit)
        if price is None:
            continue
        # ties resolve by longer context, then by id, so the answer is stable
        scored.append((price, -(model.context_length or 0), model.id, model))
    if not scored:
        raise ModelNotFoundError(
            f"no model matches: capabilities={list(capabilities or [])}, "
            f"min_context={min_context}, developer={developer}, unit={unit!r}"
        )
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][3]


def _score(model: Model, unit: str) -> Decimal | None:
    """Price used for comparison, or None when the model is not priced in it."""
    if unit != "blended":
        return model.pricing.per_million(unit)
    prompt = model.pricing.per_million("prompt")
    completion = model.pricing.per_million("completion")
    if prompt is None and completion is None:
        return None
    if model.pricing.priced_units() - {"prompt", "completion"} and not (prompt or completion):
        # billed per image or per second: not comparable with token prices
        return None
    return ((prompt or Decimal(0)) * 3 + (completion or Decimal(0))) / 4


def filter_models(
    models: ModelList,
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
    aliases: AliasFilter = "include",
) -> ModelList:
    """Filter a catalog. Pure function: no I/O, usable on an already fetched list."""
    caps = _normalize_capabilities(capabilities)
    if reasoning is True:
        caps |= {Capability.REASONING}
    if tools is True:
        caps |= {Capability.TOOLS}

    results: list[Model] = []
    for model in models:
        if aliases == "exclude" and model.is_alias:
            continue
        if aliases == "only" and not model.is_alias:
            continue
        if q and not _matches_query(model, q.lower()):
            continue
        if developer and model.author.lower() != developer.lower():
            continue
        if input_modalities is not None and not set(input_modalities).issubset(
            set(model.architecture.input_modalities)
        ):
            continue
        if output_modalities is not None and not set(output_modalities).issubset(
            set(model.architecture.output_modalities)
        ):
            continue
        if caps and not caps.issubset(model.capabilities):
            continue
        if min_context and (model.context_length is None or model.context_length < min_context):
            continue
        if max_price_prompt is not None and not _within_price(model, "prompt", max_price_prompt):
            continue
        if max_price_completion is not None and not _within_price(
            model, "completion", max_price_completion
        ):
            continue
        results.append(model)
    return results


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
