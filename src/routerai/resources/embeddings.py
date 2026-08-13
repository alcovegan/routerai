from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from .._extras import merge_extra
from ..schemas import Usage

if TYPE_CHECKING:
    from .._http import HTTPClient


class Embeddings:
    """Embeddings (``POST /api/v1/embeddings``)."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def create(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> EmbeddingsResult:
        body: dict[str, Any] = {"model": model, "input": input}
        if dimensions is not None:
            body["dimensions"] = dimensions
        merge_extra(extra, reserved=("model", "input", "dimensions"))
        if extra:
            body.update(extra)
        response = self._http.post("embeddings", json=body)
        return EmbeddingsResult.from_response(response.json())

    async def acreate(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> EmbeddingsResult:
        body: dict[str, Any] = {"model": model, "input": input}
        if dimensions is not None:
            body["dimensions"] = dimensions
        merge_extra(extra, reserved=("model", "input", "dimensions"))
        if extra:
            body.update(extra)
        response = await self._http.apost("embeddings", json=body)
        return EmbeddingsResult.from_response(response.json())


class EmbeddingsResult:
    def __init__(
        self, embeddings: list[list[float]], usage: Usage | None, raw: dict[str, Any]
    ) -> None:
        self.embeddings = embeddings
        self.usage = usage
        self.raw = raw

    @classmethod
    def from_response(cls, payload: dict[str, Any]) -> EmbeddingsResult:
        data = payload.get("data") or []
        embeddings = [list(item.get("embedding") or []) for item in data if isinstance(item, dict)]
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return cls(embeddings, usage, payload)

    @property
    def cost_rub(self) -> Decimal | None:
        return self.usage.cost_rub if self.usage else None


class Rerank:
    """Reranking (``POST /api/v1/rerank``)."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def create(
        self,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RerankResult:
        body: dict[str, Any] = {"model": model, "query": query, "documents": documents}
        if top_n is not None:
            body["top_n"] = top_n
        merge_extra(extra, reserved=("model", "query", "documents", "top_n"))
        if extra:
            body.update(extra)
        response = self._http.post("rerank", json=body)
        return RerankResult.from_response(response.json())

    async def acreate(
        self,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RerankResult:
        body: dict[str, Any] = {"model": model, "query": query, "documents": documents}
        if top_n is not None:
            body["top_n"] = top_n
        merge_extra(extra, reserved=("model", "query", "documents", "top_n"))
        if extra:
            body.update(extra)
        response = await self._http.apost("rerank", json=body)
        return RerankResult.from_response(response.json())


class RerankResult:
    def __init__(
        self, results: list[dict[str, Any]], usage: Usage | None, raw: dict[str, Any]
    ) -> None:
        self.results = results
        self.usage = usage
        self.raw = raw

    @classmethod
    def from_response(cls, payload: dict[str, Any]) -> RerankResult:
        results = payload.get("results") or payload.get("data") or []
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return cls(list(results), usage, payload)

    def top_documents(self, documents: list[str] | None = None) -> list[str]:
        ordered = sorted(
            self.results, key=lambda r: float(r.get("relevance_score", 0)), reverse=True
        )
        if documents is None:
            return [str(r.get("document") or r.get("text") or "") for r in ordered]
        return [documents[int(r["index"])] for r in ordered if isinstance(r.get("index"), int)]

    @property
    def cost_rub(self) -> Decimal | None:
        return self.usage.cost_rub if self.usage else None
