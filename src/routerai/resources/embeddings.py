from __future__ import annotations

import base64
import binascii
import struct
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from .._extras import merge_extra
from .._options import RequestOptions
from ..errors import ResponseParsingError
from ..schemas import Usage

if TYPE_CHECKING:
    from typing_extensions import Unpack

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
        **opts: Unpack[RequestOptions],
    ) -> EmbeddingsResult:
        body: dict[str, Any] = {"model": model, "input": input}
        if dimensions is not None:
            body["dimensions"] = dimensions
        merge_extra(extra, reserved=("model", "input", "dimensions"))
        if extra:
            body.update(extra)
        response = self._http.post("embeddings", json=body, **opts)
        return EmbeddingsResult.from_response(
            response.json(), generation_id=response.generation_id, request_id=response.request_id
        )

    async def acreate(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> EmbeddingsResult:
        body: dict[str, Any] = {"model": model, "input": input}
        if dimensions is not None:
            body["dimensions"] = dimensions
        merge_extra(extra, reserved=("model", "input", "dimensions"))
        if extra:
            body.update(extra)
        response = await self._http.apost("embeddings", json=body, **opts)
        return EmbeddingsResult.from_response(
            response.json(), generation_id=response.generation_id, request_id=response.request_id
        )


class EmbeddingsResult:
    def __init__(
        self,
        embeddings: list[list[float]],
        usage: Usage | None,
        raw: dict[str, Any],
        generation_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.usage = usage
        self.raw = raw
        self.generation_id = generation_id
        self.request_id = request_id

    @classmethod
    def from_response(
        cls,
        payload: dict[str, Any],
        *,
        generation_id: str | None = None,
        request_id: str | None = None,
    ) -> EmbeddingsResult:
        data = payload.get("data") or []
        embeddings = [
            _decode_embedding(item.get("embedding")) for item in data if isinstance(item, dict)
        ]
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return cls(embeddings, usage, payload, generation_id, request_id)

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
        **opts: Unpack[RequestOptions],
    ) -> RerankResult:
        body: dict[str, Any] = {"model": model, "query": query, "documents": documents}
        if top_n is not None:
            body["top_n"] = top_n
        merge_extra(extra, reserved=("model", "query", "documents", "top_n"))
        if extra:
            body.update(extra)
        response = self._http.post("rerank", json=body, **opts)
        return RerankResult.from_response(
            response.json(), generation_id=response.generation_id, request_id=response.request_id
        )

    async def acreate(
        self,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> RerankResult:
        body: dict[str, Any] = {"model": model, "query": query, "documents": documents}
        if top_n is not None:
            body["top_n"] = top_n
        merge_extra(extra, reserved=("model", "query", "documents", "top_n"))
        if extra:
            body.update(extra)
        response = await self._http.apost("rerank", json=body, **opts)
        return RerankResult.from_response(
            response.json(), generation_id=response.generation_id, request_id=response.request_id
        )


class RerankResult:
    def __init__(
        self,
        results: list[dict[str, Any]],
        usage: Usage | None,
        raw: dict[str, Any],
        generation_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.results = results
        self.usage = usage
        self.raw = raw
        self.generation_id = generation_id
        self.request_id = request_id

    @classmethod
    def from_response(
        cls,
        payload: dict[str, Any],
        *,
        generation_id: str | None = None,
        request_id: str | None = None,
    ) -> RerankResult:
        results = payload.get("results") or payload.get("data") or []
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return cls(list(results), usage, payload, generation_id, request_id)

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


def _decode_embedding(value: Any) -> list[float]:
    """Read a vector in either wire format.

    With ``encoding_format="base64"`` — which openai-python asks for by
    default to save bandwidth — the server sends the vector as a base64
    string of float32 values. Wrapping that string in list() yields a list of
    single characters instead of numbers, silently and without an error.
    """
    if isinstance(value, str):
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ResponseParsingError("embedding is not valid base64", body=value) from exc
        if len(raw) % 4:
            raise ResponseParsingError(
                "base64 embedding is not a whole number of float32 values", body=value
            )
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if isinstance(value, list):
        return list(value)
    return []
