from __future__ import annotations

import base64
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..schemas import Usage

if TYPE_CHECKING:
    from .._http import HTTPClient


class GeneratedImage:
    def __init__(
        self, data: bytes, *, b64: str | None = None, revised_prompt: str | None = None
    ) -> None:
        self.data = data
        self.b64 = b64
        self.revised_prompt = revised_prompt

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        return path


class ImageResult:
    def __init__(
        self,
        images: list[GeneratedImage],
        usage: Usage | None,
        raw: dict[str, Any],
        generation_id: str | None,
    ) -> None:
        self.images = images
        self.usage = usage
        self.raw = raw
        self.generation_id = generation_id

    @property
    def cost_rub(self) -> Decimal | None:
        return self.usage.cost_rub if self.usage else None


class Images:
    """Image generation (``POST /api/v1/images``)."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        input_references: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
        body = self._body(
            model,
            prompt,
            n=n,
            size=size,
            quality=quality,
            input_references=input_references,
            extra=extra,
        )
        response = self._http.post("images", json=body)
        return self._parse(response)

    async def agenerate(
        self,
        model: str,
        prompt: str,
        *,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        input_references: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
        body = self._body(
            model,
            prompt,
            n=n,
            size=size,
            quality=quality,
            input_references=input_references,
            extra=extra,
        )
        response = await self._http.apost("images", json=body)
        return await self._aparse(response)

    def _body(
        self,
        model: str,
        prompt: str,
        *,
        n: int,
        size: str | None,
        quality: str | None,
        input_references: list[Any] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "prompt": prompt, "n": n}
        if size:
            body["size"] = size
        if quality:
            body["quality"] = quality
        if input_references:
            body["input_references"] = input_references
        if extra:
            body.update(extra)
        return body

    def _parse(self, response: Any) -> ImageResult:
        payload = self._http._json(response)
        images = []
        for item in payload.get("data") or []:
            b64 = item.get("b64_json")
            if b64:
                images.append(GeneratedImage(base64.b64decode(b64), b64=b64))
                continue
            url = item.get("url")
            if url:
                import httpx

                fetched = httpx.get(url, timeout=60.0)
                fetched.raise_for_status()
                images.append(GeneratedImage(fetched.content))
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return ImageResult(images, usage, payload, response.headers.get("X-Generation-Id"))

    async def _aparse(self, response: Any) -> ImageResult:
        payload = self._http._json(response)
        images = []
        for item in payload.get("data") or []:
            b64 = item.get("b64_json")
            if b64:
                images.append(GeneratedImage(base64.b64decode(b64), b64=b64))
                continue
            url = item.get("url")
            if url:
                import httpx

                async with httpx.AsyncClient(timeout=60.0) as client:
                    fetched = await client.get(url)
                fetched.raise_for_status()
                images.append(GeneratedImage(fetched.content))
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return ImageResult(images, usage, payload, response.headers.get("X-Generation-Id"))
