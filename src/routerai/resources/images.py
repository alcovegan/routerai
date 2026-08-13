from __future__ import annotations

import base64
import binascii
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._extras import merge_extra
from ..errors import RouterAIError
from ..schemas import Usage

if TYPE_CHECKING:
    from .._http import HTTPClient


class GeneratedImage:
    def __init__(
        self,
        data: bytes | None = None,
        *,
        b64: str | None = None,
        url: str | None = None,
        revised_prompt: str | None = None,
    ) -> None:
        self.data = data
        self.b64 = b64
        self.url = url
        self.revised_prompt = revised_prompt

    def save(self, path: str | Path) -> Path:
        if self.data is None:
            raise RouterAIError("image has no inline data; use download() for url images")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        return path

    def download(
        self, path: str | Path, *, timeout: float = 60.0, max_bytes: int = 50 * 1024 * 1024
    ) -> Path:
        """Download a url-based image explicitly (HTTPS only, size-limited)."""
        if not self.url:
            raise RouterAIError("image has no url; use save() for inline data")
        if not self.url.startswith("https://"):
            raise RouterAIError(f"refusing to download non-https url: {self.url!r}")
        import httpx

        with httpx.stream("GET", self.url, timeout=timeout, follow_redirects=True) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise RouterAIError(f"image exceeds the {max_bytes} byte download limit")
                chunks.append(chunk)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(chunks))
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
        return self._parse(response)

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
        merge_extra(extra, reserved=("model", "prompt", "n"))
        if extra:
            body.update(extra)
        return body

    @staticmethod
    def _parse_item(item: dict[str, Any]) -> GeneratedImage:
        revised_prompt = item.get("revised_prompt")
        b64 = item.get("b64_json")
        if b64 is not None:
            if not b64:
                raise RouterAIError("provider returned an empty b64_json image payload")
            try:
                data = base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise RouterAIError("provider returned a corrupted b64_json image payload") from exc
            if not data:
                raise RouterAIError("provider returned an empty b64_json image payload")
            return GeneratedImage(data, b64=b64, revised_prompt=revised_prompt)
        url = item.get("url")
        if url:
            return GeneratedImage(url=str(url), revised_prompt=revised_prompt)
        raise RouterAIError(f"image entry has neither b64_json nor url: {item!r}")

    def _parse(self, response: Any) -> ImageResult:
        payload = self._http._json(response)
        images = [self._parse_item(item) for item in payload.get("data") or []]
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return ImageResult(images, usage, payload, response.headers.get("X-Generation-Id"))
