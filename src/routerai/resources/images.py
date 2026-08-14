from __future__ import annotations

import base64
import binascii
from collections.abc import AsyncIterator, Iterator
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .._errors import DONE_MARKER, parse_stream_event
from .._extras import merge_extra
from .._options import RequestOptions
from ..errors import RequestError, RouterAIError, StreamInterruptedError
from ..schemas import Usage

if TYPE_CHECKING:
    from typing_extensions import Unpack

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
        self,
        path: str | Path,
        *,
        timeout: float = 60.0,
        max_bytes: int = 50 * 1024 * 1024,
        max_redirects: int = 5,
    ) -> Path:
        """Download a url-based image explicitly (HTTPS only, size-limited).

        Redirects are followed manually and every hop must stay on HTTPS;
        each hop is streamed (no full-body buffering), the byte limit is
        enforced before each write, and data lands in a unique temp file
        that is atomically renamed only on success.
        """
        import urllib.parse

        import httpx

        from .._files import AtomicFileWriter

        if not self.url:
            raise RouterAIError("image has no url; use save() for inline data")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError(f"max_bytes must be a positive integer, got {max_bytes!r}")

        if (
            not isinstance(max_redirects, int)
            or isinstance(max_redirects, bool)
            or max_redirects < 0
        ):
            raise ValueError(f"max_redirects must be a non-negative integer, got {max_redirects!r}")

        url = self.url
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                for _ in range(max_redirects + 1):
                    _validate_https_url(url)
                    with client.stream("GET", url) as response:
                        if response.status_code in (301, 302, 303, 307, 308):
                            location = response.headers.get("Location")
                            if not location:
                                raise RouterAIError(
                                    f"redirect from {url!r} without a Location header"
                                )
                            url = urllib.parse.urljoin(url, location)
                            continue
                        response.raise_for_status()
                        content_length = response.headers.get("content-length", "")
                        if content_length.isdigit() and int(content_length) > max_bytes:
                            raise RouterAIError(
                                f"image exceeds the {max_bytes} byte download limit "
                                f"(content-length={content_length})"
                            )
                        with AtomicFileWriter(path, max_bytes=max_bytes) as writer:
                            for chunk in response.iter_bytes():
                                writer.write(chunk)
                            return writer.commit()
        except httpx.HTTPStatusError as exc:
            raise RouterAIError(f"image download returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RequestError("image download failed") from exc
        raise RouterAIError(f"too many redirects while downloading {self.url!r}")


def _validate_https_url(url: str) -> None:
    from .._urls import validate_public_https_url

    try:
        validate_public_https_url(url, field="image download url")
    except ValueError as exc:
        raise RouterAIError(str(exc)) from exc


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
        prompt: str | None = None,
        *,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
        seed: int | None = None,
        input_references: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> ImageResult:
        body = self._body(
            model,
            prompt,
            n=n,
            size=size,
            quality=quality,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            background=background,
            output_format=output_format,
            output_compression=output_compression,
            seed=seed,
            input_references=input_references,
            extra=extra,
        )
        response = self._http.post("images", json=body, **opts)
        return self._parse(response)

    async def agenerate(
        self,
        model: str,
        prompt: str | None = None,
        *,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
        seed: int | None = None,
        input_references: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> ImageResult:
        body = self._body(
            model,
            prompt,
            n=n,
            size=size,
            quality=quality,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            background=background,
            output_format=output_format,
            output_compression=output_compression,
            seed=seed,
            input_references=input_references,
            extra=extra,
        )
        response = await self._http.apost("images", json=body, **opts)
        return self._parse(response)

    def stream(
        self,
        model: str,
        prompt: str | None = None,
        *,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
        seed: int | None = None,
        input_references: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> Iterator[ImageStreamChunk]:
        """Streaming generation: partial previews then a completed event."""
        body = self._body(
            model,
            prompt,
            n=n,
            size=size,
            quality=quality,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            background=background,
            output_format=output_format,
            output_compression=output_compression,
            seed=seed,
            input_references=input_references,
            extra=extra,
        )
        body["stream"] = True
        with self._http.stream_request("POST", "images", json=body, **opts) as response:
            yield from _iter_image_sse(
                response, http=self._http, generation_id=response.headers.get("X-Generation-Id")
            )

    async def astream(
        self,
        model: str,
        prompt: str | None = None,
        *,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
        seed: int | None = None,
        input_references: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> AsyncIterator[ImageStreamChunk]:
        body = self._body(
            model,
            prompt,
            n=n,
            size=size,
            quality=quality,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            background=background,
            output_format=output_format,
            output_compression=output_compression,
            seed=seed,
            input_references=input_references,
            extra=extra,
        )
        body["stream"] = True
        async with self._http.astream_request("POST", "images", json=body, **opts) as response:
            async for chunk in _aiter_image_sse(
                response, http=self._http, generation_id=response.headers.get("X-Generation-Id")
            ):
                yield chunk

    def _body(
        self,
        model: str,
        prompt: str | None,
        *,
        n: int,
        size: str | None,
        quality: str | None,
        aspect_ratio: str | None,
        resolution: str | None,
        background: str | None,
        output_format: str | None,
        output_compression: int | None,
        seed: int | None,
        input_references: list[Any] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not prompt and not input_references:
            raise ValueError("prompt is required unless input_references are provided")
        body: dict[str, Any] = {"model": model, "n": n}
        if prompt:
            body["prompt"] = prompt
        if size:
            body["size"] = size
        if quality:
            body["quality"] = quality
        if aspect_ratio:
            body["aspect_ratio"] = aspect_ratio
        if resolution:
            body["resolution"] = resolution
        if background:
            body["background"] = background
        if output_format:
            body["output_format"] = output_format
        if output_compression is not None:
            body["output_compression"] = output_compression
        if seed is not None:
            body["seed"] = seed
        if input_references:
            body["input_references"] = input_references
        merge_extra(
            extra,
            reserved=(
                "model",
                "prompt",
                "n",
                "size",
                "quality",
                "aspect_ratio",
                "resolution",
                "background",
                "output_format",
                "output_compression",
                "seed",
                "input_references",
                "stream",
            ),
        )
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
        return ImageResult(images, usage, payload, response.generation_id)


class ImageStreamChunk:
    """A single SSE event of a streaming image generation."""

    def __init__(self, raw: dict[str, Any], generation_id: str | None = None) -> None:
        self.raw = raw
        self.generation_id = generation_id

    @property
    def type(self) -> str | None:
        return self.raw.get("type") if isinstance(self.raw, dict) else None

    @property
    def images(self) -> list[GeneratedImage]:
        images = []
        for item in self.raw.get("data") or []:
            if not isinstance(item, dict):
                continue
            b64 = item.get("b64_json")
            if b64:
                images.append(GeneratedImage(base64.b64decode(b64), b64=b64))
        return images

    @property
    def usage(self) -> Usage | None:
        usage = self.raw.get("usage")
        return Usage.model_validate(usage) if isinstance(usage, dict) else None

    @property
    def cost_rub(self) -> Decimal | None:
        return self.usage.cost_rub if self.usage else None

    @property
    def is_completed(self) -> bool:
        return self.type == "image_generation.completed"


def _iter_image_sse(
    response: Any, *, http: HTTPClient, generation_id: str | None = None
) -> Iterator[ImageStreamChunk]:
    chunks_received = 0
    try:
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == DONE_MARKER:
                break
            payload = parse_stream_event(data)
            if payload is None:
                continue
            chunks_received += 1
            yield ImageStreamChunk(payload, generation_id=generation_id)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise StreamInterruptedError(
            f"image stream interrupted after {chunks_received} chunks: {exc}",
            chunks_received=chunks_received,
        ) from exc


async def _aiter_image_sse(
    response: Any, *, http: HTTPClient, generation_id: str | None = None
) -> AsyncIterator[ImageStreamChunk]:
    chunks_received = 0
    try:
        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == DONE_MARKER:
                break
            payload = parse_stream_event(data)
            if payload is None:
                continue
            chunks_received += 1
            yield ImageStreamChunk(payload, generation_id=generation_id)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise StreamInterruptedError(
            f"image stream interrupted after {chunks_received} chunks: {exc}",
            chunks_received=chunks_received,
        ) from exc
