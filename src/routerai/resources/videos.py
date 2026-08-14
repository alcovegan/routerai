from __future__ import annotations

import math
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from .._extras import merge_extra
from .._urls import validate_image_source, validate_public_https_url
from ..errors import DeadlineExceededError, RequestError, VideoGenerationError
from ..schemas import Usage

if TYPE_CHECKING:
    from .._http import HTTPClient

_POLL_STATUSES = {"pending", "processing", "running", "in_progress", "queued"}
_TERMINAL_FAILURES = {"failed", "cancelled", "expired"}


def _validate_public_url(value: str) -> str:
    return validate_image_source(value)


def _validate_https_callback(value: str) -> None:
    validate_public_https_url(value, field="callback_url")


class FrameImage(BaseModel):
    """Reference frame for image-to-video (``frame_images``).

    ``frame_type`` is ``first_frame`` (the video starts from this frame) or
    ``last_frame`` (the video ends on this frame).
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    frame_type: Literal["first_frame", "last_frame"] = "first_frame"

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        return _validate_public_url(value)

    def model_dump_wire(self) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self.url},
            "frame_type": self.frame_type,
        }


class ImageReference(BaseModel):
    """Style/character reference for reference-to-video (``input_references``)."""

    model_config = ConfigDict(extra="forbid")

    url: str

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        return _validate_public_url(value)

    def model_dump_wire(self) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": self.url}}


class VideoTask:
    def __init__(self, http: HTTPClient, payload: dict[str, Any]) -> None:
        self._http = http
        self.id: str = payload.get("id", "")
        self.status: str = payload.get("status", "pending")
        self.polling_url: str | None = payload.get("polling_url")
        self.raw = payload
        self._apply(payload)

    def _apply(self, payload: dict[str, Any]) -> None:
        self.status = payload.get("status", self.status)
        self.raw = payload
        usage_payload = payload.get("usage")
        if usage_payload is None and isinstance(payload.get("data"), dict):
            usage_payload = payload["data"].get("usage")
        self._usage = (
            Usage.model_validate(usage_payload) if isinstance(usage_payload, dict) else None
        )

    @property
    def urls(self) -> list[str]:
        urls = self.raw.get("unsigned_urls") or self.raw.get("urls") or []
        return [str(url) for url in urls]

    @property
    def error(self) -> Any:
        return self.raw.get("error")

    @property
    def generation_id(self) -> str | None:
        return self.raw.get("generation_id")

    @property
    def cost_rub(self) -> Decimal | None:
        return self._usage.cost_rub if self._usage else None

    def refresh(self, *, timeout: float | None = None, deadline: float | None = None) -> VideoTask:
        # error_in_body=False: for a video task ``error`` describes why the
        # generation failed, which is a state to report through
        # VideoGenerationError — not a failure of this HTTP call.
        response = self._http.get(
            f"videos/{self.id}", timeout=timeout, deadline=deadline, error_in_body=False
        )
        self._apply(self._http._json(response))
        return self

    async def arefresh(
        self, *, timeout: float | None = None, deadline: float | None = None
    ) -> VideoTask:
        response = await self._http.aget(
            f"videos/{self.id}", timeout=timeout, deadline=deadline, error_in_body=False
        )
        self._apply(self._http._json(response))
        return self

    @property
    def done(self) -> bool:
        return self.status not in _POLL_STATUSES

    @property
    def failed(self) -> bool:
        return self.status in _TERMINAL_FAILURES

    def content(self, *, index: int = 0, timeout: float | None = None) -> bytes:
        """Download one generated video (``GET /videos/{id}/content?index=N``)."""
        self._validate_index(index)
        response = self._http.get(
            f"videos/{self.id}/content", params={"index": index}, timeout=timeout
        )
        return response.content

    async def acontent(self, *, index: int = 0, timeout: float | None = None) -> bytes:
        self._validate_index(index)
        response = await self._http.aget(
            f"videos/{self.id}/content", params={"index": index}, timeout=timeout
        )
        return response.content

    def _validate_index(self, index: int) -> None:
        if index < 0:
            raise ValueError(f"index must be >= 0, got {index!r}")
        if self.urls and index >= len(self.urls):
            raise ValueError(f"index {index} out of range: task has {len(self.urls)} output(s)")

    def save(
        self,
        path: str,
        *,
        index: int = 0,
        timeout: float | None = None,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> str:
        """Stream one video output to ``path`` atomically (unique temp + rename)."""
        from .._files import AtomicFileWriter

        self._validate_index(index)
        try:
            with AtomicFileWriter(path, max_bytes=max_bytes) as writer:
                with self._http.stream_request(
                    "GET", f"videos/{self.id}/content", params={"index": index}, timeout=timeout
                ) as response:
                    for chunk in response.iter_bytes():
                        writer.write(chunk)
                return str(writer.commit())
        except httpx.RequestError as exc:
            raise RequestError(f"video content download failed for task {self.id}") from exc

    async def asave(
        self,
        path: str,
        *,
        index: int = 0,
        timeout: float | None = None,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> str:
        """Async variant of :meth:`save` (file writes run off the event loop)."""
        import asyncio

        from .._files import AtomicFileWriter

        self._validate_index(index)
        writer = AtomicFileWriter(path, max_bytes=max_bytes)
        await asyncio.to_thread(writer.__enter__)
        try:
            async with self._http.astream_request(
                "GET", f"videos/{self.id}/content", params={"index": index}, timeout=timeout
            ) as response:
                async for chunk in response.aiter_bytes():
                    await asyncio.to_thread(writer.write, chunk)
            await asyncio.to_thread(writer.commit)
        except httpx.RequestError as exc:
            await asyncio.to_thread(writer.abort)
            raise RequestError(f"video content download failed for task {self.id}") from exc
        except BaseException:
            await asyncio.to_thread(writer.abort)
            raise
        return str(writer.target)

    def wait(
        self,
        *,
        timeout: float = 600.0,
        interval: float = 5.0,
        raise_on_failure: bool = False,
    ) -> VideoTask:
        """Poll until the task leaves the pending state.

        ``timeout`` bounds the total wait time: the sleep is clamped to the
        remaining budget and each refresh gets a request timeout capped to
        the remaining budget too. With ``raise_on_failure`` a terminal
        failure raises :class:`VideoGenerationError` instead of returning
        the task.
        """
        self._validate_wait(timeout, interval)
        deadline = time.monotonic() + timeout
        while not self.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeadlineExceededError(
                    f"video task {self.id} not finished within {timeout}s", budget=timeout
                )
            time.sleep(min(interval, remaining))
            self.refresh(timeout=None, deadline=deadline)
        if self.failed and raise_on_failure:
            raise VideoGenerationError(self.id, self.status, self.error)
        return self

    async def await_(
        self,
        *,
        timeout: float = 600.0,
        interval: float = 5.0,
        raise_on_failure: bool = False,
    ) -> VideoTask:
        import asyncio

        self._validate_wait(timeout, interval)
        deadline = time.monotonic() + timeout
        while not self.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeadlineExceededError(
                    f"video task {self.id} not finished within {timeout}s", budget=timeout
                )
            await asyncio.sleep(min(interval, remaining))
            await self.arefresh(timeout=None, deadline=deadline)
        if self.failed and raise_on_failure:
            raise VideoGenerationError(self.id, self.status, self.error)
        return self

    @staticmethod
    def _validate_wait(timeout: float, interval: float) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError(f"timeout must be a finite number >= 0, got {timeout!r}")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError(f"interval must be a finite number > 0, got {interval!r}")


class Videos:
    """Async video generation (``POST /api/v1/videos`` + polling)."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def create(
        self,
        model: str,
        prompt: str,
        *,
        aspect_ratio: str | None = None,
        duration: int | None = None,
        resolution: str | None = None,
        size: str | None = None,
        seed: int | None = None,
        generate_audio: bool | None = None,
        negative_prompt: str | None = None,
        frame_images: list[FrameImage | dict[str, Any]] | None = None,
        input_references: list[ImageReference | dict[str, Any]] | None = None,
        image_input: str | None = None,
        callback_url: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> VideoTask:
        body = self._body(
            model,
            prompt,
            aspect_ratio,
            duration,
            resolution,
            size,
            seed,
            generate_audio,
            negative_prompt,
            frame_images,
            input_references,
            image_input,
            callback_url,
            extra,
        )
        response = self._http.post("videos", json=body)
        return VideoTask(self._http, self._http._json(response))

    async def acreate(
        self,
        model: str,
        prompt: str,
        *,
        aspect_ratio: str | None = None,
        duration: int | None = None,
        resolution: str | None = None,
        size: str | None = None,
        seed: int | None = None,
        generate_audio: bool | None = None,
        negative_prompt: str | None = None,
        frame_images: list[FrameImage | dict[str, Any]] | None = None,
        input_references: list[ImageReference | dict[str, Any]] | None = None,
        image_input: str | None = None,
        callback_url: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> VideoTask:
        body = self._body(
            model,
            prompt,
            aspect_ratio,
            duration,
            resolution,
            size,
            seed,
            generate_audio,
            negative_prompt,
            frame_images,
            input_references,
            image_input,
            callback_url,
            extra,
        )
        response = await self._http.apost("videos", json=body)
        return VideoTask(self._http, self._http._json(response))

    def get(self, task_id: str) -> VideoTask:
        response = self._http.get(f"videos/{task_id}", error_in_body=False)
        return VideoTask(self._http, self._http._json(response))

    async def aget(self, task_id: str) -> VideoTask:
        response = await self._http.aget(f"videos/{task_id}", error_in_body=False)
        return VideoTask(self._http, self._http._json(response))

    def _body(
        self,
        model: str,
        prompt: str,
        aspect_ratio: str | None,
        duration: int | None,
        resolution: str | None,
        size: str | None,
        seed: int | None,
        generate_audio: bool | None,
        negative_prompt: str | None,
        frame_images: list[FrameImage | dict[str, Any]] | None,
        input_references: list[ImageReference | dict[str, Any]] | None,
        image_input: str | None,
        callback_url: str | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # normalize inputs first (incl. the deprecated image_input), then
        # enforce the single input mode over the normalized set
        normalized_frames: list[FrameImage] | None = None
        normalized_references: list[ImageReference] | None = None
        if frame_images:
            normalized_frames = [
                item if isinstance(item, FrameImage) else FrameImage.model_validate(item)
                for item in frame_images
            ]
        if input_references:
            normalized_references = [
                item if isinstance(item, ImageReference) else ImageReference.model_validate(item)
                for item in input_references
            ]
        if image_input is not None:
            import warnings

            warnings.warn(
                "image_input is deprecated; use frame_images=[FrameImage(url=...)] "
                "for image-to-video instead",
                DeprecationWarning,
                stacklevel=3,
            )
            normalized_frames = (normalized_frames or []) + [
                FrameImage(url=image_input, frame_type="first_frame")
            ]
        if normalized_frames and normalized_references:
            raise ValueError(
                "frame_images and input_references are mutually exclusive; "
                "pick one image-input mode"
            )
        body: dict[str, Any] = {"model": model, "prompt": prompt}
        if aspect_ratio:
            body["aspect_ratio"] = aspect_ratio
        if duration:
            body["duration"] = duration
        if resolution:
            body["resolution"] = resolution
        if size:
            body["size"] = size
        if seed is not None:
            body["seed"] = seed
        if generate_audio is not None:
            body["generate_audio"] = generate_audio
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if normalized_frames:
            body["frame_images"] = [item.model_dump_wire() for item in normalized_frames]
        if normalized_references:
            body["input_references"] = [item.model_dump_wire() for item in normalized_references]
        if callback_url:
            body["callback_url"] = callback_url
        if callback_url:
            _validate_https_callback(callback_url)
        merge_extra(
            extra,
            reserved=(
                "model",
                "prompt",
                "aspect_ratio",
                "duration",
                "resolution",
                "size",
                "seed",
                "generate_audio",
                "negative_prompt",
                "frame_images",
                "input_references",
                "image_input",
                "callback_url",
            ),
        )
        if extra:
            body.update(extra)
        return body
