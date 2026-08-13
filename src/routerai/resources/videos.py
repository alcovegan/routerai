from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from .._extras import merge_extra
from ..errors import RouterAIError, VideoGenerationError
from ..schemas import Usage

if TYPE_CHECKING:
    from .._http import HTTPClient

_POLL_STATUSES = {"pending", "processing", "running", "in_progress", "queued"}
_TERMINAL_FAILURES = {"failed", "cancelled", "expired"}


class FrameImage(BaseModel):
    """Reference frame for image-to-video (``frame_images``).

    ``frame_type`` is ``first_frame`` (the video starts from this frame) or
    ``last_frame`` (the video ends on this frame).
    """

    model_config = ConfigDict(extra="allow")

    url: str
    frame_type: str = "first_frame"

    def model_dump_wire(self) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self.url},
            "frame_type": self.frame_type,
        }


class ImageReference(BaseModel):
    """Style/character reference for reference-to-video (``input_references``)."""

    model_config = ConfigDict(extra="allow")

    url: str

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

    def refresh(self) -> VideoTask:
        response = self._http.get(f"videos/{self.id}")
        self._apply(self._http._json(response))
        return self

    async def arefresh(self) -> VideoTask:
        response = await self._http.aget(f"videos/{self.id}")
        self._apply(self._http._json(response))
        return self

    @property
    def done(self) -> bool:
        return self.status not in _POLL_STATUSES

    @property
    def failed(self) -> bool:
        return self.status in _TERMINAL_FAILURES

    def content(self) -> bytes:
        """Download the generated video (``GET /videos/{id}/content``)."""
        return self._http.get(f"videos/{self.id}/content").content

    async def acontent(self) -> bytes:
        return (await self._http.aget(f"videos/{self.id}/content")).content

    def save(self, path: str) -> str:
        import pathlib

        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.content())
        return str(target)

    async def asave(self, path: str) -> str:
        import pathlib

        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await self.acontent())
        return str(target)

    def wait(
        self,
        *,
        timeout: float = 600.0,
        interval: float = 5.0,
        raise_on_failure: bool = False,
    ) -> VideoTask:
        """Poll until the task leaves the pending state.

        ``timeout`` bounds the total wait time (the sleep is clamped to the
        remaining budget). With ``raise_on_failure`` a terminal failure
        raises :class:`VideoGenerationError` instead of returning the task.
        """
        self._validate_wait(timeout, interval)
        deadline = time.monotonic() + timeout
        while not self.done:
            if time.monotonic() >= deadline:
                raise RouterAIError(f"video task {self.id} not finished within {timeout}s")
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
            self.refresh()
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
            if time.monotonic() >= deadline:
                raise RouterAIError(f"video task {self.id} not finished within {timeout}s")
            await asyncio.sleep(min(interval, max(0.0, deadline - time.monotonic())))
            await self.arefresh()
        if self.failed and raise_on_failure:
            raise VideoGenerationError(self.id, self.status, self.error)
        return self

    @staticmethod
    def _validate_wait(timeout: float, interval: float) -> None:
        if timeout < 0:
            raise ValueError(f"timeout must be >= 0, got {timeout!r}")
        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval!r}")


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
        response = self._http.get(f"videos/{task_id}")
        return VideoTask(self._http, self._http._json(response))

    async def aget(self, task_id: str) -> VideoTask:
        response = await self._http.aget(f"videos/{task_id}")
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
        if frame_images and input_references:
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
        if frame_images:
            body["frame_images"] = [
                item.model_dump_wire() if isinstance(item, FrameImage) else dict(item)
                for item in frame_images
            ]
        if input_references:
            body["input_references"] = [
                item.model_dump_wire() if isinstance(item, ImageReference) else dict(item)
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
            body.setdefault("frame_images", []).append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_input},
                    "frame_type": "first_frame",
                }
            )
        if callback_url:
            body["callback_url"] = callback_url
        merge_extra(extra, reserved=("model", "prompt", "frame_images", "input_references"))
        if extra:
            body.update(extra)
        return body
