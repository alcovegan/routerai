from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ..errors import RouterAIError
from ..schemas import Usage

if TYPE_CHECKING:
    from .._http import HTTPClient

_POLL_STATUSES = {"pending", "processing", "running", "in_progress", "queued"}


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

    def wait(
        self,
        *,
        timeout: float = 600.0,
        interval: float = 5.0,
    ) -> VideoTask:
        deadline = time.monotonic() + timeout
        while not self.done and time.monotonic() < deadline:
            time.sleep(interval)
            self.refresh()
        if not self.done:
            raise RouterAIError(f"video task {self.id} not finished within {timeout}s")
        return self

    async def await_(
        self,
        *,
        timeout: float = 600.0,
        interval: float = 5.0,
    ) -> VideoTask:
        import asyncio

        deadline = time.monotonic() + timeout
        while not self.done and time.monotonic() < deadline:
            await asyncio.sleep(interval)
            await self.arefresh()
        if not self.done:
            raise RouterAIError(f"video task {self.id} not finished within {timeout}s")
        return self


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
        callback_url: str | None = None,
        image_input: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> VideoTask:
        body = self._body(
            model, prompt, aspect_ratio, duration, resolution, callback_url, image_input, extra
        )
        response = self._http.post("videos", json=body)
        return VideoTask(self._http, response.json())

    async def acreate(
        self,
        model: str,
        prompt: str,
        *,
        aspect_ratio: str | None = None,
        duration: int | None = None,
        resolution: str | None = None,
        callback_url: str | None = None,
        image_input: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> VideoTask:
        body = self._body(
            model, prompt, aspect_ratio, duration, resolution, callback_url, image_input, extra
        )
        response = await self._http.apost("videos", json=body)
        return VideoTask(self._http, response.json())

    def get(self, task_id: str) -> VideoTask:
        response = self._http.get(f"videos/{task_id}")
        return VideoTask(self._http, response.json())

    async def aget(self, task_id: str) -> VideoTask:
        response = await self._http.aget(f"videos/{task_id}")
        return VideoTask(self._http, response.json())

    def _body(
        self,
        model: str,
        prompt: str,
        aspect_ratio: str | None,
        duration: int | None,
        resolution: str | None,
        callback_url: str | None,
        image_input: str | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "prompt": prompt}
        if aspect_ratio:
            body["aspect_ratio"] = aspect_ratio
        if duration:
            body["duration"] = duration
        if resolution:
            body["resolution"] = resolution
        if callback_url:
            body["callback_url"] = callback_url
        if image_input:
            body["image_input"] = image_input
        if extra:
            body.update(extra)
        return body
