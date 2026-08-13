from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ..schemas import Usage

if TYPE_CHECKING:
    from .._http import HTTPClient


class Completions:
    """Legacy text completions (``POST /api/v1/completions``)."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def create(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CompletionsResult:
        body: dict[str, Any] = {"model": model, "prompt": prompt}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if extra:
            body.update(extra)
        response = self._http.post("completions", json=body)
        return CompletionsResult.from_response(response.json())

    async def acreate(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CompletionsResult:
        body: dict[str, Any] = {"model": model, "prompt": prompt}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if extra:
            body.update(extra)
        response = await self._http.apost("completions", json=body)
        return CompletionsResult.from_response(response.json())


class CompletionsResult:
    def __init__(self, text: str, usage: Usage | None, raw: dict[str, Any]) -> None:
        self.text = text
        self.usage = usage
        self.raw = raw

    @classmethod
    def from_response(cls, payload: dict[str, Any]) -> CompletionsResult:
        choices = payload.get("choices") or []
        text = "".join(str(c.get("text", "")) for c in choices if isinstance(c, dict))
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return cls(text, usage, payload)

    @property
    def cost_rub(self) -> Decimal | None:
        return self.usage.cost_rub if self.usage else None


class Responses:
    """OpenAI Responses API compatibility (``POST /api/v1/responses``)."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def create(self, model: str, input: Any, **kwargs: Any) -> dict[str, Any]:
        body = {"model": model, "input": input, **kwargs}
        return self._http._json(self._http.post("responses", json=body))

    async def acreate(self, model: str, input: Any, **kwargs: Any) -> dict[str, Any]:
        body = {"model": model, "input": input, **kwargs}
        return self._http._json(await self._http.apost("responses", json=body))


class Messages:
    """Anthropic-compatible Messages API (``POST /api/v1/messages``)."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def create(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        body = {"model": model, "messages": messages, **kwargs}
        return self._http._json(self._http.post("messages", json=body))

    async def acreate(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        body = {"model": model, "messages": messages, **kwargs}
        return self._http._json(await self._http.apost("messages", json=body))
