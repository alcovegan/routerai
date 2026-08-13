from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ..errors import RouterAIError
from ..schemas import ChatResult, ProviderSelection, ServiceTier, Usage

if TYPE_CHECKING:
    from .._http import HTTPClient

MessageInput = dict[str, Any] | str


def _messages(prompt: str | list[MessageInput], system: str | None = None) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages
    messages = [dict(m) for m in prompt if isinstance(m, dict)]
    if system and not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system})
    return messages


class Chat:
    """Chat completions with cost parsing and streaming."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def complete(
        self,
        model: str,
        prompt: str | list[MessageInput],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        service_tier: ServiceTier | str | None = None,
        provider: ProviderSelection | dict[str, Any] | None = None,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Send a chat completion request and return a parsed result.

        ``result.cost_rub`` contains the ruble cost when the provider reports
        it; otherwise call ``client.generation.get(result.generation_id)``.
        """
        body = self._build_body(
            model,
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            service_tier=service_tier,
            provider=provider,
            stop=stop,
            extra=extra,
        )
        response = self._http.post("chat/completions", json=body)
        generation_id = response.headers.get("X-Generation-Id")
        return ChatResult.from_response(response.json(), generation_id=generation_id)

    async def acomplete(
        self,
        model: str,
        prompt: str | list[MessageInput],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        service_tier: ServiceTier | str | None = None,
        provider: ProviderSelection | dict[str, Any] | None = None,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        body = self._build_body(
            model,
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            service_tier=service_tier,
            provider=provider,
            stop=stop,
            extra=extra,
        )
        response = await self._http.apost("chat/completions", json=body)
        generation_id = response.headers.get("X-Generation-Id")
        return ChatResult.from_response(response.json(), generation_id=generation_id)

    def _build_body(
        self,
        model: str,
        prompt: str | list[MessageInput],
        *,
        system: str | None,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        response_format: dict[str, Any] | None,
        service_tier: ServiceTier | str | None,
        provider: ProviderSelection | dict[str, Any] | None,
        stop: list[str] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": _messages(prompt, system)}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if service_tier is not None:
            body["service_tier"] = ServiceTier(service_tier).value
        if provider is not None:
            body["provider"] = (
                provider.model_dump(exclude_none=True)
                if isinstance(provider, ProviderSelection)
                else provider
            )
        if stop is not None:
            body["stop"] = stop
        if extra:
            body.update(extra)
        return body

    # --- streaming ---

    def stream(
        self,
        model: str,
        prompt: str | list[MessageInput],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        service_tier: ServiceTier | str | None = None,
        provider: ProviderSelection | dict[str, Any] | None = None,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        body = self._build_body(
            model,
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            response_format=None,
            service_tier=service_tier,
            provider=provider,
            stop=stop,
            extra={"stream": True, **(extra or {})},
        )
        with self._http.stream_request("POST", "chat/completions", json=body) as response:
            yield from _iter_sse(response, http=self._http)

    async def astream(
        self,
        model: str,
        prompt: str | list[MessageInput],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        service_tier: ServiceTier | str | None = None,
        provider: ProviderSelection | dict[str, Any] | None = None,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        body = self._build_body(
            model,
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            response_format=None,
            service_tier=service_tier,
            provider=provider,
            stop=stop,
            extra={"stream": True, **(extra or {})},
        )
        async with self._http.astream_request("POST", "chat/completions", json=body) as response:
            async for chunk in _aiter_sse(response, http=self._http):
                yield chunk


class StreamChunk:
    """A single SSE chunk of a streaming completion."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw

    @property
    def content(self) -> str:
        for choice in self.raw.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                return content
        return ""

    @property
    def reasoning(self) -> str:
        for choice in self.raw.get("choices") or []:
            delta = choice.get("delta") or {}
            for key in ("reasoning_content", "reasoning"):
                value = delta.get(key)
                if isinstance(value, str):
                    return value
        return ""

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for choice in self.raw.get("choices") or []:
            delta = choice.get("delta") or {}
            for call in delta.get("tool_calls") or []:
                calls.append(dict(call))
        return calls

    @property
    def finish_reason(self) -> str | None:
        for choice in self.raw.get("choices") or []:
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                return finish_reason
        return None

    @property
    def usage(self) -> Usage | None:
        usage = self.raw.get("usage")
        return Usage.model_validate(usage) if usage else None

    @property
    def cost_rub(self) -> Decimal | None:
        usage = self.usage
        return usage.cost_rub if usage else None


def _iter_sse(response: Any, *, http: HTTPClient) -> Iterator[StreamChunk]:
    for line in response.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            http.logger.debug("skip unparsable SSE line: %r", data)
            raise RouterAIError(f"unparsable SSE line: {data!r}") from exc
        yield StreamChunk(payload)


async def _aiter_sse(response: Any, *, http: HTTPClient) -> AsyncIterator[StreamChunk]:
    async for line in response.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            http.logger.debug("skip unparsable SSE line: %r", data)
            raise RouterAIError(f"unparsable SSE line: {data!r}") from exc
        yield StreamChunk(payload)
