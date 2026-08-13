from __future__ import annotations

import base64
import binascii
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import httpx

from .._extras import merge_extra as _merge_extra
from ..errors import APIStatusError, RouterAIError, StreamInterruptedError
from ..schemas import ChatResult, ProviderSelection, ServiceTier, Usage

if TYPE_CHECKING:
    from .._http import HTTPClient

MessageInput = dict[str, Any]

RESERVED_BODY_KEYS = (
    "model",
    "messages",
    "stream",
    "system",
    "max_tokens",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "response_format",
    "service_tier",
    "provider",
    "stop",
)


def _messages(
    prompt: str | Sequence[MessageInput], system: str | None = None
) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages
    messages = []
    for index, message in enumerate(prompt):
        if not isinstance(message, dict):
            raise ValueError(
                f"message at index {index} must be a dict, got {type(message).__name__}; "
                "string shorthand is only supported for the whole prompt"
            )
        messages.append(dict(message))
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
        prompt: str | Sequence[MessageInput],
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
        generation_id = response.generation_id
        return ChatResult.from_response(response.json(), generation_id=generation_id)

    async def acomplete(
        self,
        model: str,
        prompt: str | Sequence[MessageInput],
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
        generation_id = response.generation_id
        return ChatResult.from_response(response.json(), generation_id=generation_id)

    def _build_body(
        self,
        model: str,
        prompt: str | Sequence[MessageInput],
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
        _merge_extra(extra, reserved=RESERVED_BODY_KEYS)
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
        prompt: str | Sequence[MessageInput],
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
            extra=extra,
        )
        body["stream"] = True
        with self._http.stream_request("POST", "chat/completions", json=body) as response:
            yield from _iter_sse(
                response,
                http=self._http,
                generation_id=response.headers.get("X-Generation-Id"),
            )

    async def astream(
        self,
        model: str,
        prompt: str | Sequence[MessageInput],
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
            extra=extra,
        )
        body["stream"] = True
        async with self._http.astream_request("POST", "chat/completions", json=body) as response:
            async for chunk in _aiter_sse(
                response,
                http=self._http,
                generation_id=response.headers.get("X-Generation-Id"),
            ):
                yield chunk


class StreamChunk:
    """A single SSE chunk of a streaming completion."""

    def __init__(self, raw: dict[str, Any], generation_id: str | None = None) -> None:
        self.raw = raw
        self.generation_id = generation_id

    @property
    def error(self) -> Any:
        return self.raw.get("error") if isinstance(self.raw, dict) else None

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
    def audio(self) -> AudioDelta | None:
        """Decoded audio delta (chat audio output models)."""
        for choice in self.raw.get("choices") or []:
            delta = choice.get("delta") or {}
            audio = delta.get("audio")
            if isinstance(audio, dict) and audio.get("data"):
                try:
                    return AudioDelta(
                        data=base64.b64decode(audio["data"], validate=True),
                        format=audio.get("format"),
                        transcript=audio.get("transcript"),
                    )
                except (binascii.Error, ValueError):
                    continue
        return None

    @property
    def usage(self) -> Usage | None:
        usage = self.raw.get("usage")
        return Usage.model_validate(usage) if usage else None

    @property
    def cost_rub(self) -> Decimal | None:
        usage = self.usage
        return usage.cost_rub if usage else None


class AudioDelta:
    """A decoded audio chunk from a chat audio stream."""

    def __init__(
        self, data: bytes, *, format: str | None = None, transcript: str | None = None
    ) -> None:
        self.data = data
        self.format = format
        self.transcript = transcript


class StreamAccumulator:
    """Aggregates streamed chunks into a single result.

    Collects text, reasoning, tool calls and audio deltas across SSE chunks
    and keeps the final usage/metadata reported by the last chunk.
    """

    def __init__(self) -> None:
        self.generation_id: str | None = None
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._audio: list[AudioDelta] = []
        self._usage: Usage | None = None
        self._finish_reason: str | None = None
        self.chunks_received = 0

    def add(self, chunk: StreamChunk) -> None:
        self.chunks_received += 1
        if chunk.generation_id and not self.generation_id:
            self.generation_id = chunk.generation_id
        if chunk.content:
            self._content.append(chunk.content)
        if chunk.reasoning:
            self._reasoning.append(chunk.reasoning)
        if chunk.tool_calls:
            self._tool_calls.extend(chunk.tool_calls)
        if chunk.audio is not None:
            self._audio.append(chunk.audio)
        if chunk.usage is not None:
            self._usage = chunk.usage
        if chunk.finish_reason:
            self._finish_reason = chunk.finish_reason

    @property
    def content(self) -> str:
        return "".join(self._content)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return list(self._tool_calls)

    @property
    def audio(self) -> list[AudioDelta]:
        return list(self._audio)

    @property
    def usage(self) -> Usage | None:
        return self._usage

    @property
    def finish_reason(self) -> str | None:
        return self._finish_reason

    @property
    def cost_rub(self) -> Decimal | None:
        return self._usage.cost_rub if self._usage else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "audio_chunks": len(self._audio),
            "usage": self._usage.model_dump() if self._usage else None,
            "finish_reason": self._finish_reason,
            "chunks_received": self.chunks_received,
        }


def _parse_sse_event(data: str, chunks_received: int) -> dict[str, Any] | None:
    if data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RouterAIError(f"unparsable SSE line: {data!r}") from exc
    error = payload.get("error") if isinstance(payload, dict) else None
    if error:
        raise APIStatusError(
            str(error.get("message", error) if isinstance(error, dict) else error),
            _safe_status(error if isinstance(error, dict) else payload),
            dict(payload),
        )
    return dict(payload)


def _safe_status(payload: dict[str, Any]) -> int:
    """Normalize an SSE error status_code; malformed values default to 502."""
    for key in ("status_code", "status"):
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and 100 <= value <= 599:
            return value
        if isinstance(value, str) and value.isdigit():
            number = int(value)
            if 100 <= number <= 599:
                return number
    return 502


def _iter_sse(
    response: Any, *, http: HTTPClient, generation_id: str | None = None
) -> Iterator[StreamChunk]:
    chunks_received = 0
    try:
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            payload = _parse_sse_event(data, chunks_received)
            if payload is None:
                break
            chunks_received += 1
            yield StreamChunk(payload, generation_id=generation_id)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise StreamInterruptedError(
            f"stream interrupted after {chunks_received} chunks: {exc}",
            chunks_received=chunks_received,
        ) from exc


async def _aiter_sse(
    response: Any, *, http: HTTPClient, generation_id: str | None = None
) -> AsyncIterator[StreamChunk]:
    chunks_received = 0
    try:
        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            payload = _parse_sse_event(data, chunks_received)
            if payload is None:
                break
            chunks_received += 1
            yield StreamChunk(payload, generation_id=generation_id)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise StreamInterruptedError(
            f"stream interrupted after {chunks_received} chunks: {exc}",
            chunks_received=chunks_received,
        ) from exc
