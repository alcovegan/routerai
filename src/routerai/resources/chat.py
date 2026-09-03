from __future__ import annotations

import base64
import binascii
import inspect
import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .._errors import DONE_MARKER, parse_stream_event
from .._extras import merge_extra as _merge_extra
from .._options import RequestOptions, Unpack
from .._routing import conflicting_keys
from ..errors import (
    ConfigurationError,
    ResponseParsingError,
    RouterAIError,
    StreamInterruptedError,
)
from ..schemas import ChatResult, ProviderSelection, ServiceTier, ToolCall, Usage
from ..tools import (
    ToolFunction,
    ToolRun,
    ToolRunResult,
    assistant_message,
    normalize_tools,
    parse_arguments,
    tool_message,
)

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
        **opts: Unpack[RequestOptions],
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
        response = self._http.post("chat/completions", json=body, **opts)
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
        **opts: Unpack[RequestOptions],
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
        response = await self._http.apost("chat/completions", json=body, **opts)
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
        clashes = conflicting_keys(model, provider)
        if clashes:
            raise ConfigurationError(
                f"{', '.join(clashes)} set both in the model string and in provider=; "
                "RouterAI rejects that with 400. Keep one of the two."
            )
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

    def parse(
        self,
        model: str,
        prompt: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> ParsedResult[Any]:
        """Ask for a structured answer and validate it against ``response_model``.

        The JSON schema is derived from the model, so the shape asked for and
        the shape validated cannot drift apart.
        """
        result = self.complete(
            model,
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=_json_schema_format(response_model),
            extra=extra,
            **opts,
        )
        return ParsedResult(result, _validate_parsed(result, response_model))

    async def aparse(
        self,
        model: str,
        prompt: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> ParsedResult[Any]:
        result = await self.acomplete(
            model,
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=_json_schema_format(response_model),
            extra=extra,
            **opts,
        )
        return ParsedResult(result, _validate_parsed(result, response_model))

    def run_tools(
        self,
        model: str,
        prompt: str | Sequence[MessageInput],
        *,
        tools: Sequence[ToolFunction | dict[str, Any]] | Mapping[str, ToolFunction],
        system: str | None = None,
        max_turns: int = 5,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> ToolRunResult:
        """Answer the prompt, running any tools the model asks for.

        Loops model → function → model until the model stops asking, or until
        ``max_turns`` is reached; the last result is returned either way, so a
        model stuck in a loop costs a bounded amount of money.
        """
        schemas, registry = normalize_tools(tools)
        messages = _messages(prompt, system)
        runs: list[ToolRun] = []

        for turn in range(1, max_turns + 1):
            result = self.complete(
                model,
                messages,
                tools=schemas,
                max_tokens=max_tokens,
                temperature=temperature,
                extra=extra,
                **opts,
            )
            if not result.tool_calls:
                return ToolRunResult(result, runs, messages, turn)
            messages.append(assistant_message(result))
            for call in result.tool_calls:
                run = _execute(call, registry)
                runs.append(run)
                messages.append(
                    tool_message(call.id, run.name, run.error if run.error else run.result)
                )
        return ToolRunResult(result, runs, messages, max_turns)

    async def arun_tools(
        self,
        model: str,
        prompt: str | Sequence[MessageInput],
        *,
        tools: Sequence[ToolFunction | dict[str, Any]] | Mapping[str, ToolFunction],
        system: str | None = None,
        max_turns: int = 5,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> ToolRunResult:
        """Async :meth:`run_tools`; async tool functions are awaited."""
        schemas, registry = normalize_tools(tools)
        messages = _messages(prompt, system)
        runs: list[ToolRun] = []

        for turn in range(1, max_turns + 1):
            result = await self.acomplete(
                model,
                messages,
                tools=schemas,
                max_tokens=max_tokens,
                temperature=temperature,
                extra=extra,
                **opts,
            )
            if not result.tool_calls:
                return ToolRunResult(result, runs, messages, turn)
            messages.append(assistant_message(result))
            for call in result.tool_calls:
                run = await _aexecute(call, registry)
                runs.append(run)
                messages.append(
                    tool_message(call.id, run.name, run.error if run.error else run.result)
                )
        return ToolRunResult(result, runs, messages, max_turns)

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
        response_format: dict[str, Any] | None = None,
        service_tier: ServiceTier | str | None = None,
        provider: ProviderSelection | dict[str, Any] | None = None,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        **opts: Unpack[RequestOptions],
    ) -> ChatStream:
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
        body["stream"] = True
        return ChatStream(self._stream_chunks(body, opts))

    def _stream_chunks(self, body: dict[str, Any], opts: RequestOptions) -> Iterator[StreamChunk]:
        with self._http.stream_request("POST", "chat/completions", json=body, **opts) as response:
            yield from _iter_sse(
                response,
                http=self._http,
                generation_id=response.generation_id,
            )

    def astream(
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
        **opts: Unpack[RequestOptions],
    ) -> AsyncChatStream:
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
        body["stream"] = True
        return AsyncChatStream(self._astream_chunks(body, opts))

    async def _astream_chunks(
        self, body: dict[str, Any], opts: RequestOptions
    ) -> AsyncIterator[StreamChunk]:
        async with self._http.astream_request(
            "POST", "chat/completions", json=body, **opts
        ) as response:
            async for chunk in _aiter_sse(
                response,
                http=self._http,
                generation_id=response.generation_id,
            ):
                yield chunk


ParsedT = TypeVar("ParsedT", bound=BaseModel)


class ParsedResult(Generic[ParsedT]):
    """A chat result plus the object it was validated into."""

    def __init__(self, result: ChatResult, parsed: ParsedT) -> None:
        self.result = result
        self.parsed = parsed

    @property
    def content(self) -> str | None:
        return self.result.content

    @property
    def usage(self) -> Usage | None:
        return self.result.usage

    @property
    def cost_rub(self) -> Decimal | None:
        return self.result.cost_rub

    @property
    def generation_id(self) -> str | None:
        return self.result.generation_id


def _json_schema_format(response_model: type[BaseModel]) -> dict[str, Any]:
    schema = response_model.model_json_schema()
    schema.setdefault("additionalProperties", False)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": schema,
            "strict": True,
        },
    }


def _validate_parsed(result: ChatResult, response_model: type[ParsedT]) -> ParsedT:
    if not result.content:
        raise ResponseParsingError("model returned no content to parse", body=result.raw)
    try:
        payload = json.loads(result.content)
    except ValueError as exc:
        raise ResponseParsingError(
            "model did not return JSON despite the requested schema", body=result.content
        ) from exc
    try:
        return response_model.model_validate(payload)
    except ValidationError as exc:
        raise ResponseParsingError(
            f"model output does not match {response_model.__name__}: {exc}", body=payload
        ) from exc


def _execute(call: ToolCall, registry: dict[str, ToolFunction]) -> ToolRun:
    name = call.name or ""
    function = registry.get(name)
    if function is None:
        return ToolRun(name, {}, None, error=f"no tool named {name!r} was provided")
    try:
        arguments = parse_arguments(call.arguments)
    except RouterAIError as exc:
        return ToolRun(name, {}, None, error=str(exc))
    try:
        return ToolRun(name, arguments, function(**arguments))
    except Exception as exc:
        # A failing tool is information for the model, not a crash for the caller.
        return ToolRun(name, arguments, None, error=f"{type(exc).__name__}: {exc}")


async def _aexecute(call: ToolCall, registry: dict[str, ToolFunction]) -> ToolRun:
    run = _execute(call, registry)
    if inspect.isawaitable(run.result):
        try:
            run.result = await run.result
        except Exception as exc:
            run.result, run.error = None, f"{type(exc).__name__}: {exc}"
    return run


class ChatStream:
    """An open chat stream.

    Iterating works exactly as before. The context-manager form closes the
    connection deterministically::

        with client.chat.stream(model, prompt) as stream:
            for chunk in stream:
                ...
    """

    def __init__(self, chunks: Iterator[StreamChunk]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[StreamChunk]:
        return self._chunks

    def __next__(self) -> StreamChunk:
        return next(self._chunks)

    def __enter__(self) -> ChatStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        close = getattr(self._chunks, "close", None)
        if close is not None:
            close()


class AsyncChatStream:
    """The async counterpart of :class:`ChatStream`.

    Here the context manager matters more than convenience: reference counting
    closes an abandoned sync generator, but an abandoned async one keeps the
    connection until the loop shuts down its async generators.
    """

    def __init__(self, chunks: AsyncIterator[StreamChunk]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        return self._chunks

    async def __anext__(self) -> StreamChunk:
        return await self._chunks.__anext__()

    async def __aenter__(self) -> AsyncChatStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        aclose = getattr(self._chunks, "aclose", None)
        if aclose is not None:
            await aclose()


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
        self._tool_calls: dict[int, dict[str, Any]] = {}
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
        for delta in chunk.tool_calls:
            self._merge_tool_call(delta)
        if chunk.audio is not None:
            self._audio.append(chunk.audio)
        if chunk.usage is not None:
            self._usage = chunk.usage
        if chunk.finish_reason:
            self._finish_reason = chunk.finish_reason

    def _merge_tool_call(self, delta: dict[str, Any]) -> None:
        """Fold one tool-call delta into the call it belongs to.

        A tool call arrives split across chunks: the first carries ``id`` and
        the function name, the rest append a character or two of arguments,
        all under the same ``index``. Collecting them as separate calls leaves
        the caller with fragments that no JSON parser will accept.
        """
        index = delta.get("index")
        if not isinstance(index, int):
            index = len(self._tool_calls)
        call = self._tool_calls.setdefault(index, {"index": index, "function": {}})

        for key, value in delta.items():
            if key in ("index", "function"):
                continue
            if value is not None:
                call[key] = value

        function = delta.get("function")
        if not isinstance(function, dict):
            return
        merged = call["function"]
        for key, value in function.items():
            if value is None:
                continue
            if key == "arguments":
                merged["arguments"] = f"{merged.get('arguments', '')}{value}"
            else:
                merged[key] = value

    @property
    def content(self) -> str:
        return "".join(self._content)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Complete tool calls, ordered by index and ready to execute."""
        return [self._tool_calls[index] for index in sorted(self._tool_calls)]

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


def _iter_sse(
    response: Any, *, http: HTTPClient, generation_id: str | None = None
) -> Iterator[StreamChunk]:
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
            response.note_chunk(payload)
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
            if data == DONE_MARKER:
                break
            payload = parse_stream_event(data)
            if payload is None:
                continue
            chunks_received += 1
            response.note_chunk(payload)
            yield StreamChunk(payload, generation_id=generation_id)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise StreamInterruptedError(
            f"stream interrupted after {chunks_received} chunks: {exc}",
            chunks_received=chunks_received,
        ) from exc
