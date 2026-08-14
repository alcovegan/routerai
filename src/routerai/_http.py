from __future__ import annotations

import logging
import platform
import random
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass
from json import loads as _json_loads
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from ._errors import build_error
from ._options import RequestOptions
from ._version import __version__
from .errors import (
    APIConnectionError,
    APITimeoutError,
    ConfigurationError,
    DeadlineExceededError,
    RequestError,
    ResponseParsingError,
)
from .logging import log_request, mask_key
from .usage import UsageHook, UsageTracker, record_from

if TYPE_CHECKING:
    from typing_extensions import Unpack

DEFAULT_BASE_URL = "https://routerai.ru/api/v1"
DEFAULT_TIMEOUT = 60.0

# Identify the SDK to the server: without this RouterAI sees "python-httpx"
# and cannot tell its own client from a hand-rolled request.
USER_AGENT = (
    f"routerai-python/{__version__} (python/{platform.python_version()}; httpx/{httpx.__version__})"
)

# Statuses retried for idempotent (safe) methods.
RETRYABLE_SAFE_STATUSES = {408, 425, 429, 500, 502, 503, 504}
# Statuses retried for non-idempotent methods (POST/PATCH/DELETE).
# Only 429 is safe: the request was rejected before being processed.
RETRYABLE_UNSAFE_STATUSES = {429}
SAFE_METHODS = {"GET", "HEAD"}


def _raise_for_status(
    status: int,
    body: Any,
    *,
    headers: Any = None,
    error_in_body: bool = True,
) -> None:
    """Raise the typed error this response describes, if it describes one."""
    error = build_error(http_status=status, body=body, headers=headers, error_in_body=error_in_body)
    if error is not None:
        raise error


def _transport_error(message: str, exc: Exception | None) -> RequestError:
    """Distinguish a timeout from a connection failure — they need different fixes."""
    if isinstance(exc, httpx.TimeoutException):
        return APITimeoutError(message)
    if isinstance(exc, httpx.ConnectError):
        return APIConnectionError(message)
    return RequestError(message)


@dataclass(frozen=True)
class _Call:
    """Settings resolved for one request."""

    timeout: float
    max_retries: int
    deadline: float | None
    headers: Mapping[str, str] | None
    error_in_body: bool


class ResponseEnvelope:
    """A response with its JSON body decoded exactly once.

    ``HTTPClient`` returns this for buffered (non-streaming) responses.
    Resources read ``.json()``/``.content``/``.generation_id`` from it, so
    the body is parsed a single time and shared by logging, error mapping
    and the resource parser.
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._body: Any = None
        self._is_json = False
        self.status_code = response.status_code
        self.headers = response.headers
        self.generation_id = response.headers.get("X-Generation-Id")
        self.request_id = response.headers.get("X-Request-Id") or response.headers.get("Request-Id")
        content_type = response.headers.get("content-type", "").lower()
        if "json" in content_type and response.status_code != 204:
            # Decode, but do not force the result into a dict: a JSON array of
            # two-key objects would be read as key/value pairs and turn into a
            # dictionary that never existed in the response.
            try:
                self._body = response.json()
                self._is_json = True
            except ValueError:
                pass

    @property
    def body(self) -> Any:
        """The decoded body (mapping for JSON objects, bytes otherwise)."""
        if self._body is None:
            if self._is_json:
                try:
                    self._body = self._response.json()
                except ValueError:
                    self._is_json = False
                    self._body = self._response.content
            else:
                self._body = self._response.content
        return self._body

    def json(self) -> dict[str, Any]:
        """The body as a JSON object.

        An empty body — which is what a 204 on DELETE looks like — reads as an
        empty object rather than an error, so deleting a key does not fail
        after the server already deleted it.
        """
        if self.status_code == 204:
            return {}
        body = self.body
        if isinstance(body, dict):
            return dict(body)
        if body in (None, b"", ""):
            return {}
        raise ResponseParsingError(f"expected a JSON object, got {type(body).__name__}", body=body)

    @property
    def content(self) -> bytes:
        return self._response.content

    @property
    def text(self) -> str:
        return self._response.text

    @property
    def raw_response(self) -> httpx.Response:
        return self._response

    def close(self) -> None:
        self._response.close()

    async def aclose(self) -> None:
        await self._response.aclose()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


class StreamEnvelope:
    """An open stream that remembers what the last chunk said it cost.

    A context variable cannot be used for this: generators do not carry their
    own context, so interleaving two streams would file one stream's usage
    under the other. The chunk reader hands its payload here instead.
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.usage_payload: dict[str, Any] | None = None
        self.model: str | None = None

    @property
    def generation_id(self) -> str | None:
        value = self._response.headers.get("X-Generation-Id")
        return str(value) if value else None

    def note_chunk(self, payload: Mapping[str, Any]) -> None:
        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            self.usage_payload = dict(usage)
        model = payload.get("model")
        if isinstance(model, str):
            self.model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


def _stream_body(response: httpx.Response) -> Any:
    """Decoded body of a streamed response, for error mapping."""
    try:
        text = response.text
    except httpx.ResponseNotRead:
        text = response.read().decode("utf-8", errors="replace")
    try:
        return _json_loads(text)
    except ValueError:
        return text


def _retry_after_seconds(response: httpx.Response, *, max_retry_after: float) -> float | None:
    """Parse Retry-After per RFC 9110 and clamp it to ``max_retry_after``.

    Only strict integer delay-seconds or a valid HTTP-date are accepted;
    anything else (fractions, nan, inf, garbage) falls back to the regular
    exponential backoff.
    """
    import math

    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None
    if value.isdigit():
        if len(value) > 9:
            return max_retry_after
        delay = float(int(value))
    else:
        try:
            import email.utils

            date = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if date.tzinfo is None:
            return None
        delay = date.timestamp() - time.time()
    if not math.isfinite(delay) or delay < 0:
        return None
    return min(delay, max_retry_after)


class HTTPClient:
    """httpx wrapper: base URL, auth, retries, logging, error mapping.

    Sync and async transports are kept in separate slots, so the same
    :class:`routerai.RouterAI` instance can be used from both a sync context
    and an async loop without crashing or leaking connections.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        retry_unsafe_methods: bool = False,
        max_retry_after: float = 60.0,
        logger: logging.Logger | str | None = None,
        http_client: httpx.Client | httpx.AsyncClient | None = None,
        async_http_client: httpx.AsyncClient | None = None,
        default_headers: dict[str, str] | None = None,
        app_info: str | None = None,
    ) -> None:
        if async_http_client is not None and isinstance(http_client, httpx.AsyncClient):
            raise ConfigurationError("pass async transport via async_http_client, not http_client")
        if isinstance(http_client, httpx.AsyncClient):
            async_http_client = http_client
            http_client = None

        self._validate_config(timeout, max_retries, retry_backoff, max_retry_after)

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._retry_unsafe_methods = retry_unsafe_methods
        self._max_retry_after = max_retry_after
        if logger is None:
            from .logging import get_logger

            logger = get_logger()
        elif isinstance(logger, str):
            from .logging import get_logger

            logger = get_logger(logger)
        self._logger = logger
        user_agent = USER_AGENT if app_info is None else f"{USER_AGENT} {app_info}"
        self._headers = dict(
            httpx.Headers(
                {
                    "Content-Type": "application/json",
                    "User-Agent": user_agent,
                    **(default_headers or {}),
                }
            )
        )
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._sync_client = http_client if isinstance(http_client, httpx.Client) else None
        self._async_client = async_http_client
        self._owns_sync = self._sync_client is None
        self._owns_async = self._async_client is None
        self._sync_closed = False
        self._async_closed = False
        self.usage = UsageTracker()
        self._trackers: ContextVar[tuple[UsageTracker, ...]] = ContextVar(
            f"routerai_trackers_{id(self)}", default=()
        )
        self._usage_hooks: list[UsageHook] = []
        # Polling a video returns the same usage on every refresh; without this
        # the cost of one generation would be counted once per poll.
        self._counted_generations: OrderedDict[str, None] = OrderedDict()

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @staticmethod
    def _validate_timeout(timeout: float) -> None:
        import math

        if not math.isfinite(timeout) or timeout <= 0:
            raise ConfigurationError(f"timeout must be a positive finite number, got {timeout!r}")

    @staticmethod
    def _validate_max_retries(max_retries: int) -> None:
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ConfigurationError(
                f"max_retries must be a non-negative integer, got {max_retries!r}"
            )

    @classmethod
    def _validate_config(
        cls, timeout: float, max_retries: int, retry_backoff: float, max_retry_after: float
    ) -> None:
        import math

        cls._validate_timeout(timeout)
        cls._validate_max_retries(max_retries)
        if not math.isfinite(retry_backoff) or retry_backoff < 0:
            raise ConfigurationError(
                f"retry_backoff must be a non-negative finite number, got {retry_backoff!r}"
            )
        if not math.isfinite(max_retry_after) or max_retry_after < 0:
            raise ConfigurationError(
                f"max_retry_after must be a non-negative finite number, got {max_retry_after!r}"
            )

    def _ensure_sync_client(self) -> httpx.Client:
        if self._sync_closed:
            raise RuntimeError("client is closed")
        if self._sync_client is None:
            self._sync_client = httpx.Client()
            self._owns_sync = True
        return self._sync_client

    def _ensure_async_client(self) -> httpx.AsyncClient:
        if self._async_closed:
            raise RuntimeError("client is closed")
        if self._async_client is None:
            self._async_client = httpx.AsyncClient()
            self._owns_async = True
        return self._async_client

    def _resolve(self, opts: RequestOptions) -> _Call:
        """Merge per-call options with the client defaults.

        Reading self._max_retries and self._timeout directly in a dozen places
        is what made per-call overrides impossible; everything goes through here
        now.
        """
        timeout = opts.get("timeout")
        if timeout is None:
            timeout = self._timeout
        else:
            self._validate_timeout(timeout)
        retries = opts.get("max_retries")
        if retries is None:
            retries = self._max_retries
        else:
            self._validate_max_retries(retries)
        return _Call(
            timeout=timeout,
            max_retries=retries,
            deadline=opts.get("deadline"),
            headers=opts.get("headers"),
            error_in_body=opts.get("error_in_body", True),
        )

    def _build_url(self, path: str) -> str:
        """Join the path to the base url, escaping each segment.

        Identifiers come from callers and sometimes from other systems. Without
        escaping, a task id of "../keys" walks out of /api/v1 and sends the
        Authorization header to a different endpoint entirely. Dot segments are
        escaped explicitly — quote() leaves them alone, and httpx would resolve
        them away.
        """
        segments = []
        for segment in path.strip("/").split("/"):
            if not segment:
                continue
            if segment in (".", ".."):
                segment = segment.replace(".", "%2E")
            else:
                segment = quote(segment, safe="")
            segments.append(segment)
        return f"{self._base_url}/{'/'.join(segments)}"

    def _merge_headers(
        self, headers: Mapping[str, str] | None, *, content: bytes | None = None
    ) -> dict[str, str]:
        """Client headers with per-call ones on top.

        httpx.Headers matches case-insensitively, so a caller passing
        "user-agent" replaces the default instead of sending it twice.
        """
        merged = httpx.Headers(self._headers)
        if headers:
            merged.update(headers)
        if content is not None:
            merged.pop("content-type", None)
        return dict(merged)

    # --- retry policy ---

    def _should_retry_status(self, method: str, status: int) -> bool:
        if method in SAFE_METHODS:
            return status in RETRYABLE_SAFE_STATUSES
        if self._retry_unsafe_methods:
            return status in RETRYABLE_SAFE_STATUSES
        return status in RETRYABLE_UNSAFE_STATUSES

    def _should_retry_transport(self, method: str, exc: Exception) -> bool:
        if method in SAFE_METHODS:
            return True
        # For unsafe methods retry only if the connection was never established,
        # so the server cannot have processed the request.
        return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))

    def _wait(
        self, attempt: int, response: httpx.Response | None = None, *, deadline: float | None = None
    ) -> None:
        retry_after = (
            _retry_after_seconds(response, max_retry_after=self._max_retry_after)
            if response is not None
            else None
        )
        delay = (
            retry_after
            if retry_after is not None
            else (self._retry_backoff * (2**attempt) * random.uniform(0.5, 1.0))
        )
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeadlineExceededError("deadline exceeded during retry backoff")
            if delay >= remaining:
                time.sleep(remaining)
                raise DeadlineExceededError("deadline exceeded during retry backoff")
        time.sleep(delay)

    async def _await(
        self, attempt: int, response: httpx.Response | None = None, *, deadline: float | None = None
    ) -> None:
        import asyncio

        retry_after = (
            _retry_after_seconds(response, max_retry_after=self._max_retry_after)
            if response is not None
            else None
        )
        delay = (
            retry_after
            if retry_after is not None
            else (self._retry_backoff * (2**attempt) * random.uniform(0.5, 1.0))
        )
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeadlineExceededError("deadline exceeded during retry backoff")
            if delay >= remaining:
                await asyncio.sleep(remaining)
                raise DeadlineExceededError("deadline exceeded during retry backoff")
        await asyncio.sleep(delay)

    def _observe(
        self,
        *,
        method: str,
        url: str,
        status: int | None,
        elapsed: float,
        body: Any = None,
        label: str | None = None,
        streamed: bool = False,
        generation_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Account for one request, then log it.

        Deliberately not gated on the log level: accounting that only works
        when INFO logging happens to be on is accounting nobody can trust.
        """
        duplicate = False
        if generation_id:
            if generation_id in self._counted_generations:
                duplicate = True
            else:
                self._counted_generations[generation_id] = None
                if len(self._counted_generations) > 4096:
                    self._counted_generations.popitem(last=False)
        record = record_from(
            method=method,
            path=url,
            status=status,
            elapsed=elapsed,
            body=body,
            label=label,
            streamed=streamed,
            generation_id=generation_id,
            request_id=request_id,
            duplicate=duplicate,
        )
        self.usage.add(record)
        for tracker in self._trackers.get():
            tracker.add(record)
        for hook in tuple(self._usage_hooks):
            try:
                hook(record)
            except Exception:  # a callback must never break the call
                self._logger.warning("usage callback failed", exc_info=True)
        log_request(
            self._logger,
            method,
            url,
            elapsed=elapsed,
            status=status,
            tokens=record.total_tokens or None,
            cost=record.cost_rub,
        )

    def track(self, label: str | None = None) -> UsageTracker:
        """Start a tracker for the current context; see RouterAI.track()."""
        return UsageTracker(label=label)

    def push_tracker(self, tracker: UsageTracker) -> Token[tuple[UsageTracker, ...]]:
        return self._trackers.set((*self._trackers.get(), tracker))

    def pop_tracker(self, token: Token[tuple[UsageTracker, ...]]) -> None:
        self._trackers.reset(token)

    def current_label(self) -> str | None:
        for tracker in reversed(self._trackers.get()):
            if tracker.label:
                return tracker.label
        return None

    def on_usage(self, hook: UsageHook) -> Callable[[], None]:
        self._usage_hooks.append(hook)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._usage_hooks.remove(hook)

        return unsubscribe

    # --- sync ---

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        content: bytes | None = None,
        **opts: Unpack[RequestOptions],
    ) -> ResponseEnvelope:
        url = self._build_url(path)
        call = self._resolve(opts)
        request_headers = self._merge_headers(call.headers, content=content)
        client = self._ensure_sync_client()
        deadline = call.deadline

        last_exc: Exception | None = None
        for attempt in range(call.max_retries + 1):
            attempt_timeout = call.timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeadlineExceededError(f"deadline exceeded before attempt {attempt + 1}")
                attempt_timeout = min(attempt_timeout, remaining)
            started = time.monotonic()
            try:
                response = client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                    timeout=attempt_timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < call.max_retries and self._should_retry_transport(method, exc):
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    self._wait(attempt, deadline=deadline)
                    continue
                break
            if deadline is not None and time.monotonic() >= deadline:
                response.close()
                raise DeadlineExceededError(
                    f"deadline exceeded while waiting for attempt {attempt + 1}"
                )
            elapsed = time.monotonic() - started
            if (
                self._should_retry_status(method, response.status_code)
                and attempt < call.max_retries
            ):
                response.read()
                response.close()
                self._logger.warning(
                    "retry %s %s: status=%s (attempt %d)",
                    method,
                    url,
                    response.status_code,
                    attempt + 1,
                )
                self._wait(attempt, response, deadline=deadline)
                continue
            envelope = ResponseEnvelope(response)
            self._observe(
                method=method,
                url=url,
                status=envelope.status_code,
                elapsed=elapsed,
                body=envelope.body,
                label=self.current_label(),
                generation_id=envelope.generation_id,
                request_id=envelope.request_id,
            )
            _raise_for_status(
                envelope.status_code,
                envelope.body,
                headers=envelope.headers,
                error_in_body=call.error_in_body,
            )
            return envelope

        raise _transport_error(
            f"{method} {url} failed: {mask_key(last_exc)}", last_exc
        ) from last_exc

    @contextmanager
    def stream_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        content: bytes | None = None,
        **opts: Unpack[RequestOptions],
    ) -> Iterator[StreamEnvelope]:
        """Open a streaming response.

        Retries happen only before the response is handed to the caller.
        Once the first chunk is delivered, transport errors are re-raised
        without retry (the request may already be billed).
        """
        url = self._build_url(path)
        call = self._resolve(opts)
        request_headers = self._merge_headers(call.headers, content=content)
        client = self._ensure_sync_client()

        yielded = False
        last_exc: Exception | None = None
        for attempt in range(call.max_retries + 1):
            started = time.monotonic()
            try:
                with client.stream(
                    method,
                    url,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                    timeout=call.timeout,
                ) as response:
                    if (
                        self._should_retry_status(method, response.status_code)
                        and attempt < call.max_retries
                    ):
                        response.read()
                        self._logger.warning(
                            "retry %s %s: status=%s (attempt %d)",
                            method,
                            url,
                            response.status_code,
                            attempt + 1,
                        )
                        self._wait(attempt, response)
                        continue
                    if response.status_code >= 400:
                        _raise_for_status(
                            response.status_code,
                            _stream_body(response),
                            headers=response.headers,
                        )
                    yielded = True
                    envelope = StreamEnvelope(response)
                    try:
                        yield envelope
                    finally:
                        # finally, not after the yield: a caller that breaks out
                        # early still paid for what the stream produced.
                        self._observe(
                            method=method,
                            url=url,
                            status=response.status_code,
                            elapsed=time.monotonic() - started,
                            body={"usage": envelope.usage_payload, "model": envelope.model}
                            if envelope.usage_payload
                            else None,
                            label=self.current_label(),
                            streamed=True,
                            generation_id=envelope.generation_id,
                        )
                    return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if yielded:
                    raise
                last_exc = exc
                if attempt < call.max_retries and self._should_retry_transport(method, exc):
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    self._wait(attempt)
                    continue
                break
        raise _transport_error(
            f"{method} {url} failed: {mask_key(last_exc)}", last_exc
        ) from last_exc

    # --- async ---

    async def arequest(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        content: bytes | None = None,
        **opts: Unpack[RequestOptions],
    ) -> ResponseEnvelope:
        url = self._build_url(path)
        call = self._resolve(opts)
        request_headers = self._merge_headers(call.headers, content=content)
        client = self._ensure_async_client()
        deadline = call.deadline

        last_exc: Exception | None = None
        for attempt in range(call.max_retries + 1):
            attempt_timeout = call.timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeadlineExceededError(f"deadline exceeded before attempt {attempt + 1}")
                attempt_timeout = min(attempt_timeout, remaining)
            started = time.monotonic()
            try:
                request = client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                    timeout=attempt_timeout,
                )
                if deadline is None:
                    response = await request
                else:
                    import asyncio

                    try:
                        response = await asyncio.wait_for(request, timeout=remaining)
                    except asyncio.TimeoutError as exc:
                        raise DeadlineExceededError(
                            f"deadline exceeded while waiting for attempt {attempt + 1}"
                        ) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < call.max_retries and self._should_retry_transport(method, exc):
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    await self._await(attempt, deadline=deadline)
                    continue
                break
            if deadline is not None and time.monotonic() >= deadline:
                await response.aclose()
                raise DeadlineExceededError(
                    f"deadline exceeded while waiting for attempt {attempt + 1}"
                )
            elapsed = time.monotonic() - started
            if (
                self._should_retry_status(method, response.status_code)
                and attempt < call.max_retries
            ):
                await response.aread()
                await response.aclose()
                self._logger.warning(
                    "retry %s %s: status=%s (attempt %d)",
                    method,
                    url,
                    response.status_code,
                    attempt + 1,
                )
                await self._await(attempt, response, deadline=deadline)
                continue
            envelope = ResponseEnvelope(response)
            self._observe(
                method=method,
                url=url,
                status=envelope.status_code,
                elapsed=elapsed,
                body=envelope.body,
                label=self.current_label(),
                generation_id=envelope.generation_id,
                request_id=envelope.request_id,
            )
            _raise_for_status(
                envelope.status_code,
                envelope.body,
                headers=envelope.headers,
                error_in_body=call.error_in_body,
            )
            return envelope

        raise _transport_error(
            f"{method} {url} failed: {mask_key(last_exc)}", last_exc
        ) from last_exc

    @asynccontextmanager
    async def astream_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        content: bytes | None = None,
        **opts: Unpack[RequestOptions],
    ) -> AsyncIterator[StreamEnvelope]:
        url = self._build_url(path)
        call = self._resolve(opts)
        request_headers = self._merge_headers(call.headers, content=content)
        client = self._ensure_async_client()

        yielded = False
        last_exc: Exception | None = None
        for attempt in range(call.max_retries + 1):
            started = time.monotonic()
            try:
                async with client.stream(
                    method,
                    url,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                    timeout=call.timeout,
                ) as response:
                    if (
                        self._should_retry_status(method, response.status_code)
                        and attempt < call.max_retries
                    ):
                        await response.aread()
                        self._logger.warning(
                            "retry %s %s: status=%s (attempt %d)",
                            method,
                            url,
                            response.status_code,
                            attempt + 1,
                        )
                        await self._await(attempt, response)
                        continue
                    if response.status_code >= 400:
                        # read the (async) body before mapping so the sync
                        # text/json accessors work on unread streams too
                        await response.aread()
                        _raise_for_status(
                            response.status_code,
                            _stream_body(response),
                            headers=response.headers,
                        )
                    yielded = True
                    envelope = StreamEnvelope(response)
                    try:
                        yield envelope
                    finally:
                        # finally, not after the yield: a caller that breaks out
                        # early still paid for what the stream produced.
                        self._observe(
                            method=method,
                            url=url,
                            status=response.status_code,
                            elapsed=time.monotonic() - started,
                            body={"usage": envelope.usage_payload, "model": envelope.model}
                            if envelope.usage_payload
                            else None,
                            label=self.current_label(),
                            streamed=True,
                            generation_id=envelope.generation_id,
                        )
                    return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if yielded:
                    raise
                last_exc = exc
                if attempt < call.max_retries and self._should_retry_transport(method, exc):
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    await self._await(attempt)
                    continue
                break
        raise _transport_error(
            f"{method} {url} failed: {mask_key(last_exc)}", last_exc
        ) from last_exc

    # --- helpers ---

    @staticmethod
    def _json(response: httpx.Response | ResponseEnvelope) -> dict[str, Any]:
        if isinstance(response, ResponseEnvelope):
            return response.json()
        return dict(response.json())

    def get(
        self, path: str, *, params: dict[str, Any] | None = None, **opts: Unpack[RequestOptions]
    ) -> ResponseEnvelope:
        return self.request("GET", path, params=params, **opts)

    def post(
        self, path: str, *, json: Any = None, **opts: Unpack[RequestOptions]
    ) -> ResponseEnvelope:
        return self.request("POST", path, json=json, **opts)

    async def aget(
        self, path: str, *, params: dict[str, Any] | None = None, **opts: Unpack[RequestOptions]
    ) -> ResponseEnvelope:
        return await self.arequest("GET", path, params=params, **opts)

    async def apost(
        self, path: str, *, json: Any = None, **opts: Unpack[RequestOptions]
    ) -> ResponseEnvelope:
        return await self.arequest("POST", path, json=json, **opts)

    def close(self) -> None:
        """Close the sync side. Requests after this raise RuntimeError.

        Only the sync transport is affected: the async one has its own
        lifecycle and its own :meth:`aclose`. An injected transport is never
        closed here, but it is not silently replaced either — before this the
        next request quietly opened a brand new pool, discarding proxy, mTLS
        and timeout settings the caller had configured.
        """
        if self._sync_client is not None and self._owns_sync:
            self._sync_client.close()
        self._sync_closed = True

    async def aclose(self) -> None:
        """Close the async side. Requests after this raise RuntimeError."""
        if self._async_client is not None and self._owns_async:
            await self._async_client.aclose()
        self._async_closed = True
