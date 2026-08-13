from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from json import loads as _json_loads
from typing import Any

import httpx

from .errors import (
    APIStatusError,
    AuthenticationError,
    ConfigurationError,
    DeadlineExceededError,
    InsufficientFundsError,
    NoProviderError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RequestError,
    RouterAIError,
)
from .logging import log_request, mask_key

DEFAULT_BASE_URL = "https://routerai.ru/api/v1"
DEFAULT_TIMEOUT = 60.0

# Statuses retried for idempotent (safe) methods.
RETRYABLE_SAFE_STATUSES = {408, 425, 429, 500, 502, 503, 504}
# Statuses retried for non-idempotent methods (POST/PATCH/DELETE).
# Only 429 is safe: the request was rejected before being processed.
RETRYABLE_UNSAFE_STATUSES = {429}
SAFE_METHODS = {"GET", "HEAD"}

_STATUS_TO_ERROR: dict[int, type[RouterAIError]] = {
    401: AuthenticationError,
    402: InsufficientFundsError,
    403: PermissionDeniedError,
    404: NotFoundError,
    429: RateLimitError,
}


def _parse_error_payload(payload: Any) -> str:
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str) and error:
        return error
    return ""


def _error_message_from_body(status: int, body: Any) -> str:
    """Extract the error message from an already decoded body, or ''.

    For HTTP 200 responses only JSON error payloads count — plain-text or
    binary 200 bodies (e.g. TTS audio) are not errors.
    """
    message = _parse_error_payload(body)
    if not message and isinstance(body, str) and status >= 400:
        message = body.strip()
    return message


def _raise_for_status(status: int, message: str, body: Any) -> None:
    if status < 400:
        # RouterAI sometimes wraps provider errors in an HTTP 200 body
        if message:
            raise APIStatusError(message, status, body)
        return
    text = message or f"HTTP {status}"
    if status == 503 and "provider" in text.lower() and "available" in text.lower():
        raise NoProviderError(text)
    error_cls = _STATUS_TO_ERROR.get(status, APIStatusError)
    if error_cls is APIStatusError:
        raise APIStatusError(text, status, body)
    raise error_cls(text)


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
            try:
                self._body = dict(response.json())
                self._is_json = True
            except ValueError:
                pass

    @property
    def body(self) -> Any:
        """The decoded body (dict for JSON responses, bytes otherwise)."""
        if self._body is None:
            if self._is_json:
                try:
                    self._body = dict(self._response.json())
                except ValueError:
                    self._is_json = False
                    self._body = self._response.content
            else:
                self._body = self._response.content
        return self._body

    def json(self) -> dict[str, Any]:
        body = self.body
        if not isinstance(body, dict):
            raise ValueError("response is not a JSON object")
        return dict(body)

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


def _error_message_for_stream(response: httpx.Response) -> str:
    """Error message for streamed responses (reads the body once)."""
    if (
        response.status_code < 400
        and "json" not in response.headers.get("content-type", "").lower()
    ):
        return ""
    try:
        body = response.text
    except httpx.ResponseNotRead:
        body = response.read().decode("utf-8", errors="replace")
    message = _parse_error_payload(json.loads(body)) if _looks_json(body) else ""
    if not message and response.status_code >= 400:
        message = body.strip()
    return message


def _looks_json(body: str) -> bool:
    try:
        json.loads(body)
    except ValueError:
        return False
    return True


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
        self._headers = {
            "Content-Type": "application/json",
            **(default_headers or {}),
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._sync_client = http_client if isinstance(http_client, httpx.Client) else None
        self._async_client = async_http_client
        self._owns_sync = self._sync_client is None
        self._owns_async = self._async_client is None

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @staticmethod
    def _validate_config(
        timeout: float, max_retries: int, retry_backoff: float, max_retry_after: float
    ) -> None:
        import math

        if not math.isfinite(timeout) or timeout <= 0:
            raise ConfigurationError(f"timeout must be a positive finite number, got {timeout!r}")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ConfigurationError(
                f"max_retries must be a non-negative integer, got {max_retries!r}"
            )
        if not math.isfinite(retry_backoff) or retry_backoff < 0:
            raise ConfigurationError(
                f"retry_backoff must be a non-negative finite number, got {retry_backoff!r}"
            )
        if not math.isfinite(max_retry_after) or max_retry_after < 0:
            raise ConfigurationError(
                f"max_retry_after must be a non-negative finite number, got {max_retry_after!r}"
            )

    def _ensure_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client()
            self._owns_sync = True
        return self._sync_client

    def _ensure_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient()
            self._owns_async = True
        return self._async_client

    def _build_url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _merge_headers(
        self, headers: dict[str, str] | None, *, content: bytes | None = None
    ) -> dict[str, str]:
        merged = dict(self._headers)
        if headers:
            merged.update(headers)
        if content is not None:
            merged.pop("Content-Type", None)
        return merged

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
            delay = min(delay, remaining)
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
            delay = min(delay, remaining)
        await asyncio.sleep(delay)

    def _log_result(
        self, response: httpx.Response | ResponseEnvelope, method: str, url: str, elapsed: float
    ) -> None:
        if not self._logger.isEnabledFor(logging.INFO):
            return
        tokens = cost = None
        if isinstance(response, ResponseEnvelope):
            body = response.body
            if isinstance(body, dict):
                usage = body.get("usage")
                if isinstance(usage, dict):
                    tokens = usage.get("total_tokens")
                    cost = usage.get("cost")
        else:
            content_type = response.headers.get("content-type", "").lower()
            if "json" in content_type:
                try:
                    payload = response.json()
                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        tokens = usage.get("total_tokens")
                        cost = usage.get("cost")
                except ValueError:
                    pass
        log_request(
            self._logger,
            method,
            url,
            elapsed=elapsed,
            status=response.status_code,
            tokens=tokens,
            cost=cost,
        )

    # --- sync ---

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        deadline: float | None = None,
    ) -> ResponseEnvelope:
        url = self._build_url(path)
        request_headers = self._merge_headers(headers, content=content)
        client = self._ensure_sync_client()

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            attempt_timeout = self._timeout if timeout is None else timeout
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
                if attempt < self._max_retries and self._should_retry_transport(method, exc):
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    self._wait(attempt, deadline=deadline)
                    continue
                break
            elapsed = time.monotonic() - started
            if (
                self._should_retry_status(method, response.status_code)
                and attempt < self._max_retries
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
                self._wait(attempt, response)
                continue
            envelope = ResponseEnvelope(response)
            self._log_result(envelope, method, url, elapsed)
            _raise_for_status(
                envelope.status_code,
                _error_message_from_body(envelope.status_code, envelope.body),
                envelope.body,
            )
            return envelope

        raise RequestError(f"{method} {url} failed: {mask_key(last_exc)}")

    @contextmanager
    def stream_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Iterator[httpx.Response]:
        """Open a streaming response.

        Retries happen only before the response is handed to the caller.
        Once the first chunk is delivered, transport errors are re-raised
        without retry (the request may already be billed).
        """
        url = self._build_url(path)
        request_headers = self._merge_headers(headers, content=content)
        client = self._ensure_sync_client()

        yielded = False
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.monotonic()
            try:
                with client.stream(
                    method,
                    url,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                    timeout=self._timeout if timeout is None else timeout,
                ) as response:
                    if (
                        self._should_retry_status(method, response.status_code)
                        and attempt < self._max_retries
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
                    log_request(
                        self._logger,
                        method,
                        url,
                        elapsed=time.monotonic() - started,
                        status=response.status_code,
                    )
                    if response.status_code >= 400:
                        message = _error_message_for_stream(response)
                        body: Any = None
                        try:
                            body = _json_loads(response.text)
                        except ValueError:
                            body = response.text
                        _raise_for_status(response.status_code, message, body)
                    yielded = True
                    yield response
                    return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if yielded:
                    raise
                last_exc = exc
                if attempt < self._max_retries and self._should_retry_transport(method, exc):
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    self._wait(attempt)
                    continue
                break
        raise RequestError(f"{method} {url} failed: {mask_key(last_exc)}")

    # --- async ---

    async def arequest(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        deadline: float | None = None,
    ) -> ResponseEnvelope:
        url = self._build_url(path)
        request_headers = self._merge_headers(headers, content=content)
        client = self._ensure_async_client()

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            attempt_timeout = self._timeout if timeout is None else timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeadlineExceededError(f"deadline exceeded before attempt {attempt + 1}")
                attempt_timeout = min(attempt_timeout, remaining)
            started = time.monotonic()
            try:
                response = await client.request(
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
                if attempt < self._max_retries and self._should_retry_transport(method, exc):
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    await self._await(attempt, deadline=deadline)
                    continue
                break
            elapsed = time.monotonic() - started
            if (
                self._should_retry_status(method, response.status_code)
                and attempt < self._max_retries
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
                await self._await(attempt, response)
                continue
            envelope = ResponseEnvelope(response)
            self._log_result(envelope, method, url, elapsed)
            _raise_for_status(
                envelope.status_code,
                _error_message_from_body(envelope.status_code, envelope.body),
                envelope.body,
            )
            return envelope

        raise RequestError(f"{method} {url} failed: {mask_key(last_exc)}")

    @asynccontextmanager
    async def astream_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[httpx.Response]:
        url = self._build_url(path)
        request_headers = self._merge_headers(headers, content=content)
        client = self._ensure_async_client()

        yielded = False
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.monotonic()
            try:
                async with client.stream(
                    method,
                    url,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                    timeout=self._timeout if timeout is None else timeout,
                ) as response:
                    if (
                        self._should_retry_status(method, response.status_code)
                        and attempt < self._max_retries
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
                    log_request(
                        self._logger,
                        method,
                        url,
                        elapsed=time.monotonic() - started,
                        status=response.status_code,
                    )
                    if response.status_code >= 400:
                        # read the (async) body before mapping so the sync
                        # text/json accessors work on unread streams too
                        await response.aread()
                        message = _error_message_for_stream(response)
                        body: Any = None
                        try:
                            body = _json_loads(response.text)
                        except ValueError:
                            body = response.text
                        _raise_for_status(response.status_code, message, body)
                    yielded = True
                    yield response
                    return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if yielded:
                    raise
                last_exc = exc
                if attempt < self._max_retries and self._should_retry_transport(method, exc):
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    await self._await(attempt)
                    continue
                break
        raise RequestError(f"{method} {url} failed: {mask_key(last_exc)}")

    # --- helpers ---

    @staticmethod
    def _json(response: httpx.Response | ResponseEnvelope) -> dict[str, Any]:
        if isinstance(response, ResponseEnvelope):
            return response.json()
        return dict(response.json())

    def get(self, path: str, **kwargs: Any) -> ResponseEnvelope:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> ResponseEnvelope:
        return self.request("POST", path, **kwargs)

    async def aget(self, path: str, **kwargs: Any) -> ResponseEnvelope:
        return await self.arequest("GET", path, **kwargs)

    async def apost(self, path: str, **kwargs: Any) -> ResponseEnvelope:
        return await self.arequest("POST", path, **kwargs)

    def close(self) -> None:
        if self._sync_client is not None and self._owns_sync:
            self._sync_client.close()
        self._sync_client = None
        self._owns_sync = True

    async def aclose(self) -> None:
        if self._async_client is not None and self._owns_async:
            await self._async_client.aclose()
        self._async_client = None
        self._owns_async = True
