from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx

from .errors import (
    APIStatusError,
    AuthenticationError,
    ConfigurationError,
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


def _raise_for_status(status: int, message: str, body: Any = None) -> None:
    if status < 400:
        return
    if status == 503 and "provider" in message.lower() and "available" in message.lower():
        raise NoProviderError(message)
    error_cls = _STATUS_TO_ERROR.get(status, APIStatusError)
    if error_cls is APIStatusError:
        raise APIStatusError(message, status, body)
    raise error_cls(message)


def _parse_error_payload(payload: Any) -> str:
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str) and error:
        return error
    return ""


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.text
    except httpx.ResponseNotRead:
        body = response.read().decode("utf-8", errors="replace")
    message = ""
    try:
        payload = json.loads(body)
        message = _parse_error_payload(payload)
    except ValueError:
        message = body.strip()
    if not message:
        message = f"HTTP {response.status_code}"
    return message


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            import email.utils

            date = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if date.tzinfo is None:
            return None
        return max(0.0, date.timestamp() - time.time())


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

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._retry_unsafe_methods = retry_unsafe_methods
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

    def _wait(self, attempt: int, response: httpx.Response | None = None) -> None:
        retry_after = _retry_after_seconds(response) if response is not None else None
        if retry_after is not None:
            time.sleep(retry_after)
            return
        jitter = random.uniform(0.5, 1.0)
        time.sleep(self._retry_backoff * (2**attempt) * jitter)

    async def _await(self, attempt: int, response: httpx.Response | None = None) -> None:
        import asyncio

        retry_after = _retry_after_seconds(response) if response is not None else None
        if retry_after is not None:
            await asyncio.sleep(retry_after)
            return
        jitter = random.uniform(0.5, 1.0)
        await asyncio.sleep(self._retry_backoff * (2**attempt) * jitter)

    def _log_result(self, response: httpx.Response, method: str, url: str, elapsed: float) -> None:
        tokens = cost = None
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
    ) -> httpx.Response:
        url = self._build_url(path)
        request_headers = self._merge_headers(headers, content=content)
        client = self._ensure_sync_client()

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.monotonic()
            try:
                response = client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                    timeout=self._timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self._max_retries and self._should_retry_transport(method, exc):
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    self._wait(attempt)
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
            self._log_result(response, method, url, elapsed)
            _raise_for_status(
                response.status_code,
                _error_message(response) if response.status_code >= 400 else "",
            )
            return response

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
                    timeout=self._timeout,
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
                    _raise_for_status(
                        response.status_code,
                        _error_message(response) if response.status_code >= 400 else "",
                    )
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
    ) -> httpx.Response:
        url = self._build_url(path)
        request_headers = self._merge_headers(headers, content=content)
        client = self._ensure_async_client()

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.monotonic()
            try:
                response = await client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                    timeout=self._timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self._max_retries and self._should_retry_transport(method, exc):
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    await self._await(attempt)
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
            self._log_result(response, method, url, elapsed)
            _raise_for_status(
                response.status_code,
                _error_message(response) if response.status_code >= 400 else "",
            )
            return response

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
                    timeout=self._timeout,
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
                    _raise_for_status(
                        response.status_code,
                        _error_message(response) if response.status_code >= 400 else "",
                    )
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
    def _json(response: httpx.Response) -> dict[str, Any]:
        return dict(response.json())

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    async def aget(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self.arequest("GET", path, **kwargs)

    async def apost(self, path: str, **kwargs: Any) -> httpx.Response:
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
