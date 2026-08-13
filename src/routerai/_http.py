from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx

from .errors import (
    APIStatusError,
    AuthenticationError,
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

RETRYABLE_STATUSES = {408, 409, 429, 500, 502, 503, 504}
_STATUS_TO_ERROR: dict[int, type[RouterAIError]] = {
    401: AuthenticationError,
    402: InsufficientFundsError,
    403: PermissionDeniedError,
    404: NotFoundError,
    429: RateLimitError,
}


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status < 400:
        return
    message = _extract_error_message(response)
    if status == 503 and "provider" in message.lower() and "available" in message.lower():
        raise NoProviderError(message)
    error_cls = _STATUS_TO_ERROR.get(status, APIStatusError)
    if error_cls is APIStatusError:
        raise APIStatusError(message, status, response.text)
    raise error_cls(message)


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    error = payload.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str) and error:
        return error
    return f"HTTP {response.status_code}: {response.text.strip()[:500]}"


class HTTPClient:
    """Thin httpx wrapper: base URL, auth, retries, logging, error mapping."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        logger: logging.Logger | str | None = None,
        http_client: httpx.Client | httpx.AsyncClient | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
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
        self._client = http_client
        self._owns_client = http_client is None

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def _ensure_client(self, async_mode: bool) -> httpx.Client | httpx.AsyncClient:
        if self._client is None:
            if async_mode:
                self._client = httpx.AsyncClient()
            else:
                self._client = httpx.Client()
        return self._client

    def _build_url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _should_retry(self, response: httpx.Response) -> bool:
        if response.status_code not in RETRYABLE_STATUSES:
            return False
        try:
            payload = response.json()
        except ValueError:
            return True
        error = payload.get("error")
        return not (isinstance(error, dict) and error.get("type") == "insufficient_funds")

    def _sleep(self, attempt: int) -> None:
        time.sleep(self._retry_backoff * (2**attempt))

    async def _asleep(self, attempt: int) -> None:
        import asyncio

        await asyncio.sleep(self._retry_backoff * (2**attempt))

    def _log_result(
        self,
        response: httpx.Response,
        method: str,
        url: str,
        elapsed: float,
        *,
        streaming: bool = False,
    ) -> None:
        if streaming:
            log_request(self._logger, method, url, elapsed=elapsed, status=response.status_code)
            return
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

    def _merge_headers(self, headers: dict[str, str] | None) -> dict[str, str]:
        merged = dict(self._headers)
        if headers:
            merged.update(headers)
        return merged

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
        request_headers = self._merge_headers(headers)
        if content is not None:
            request_headers.pop("Content-Type", None)
        client = self._ensure_client(async_mode=False)
        assert isinstance(client, httpx.Client)

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
                if attempt < self._max_retries:
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    self._sleep(attempt)
                    continue
                break
            elapsed = time.monotonic() - started
            if self._should_retry(response) and attempt < self._max_retries:
                response.read()
                response.close()
                self._logger.warning(
                    "retry %s %s: status=%s (attempt %d)",
                    method,
                    url,
                    response.status_code,
                    attempt + 1,
                )
                self._sleep(attempt)
                continue
            self._log_result(response, method, url, elapsed)
            _raise_for_status(response)
            return response

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
        request_headers = self._merge_headers(headers)
        if content is not None:
            request_headers.pop("Content-Type", None)
        client = self._ensure_client(async_mode=True)
        assert isinstance(client, httpx.AsyncClient)

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
                if attempt < self._max_retries:
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    await self._asleep(attempt)
                    continue
                break
            elapsed = time.monotonic() - started
            if self._should_retry(response) and attempt < self._max_retries:
                await response.aread()
                await response.aclose()
                self._logger.warning(
                    "retry %s %s: status=%s (attempt %d)",
                    method,
                    url,
                    response.status_code,
                    attempt + 1,
                )
                await self._asleep(attempt)
                continue
            self._log_result(response, method, url, elapsed)
            _raise_for_status(response)
            return response

        raise RequestError(f"{method} {url} failed: {mask_key(last_exc)}")

    # --- helpers ---

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    async def aget(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self.arequest("GET", path, **kwargs)

    async def apost(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self.arequest("POST", path, **kwargs)

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        return dict(response.json())

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
        url = self._build_url(path)
        request_headers = self._merge_headers(headers)
        client = self._ensure_client(async_mode=False)
        assert isinstance(client, httpx.Client)
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
                    if response.status_code in RETRYABLE_STATUSES and attempt < self._max_retries:
                        self._logger.warning(
                            "retry %s %s: status=%s (attempt %d)",
                            method,
                            url,
                            response.status_code,
                            attempt + 1,
                        )
                        self._sleep(attempt)
                        continue
                    self._log_result(
                        response, method, url, time.monotonic() - started, streaming=True
                    )
                    _raise_for_status(response)
                    yield response
                    return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    self._sleep(attempt)
                    continue
                break
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
        request_headers = self._merge_headers(headers)
        client = self._ensure_client(async_mode=True)
        assert isinstance(client, httpx.AsyncClient)
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
                    if response.status_code in RETRYABLE_STATUSES and attempt < self._max_retries:
                        self._logger.warning(
                            "retry %s %s: status=%s (attempt %d)",
                            method,
                            url,
                            response.status_code,
                            attempt + 1,
                        )
                        await self._asleep(attempt)
                        continue
                    self._log_result(
                        response, method, url, time.monotonic() - started, streaming=True
                    )
                    _raise_for_status(response)
                    yield response
                    return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    self._logger.warning(
                        "retry %s %s after %s (attempt %d)", method, url, mask_key(exc), attempt + 1
                    )
                    await self._asleep(attempt)
                    continue
                break
        raise RequestError(f"{method} {url} failed: {mask_key(last_exc)}")

    def close(self) -> None:
        if isinstance(self._client, httpx.Client):
            self._client.close()
        self._client = None

    async def aclose(self) -> None:
        if isinstance(self._client, httpx.AsyncClient):
            await self._client.aclose()
        self._client = None
