"""Turning a RouterAI error body into a typed exception.

RouterAI reports failures in several shapes, and the transport status often
disagrees with the real cause. All three were observed against the live API:

    HTTP 200  {"error": "{\\"error\\":{\\"message\\":...,\\"code\\":429,...}}"}
    HTTP 200  {"error": "{\\"error\\":{...,\\"code\\":400,...}}"}
    HTTP 503  {"error": "{\\"error\\":{\\"message\\":\\"HTTP 429: {...}\\",\\"code\\":429}}"}

So the code that matters can sit one or two JSON strings deep, while the HTTP
line says 200 or 503. This module unwraps that, decides which status is the
effective one, and picks the exception class. It is the single place where
those decisions are made — buffered responses and SSE events both go through it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from .errors import (
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ErrorInfo,
    InsufficientFundsError,
    NoProviderError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RouterAIError,
    ServerError,
    UnprocessableEntityError,
)

# A 2xx body that claims failure without naming a code: the gateway broke and
# did not say how, which is what 502 means.
SYNTHETIC_UPSTREAM_STATUS = 502

DONE_MARKER = "[DONE]"

_MAX_UNWRAP_DEPTH = 4
_HTTP_PREFIX = re.compile(r"^\s*HTTP[ :/]?(\d{3})\b")

_STATUS_TO_ERROR: dict[int, type[APIStatusError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    402: InsufficientFundsError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
}

# Machine-readable markers beat prose: the previous substring match on
# "provider" + "available" stopped working the moment a message was translated.
NO_PROVIDER_MARKERS = frozenset(
    {
        "no_provider_available",
        "no_providers_available",
        "provider_unavailable",
        "no_endpoints_found",
        "no_allowed_providers",
    }
)


def _loads(text: str) -> Any | None:
    """Decode a JSON string, but only if it really looks like JSON.

    Without the shape check a plain message such as "Model 'x' not found"
    would be fed to the parser on every error.
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except ValueError:
        return None


def _unwrap(node: Any) -> list[Any]:
    """Peel JSON-in-a-string and nested ``error`` keys, outermost first."""
    chain: list[Any] = []
    for _ in range(_MAX_UNWRAP_DEPTH):
        chain.append(node)
        if isinstance(node, str):
            decoded = _loads(node)
            if decoded is None:
                break
            node = decoded
            continue
        if isinstance(node, Mapping) and isinstance(node.get("error"), (dict, str)):
            node = node["error"]
            continue
        break
    return chain


def _status_in(mapping: Mapping[str, Any]) -> int | None:
    """A status-looking code, if the mapping carries one.

    Only integers in the HTTP range count: providers also use ``code`` for
    slugs ("context_length_exceeded") and for trace ids.
    """
    for key in ("status_code", "status", "code"):
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    return None


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return ""


def extract_error(body: Any, *, status: int, error_in_body: bool = True) -> ErrorInfo | None:
    """Describe the failure in ``body``, or None when it is not a failure.

    Args:
        body: decoded response body (mapping, text or bytes).
        status: the HTTP status the body arrived with.
        error_in_body: whether an ``error`` field inside a successful response
            means the request failed. False for video polling, where ``error``
            describes the state of the task rather than the HTTP call.
    """
    if not isinstance(body, Mapping):
        text = _text_of(body).strip()
        if status >= 400:
            return ErrorInfo(message=text[:2048] or f"HTTP {status}")
        return None

    raw = body.get("error")
    if not raw:
        return ErrorInfo(message=f"HTTP {status}", error=body) if status >= 400 else None
    if status < 400 and not error_in_body:
        return None

    chain = _unwrap(raw)
    innermost_first = [node for node in reversed(chain) if isinstance(node, Mapping)]

    provider_code: int | None = None
    message = ""
    error_type: str | None = None
    error_code: str | None = None
    metadata: dict[str, Any] | None = None

    for node in innermost_first:
        if provider_code is None:
            provider_code = _status_in(node)
        if not message and isinstance(node.get("message"), str):
            message = node["message"]
        if error_type is None and isinstance(node.get("type"), str):
            error_type = node["type"]
        if error_code is None:
            code = node.get("code")
            if isinstance(code, str) and not code.isdigit():
                error_code = code
        if metadata is None and isinstance(node.get("metadata"), Mapping):
            metadata = dict(node["metadata"])

    if provider_code is None:
        provider_code = _status_in(body)
    if not message:
        message = next((node for node in reversed(chain) if isinstance(node, str)), "")
    if provider_code is None and message:
        match = _HTTP_PREFIX.match(message)
        if match:
            provider_code = int(match.group(1))

    provider_message = None
    provider_name = None
    if metadata:
        raw_text = metadata.get("raw")
        provider_message = raw_text if isinstance(raw_text, str) else None
        name = metadata.get("provider_name")
        provider_name = name if isinstance(name, str) else None

    return ErrorInfo(
        message=message or f"HTTP {status}",
        provider_code=provider_code,
        error_type=error_type,
        error_code=error_code,
        provider_name=provider_name,
        provider_message=provider_message,
        metadata=metadata,
        error=chain[-1] if chain else raw,
    )


def resolve_status(
    http_status: int, provider_code: int | None
) -> tuple[int, Literal["http", "provider", "synthesized"]]:
    """The effective status and where it came from.

    A 4xx is produced by the gateway itself (bad key, unknown model) and stays
    authoritative. A 2xx cannot describe a failure, and RouterAI's own 5xx is a
    wrapper around whatever the provider said — in both cases the body wins.
    """
    wrapped = http_status < 400 or http_status >= 500
    if provider_code is not None and 400 <= provider_code <= 599 and wrapped:
        return provider_code, "provider"
    if http_status >= 400:
        return http_status, "http"
    return SYNTHETIC_UPSTREAM_STATUS, "synthesized"


def _is_no_provider(status: int, info: ErrorInfo) -> bool:
    slug = (info.error_code or info.error_type or "").strip().lower()
    if slug in NO_PROVIDER_MARKERS:
        return True
    # Wrapped 503s already resolved to their real code above, so a 503 that
    # still reads as 503 means the router found nobody to serve the request.
    return status == 503 and info.provider_code is None


def error_class_for(status: int, info: ErrorInfo) -> type[APIStatusError]:
    if _is_no_provider(status, info):
        return NoProviderError
    known = _STATUS_TO_ERROR.get(status)
    if known is not None:
        return known
    return ServerError if status >= 500 else APIStatusError


def build_error(
    *,
    http_status: int,
    body: Any,
    headers: Mapping[str, str] | None = None,
    error_in_body: bool = True,
) -> APIStatusError | None:
    """Build the exception for this response, or None when it is not an error."""
    info = extract_error(body, status=http_status, error_in_body=error_in_body)
    if info is None:
        return None
    status, source = resolve_status(http_status, info.provider_code)
    error_cls = error_class_for(status, info)
    return error_cls(
        info.message,
        status,
        body,
        http_status=http_status,
        provider_code=info.provider_code,
        status_source=source,
        error_info=info,
        request_id=_header(headers, "X-Request-Id", "Request-Id"),
        generation_id=_header(headers, "X-Generation-Id"),
    )


def _header(headers: Mapping[str, str] | None, *names: str) -> str | None:
    if not headers:
        return None
    for name in names:
        value = headers.get(name)
        if value:
            return value
    return None


def parse_stream_event(data: str, *, http_status: int = 200) -> dict[str, Any] | None:
    """Decode one SSE ``data:`` payload, or None when it carries no event.

    Raises the typed error if the event carries one — the same unwrapping the
    buffered path uses, so a rate limit reads as RateLimitError whether it
    arrived in a body or in a stream. The end-of-stream marker is the caller's
    business; here an empty payload is simply a keep-alive.
    """
    if not data:
        return None
    try:
        payload = json.loads(data)
    except ValueError as exc:
        raise RouterAIError(f"unparsable SSE line: {data!r}") from exc
    if not isinstance(payload, Mapping):
        raise RouterAIError(f"unexpected SSE payload: {data!r}")
    error = build_error(http_status=http_status, body=payload)
    if error is not None:
        raise error
    return dict(payload)
