from __future__ import annotations

from typing import Any


class RouterAIError(Exception):
    """Base class for all RouterAI client errors."""


class AuthenticationError(RouterAIError):
    """Raised on 401 — the API key is missing, invalid, or revoked."""


class InsufficientFundsError(RouterAIError):
    """Raised on 402 — not enough funds on the balance."""


class PermissionDeniedError(RouterAIError):
    """Raised on 403 — the key has no access to the resource."""


class NotFoundError(RouterAIError):
    """Raised on 404 — the resource does not exist."""


class RateLimitError(RouterAIError):
    """Raised on 429 — too many requests."""


class NoProviderError(RouterAIError):
    """Raised on 503 when no provider is available for the requested model."""


class APIStatusError(RouterAIError):
    """Raised on other 4xx/5xx statuses."""

    def __init__(self, message: str, status_code: int, body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class RequestError(RouterAIError):
    """Raised on transport-level errors (timeout, connection failure)."""


class StreamInterruptedError(RouterAIError):
    """Raised when a streaming response breaks after it was already opened.

    Retries stop as soon as a successful HTTP response stream is opened —
    even before the first SSE chunk — because the request may have been
    billed and retrying could duplicate the generation. ``chunks_received``
    can therefore legitimately be 0.
    """

    def __init__(self, message: str, *, chunks_received: int = 0) -> None:
        super().__init__(message)
        self.chunks_received = chunks_received


class ConfigurationError(RouterAIError):
    """Raised when the client is configured inconsistently (e.g. wrong transport)."""


class VideoGenerationError(RouterAIError):
    """Raised when a video task reaches a terminal failure state."""

    def __init__(self, task_id: str, status: str, error: Any = None) -> None:
        super().__init__(f"video task {task_id} failed with status {status!r}")
        self.task_id = task_id
        self.status = status
        self.error = error


class WebhookVerificationError(RouterAIError):
    """Raised when a video webhook fails signature or freshness checks."""


class ModelNotFoundError(RouterAIError):
    """Raised when a model is not found in the catalog."""
