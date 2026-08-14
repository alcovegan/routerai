from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal


class RouterAIError(Exception):
    """Base class for all RouterAI client errors."""


@dataclass(frozen=True)
class ErrorInfo:
    """What could be read out of an error body once every wrapper is removed.

    RouterAI reports upstream failures by putting a JSON *string* in the
    ``error`` field, sometimes nested twice, with the real status inside. This
    carries the pieces that survive that unwrapping.
    """

    message: str
    provider_code: int | None = None
    error_type: str | None = None
    error_code: str | None = None
    provider_name: str | None = None
    provider_message: str | None = None
    metadata: dict[str, Any] | None = field(default=None)
    error: Any = None


class APIStatusError(RouterAIError):
    """The server answered, and the answer means failure.

    ``status_code`` is the *effective* status — the one that decides which
    subclass is raised. When RouterAI wraps an upstream failure in HTTP 200 or
    HTTP 503, the real code lives in the body; then ``status_code`` is that
    code, ``http_status`` keeps the transport status, and ``status_source``
    says which one won.
    """

    status: ClassVar[int | None] = None

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        body: Any = None,
        *,
        http_status: int | None = None,
        provider_code: int | None = None,
        status_source: Literal["http", "provider", "synthesized"] = "http",
        error_info: ErrorInfo | None = None,
        request_id: str | None = None,
        generation_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code if status_code is not None else self.status
        self.http_status = http_status if http_status is not None else self.status_code
        self.provider_code = provider_code
        self.status_source = status_source
        self.body = body
        self.error_info = error_info
        self.request_id = request_id
        self.generation_id = generation_id

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status_code={self.status_code}, "
            f"http_status={self.http_status}, message={self.message!r})"
        )


class BadRequestError(APIStatusError):
    """Raised on 400 — the request was rejected before reaching a provider."""

    status = 400


class AuthenticationError(APIStatusError):
    """Raised on 401 — the API key is missing, invalid, or revoked."""

    status = 401


class InsufficientFundsError(APIStatusError):
    """Raised on 402 — not enough funds on the balance."""

    status = 402


class PermissionDeniedError(APIStatusError):
    """Raised on 403 — the key has no access to the resource."""

    status = 403


class NotFoundError(APIStatusError):
    """Raised on 404 — the resource does not exist."""

    status = 404


class ConflictError(APIStatusError):
    """Raised on 409 — the resource is in a conflicting state."""

    status = 409


class UnprocessableEntityError(APIStatusError):
    """Raised on 422 — the request was understood but is not valid."""

    status = 422


class RateLimitError(APIStatusError):
    """Raised on 429 — too many requests, ours or an upstream provider's."""

    status = 429


class ServerError(APIStatusError):
    """Raised on 5xx — RouterAI or an upstream provider failed."""


class NoProviderError(ServerError):
    """Raised when no provider could serve the requested model."""

    status = 503


class ResponseParsingError(RouterAIError):
    """Raised when a successful response cannot be turned into the expected shape."""

    def __init__(self, message: str, *, body: Any = None) -> None:
        super().__init__(message)
        self.body = body


class RequestError(RouterAIError):
    """Raised on transport-level errors (timeout, connection failure)."""


class APIConnectionError(RequestError):
    """Raised when the connection could not be established (DNS, TCP, TLS)."""


class APITimeoutError(APIConnectionError):
    """Raised when a request timed out."""


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


class DeadlineExceededError(RouterAIError):
    """Raised when an absolute operation deadline passes before completion.

    Carries the remaining budget information and is raised instead of
    starting a new attempt or sleeping past the caller's deadline.
    """

    def __init__(self, message: str, *, budget: float | None = None) -> None:
        super().__init__(message)
        self.budget = budget


class ModelNotFoundError(RouterAIError):
    """Raised when a model is not found in the catalog."""
