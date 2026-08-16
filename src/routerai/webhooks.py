"""Verification of RouterAI video webhooks.

RouterAI signs webhook requests with HMAC-SHA256 over the raw body:

    signature = HMAC_SHA256(secret, "<X-RouterAI-Timestamp>.<raw_body>")

where ``secret`` is the SHA-256 hex digest of the API key that created the
task. Feed this module the **raw request body bytes** — not a re-serialized
JSON object, or the digest will not match.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

from .errors import WebhookVerificationError

DEFAULT_MAX_AGE_SECONDS = 300.0


def signing_secret(api_key: str) -> str:
    """The RouterAI webhook secret derived from an API key (sha256 hex)."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_video(
    raw_body: bytes | str,
    signature: str,
    api_key: str,
    timestamp: str,
    *,
    max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Verify a video webhook request and return its parsed payload.

    Args:
        raw_body: the exact request body bytes (before any JSON parsing).
        signature: the ``X-RouterAI-Signature`` header value (hex digest).
        api_key: the API key the video task was created with.
        timestamp: the ``X-RouterAI-Timestamp`` header value (unix seconds).
        max_age_seconds: maximum allowed clock skew; also bounds the replay
            window (RouterAI may re-deliver the same event).

    Raises:
        WebhookVerificationError: signature mismatch, malformed timestamp or
            a timestamp outside the freshness window.
    """
    try:
        ts = int(timestamp)
    except (TypeError, ValueError) as exc:
        raise WebhookVerificationError(
            f"invalid X-RouterAI-Timestamp header: {timestamp!r}"
        ) from exc

    age = abs(time.time() - ts)
    if age > max_age_seconds:
        raise WebhookVerificationError(
            f"webhook timestamp {ts} is outside the {max_age_seconds}s freshness window"
        )

    body_bytes = raw_body if isinstance(raw_body, bytes) else raw_body.encode("utf-8")
    # Sign the header exactly as it arrived: int() would normalise "1_700" or
    # leading zeros into a different number and the digest would never match.
    expected = hmac.new(
        signing_secret(api_key).encode("ascii"),
        timestamp.strip().encode("utf-8") + b"." + body_bytes,
        hashlib.sha256,
    ).hexdigest()
    # Compare bytes: compare_digest raises TypeError on non-ASCII text, and
    # the signature header is attacker-controlled and unauthenticated here.
    if not hmac.compare_digest(expected.encode("ascii"), signature.strip().encode("utf-8")):
        raise WebhookVerificationError("webhook signature mismatch")

    try:
        payload = json.loads(body_bytes)
    except ValueError as exc:
        raise WebhookVerificationError("webhook body is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise WebhookVerificationError("webhook body is not a JSON object")
    return payload
