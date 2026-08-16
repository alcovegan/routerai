from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation
from typing import Any

DEFAULT_LOGGER_NAME = "routerai"
KEY_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{4})[A-Za-z0-9_\-]+")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``routerai`` namespace.

    A custom child name (e.g. ``"myapp.rai"``) attaches to the same tree,
    so configuring ``logging.getLogger("routerai")`` affects all clients.
    """
    base = name or DEFAULT_LOGGER_NAME
    return logging.getLogger(
        DEFAULT_LOGGER_NAME if base == DEFAULT_LOGGER_NAME else f"{DEFAULT_LOGGER_NAME}.{base}"
    )


def mask_key(value: Any) -> str:
    """Redact API keys in arbitrary payloads before logging."""
    return KEY_PATTERN.sub(r"\1...", str(value))


def format_cost(value: Decimal | float | None) -> str:
    """Format a cost for logging, never raising.

    The response is already paid for by the time it is logged, so a value the
    provider reported in an unexpected shape must not turn into an exception
    that loses the result.
    """
    if value is None:
        return "?"
    try:
        return f"{Decimal(str(value)).normalize():f}₽"
    except (InvalidOperation, ValueError):
        return f"{value}"


def log_request(
    logger: logging.Logger,
    method: str,
    url: str,
    *,
    payload: Any = None,
    elapsed: float | None = None,
    status: int | None = None,
    tokens: Any = None,
    cost: Decimal | float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Single entry point for request logging across the library."""
    if not logger.isEnabledFor(logging.INFO):
        return

    if logger.isEnabledFor(logging.DEBUG) and payload is not None:
        logger.debug("request %s %s body=%s", method, url, mask_key(payload))

    parts: list[str] = [f"{method} {url}"]
    if status is not None:
        parts.append(f"status={status}")
    if elapsed is not None:
        parts.append(f"t={elapsed:.2f}s")
    if tokens is not None:
        parts.append(f"tokens={tokens}")
    if cost is not None:
        parts.append(f"cost={format_cost(cost)}")

    if logger.isEnabledFor(logging.INFO):
        logger.info(" ".join(parts), extra=extra)
