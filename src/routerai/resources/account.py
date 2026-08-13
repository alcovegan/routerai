from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._http import HTTPClient


class Account:
    """Account-level endpoints (currently: balance in rubles)."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def credits(self) -> Decimal:
        """Return the current balance in rubles (``GET /api/v1/credits``)."""
        return self._parse(self._http._json(self._http.get("credits")))

    async def acredits(self) -> Decimal:
        return self._parse(self._http._json(await self._http.aget("credits")))

    @staticmethod
    def _parse(payload: dict[str, Any]) -> Decimal:
        data = payload.get("data") or payload
        value = data.get("credits")
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(f"unexpected credits response: {payload!r}") from exc
