from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._http import HTTPClient


class Generation:
    """Post-hoc lookup of a generation by its ``X-Generation-Id``.

    Useful when a response does not include ``usage.cost``::

        cost = client.generation.cost(result.generation_id)
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def get(self, generation_id: str) -> dict[str, Any]:
        response = self._http.get("generation", params={"id": generation_id})
        return self._http._json(response)

    async def aget(self, generation_id: str) -> dict[str, Any]:
        response = await self._http.aget("generation", params={"id": generation_id})
        return self._http._json(response)

    def cost(self, generation_id: str) -> Decimal | None:
        payload = self.get(generation_id)
        return self._extract_cost(payload)

    async def acost(self, generation_id: str) -> Decimal | None:
        payload = await self.aget(generation_id)
        return self._extract_cost(payload)

    def _extract_cost(self, payload: dict[str, Any]) -> Decimal | None:
        for key in ("total_cost", "cost"):
            value = payload.get(key)
            if value is not None:
                return Decimal(str(value))
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("total_cost", "cost"):
                value = data.get(key)
                if value is not None:
                    return Decimal(str(value))
        usage = payload.get("usage")
        if isinstance(usage, dict) and usage.get("cost") is not None:
            return Decimal(str(usage["cost"]))
        return None
