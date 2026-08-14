from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from .._options import RequestOptions
from ..schemas import GenerationInfo

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from .._http import HTTPClient


class Generation:
    """Post-hoc lookup of a generation by its ``X-Generation-Id``.

    Useful when a response does not include ``usage.cost``::

        cost = client.generation.cost(result.generation_id)
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def get(self, generation_id: str, **opts: Unpack[RequestOptions]) -> GenerationInfo:
        response = self._http.get("generation", params={"id": generation_id}, **opts)
        payload = self._http._json(response)
        return GenerationInfo.model_validate(_unwrap(payload))

    async def aget(self, generation_id: str, **opts: Unpack[RequestOptions]) -> GenerationInfo:
        response = await self._http.aget("generation", params={"id": generation_id}, **opts)
        payload = self._http._json(response)
        return GenerationInfo.model_validate(_unwrap(payload))

    def cost(self, generation_id: str) -> Decimal | None:
        return self.get(generation_id).total_cost

    async def acost(self, generation_id: str) -> Decimal | None:
        return (await self.aget(generation_id)).total_cost


def _unwrap(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        return {**payload, **data}
    return payload
