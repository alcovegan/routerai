from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._extras import merge_extra
from ..schemas import KeyInfo

KeyInfoList = list[KeyInfo]

if TYPE_CHECKING:
    from .._http import HTTPClient


class Keys:
    """API key management via the master key (``/api/v1/keys``).

    Requires a client initialized with a master key; regular API keys get 403.
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def list(self) -> KeyInfoList:
        payload = self._http._json(self._http.get("keys"))
        return [KeyInfo.model_validate(item) for item in payload.get("data") or []]

    async def alist(self) -> KeyInfoList:
        payload = self._http._json(await self._http.aget("keys"))
        return [KeyInfo.model_validate(item) for item in payload.get("data") or []]

    def create(
        self,
        name: str,
        *,
        limit: float | None = None,
        expires_at: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> KeyInfo:
        body: dict[str, Any] = {"name": name}
        if limit is not None:
            body["limit"] = limit
        if expires_at:
            body["expires_at"] = expires_at
        merge_extra(extra, reserved=("name", "limit", "expires_at"))
        if extra:
            body.update(extra)
        payload = self._http._json(self._http.post("keys", json=body))
        return KeyInfo.model_validate(payload.get("data") or payload)

    async def acreate(
        self,
        name: str,
        *,
        limit: float | None = None,
        expires_at: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> KeyInfo:
        body: dict[str, Any] = {"name": name}
        if limit is not None:
            body["limit"] = limit
        if expires_at:
            body["expires_at"] = expires_at
        merge_extra(extra, reserved=("name", "limit", "expires_at"))
        if extra:
            body.update(extra)
        payload = self._http._json(await self._http.apost("keys", json=body))
        return KeyInfo.model_validate(payload.get("data") or payload)

    def delete(self, key_id: str) -> dict[str, Any]:
        return self._http._json(self._http.request("DELETE", f"keys/{key_id}"))

    async def adelete(self, key_id: str) -> dict[str, Any]:
        return self._http._json(await self._http.arequest("DELETE", f"keys/{key_id}"))
