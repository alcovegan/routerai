from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._extras import merge_extra

if TYPE_CHECKING:
    from .._http import HTTPClient


class Team:
    """Team management (``/api/v1/team``) — requires an admin master key."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def members(self) -> dict[str, Any]:
        return self._http._json(self._http.get("team/members"))

    async def amembers(self) -> dict[str, Any]:
        return self._http._json(await self._http.aget("team/members"))

    def create_member(
        self,
        email: str,
        *,
        role: str = "member",
        monthly_spending_limit: float | None = None,
        spending_limit_period: str | None = None,
        send_email: bool = False,
        password: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"email": email, "role": role, "send_email": send_email}
        if monthly_spending_limit is not None:
            body["monthly_spending_limit"] = monthly_spending_limit
        if spending_limit_period:
            body["spending_limit_period"] = spending_limit_period
        if password:
            body["password"] = password
        merge_extra(
            extra,
            reserved=(
                "email",
                "role",
                "monthly_spending_limit",
                "spending_limit_period",
                "send_email",
                "password",
            ),
        )
        if extra:
            body.update(extra)
        return self._http._json(self._http.post("team/members", json=body))

    async def acreate_member(
        self,
        email: str,
        *,
        role: str = "member",
        monthly_spending_limit: float | None = None,
        spending_limit_period: str | None = None,
        send_email: bool = False,
        password: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"email": email, "role": role, "send_email": send_email}
        if monthly_spending_limit is not None:
            body["monthly_spending_limit"] = monthly_spending_limit
        if spending_limit_period:
            body["spending_limit_period"] = spending_limit_period
        if password:
            body["password"] = password
        merge_extra(
            extra,
            reserved=(
                "email",
                "role",
                "monthly_spending_limit",
                "spending_limit_period",
                "send_email",
                "password",
            ),
        )
        if extra:
            body.update(extra)
        return self._http._json(await self._http.apost("team/members", json=body))

    def update_member(self, member_id: int, **changes: Any) -> dict[str, Any]:
        return self._http._json(
            self._http.request("PATCH", f"team/members/{member_id}", json=changes)
        )

    async def aupdate_member(self, member_id: int, **changes: Any) -> dict[str, Any]:
        return self._http._json(
            await self._http.arequest("PATCH", f"team/members/{member_id}", json=changes)
        )

    def delete_member(self, member_id: int) -> dict[str, Any]:
        return self._http._json(self._http.request("DELETE", f"team/members/{member_id}"))

    async def adelete_member(self, member_id: int) -> dict[str, Any]:
        return self._http._json(await self._http.arequest("DELETE", f"team/members/{member_id}"))

    def invite(
        self,
        email: str,
        *,
        role: str = "member",
        send_email: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"email": email, "role": role, "send_email": send_email}
        merge_extra(extra, reserved=("email", "role", "send_email"))
        if extra:
            body.update(extra)
        return self._http._json(self._http.post("team/invitations", json=body))

    async def ainvite(
        self,
        email: str,
        *,
        role: str = "member",
        send_email: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"email": email, "role": role, "send_email": send_email}
        merge_extra(extra, reserved=("email", "role", "send_email"))
        if extra:
            body.update(extra)
        return self._http._json(await self._http.apost("team/invitations", json=body))
