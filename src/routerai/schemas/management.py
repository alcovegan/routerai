from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .usage import Usage


class KeyInfo(BaseModel):
    """An API key managed via the master key (``/api/v1/keys``)."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    # SecretStr so printing or logging the object cannot leak a live key:
    # the value stays reachable through key.get_secret_value().
    key: SecretStr | None = None
    limit: Decimal | None = None
    expires_at: str | None = None
    created_at: str | None = None

    @field_validator("limit", mode="before")
    @classmethod
    def _limit(cls, value: Any) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value))


class TeamMember(BaseModel):
    """A team member managed via ``/api/v1/team``."""

    model_config = ConfigDict(extra="allow")

    id: int | None = None
    user_id: int | None = None
    email: str | None = None
    role: str | None = None
    active: bool | None = None
    is_owner: bool | None = None
    monthly_spending_limit: Decimal | None = None
    spending_limit_period: str | None = None
    monthly_spending: Decimal | None = None
    period_spending: Decimal | None = None
    email_confirmed: bool | None = None
    created_at: str | None = None

    @field_validator("monthly_spending_limit", "monthly_spending", "period_spending", mode="before")
    @classmethod
    def _decimals(cls, value: Any) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value))


class MemberCreation(BaseModel):
    """Result of ``POST /api/v1/team/members``."""

    model_config = ConfigDict(extra="allow")

    data: TeamMember | None = None
    password_setup_url: str | None = None
    raw: dict[str, Any] | None = None


class TeamInvitation(BaseModel):
    """Result of ``POST /api/v1/team/invitations``."""

    model_config = ConfigDict(extra="allow")

    id: int | None = None
    email: str | None = None
    role: str | None = None
    invite_url: str | None = None
    expires_at: str | None = None
    raw: dict[str, Any] | None = None


class ResponsesResult(BaseModel):
    """Typed result of the OpenAI-compatible ``/responses`` endpoint."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    object: str | None = None
    model: str | None = None
    status: str | None = None
    created_at: int | None = None
    output: list[dict[str, Any]] = Field(default_factory=list)
    usage: Usage | None = None
    error: Any = None
    raw: dict[str, Any] | None = None

    @property
    def output_text(self) -> str:
        parts: list[str] = []
        for item in self.output:
            for part in item.get("content") or []:
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
        return "\n".join(parts)

    @property
    def cost_rub(self) -> Decimal | None:
        return self.usage.cost_rub if self.usage else None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ResponsesResult:
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return cls(
            id=payload.get("id"),
            object=payload.get("object"),
            model=payload.get("model"),
            status=payload.get("status"),
            created_at=payload.get("created_at"),
            output=payload.get("output") or [],
            usage=usage,
            error=payload.get("error"),
            raw=payload,
        )


class MessagesResult(BaseModel):
    """Typed result of the Anthropic-compatible ``/messages`` endpoint."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    type: str | None = None
    role: str | None = None
    model: str | None = None
    content: list[dict[str, Any]] = Field(default_factory=list)
    stop_reason: str | None = None
    stop_sequence: Any = None
    usage: Usage | None = None
    raw: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        parts: list[str] = []
        for block in self.content:
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)

    @property
    def cost_rub(self) -> Decimal | None:
        return self.usage.cost_rub if self.usage else None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> MessagesResult:
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return cls(
            id=payload.get("id"),
            type=payload.get("type"),
            role=payload.get("role"),
            model=payload.get("model"),
            content=payload.get("content") or [],
            stop_reason=payload.get("stop_reason"),
            stop_sequence=payload.get("stop_sequence"),
            usage=usage,
            raw=payload,
        )
