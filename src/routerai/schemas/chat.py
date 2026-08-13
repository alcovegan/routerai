from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from .usage import Usage


class ProviderSelection(BaseModel):
    """Controls RouterAI multi-provider routing.

    Passed as the ``provider`` object in request bodies; see
    https://routerai.ru/docs/guides/overview/provider-selection
    """

    order: list[str] | None = None
    only: list[str] | None = None
    ignore: list[str] | None = None
    allow_fallbacks: bool | None = None
    country: str | None = None


class ServiceTier(str, Enum):
    DEFAULT = "default"
    FLEX = "flex"
    PRIORITY = "priority"


class Message(BaseModel):
    """OpenAI-style chat message."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None
    name: str | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str = "function"
    name: str | None = None
    arguments: str | dict[str, Any] | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: Message | None = None
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    """Parsed chat completion response."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    model: str | None = None
    created: int | None = None
    choices: list[Choice] = []
    usage: Usage | None = None
    service_tier: str | None = None
    generation_id: str | None = None


class ChatResult(BaseModel):
    """Convenience wrapper around a chat completion."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    id: str | None = None
    model: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = []
    finish_reason: str | None = None
    usage: Usage | None = None
    service_tier: str | None = None
    generation_id: str | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_response(cls, payload: dict[str, Any], generation_id: str | None = None) -> ChatResult:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason: str | None = None

        for choice in payload.get("choices") or []:
            message = choice.get("message") or {}
            finish_reason = choice.get("finish_reason") or finish_reason
            if isinstance(message.get("content"), str):
                content_parts.append(message["content"])
            if isinstance(message.get("reasoning_content"), str):
                reasoning_parts.append(message["reasoning_content"])
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                tool_calls.append(
                    ToolCall(
                        id=call.get("id", ""),
                        type=call.get("type", "function"),
                        name=function.get("name"),
                        arguments=function.get("arguments"),
                    )
                )

        return cls(
            id=payload.get("id"),
            model=payload.get("model"),
            content="\n".join(part for part in content_parts if part) or None,
            reasoning="\n".join(part for part in reasoning_parts if part) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=Usage.model_validate(payload["usage"]) if payload.get("usage") else None,
            service_tier=payload.get("service_tier"),
            generation_id=generation_id,
            raw=payload,
        )

    @property
    def cost_rub(self) -> Decimal | None:
        return self.usage.cost_rub if self.usage else None
