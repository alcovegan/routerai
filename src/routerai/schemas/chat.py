from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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

    role: str | None = None
    content: Any = None
    name: str | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None
    reasoning_content: Any = None


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
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage | None = None
    service_tier: str | None = None
    generation_id: str | None = None


class ChatResult(BaseModel):
    """Typed wrapper around a chat completion.

    ``choices`` preserves every alternative returned by the API (e.g. when
    ``n > 1``); the convenience fields ``content``, ``reasoning``,
    ``tool_calls`` and ``finish_reason`` refer to ``choices[0]`` only.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    model: str | None = None
    created: int | None = None
    choices: list[Choice] = Field(default_factory=list)
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    service_tier: str | None = None
    generation_id: str | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_response(cls, payload: dict[str, Any], generation_id: str | None = None) -> ChatResult:
        choices = []
        for item in payload.get("choices") or []:
            message_payload = item.get("message")
            message = (
                Message.model_validate(message_payload)
                if isinstance(message_payload, dict)
                else None
            )
            choices.append(
                Choice(
                    index=item.get("index", 0),
                    message=message,
                    finish_reason=item.get("finish_reason"),
                )
            )

        first = choices[0] if choices else None
        first_message = first.message if first else None

        content: str | None = None
        if first_message is not None and isinstance(first_message.content, str):
            content = first_message.content
        reasoning: str | None = None
        if first_message is not None:
            for key in ("reasoning_content", "reasoning"):
                value = getattr(first_message, key, None)
                if value is None and first_message.model_extra:
                    value = first_message.model_extra.get(key)
                if isinstance(value, str):
                    reasoning = value
                    break

        tool_calls = []
        for call in (first_message.tool_calls if first_message else None) or []:
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
            created=payload.get("created"),
            choices=choices,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            finish_reason=first.finish_reason if first else None,
            usage=Usage.model_validate(payload["usage"]) if payload.get("usage") else None,
            service_tier=payload.get("service_tier"),
            generation_id=generation_id,
            raw=payload,
        )

    @property
    def cost_rub(self) -> Decimal | None:
        return self.usage.cost_rub if self.usage else None
