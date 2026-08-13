from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Usage(BaseModel):
    """Token usage and cost of a single request.

    ``cost_rub`` is the amount charged by RouterAI in rubles; it is present
    in most responses under ``usage.cost``. When it is missing it can be
    fetched afterwards via ``client.generation.get(generation_id)``.
    """

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_rub: Decimal | None = Field(default=None, alias="cost")
    seconds: float | None = None

    @field_validator("cost_rub", mode="before")
    @classmethod
    def _validate_cost(cls, value: Any) -> Decimal | None:
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None

    def tokens(self) -> int:
        return self.total_tokens or (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class GenerationInfo(BaseModel):
    """Post-hoc generation details from ``GET /api/v1/generation``."""

    model_config = ConfigDict(extra="allow")

    id: str
    total_cost: Decimal | None = None
    created_at: str | None = None
