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

    # populate_by_name keeps the field readable under its own name, so a value
    # produced by model_dump() survives model_validate() — without it the cost
    # silently lands in model_extra and reads back as None.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_rub: Decimal | None = Field(default=None, alias="cost", serialization_alias="cost")
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
        """Total tokens, whichever naming the endpoint used.

        Chat reports ``prompt_tokens``/``completion_tokens``; transcription and
        ``GET /generation`` report ``input_tokens``/``output_tokens``.
        """
        if self.total_tokens:
            return self.total_tokens
        prompt = self.prompt_tokens if self.prompt_tokens is not None else self.input_tokens
        completion = (
            self.completion_tokens if self.completion_tokens is not None else self.output_tokens
        )
        return (prompt or 0) + (completion or 0)


class GenerationInfo(BaseModel):
    """Post-hoc generation details from ``GET /api/v1/generation``."""

    model_config = ConfigDict(extra="allow")

    id: str
    total_cost: Decimal | None = None
    created_at: str | None = None
