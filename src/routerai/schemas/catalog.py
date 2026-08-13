from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MILLION = Decimal("1000000")


class Capability(str, Enum):
    """High-level capabilities derived from the catalog entry."""

    TEXT = "text"
    REASONING = "reasoning"
    VISION = "vision"
    IMAGE_GENERATION = "image"
    VIDEO_INPUT = "video-input"
    VIDEO_GENERATION = "video"
    AUDIO_INPUT = "audio-input"
    SPEECH = "speech"
    TRANSCRIPTION = "transcription"
    EMBEDDINGS = "embeddings"
    RERANK = "rerank"
    TOOLS = "tools"


class Architecture(BaseModel):
    model_config = ConfigDict(extra="allow")

    modality: str | None = None
    tokenizer: str | None = None
    instruct_type: str | None = None
    input_modalities: list[str] = []
    output_modalities: list[str] = []


class ModelPricing(BaseModel):
    """Prices in rubles per token (or per unit for non-token models)."""

    model_config = ConfigDict(extra="allow")

    prompt: Decimal | None = None
    completion: Decimal | None = None

    def per_million(self, field: str = "prompt") -> Decimal | None:
        value: Decimal | None = getattr(self, field, None)
        if value is None:
            return None
        return value * MILLION


class Model(BaseModel):
    """A single catalog entry from ``GET /api/v1/models``."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str | None = None
    created: int | None = None
    description: str | None = None
    context_length: int | None = None
    architecture: Architecture = Field(default_factory=Architecture)
    pricing: ModelPricing = Field(default_factory=ModelPricing)
    per_request_limits: Any = None
    supported_parameters: list[str] = Field(default_factory=list)
    default_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("default_parameters", mode="before")
    @classmethod
    def _default_parameters(cls, value: Any) -> dict[str, Any]:
        return value or {}

    @field_validator("supported_parameters", mode="before")
    @classmethod
    def _supported_parameters(cls, value: Any) -> list[str]:
        return value or []

    @property
    def author(self) -> str:
        return self.id.split("/", 1)[0]

    @property
    def slug(self) -> str:
        return self.id.split("/", 1)[1] if "/" in self.id else self.id

    @property
    def capabilities(self) -> set[Capability]:
        caps: set[Capability] = set()
        inputs = self.architecture.input_modalities
        outputs = self.architecture.output_modalities
        params = self.supported_parameters
        if "text" in inputs or "text" in outputs:
            caps.add(Capability.TEXT)
        if "image" in inputs:
            caps.add(Capability.VISION)
        if "image" in outputs:
            caps.add(Capability.IMAGE_GENERATION)
        if "video" in inputs:
            caps.add(Capability.VIDEO_INPUT)
        if "video" in outputs:
            caps.add(Capability.VIDEO_GENERATION)
        if "audio" in inputs:
            caps.add(Capability.AUDIO_INPUT)
        if "speech" in outputs:
            caps.add(Capability.SPEECH)
        if "transcription" in outputs:
            caps.add(Capability.TRANSCRIPTION)
        if "embeddings" in outputs:
            caps.add(Capability.EMBEDDINGS)
        if "rerank" in outputs:
            caps.add(Capability.RERANK)
        if "tools" in params:
            caps.add(Capability.TOOLS)
        if "include_reasoning" in params or "reasoning" in params:
            caps.add(Capability.REASONING)
        if not caps:
            caps.add(Capability.TEXT)
        return caps


class Endpoint(BaseModel):
    """A single provider endpoint for a model."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    provider_name: str | None = None
    tag: str | None = None
    country: str | None = None
    context_length: int | None = None
    quantization: str | None = None
    max_completion_tokens: int | None = None
    max_prompt_tokens: int | None = None
    supported_parameters: list[str] = []
    supported_apis: list[str] = []
    status: int | None = None
    pricing: ModelPricing = Field(default_factory=ModelPricing)
    variable_pricings: list[dict[str, Any]] = []


class ModelDetail(BaseModel):
    """Response of ``GET /api/v1/models/{author}/{slug}/endpoints``."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str | None = None
    created: int | None = None
    description: str | None = None
    architecture: Architecture = Field(default_factory=Architecture)
    endpoints: list[Endpoint] = []


class VariablePricing(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str | None = None
    prompt: Decimal | None = None
    completion: Decimal | None = None
    threshold: int | None = None


class EndpointPricing(BaseModel):
    """Helper alias for the pricing payload on endpoints."""

    prompt: Decimal | None = None
    completion: Decimal | None = None

    @field_validator("prompt", "completion", mode="before")
    @classmethod
    def _to_decimal(cls, value: Any) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value))
