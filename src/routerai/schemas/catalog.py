from __future__ import annotations

from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .._routing import ALIAS_PREFIX

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
    AUDIO_GENERATION = "audio"
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
    """Prices in rubles per unit.

    Token models are priced per token via ``prompt``/``completion``. Others are
    priced in their own unit: image generators use ``image_output``, rerank uses
    ``search_units``, video uses ``seconds``. In the live catalog 64 of 458
    models have no token price at all, so ``prompt == 0`` must never be read as
    "free" — use :meth:`priced_units` to see what a model actually charges for.
    """

    model_config = ConfigDict(extra="allow")

    prompt: Decimal | None = None
    completion: Decimal | None = None
    request: Decimal | None = None
    image: Decimal | None = None
    image_output: Decimal | None = None
    image_token: Decimal | None = None
    audio: Decimal | None = None
    audio_output: Decimal | None = None
    seconds: Decimal | None = None
    search_units: Decimal | None = None
    web_search: Decimal | None = None
    input_cache_read: Decimal | None = None
    input_cache_write: Decimal | None = None
    input_audio_cache: Decimal | None = None
    internal_reasoning: Decimal | None = None

    @model_validator(mode="before")
    @classmethod
    def _decimalize(cls, data: Any) -> Any:
        """Coerce every numeric price to Decimal, including unknown units.

        ``extra="allow"`` keeps unknown fields as floats, and multiplying a
        float by a Decimal raises TypeError — so a price the SDK does not know
        about yet would break :meth:`per_million` instead of just working.
        """
        if not isinstance(data, dict):
            return data
        coerced: dict[Any, Any] = {}
        for key, value in data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    coerced[key] = Decimal(str(value))
                except InvalidOperation:
                    coerced[key] = value
            else:
                coerced[key] = value
        return coerced

    def price(self, unit: str = "prompt") -> Decimal | None:
        """Price per single unit, reading declared fields and unknown ones alike."""
        value = getattr(self, unit, None)
        if value is None and self.model_extra:
            value = self.model_extra.get(unit)
        return value if isinstance(value, Decimal) else None

    def per_million(self, field: str = "prompt") -> Decimal | None:
        """Price per million units. For non-token models the unit is not a token."""
        value = self.price(field)
        if value is None:
            return None
        return value * MILLION

    def priced_units(self) -> frozenset[str]:
        """Units this model actually charges for."""
        units = {
            name
            for name in type(self).model_fields
            if isinstance(getattr(self, name, None), Decimal) and getattr(self, name) > 0
        }
        for name, value in (self.model_extra or {}).items():
            if isinstance(value, Decimal) and value > 0:
                units.add(name)
        return frozenset(units)

    def is_free(self) -> bool:
        return not self.priced_units()


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
    def is_alias(self) -> bool:
        """Whether this entry is an alias for the newest release in a line.

        Aliases look exactly like ordinary entries — same fields, real pricing,
        and a ``name`` describing whatever they currently point at. The leading
        ``~`` is the only thing that marks one. A ``-latest`` suffix does not:
        ``openai/gpt-chat-latest`` is a normal model.
        """
        return self.id.startswith(ALIAS_PREFIX)

    @property
    def author(self) -> str:
        """The developer part of the id, without the alias marker.

        Stripping ``~`` here keeps ``developer="anthropic"`` matching both
        ``anthropic/claude-opus-5`` and ``~anthropic/claude-opus-latest``.
        """
        return self.id.lstrip(ALIAS_PREFIX).split("/", 1)[0]

    @property
    def slug(self) -> str:
        base = self.id.lstrip(ALIAS_PREFIX)
        return base.split("/", 1)[1] if "/" in base else base

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
        if "audio" in outputs:
            caps.add(Capability.AUDIO_GENERATION)
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
