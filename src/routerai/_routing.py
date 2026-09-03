"""Parsing and building RouterAI model strings.

A model string carries three things at once:

* the catalog id, ``anthropic/claude-opus-5``;
* an optional alias marker, the leading ``~`` in ``~anthropic/claude-opus-latest``,
  which tells the router to resolve the newest release in that line;
* optional routing parameters after ``@``, as in
  ``anthropic/claude-opus-5@provider=amazon-bedrock&allow_fallbacks=false``.

The ``@`` suffix is only understood by the chat-shaped endpoints. Sending it
anywhere else does not fail loudly on the server: the whole string is treated
as a model name, and the reply is ``Model '<id>@provider=x' not found``. That
is why :func:`split_model` is used to check the suffix before a request goes
out, instead of letting the caller pay a round trip to learn it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ALIAS_PREFIX = "~"
ROUTE_SEPARATOR = "@"

#: Endpoints that understand the ``@`` suffix, per RouterAI's provider-selection
#: guide. Paths are matched after the API root, without a leading slash.
ROUTABLE_PATHS = frozenset({"chat/completions", "responses", "messages"})

#: Routing keys the server accepts in the ``@`` suffix.
ROUTE_KEYS = ("provider", "allow_fallbacks")


def is_alias(model: str) -> bool:
    """Whether a model id is an alias that resolves to the newest release.

    The leading ``~`` is the only marker. A ``-latest`` suffix is not one:
    ``openai/gpt-chat-latest`` is an ordinary catalog entry.
    """
    return model.startswith(ALIAS_PREFIX)


@dataclass(frozen=True)
class ModelRoute:
    """A model string split into its catalog id and its routing parameters."""

    model: str
    """The catalog id, alias marker kept, routing suffix removed."""

    provider: str | None = None
    allow_fallbacks: bool | None = None
    unknown: dict[str, str] = field(default_factory=dict)
    """Routing keys the SDK does not know, passed through untouched."""

    @property
    def is_alias(self) -> bool:
        return is_alias(self.model)

    @property
    def has_routing(self) -> bool:
        return self.provider is not None or self.allow_fallbacks is not None or bool(self.unknown)

    def __str__(self) -> str:
        return build_model(
            self.model,
            provider=self.provider,
            allow_fallbacks=self.allow_fallbacks,
            **self.unknown,
        )


def split_model(model: str) -> ModelRoute:
    """Split ``model@key=value&key=value`` into its parts.

    Unparseable suffixes are kept verbatim in :attr:`ModelRoute.unknown` rather
    than dropped, so a string the SDK does not understand still reaches the
    server unchanged.
    """
    base, separator, suffix = model.partition(ROUTE_SEPARATOR)
    if not separator:
        return ModelRoute(model=model)

    provider: str | None = None
    allow_fallbacks: bool | None = None
    unknown: dict[str, str] = {}
    for pair in suffix.split("&"):
        if not pair:
            continue
        key, eq, value = pair.partition("=")
        if not eq:
            unknown[key] = ""
        elif key == "provider":
            provider = value
        elif key == "allow_fallbacks":
            if value.lower() in ("true", "false"):
                allow_fallbacks = value.lower() == "true"
            else:
                unknown[key] = value
        else:
            unknown[key] = value
    return ModelRoute(
        model=base, provider=provider, allow_fallbacks=allow_fallbacks, unknown=unknown
    )


def build_model(
    model: str,
    *,
    provider: str | None = None,
    allow_fallbacks: bool | None = None,
    **extra: Any,
) -> str:
    """Build a model string with a routing suffix.

    Any suffix already on ``model`` is replaced, so building on top of an
    already-routed string does not produce two ``@`` sections.
    """
    base = split_model(model).model
    parts = []
    if provider is not None:
        parts.append(f"provider={provider}")
    if allow_fallbacks is not None:
        parts.append(f"allow_fallbacks={'true' if allow_fallbacks else 'false'}")
    for key, value in extra.items():
        parts.append(f"{key}={value}")
    if not parts:
        return base
    return f"{base}{ROUTE_SEPARATOR}{'&'.join(parts)}"


def route(
    model: str,
    *,
    provider: str | None = None,
    allow_fallbacks: bool | None = None,
) -> str:
    """Pin a request to one provider without touching the request body.

    ``allow_fallbacks=False`` turns a silent switch to another provider into a
    ``NotFoundError``, which is the point of pinning::

        from routerai import route

        client.chat.complete(
            route("anthropic/claude-opus-5", provider="amazon-bedrock", allow_fallbacks=False),
            "hello",
        )

    Only ``chat.complete``/``chat.stream`` accept a routed string; other
    endpoints read the whole thing as a model name.
    """
    return build_model(model, provider=provider, allow_fallbacks=allow_fallbacks)


def conflicting_keys(model: str, provider_body: Any) -> list[str]:
    """Routing keys present both in the model string and in the request body.

    RouterAI answers ``400 Parameter 'provider' specified both in model string
    and request body`` for these, so the SDK checks first and reports the clash
    without spending a request.
    """
    parsed = split_model(model)
    if not parsed.has_routing or not provider_body:
        return []

    if isinstance(provider_body, dict):
        body_keys = {key for key, value in provider_body.items() if value is not None}
    else:  # a pydantic ProviderSelection
        dump = getattr(provider_body, "model_dump", None)
        body_keys = set(dump(exclude_none=True)) if callable(dump) else set()

    clashes = []
    # `provider` in the string selects one provider; order/only/ignore in the
    # body do the same job, and the server rejects the pair.
    if parsed.provider is not None and body_keys & {"order", "only", "ignore"}:
        clashes.append("provider")
    if parsed.allow_fallbacks is not None and "allow_fallbacks" in body_keys:
        clashes.append("allow_fallbacks")
    return clashes
