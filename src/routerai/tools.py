"""Running tool calls for you.

Calling a tool by hand means four steps every time: describe the function as
JSON schema, notice the model asked for it, parse the arguments, append the
result and ask again. This does that loop::

    def get_weather(city: str) -> str:
        return f"в городе {city} +17"

    answer = client.chat.run_tools(model, "Погода в Москве?", tools=[get_weather])
    answer.content

Schemas are derived from the signature, so the description the model sees and
the function actually called cannot drift apart.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, create_model

from .errors import RouterAIError
from .schemas import ChatResult

ToolFunction = Callable[..., Any]


@dataclass
class ToolRun:
    """One executed tool call, kept for inspection and debugging."""

    name: str
    arguments: dict[str, Any]
    result: Any
    error: str | None = None


@dataclass
class ToolRunResult:
    """The final answer plus everything that happened on the way to it."""

    result: ChatResult
    runs: list[ToolRun] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0

    @property
    def content(self) -> str | None:
        return self.result.content

    @property
    def usage(self) -> Any:
        return self.result.usage


def tool_schema(function: ToolFunction) -> dict[str, Any]:
    """Describe a python function the way the tools API expects.

    Parameter types come from the annotations, the description from the
    docstring's first line.
    """
    signature = inspect.signature(function)
    fields: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = (
            parameter.annotation if parameter.annotation is not inspect.Parameter.empty else str
        )
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (annotation, default)

    model: type[BaseModel] = create_model(f"{function.__name__}_arguments", **fields)
    schema = model.model_json_schema()
    schema.pop("title", None)
    doc = inspect.getdoc(function) or ""
    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": doc.split("\n\n")[0].strip(),
            "parameters": schema,
        },
    }


def normalize_tools(
    tools: Sequence[ToolFunction | dict[str, Any]] | Mapping[str, ToolFunction],
) -> tuple[list[dict[str, Any]], dict[str, ToolFunction]]:
    """Split what the caller passed into wire schemas and callables.

    Accepts plain functions, a name → function mapping, or ready-made schemas
    (which are passed through and simply cannot be executed here).
    """
    schemas: list[dict[str, Any]] = []
    registry: dict[str, ToolFunction] = {}

    items: list[tuple[str | None, ToolFunction | dict[str, Any]]]
    if isinstance(tools, Mapping):
        items = [(name, tool) for name, tool in tools.items()]
    else:
        items = [(None, tool) for tool in tools]

    for name, tool in items:
        if isinstance(tool, dict):
            schemas.append(tool)
            continue
        schema = tool_schema(tool)
        if name is not None:
            schema["function"]["name"] = name
        schemas.append(schema)
        registry[schema["function"]["name"]] = tool
    return schemas, registry


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments as a dict, whichever way the provider sent them."""
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise RouterAIError(f"tool arguments are not valid JSON: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise RouterAIError(f"tool arguments are not an object: {raw!r}")
    return parsed


def assistant_message(result: ChatResult) -> dict[str, Any]:
    """The assistant turn to append before the tool results."""
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    raw_calls = []
    for call in result.tool_calls:
        raw_calls.append(
            {
                "id": call.id,
                "type": call.type,
                "function": {"name": call.name, "arguments": call.arguments},
            }
        )
    if raw_calls:
        message["tool_calls"] = raw_calls
    return message


def tool_message(call_id: str, name: str, content: Any) -> dict[str, Any]:
    """The turn carrying a tool's result back to the model."""
    text = (
        content
        if isinstance(content, str)
        else json.dumps(content, ensure_ascii=False, default=str)
    )
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": text}
