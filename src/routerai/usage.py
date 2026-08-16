"""Counting what the calls cost.

RouterAI reports the price of every request in rubles. The SDK already read
that number to write it into a log line and then dropped it; this keeps it.

    with client.track("ingest") as spent:
        client.chat.complete(model, prompt)
    print(spent.cost_rub, spent.total_tokens)

Counters are aggregates, never a list of records: a long-running process must
not grow a log of every call it ever made.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from .schemas import Usage

UsageHook = Callable[["UsageRecord"], None]

# Cardinality guard: a label or model name comes from the caller, and an
# unbounded dict of them is a slow memory leak.
MAX_GROUPS = 512
OTHER_GROUP = "…other"


@dataclass(frozen=True)
class UsageRecord:
    """What one request cost."""

    method: str
    path: str
    status: int | None
    elapsed: float
    model: str | None = None
    label: str | None = None
    streamed: bool = False
    duplicate: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_rub: Decimal | None = None
    generation_id: str | None = None
    request_id: str | None = None
    usage: Usage | None = None


@dataclass(frozen=True)
class UsageStats:
    """A snapshot of accumulated usage."""

    requests: int = 0
    streamed: int = 0
    failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_rub: Decimal = Decimal(0)
    unknown_cost: int = 0
    by_model: Mapping[str, UsageStats] = field(default_factory=dict)
    by_label: Mapping[str, UsageStats] = field(default_factory=dict)


def _add(stats: UsageStats, record: UsageRecord, *, grouped: bool = False) -> UsageStats:
    cost = stats.cost_rub
    unknown = stats.unknown_cost
    if record.duplicate:
        # The same generation polled again: it is another request, but the
        # money was already counted the first time.
        return replace(
            stats,
            requests=stats.requests + 1,
            failed=stats.failed + (1 if (record.status or 0) >= 400 else 0),
        )
    if record.cost_rub is None:
        unknown += 1
    else:
        cost = cost + record.cost_rub
    updated = replace(
        stats,
        requests=stats.requests + 1,
        streamed=stats.streamed + (1 if record.streamed else 0),
        failed=stats.failed + (1 if (record.status or 0) >= 400 else 0),
        prompt_tokens=stats.prompt_tokens + record.prompt_tokens,
        completion_tokens=stats.completion_tokens + record.completion_tokens,
        total_tokens=stats.total_tokens + record.total_tokens,
        cost_rub=cost,
        unknown_cost=unknown,
    )
    if grouped:
        return updated
    return replace(
        updated,
        by_model=_grouped(dict(stats.by_model), record.model, record),
        by_label=_grouped(dict(stats.by_label), record.label, record),
    )


def _grouped(
    groups: dict[str, UsageStats], key: str | None, record: UsageRecord
) -> dict[str, UsageStats]:
    if key is None:
        return groups
    if key not in groups and len(groups) >= MAX_GROUPS:
        key = OTHER_GROUP
    groups[key] = _add(groups.get(key, UsageStats()), record, grouped=True)
    return groups


class UsageTracker:
    """Thread-safe counters for requests, tokens and rubles."""

    def __init__(self, *, label: str | None = None) -> None:
        self.label = label
        self._lock = threading.Lock()
        self._stats = UsageStats()

    def add(self, record: UsageRecord) -> None:
        with self._lock:
            self._stats = _add(self._stats, record)

    def snapshot(self) -> UsageStats:
        with self._lock:
            return self._stats

    def reset(self) -> None:
        with self._lock:
            self._stats = UsageStats()

    @property
    def requests(self) -> int:
        return self.snapshot().requests

    @property
    def total_tokens(self) -> int:
        return self.snapshot().total_tokens

    @property
    def cost_rub(self) -> Decimal:
        return self.snapshot().cost_rub

    def __repr__(self) -> str:
        stats = self.snapshot()
        return (
            f"UsageTracker(requests={stats.requests}, tokens={stats.total_tokens}, "
            f"cost_rub={stats.cost_rub})"
        )


def record_from(
    *,
    method: str,
    path: str,
    status: int | None,
    elapsed: float,
    body: Any,
    label: str | None,
    streamed: bool,
    generation_id: str | None = None,
    request_id: str | None = None,
    duplicate: bool = False,
) -> UsageRecord:
    """Build a record from a decoded response body."""
    usage: Usage | None = None
    model: str | None = None
    if isinstance(body, Mapping):
        raw = body.get("usage")
        if isinstance(raw, Mapping):
            try:
                usage = Usage.model_validate(dict(raw))
            except Exception:  # accounting must never break a paid call
                usage = None
        name = body.get("model")
        model = name if isinstance(name, str) else None
    prompt = completion = 0
    if usage is not None:
        prompt = usage.prompt_tokens or usage.input_tokens or 0
        completion = usage.completion_tokens or usage.output_tokens or 0
    return UsageRecord(
        method=method,
        path=path,
        status=status,
        elapsed=elapsed,
        model=model,
        label=label,
        streamed=streamed,
        duplicate=duplicate,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=usage.tokens() if usage else 0,
        cost_rub=usage.cost_rub if usage else None,
        generation_id=generation_id,
        request_id=request_id,
        usage=usage,
    )


def iter_trackers(trackers: tuple[UsageTracker, ...]) -> Iterator[UsageTracker]:
    return iter(trackers)
