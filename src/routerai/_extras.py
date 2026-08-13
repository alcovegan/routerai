from __future__ import annotations

from typing import Any


def merge_extra(extra: dict[str, Any] | None, *, reserved: tuple[str, ...]) -> None:
    """Reject ``extra`` collisions with library-managed body keys.

    ``extra`` is an escape hatch for provider-specific fields only; keys
    managed by typed parameters can never be overridden by accident.
    """
    if not extra:
        return
    collisions = sorted(set(extra) & set(reserved))
    if collisions:
        raise ValueError(
            f"extra cannot override library-managed keys {collisions}; "
            "use the dedicated parameters instead"
        )
