"""Per-call request options.

A TypedDict rather than a parameter object: adding a fourth option later is
one line here and zero edits across the resource methods, while the call site
still reads as plain keyword arguments::

    client.chat.complete(model, prompt, timeout=10, max_retries=0)

``Unpack`` is imported for real, not only under TYPE_CHECKING: otherwise
``typing.get_type_hints()`` on any public method raises NameError, which breaks
FastAPI, sphinx and anything else that resolves annotations at runtime.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TypedDict

if sys.version_info >= (3, 11):  # pragma: no cover - version split
    from typing import Unpack
else:  # pragma: no cover - version split
    from typing_extensions import Unpack

__all__ = ["OPTION_KEYS", "RequestOptions", "Unpack"]


class RequestOptions(TypedDict, total=False):
    """Options that apply to a single request.

    timeout: network inactivity timeout for this call, in seconds.
    max_retries: retry attempts for this call (0 disables retries).
    headers: headers layered on top of the client's, matched case-insensitively.
    deadline: absolute deadline on the ``time.monotonic()`` scale.
    error_in_body: whether an ``error`` field in a successful response means
        the request failed. False for video polling, where it is task state.
    """

    timeout: float | None
    max_retries: int | None
    headers: Mapping[str, str] | None
    deadline: float | None
    error_in_body: bool


OPTION_KEYS: frozenset[str] = frozenset(RequestOptions.__optional_keys__)
