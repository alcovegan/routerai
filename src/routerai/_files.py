"""Shared helpers for bounded atomic downloads.

A single implementation of the temp-file lifecycle is used by image and
video downloads: unique exclusive temp file in the target directory,
byte-count limiting before each write, cleanup on any failure and an
atomic rename only after the stream completes.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

from .errors import RouterAIError


class AtomicFileWriter:
    """Writes chunks to a unique temp file and renames it into place.

    Usage::

        with AtomicFileWriter("out.mp4", max_bytes=512 * 1024 * 1024) as writer:
            for chunk in response.iter_bytes():
                writer.write(chunk)
            return writer.commit()

    On exception, cancellation or size-limit breach the temp file is
    removed; ``commit()`` atomically renames it to the target.
    """

    def __init__(self, target: str | Path, *, max_bytes: int) -> None:
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError(f"max_bytes must be a positive integer, got {max_bytes!r}")
        self.target = Path(target)
        self.max_bytes = max_bytes
        self._tmp: Path | None = None
        self._handle: BinaryIO | None = None
        self._total = 0

    def __enter__(self) -> AtomicFileWriter:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.target.name}.", dir=self.target.parent)
        self._tmp = Path(name)
        self._handle = os.fdopen(fd, "wb")
        return self

    def write(self, chunk: bytes) -> None:
        if self._handle is None:
            raise RuntimeError("AtomicFileWriter is not open")
        self._total += len(chunk)
        if self._total > self.max_bytes:
            raise RouterAIError(f"download exceeds the {self.max_bytes} byte limit")
        self._handle.write(chunk)

    @property
    def total(self) -> int:
        return self._total

    @property
    def tmp_path(self) -> Path | None:
        return self._tmp

    def abort(self) -> None:
        """Close the handle and remove the temporary file on a best-effort basis."""
        if self._handle is not None:
            with suppress(OSError):
                self._handle.close()
            self._handle = None
        if self._tmp is not None:
            with suppress(OSError):
                self._tmp.unlink(missing_ok=True)
            self._tmp = None

    def commit(self) -> Path:
        if self._handle is None or self._tmp is None:
            raise RuntimeError("AtomicFileWriter has nothing to commit")
        self._handle.close()
        self._handle = None
        os.replace(self._tmp, self.target)
        self._tmp = None
        return self.target

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None or self._handle is not None:
            self.abort()
