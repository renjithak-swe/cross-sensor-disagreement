"""SHA-256 helpers for files, bytes and JSON-serialisable values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK_BYTES = 1 << 20


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash of a canonical JSON encoding: sorted keys, no whitespace."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(canonical.encode("utf-8"))
