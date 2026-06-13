from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class PutObjectResult:
    object_key: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class LocalObjectMeta:
    content_type: str


class LocalObjectStore:
    """
    Minimal dev-friendly object store.

    Production should use S3 (presigned PUT) so the API never handles raw audio bytes.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put_base64(self, *, object_key: str, content_base64: str) -> PutObjectResult:
        s = (content_base64 or "").strip()
        if s.startswith("data:") and "base64," in s:
            s = s.split("base64,", 1)[-1].strip()
        pad = (-len(s)) % 4
        if pad:
            s += "=" * pad
        raw = base64.b64decode(s, validate=False)
        digest = hashlib.sha256(raw).hexdigest()
        path = self.base_dir / object_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return PutObjectResult(object_key=object_key, sha256=digest, size_bytes=len(raw))

    def get_path(self, object_key: str) -> Path:
        return self.base_dir / object_key

    def read_bytes(self, object_key: str, *, content_type: str = "application/octet-stream") -> Tuple[bytes, LocalObjectMeta]:
        path = self.get_path(object_key)
        if not path.is_file():
            raise FileNotFoundError(object_key)
        return path.read_bytes(), LocalObjectMeta(content_type=content_type)


def resolve_local_store() -> LocalObjectStore:
    base = os.environ.get("LOCAL_OBJECT_STORE_DIR", "").strip()
    if base:
        return LocalObjectStore(Path(base))
    # default: backend/objects
    here = Path(__file__).resolve().parents[1]
    return LocalObjectStore(here / "objects")


def sniff_audio_content_type(filename: str, provided: Optional[str] = None) -> str:
    if provided and "/" in provided:
        return provided
    name = (filename or "").lower()
    if name.endswith(".m4a"):
        return "audio/mp4"
    if name.endswith(".wav"):
        return "audio/wav"
    if name.endswith(".mp3"):
        return "audio/mpeg"
    return "application/octet-stream"

