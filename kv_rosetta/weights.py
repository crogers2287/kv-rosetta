"""Content-derived weight identity, with a cache so it is affordable.

A filesystem path is not identity. The same weights moved or renamed are the same model,
and different weights written behind the same name are a different model - and a KV cache
restored against the wrong weights is silently wrong rather than loudly broken.

Hashing the bytes is the only honest answer, but a 15 GB GGUF takes about a minute to read
and a 100 GB model far longer, so a digest is computed once and cached against strong file
metadata. Rename does not disturb (device, inode, size, mtime); rewriting the file does.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 4 << 20
_SCHEMA = """
CREATE TABLE IF NOT EXISTS digests (
    device INTEGER NOT NULL,
    inode  INTEGER NOT NULL,
    size   INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY (device, inode, size, mtime_ns)
);
"""


class WeightsError(ValueError):
    pass


@dataclass(frozen=True)
class FileStamp:
    """Strong metadata identifying one file instance.

    Deliberately excludes the path: renaming a file changes none of these, which is exactly
    the behaviour required of a content identity.
    """

    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def of(cls, path: Path | str) -> "FileStamp":
        try:
            info = os.stat(path)
        except OSError as exc:
            raise WeightsError(f"cannot stat {path}: {exc}") from exc
        return cls(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def default_cache() -> Path:
    root = os.environ.get("KVROSETTA_CACHE") or (Path.home() / ".kvrosetta")
    return Path(root) / "digests.sqlite"


class DigestCache:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_cache()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, stamp: FileStamp) -> str | None:
        row = self._conn.execute(
            "SELECT digest FROM digests WHERE device=? AND inode=? AND size=? AND mtime_ns=?",
            (stamp.device, stamp.inode, stamp.size, stamp.mtime_ns),
        ).fetchone()
        return row[0] if row else None

    def put(self, stamp: FileStamp, digest: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO digests VALUES (?,?,?,?,?)",
            (stamp.device, stamp.inode, stamp.size, stamp.mtime_ns, digest),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def hash_file(path: Path | str, chunk: int = _CHUNK) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def content_digest(path: Path | str, cache: DigestCache | None = None) -> str:
    """sha256 of a file's contents, cached against its metadata stamp."""
    path = Path(path)
    stamp = FileStamp.of(path)
    owned = cache is None
    cache = cache or DigestCache()
    try:
        cached = cache.get(stamp)
        if cached:
            return cached
        digest = hash_file(path)
        cache.put(stamp, digest)
        return digest
    finally:
        if owned:
            cache.close()


def shards_of(path: Path | str) -> tuple[Path, ...]:
    """All files belonging to a model, including split GGUF shards.

    A sharded model whose identity came from only its first shard would collide with any
    other model sharing that shard.
    """
    path = Path(path)
    name = path.name
    marker = "-00001-of-"
    if marker in name:
        stem, _, tail = name.partition(marker)
        total = tail.split(".")[0]
        found = sorted(path.parent.glob(f"{stem}-*-of-{total}*"))
        if found:
            return tuple(found)
    return (path,)


def model_content_digest(path: Path | str, cache: DigestCache | None = None) -> str:
    """Identity of a model's bytes: a digest over its ordered shard inventory.

    Each shard contributes its ORDINAL POSITION, size and content digest - never its name
    or directory. Including the filename would make a rename change identity, which is
    precisely what content identity must not do; position still makes shard reordering or
    substitution detectable.
    """
    owned = cache is None
    cache = cache or DigestCache()
    try:
        inventory = []
        for index, shard in enumerate(shards_of(path)):
            # FileStamp.of raises WeightsError for an unreadable file, so callers get one
            # error type rather than a bare OSError from an incidental stat().
            stamp = FileStamp.of(shard)
            inventory.append({
                "index": index,
                "size": stamp.size,
                "sha256": content_digest(shard, cache),
            })
        payload = json.dumps({"shards": inventory}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
    finally:
        if owned:
            cache.close()
