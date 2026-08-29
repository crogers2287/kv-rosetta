"""EXPERIMENTAL, LOCAL ONLY: an adapter-owned store of pre-admitted raw state objects.

The portable KVX path verifies every payload byte on import, which costs a full pass over a
gigabyte-scale artifact on the request path. This store trades that for a two-phase contract:

1. **Admission**, off the request path. Every byte and every identity claim is validated,
   then the object is published atomically into a private directory.
2. **Restore**, on the request path. Only an already-admitted object may be resolved, and it
   is restored in place with no payload copy and no full payload read.

Trust boundary, stated rather than assumed
------------------------------------------
An admitted object may skip rehashing ONLY while all of these hold:

* it lives in a store directory owned by this user with mode 0700;
* the path resolves inside that directory, is a regular file, and is not a symlink;
* every pinned fact still matches: device, inode, size, mtime_ns, ctime_ns, link count;
* its manifest is present, parses, and is bound to the full SHA-256 content address.

A basename, a read-only bit, a prior hash, a content-looking filename, or a manifest alone
is NOT admission. None of those establish that the bytes on disk are the bytes admitted.

Residual risk that this contract does NOT close
-----------------------------------------------
llama-server restores by basename relative to its slot directory, so it opens the file
itself, after we have checked it. Between our final stat and the server's open there is a
window no user-space check can remove. The pinned-fact check narrows it to an attacker who
can write inside a 0700 directory owned by this user - i.e. this user or root - which is the
stated local threat model, not a general one.

Closing it properly needs something this layer cannot provide: fs-verity on the object, or
an upstream restore-from-open-fd / verified-offset primitive. That is recorded here rather
than papered over, because narrowing the threat model silently would be the failure this
whole project exists to avoid.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The store is private to this user. Anything looser and the pinned facts prove nothing.
STORE_MODE = 0o700

MANIFEST_SUFFIX = ".manifest.json"
OBJECT_SUFFIX = ".state"
_CHUNK = 4 << 20


class AdmissionError(ValueError):
    """Raised when a raw state cannot be admitted, or an admitted object no longer holds."""


def _file_facts(fd_or_path) -> dict[str, int]:
    st = os.fstat(fd_or_path) if isinstance(fd_or_path, int) else os.stat(fd_or_path)
    return {"device": st.st_dev, "inode": st.st_ino, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns, "nlink": st.st_nlink}


@dataclass(frozen=True)
class AdmittedObject:
    """A state object that passed admission and still satisfies its pinned facts."""

    digest: str
    path: Path
    manifest: dict[str, Any]
    facts: dict[str, int] = field(default_factory=dict)

    @property
    def basename(self) -> str:
        """What llama-server is given, relative to the slot directory."""
        return self.path.name


class AdmittedStore:
    """A private directory of admitted raw state objects, addressed by full SHA-256."""

    def __init__(self, root: Path | str, *, create: bool = True) -> None:
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            os.chmod(self.root, STORE_MODE)
        self._require_private()

    def _require_private(self) -> None:
        if not self.root.is_dir():
            raise AdmissionError(f"store {self.root} is not a directory")
        mode = stat.S_IMODE(self.root.stat().st_mode)
        if mode & 0o077:
            raise AdmissionError(
                f"store {self.root} has mode {mode:04o}; group/other access means the "
                f"pinned facts prove nothing about who wrote the bytes")

    # -- admission -----------------------------------------------------------------

    def admit(self, raw: Path | str, manifest: dict[str, Any]) -> AdmittedObject:
        """Copy-verify a raw state into the store and publish it atomically.

        The digest is computed while writing, so the bytes hashed are the bytes stored -
        hashing the source separately would leave a window where they could differ. The
        manifest is written only after the object is durable, and the object is removed if
        anything fails, so a partial publish is never resolvable.
        """
        raw = Path(raw)
        digest = hashlib.sha256()
        tmp = self.root / f".incoming.{os.getpid()}.tmp"
        written = 0
        try:
            with open(raw, "rb") as src, open(tmp, "wb") as dst:
                while True:
                    block = src.read(_CHUNK)
                    if not block:
                        break
                    digest.update(block)
                    dst.write(block)
                    written += len(block)
                dst.flush()
                os.fsync(dst.fileno())
            content = digest.hexdigest()
            if manifest.get("raw_sha256") not in (None, content):
                raise AdmissionError(
                    f"declared raw_sha256 {manifest['raw_sha256'][:16]} does not match the "
                    f"bytes written ({content[:16]})")
            if manifest.get("raw_size") not in (None, written):
                raise AdmissionError(
                    f"declared raw_size {manifest['raw_size']} does not match {written}")
            target = self.root / f"{content}{OBJECT_SUFFIX}"
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
            self._fsync_dir()
            published = dict(manifest, raw_sha256=content, raw_size=written)
            published["file_facts"] = _file_facts(target)
            self._write_manifest(content, published)
            return AdmittedObject(digest=content, path=target, manifest=published,
                                  facts=published["file_facts"])
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _write_manifest(self, digest: str, manifest: dict[str, Any]) -> None:
        path = self.root / f"{digest}{MANIFEST_SUFFIX}"
        tmp = path.with_suffix(".tmp")
        body = json.dumps(manifest, indent=2, sort_keys=True).encode()
        with open(tmp, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        self._fsync_dir()

    def _fsync_dir(self) -> None:
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    # -- resolution ----------------------------------------------------------------

    def resolve(self, digest: str) -> AdmittedObject:
        """Return an admitted object, or refuse. Never rehashes; never trusts the name.

        Opened with O_NOFOLLOW so a symlink substituted for the object is refused rather
        than followed, and every check is made against the open descriptor so the file
        inspected is the file opened.
        """
        if not isinstance(digest, str) or len(digest) != 64 or \
                any(c not in "0123456789abcdef" for c in digest):
            raise AdmissionError(f"not a full sha256 content address: {digest!r}")
        self._require_private()
        path = self.root / f"{digest}{OBJECT_SUFFIX}"
        # Resolve and confine before opening: a traversal cannot be checked afterwards.
        resolved = Path(os.path.realpath(path))
        if resolved.parent != Path(os.path.realpath(self.root)):
            raise AdmissionError(f"{path} resolves outside the store to {resolved}")
        manifest_path = self.root / f"{digest}{MANIFEST_SUFFIX}"
        if not manifest_path.is_file():
            raise AdmissionError(f"no manifest for {digest[:16]}; not admitted")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            raise AdmissionError(f"manifest for {digest[:16]} is unreadable: {exc}") from exc
        if manifest.get("raw_sha256") != digest:
            raise AdmissionError(
                f"manifest is bound to {str(manifest.get('raw_sha256'))[:16]}, not "
                f"{digest[:16]}; refusing a substituted manifest")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            raise AdmissionError(f"cannot open admitted object {digest[:16]}: {exc}") from exc
        try:
            mode = os.fstat(fd).st_mode
            if not stat.S_ISREG(mode):
                raise AdmissionError(f"{path} is not a regular file")
            facts = _file_facts(fd)
        finally:
            os.close(fd)
        pinned = manifest.get("file_facts")
        if not isinstance(pinned, dict):
            raise AdmissionError(f"manifest for {digest[:16]} pins no file facts")
        differing = [f"{k}: admitted {pinned.get(k)!r} now {facts.get(k)!r}"
                     for k in ("device", "inode", "size", "mtime_ns", "ctime_ns", "nlink")
                     if pinned.get(k) != facts.get(k)]
        if differing:
            raise AdmissionError(
                "admitted object no longer matches what was admitted: "
                + "; ".join(differing))
        return AdmittedObject(digest=digest, path=path, manifest=manifest, facts=facts)

    def verify_bytes(self, digest: str) -> bool:
        """Rehash an object. NOT used on the request path; for tests and audits only."""
        path = self.root / f"{digest}{OBJECT_SUFFIX}"
        running = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                block = handle.read(_CHUNK)
                if not block:
                    break
                running.update(block)
        return running.hexdigest() == digest
