"""KVX binary container.

One file carries a manifest header plus a single contiguous payload. The payload is
either canonical tensors (``encoding == "raw"``) or an engine-native blob
(``encoding == "opaque"``), which is what lets an adapter that can only save an opaque
state and an adapter that can emit canonical tensors share one artifact format.

Layout, little-endian throughout::

    0                        magic b"KVX1"                       4 bytes
    4                        uint32 container_version = 1
    8                        uint32 header_len
    12                       header JSON, UTF-8, header_len bytes
                             pad 0x20 up to the next 64-byte boundary
    align64(12+header_len)   payload
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kv_rosetta import dtypes

MAGIC = b"KVX1"
CONTAINER_VERSION = 1
ALIGNMENT = 64
_PREAMBLE = 12
_MAX_OFFSET_PASSES = 5
_ZERO_DIGEST = "0" * 64
_DIGEST_KEY = b'"header_sha256":"'


def _digest_placeholder(header: bytes) -> tuple[bytes, int]:
    """Locate the header digest field so it can be zeroed for hashing.

    The digest covers the header that contains it, so it is written as a fixed-width
    placeholder, hashed with the placeholder in place, then overwritten. Length never
    changes, so the payload offset stays valid.
    """
    start = header.find(_DIGEST_KEY)
    if start < 0:
        raise ContainerError("header is missing integrity.header_sha256")
    value_at = start + len(_DIGEST_KEY)
    return header[:value_at] + _ZERO_DIGEST.encode() + header[value_at + 64:], value_at


class ContainerError(ValueError):
    pass


def _align(value: int) -> int:
    remainder = value % ALIGNMENT
    return value if remainder == 0 else value + (ALIGNMENT - remainder)


def _encode_header(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _resolve_offset(manifest: dict[str, Any]) -> tuple[bytes, int]:
    """Settle the circular dependency between header length and payload offset.

    ``blob.offset`` lives inside the header, so writing it changes the header's length,
    which changes the offset. Iterate until it stops moving; a couple of passes always
    suffice, and the cap turns a pathological case into an error rather than a hang.
    """
    offset = 0
    for _ in range(_MAX_OFFSET_PASSES):
        manifest["blob"]["offset"] = offset
        header = _encode_header(manifest)
        settled = _align(_PREAMBLE + len(header))
        if settled == offset:
            return header, offset
        offset = settled
    raise ContainerError("payload offset did not converge")


def _write(path: Path | str, manifest: dict[str, Any], payload: bytes) -> Path:
    path = Path(path)
    # Integrity must cover the semantic header too, not only the payload: dtype,
    # shape, offsets and model identity must not be mutable without detection.
    manifest = dict(manifest)
    manifest["integrity"] = {"header_sha256": _ZERO_DIGEST}
    header, offset = _resolve_offset(manifest)
    zeroed, value_at = _digest_placeholder(header)
    digest = hashlib.sha256(zeroed).hexdigest()
    header = header[:value_at] + digest.encode() + header[value_at + 64:]
    padding = offset - (_PREAMBLE + len(header))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(MAGIC)
        handle.write(CONTAINER_VERSION.to_bytes(4, "little"))
        handle.write(len(header).to_bytes(4, "little"))
        handle.write(header)
        handle.write(b"\x20" * padding)
        handle.write(payload)
    os.replace(tmp, path)
    return path


def write_raw(path: Path | str, manifest: dict[str, Any], tensor: np.ndarray) -> Path:
    if tensor.ndim != 5:
        raise ContainerError(f"raw tensor must be 5-D (layer,kv,token,head,dim), got {tensor.ndim}-D")
    if tensor.shape[1] != 2:
        raise ContainerError(f"axis 1 must be the kv axis of size 2, got {tensor.shape[1]}")
    tensor = np.ascontiguousarray(tensor)
    payload = tensor.tobytes()
    manifest = dict(manifest)
    manifest["blob"] = {
        "encoding": "raw",
        "offset": 0,
        "nbytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shape": list(tensor.shape),
        "opaque_format": None,
    }
    return _write(path, manifest, payload)


def write_opaque(path: Path | str, manifest: dict[str, Any], blob: bytes,
                 opaque_format: str) -> Path:
    if not isinstance(opaque_format, str) or not opaque_format.strip():
        raise ContainerError("opaque_format must be a non-empty string")
    manifest = dict(manifest)
    manifest["blob"] = {
        "encoding": "opaque",
        "offset": 0,
        "nbytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "shape": None,
        "opaque_format": opaque_format,
    }
    return _write(path, manifest, bytes(blob))


@dataclass
class KVXArtifact:
    header: dict[str, Any]
    path: Path | None = None
    buffer: Any = field(default=None, repr=False)

    @property
    def blob(self) -> dict[str, Any]:
        return self.header["blob"]

    @property
    def encoding(self) -> str:
        return self.blob["encoding"]

    @property
    def tensor(self) -> np.ndarray:
        if self.encoding != "raw":
            raise ContainerError(f"artifact encoding is {self.encoding!r}, not 'raw'")
        kv = self.header["kv"]
        name = kv["dtype"]
        if not dtypes.is_canonical(name):
            raise ContainerError(f"kv.dtype {name!r} is opaque-only and has no canonical tensor")
        raw = np.asarray(self.buffer, dtype=np.uint8)
        return raw.view(np.dtype(dtypes.NUMPY_DTYPE[name])).reshape(tuple(self.blob["shape"]))

    @property
    def opaque(self) -> bytes:
        if self.encoding != "opaque":
            raise ContainerError(f"artifact encoding is {self.encoding!r}, not 'opaque'")
        return bytes(np.asarray(self.buffer, dtype=np.uint8).tobytes())


def _read_header(handle) -> tuple[dict[str, Any], int]:
    preamble = handle.read(_PREAMBLE)
    if len(preamble) < _PREAMBLE:
        raise ContainerError("file is shorter than the 12-byte preamble")
    if preamble[:4] != MAGIC:
        raise ContainerError(f"bad magic {preamble[:4]!r}, expected {MAGIC!r}")
    version = int.from_bytes(preamble[4:8], "little")
    if version != CONTAINER_VERSION:
        raise ContainerError(f"unsupported container version {version}")
    header_len = int.from_bytes(preamble[8:12], "little")
    raw = handle.read(header_len)
    if len(raw) < header_len:
        raise ContainerError("header truncated")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContainerError(f"header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ContainerError("header must be a JSON object")
    if not isinstance(header.get("blob"), dict):
        raise ContainerError("header is missing a 'blob' object")
    return header, header_len


def read(path: Path | str, mmap: bool = True) -> KVXArtifact:
    path = Path(path)
    with open(path, "rb") as handle:
        header, _ = _read_header(handle)
        blob = header["blob"]
        offset, nbytes = int(blob["offset"]), int(blob["nbytes"])
        if path.stat().st_size < offset + nbytes:
            raise ContainerError("file is shorter than the declared payload")
        if mmap and nbytes:
            buffer = np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(nbytes,))
        else:
            handle.seek(offset)
            buffer = np.frombuffer(handle.read(nbytes), dtype=np.uint8)
    return KVXArtifact(header=header, path=path, buffer=buffer)


def verify(path: Path | str) -> tuple[bool, str]:
    """Structural + integrity check. Never raises: callers get a reason, not a traceback."""
    path = Path(path)
    try:
        if not path.is_file():
            return False, "not a file"
        with open(path, "rb") as handle:
            header, header_len = _read_header(handle)
            blob = header["blob"]
            for key in ("encoding", "offset", "nbytes", "sha256"):
                if key not in blob:
                    return False, f"blob.{key} missing"
            integrity = header.get("integrity")
            if not isinstance(integrity, dict) or not isinstance(integrity.get("header_sha256"), str):
                return False, "integrity.header_sha256 missing"
            handle.seek(0)
            raw_header = handle.read(_PREAMBLE + header_len)[_PREAMBLE:]
            try:
                zeroed, _ = _digest_placeholder(raw_header)
            except ContainerError as exc:
                return False, str(exc)
            if hashlib.sha256(zeroed).hexdigest() != integrity["header_sha256"]:
                return False, "header sha256 mismatch"
            offset, nbytes = int(blob["offset"]), int(blob["nbytes"])
            if offset % ALIGNMENT:
                return False, f"payload offset {offset} is not {ALIGNMENT}-byte aligned"
            if path.stat().st_size < offset + nbytes:
                return False, "file is shorter than the declared payload"
            handle.seek(offset)
            digest = hashlib.sha256()
            remaining = nbytes
            while remaining:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    return False, "payload truncated"
                digest.update(chunk)
                remaining -= len(chunk)
        if digest.hexdigest() != blob["sha256"]:
            return False, "payload sha256 mismatch"
        return True, "ok"
    except ContainerError as exc:
        return False, str(exc)
    except OSError as exc:
        return False, f"unreadable: {exc}"
