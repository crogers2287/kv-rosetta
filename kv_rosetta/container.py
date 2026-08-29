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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from kv_rosetta import dtypes
from kv_rosetta.segments import Segment, table_digest
from kv_rosetta import segments as _segments

MAGIC = b"KVX1"
CONTAINER_VERSION = 1
SEGMENTED_VERSION = 2
SUPPORTED_VERSIONS = frozenset({1, 2})
ALIGNMENT = 64
_PREAMBLE = 12
_MAX_OFFSET_PASSES = 5
_CHUNK = 4 << 20   # 4 MiB: bounded staging for artifacts that reach gigabytes
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


def write_segments(path: Path | str, manifest: dict[str, Any],
                   parts: list[tuple[Segment, bytes]]) -> Path:
    """Write a segmented artifact (container version 2).

    Segment offsets are stored RELATIVE to the payload start. Absolute offsets would
    depend on the header length, which depends on the offsets written into it; keeping
    them relative removes that circularity so only ``blob.offset`` needs resolving.
    """
    if not parts:
        raise ContainerError("a segmented artifact needs at least one segment")
    placed: list[Segment] = []
    chunks: list[bytes] = []
    cursor = 0
    for seg, data in parts:
        pad = (-cursor) % ALIGNMENT
        if pad:
            chunks.append(b"\x00" * pad)
            cursor += pad
        if seg.nbytes != len(data):
            raise ContainerError(
                f"segment {seg.name!r} declares {seg.nbytes} bytes but carries {len(data)}")
        placed.append(replace(seg, offset=cursor, sha256=hashlib.sha256(data).hexdigest()))
        chunks.append(data)
        cursor += len(data)
    payload = b"".join(chunks)

    ok, reason = _segments.validate(placed, payload_start=0, file_size=len(payload),
                                    alignment=ALIGNMENT)
    if not ok:
        raise ContainerError(f"invalid segment table: {reason}")

    manifest = dict(manifest)
    manifest["blob"] = {
        "encoding": "segmented",
        "offset": 0,
        "nbytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shape": None,
        "opaque_format": None,
        "segment_schema": _segments.SEGMENT_SCHEMA,
        "segments": [seg.as_dict() for seg in placed],
        "representation_digest": table_digest(placed),
    }
    return _write(path, manifest, payload, version=SEGMENTED_VERSION)


def _write(path: Path | str, manifest: dict[str, Any], payload: bytes,
           version: int = CONTAINER_VERSION) -> Path:
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
        handle.write(version.to_bytes(4, "little"))
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


def _hash_file(src: Path, chunk: int = _CHUNK) -> tuple[str, int]:
    """sha256 and size of a file without holding it in memory."""
    digest = hashlib.sha256()
    total = 0
    with open(src, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def write_opaque_from_file(path: Path | str, manifest: dict[str, Any], src: Path | str,
                           opaque_format: str, chunk: int = _CHUNK) -> Path:
    """Wrap an engine-native state file without materialising it in memory.

    A 32K llama.cpp slot state is over a gigabyte; reading it with read_bytes and then
    joining it into the container payload holds two full copies of it in the Python heap
    at once. This streams src into the artifact and hashes it incrementally.
    """
    if not isinstance(opaque_format, str) or not opaque_format.strip():
        raise ContainerError("opaque_format must be a non-empty string")
    src = Path(src)
    if not src.is_file():
        raise ContainerError(f"source state file does not exist: {src}")
    sha256, nbytes = _hash_file(src, chunk)

    path = Path(path)
    manifest = dict(manifest)
    manifest["blob"] = {
        "encoding": "opaque", "offset": 0, "nbytes": nbytes, "sha256": sha256,
        "shape": None, "opaque_format": opaque_format,
    }
    manifest["integrity"] = {"header_sha256": _ZERO_DIGEST}
    header, offset = _resolve_offset(manifest)
    zeroed, value_at = _digest_placeholder(header)
    header = header[:value_at] + hashlib.sha256(zeroed).hexdigest().encode() + header[value_at + 64:]
    padding = offset - (_PREAMBLE + len(header))

    # A unique temporary name: two exports of the same artifact must not race on one path.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{id(manifest):x}.tmp")
    try:
        with open(tmp, "wb") as out, open(src, "rb") as handle:
            out.write(MAGIC)
            out.write(CONTAINER_VERSION.to_bytes(4, "little"))
            out.write(len(header).to_bytes(4, "little"))
            out.write(header)
            out.write(b"\x20" * padding)
            while True:
                block = handle.read(chunk)
                if not block:
                    break
                out.write(block)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def extract_payload(path: Path | str, dest: Path | str, chunk: int = _CHUNK) -> int:
    """Stream an artifact's payload to a file. Returns the byte count written.

    Used instead of `.opaque` when handing a blob back to a runtime, so a gigabyte-scale
    cache never becomes a single Python bytes object.
    """
    path, dest = Path(path), Path(dest)
    with open(path, "rb") as handle:
        header, _ = _read_header(handle)
        blob = header["blob"]
        offset, nbytes = int(blob["offset"]), int(blob["nbytes"])
        handle.seek(offset)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
        digest = hashlib.sha256()
        written = 0
        try:
            with open(tmp, "wb") as out:
                remaining = nbytes
                while remaining:
                    block = handle.read(min(chunk, remaining))
                    if not block:
                        raise ContainerError("payload truncated while extracting")
                    digest.update(block)
                    out.write(block)
                    remaining -= len(block)
                    written += len(block)
            if digest.hexdigest() != blob["sha256"]:
                raise ContainerError("payload sha256 mismatch while extracting")
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return written


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
    def segments(self) -> tuple[Segment, ...]:
        """Segments with offsets resolved to absolute file positions."""
        if self.encoding != "segmented":
            raise ContainerError(f"artifact encoding is {self.encoding!r}, not 'segmented'")
        base = int(self.blob["offset"])
        return tuple(
            replace(Segment.from_dict(d), offset=base + int(d["offset"]))
            for d in self.blob.get("segments", [])
        )

    def segment(self, name: str) -> np.ndarray:
        """Return one segment's data as an array of its declared dtype and shape."""
        for seg in self.segments:
            if seg.name == name:
                if not dtypes.is_canonical(seg.dtype):
                    raise ContainerError(
                        f"segment {name!r} dtype {seg.dtype!r} is opaque-only")
                start = seg.offset - int(self.blob["offset"])
                raw = np.asarray(self.buffer, dtype=np.uint8)[start:start + seg.nbytes]
                return raw.view(np.dtype(dtypes.NUMPY_DTYPE[seg.dtype])).reshape(seg.shape)
        raise ContainerError(f"no segment named {name!r}")

    def header_bytes(self, limit: int) -> bytes:
        """Read at most `limit` bytes from the start of the payload.

        Lets a caller parse an engine envelope header without materialising a payload
        that may be gigabytes.
        """
        if self.path is None:
            return bytes(np.asarray(self.buffer, dtype=np.uint8)[:limit].tobytes())
        offset = int(self.blob["offset"])
        with open(self.path, "rb") as handle:
            handle.seek(offset)
            return handle.read(min(limit, int(self.blob["nbytes"])))

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
    if version not in SUPPORTED_VERSIONS:
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


def read_header(path: Path | str) -> dict[str, Any]:
    """Parse only the header. Never touches the payload, which may be gigabytes."""
    with open(Path(path), "rb") as handle:
        header, _ = _read_header(handle)
    return header


def read_payload_prefix(path: Path | str, limit: int) -> bytes:
    """Read at most `limit` bytes of the payload, for parsing an engine envelope header."""
    with open(Path(path), "rb") as handle:
        header, _ = _read_header(handle)
        blob = header["blob"]
        handle.seek(int(blob["offset"]))
        return handle.read(min(limit, int(blob["nbytes"])))


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


def verify_header(path: Path | str) -> tuple[bool, str]:
    """Everything verify() checks except reading the payload.

    Structure, the header digest, blob field presence, payload alignment, and that the file
    is long enough for the payload it declares. What is deliberately NOT done here is the
    full payload hash - extract_payload() already hashes every byte as it writes the staged
    copy and refuses before publishing it, so verifying the payload separately means reading
    a gigabyte-scale artifact twice to learn the same fact once.

    A segmented artifact still needs the whole-payload pass, because its per-segment digests
    are not covered by extraction; callers are told to use verify() for those.
    """
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
            if not isinstance(integrity, dict) or \
                    not isinstance(integrity.get("header_sha256"), str):
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
        if blob.get("encoding") == "segmented":
            return False, ("segmented artifacts carry per-segment digests that extraction "
                           "does not check; use verify()")
        return True, "ok"
    except ContainerError as exc:
        return False, str(exc)
    except OSError as exc:
        return False, f"unreadable: {exc}"


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

        if blob.get("encoding") == "segmented":
            try:
                segs = [Segment.from_dict(d) for d in blob.get("segments", [])]
            except _segments.SegmentError as exc:
                return False, str(exc)
            ok, reason = _segments.validate(
                segs, payload_start=0, file_size=nbytes, alignment=ALIGNMENT)
            if not ok:
                return False, reason
            if blob.get("representation_digest") not in (None, "", table_digest(segs)):
                return False, "representation_digest does not match the segment table"
            with open(path, "rb") as body:
                for seg in segs:
                    body.seek(offset + seg.offset)
                    data = body.read(seg.nbytes)
                    if len(data) != seg.nbytes:
                        return False, f"segment {seg.name!r} truncated"
                    if seg.sha256 and hashlib.sha256(data).hexdigest() != seg.sha256:
                        return False, f"segment {seg.name!r} sha256 mismatch"
        return True, "ok"
    except ContainerError as exc:
        return False, str(exc)
    except OSError as exc:
        return False, f"unreadable: {exc}"
