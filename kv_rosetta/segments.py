"""Segment table: the model-agnostic payload description.

A single homogeneous ``layer,kv,token,head,dim`` tensor describes standard MHA/GQA
attention and nothing else. It cannot express different K and V dtypes, quantized
values with their scales, DeepSeek MLA compressed state, hybrid attention/recurrent
state, cross-attention state, or per-layer differences in shape or cache kind. A format
that can only describe one attention family is not a model-agnostic boundary, whatever
its manifest claims.

A KVX file therefore still carries one contiguous payload, but the header describes
named, non-overlapping segments within it. Adding a cache kind becomes a new segment
role rather than a new file format.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from math import prod
from typing import Any

from kv_rosetta import dtypes

SEGMENT_SCHEMA = "kvx-segments/1"

#: Recognised roles. Unknown roles are refused rather than passed through: an importer
#: that silently ignores a segment it does not understand would restore a partial cache
#: and call it a success.
ROLES = frozenset({
    "k",            # keys
    "v",            # values
    "kv",           # interleaved keys and values
    "scales",       # quantization scales / zero points for a sibling segment
    "compressed",   # MLA-style compressed latent state
    "recurrent",    # SSM / Mamba recurrent state
    "conv",         # short convolution state
    "cross",        # cross-attention state
    "aux",          # runtime-specific auxiliary state
})


class SegmentError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    """One described region of the payload.

    ``layer_start``/``layer_end`` are a half-open range so a single segment can cover one
    layer, a contiguous band, or a whole model without a separate encoding.
    """

    name: str
    role: str
    layer_start: int
    layer_end: int
    dtype: str
    shape: tuple[int, ...]
    layout: str
    offset: int
    nbytes: int
    sha256: str = ""
    byte_order: str = "little"
    quant: dict[str, Any] = field(default_factory=dict)
    position: dict[str, Any] = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.offset + self.nbytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": self.layout,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "sha256": self.sha256,
            "byte_order": self.byte_order,
            "quant": self.quant,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Segment":
        if not isinstance(data, dict):
            raise SegmentError("segment must be a JSON object")
        try:
            return cls(
                name=data["name"],
                role=data["role"],
                layer_start=data["layer_start"],
                layer_end=data["layer_end"],
                dtype=data["dtype"],
                shape=tuple(data["shape"]),
                layout=data["layout"],
                offset=data["offset"],
                nbytes=data["nbytes"],
                sha256=data.get("sha256", ""),
                byte_order=data.get("byte_order", "little"),
                quant=data.get("quant", {}) or {},
                position=data.get("position", {}) or {},
            )
        except (KeyError, TypeError) as exc:
            raise SegmentError(f"malformed segment: {exc}") from exc


def _check_one(seg: Segment, index: int) -> str | None:
    where = f"segment[{index}] {seg.name!r}"
    if not isinstance(seg.name, str) or not seg.name.strip():
        return f"{where}: name must be a non-empty string"
    if seg.role not in ROLES:
        return f"{where}: unknown role {seg.role!r}, expected one of {sorted(ROLES)}"
    if seg.byte_order != "little":
        return f"{where}: byte_order must be 'little', got {seg.byte_order!r}"
    for name, value in (("layer_start", seg.layer_start), ("layer_end", seg.layer_end),
                        ("offset", seg.offset), ("nbytes", seg.nbytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f"{where}: {name} must be a non-negative integer, got {value!r}"
    if seg.layer_end < seg.layer_start:
        return f"{where}: layer_end {seg.layer_end} precedes layer_start {seg.layer_start}"
    if not isinstance(seg.shape, tuple) or not seg.shape:
        return f"{where}: shape must be a non-empty tuple"
    for dim in seg.shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            return f"{where}: shape entries must be non-negative integers, got {seg.shape!r}"
    if not isinstance(seg.layout, str) or not seg.layout.strip():
        return f"{where}: layout must be a non-empty string"
    # A declared size that disagrees with dtype x shape means one of the two is a lie, and
    # the importer would read the wrong number of bytes for every following segment.
    if dtypes.is_canonical(seg.dtype):
        expected = dtypes.itemsize(seg.dtype) * prod(seg.shape)
        if seg.nbytes != expected:
            return (f"{where}: nbytes {seg.nbytes} does not match dtype {seg.dtype} "
                    f"x shape {seg.shape} = {expected}")
    elif seg.dtype not in dtypes.OPAQUE_ONLY:
        return f"{where}: unknown dtype {seg.dtype!r}"
    return None


def validate(segments: list[Segment] | tuple[Segment, ...], *, payload_start: int,
             file_size: int, alignment: int = 64) -> tuple[bool, str]:
    """Structural validation. Returns ``(ok, reason)`` and never raises.

    Checked before any segment is exposed to an adapter, because importing first and
    verifying afterward means the runtime has already seen bad state.
    """
    try:
        if not segments:
            return False, "segment table is empty"
        seen: set[str] = set()
        for index, seg in enumerate(segments):
            if not isinstance(seg, Segment):
                return False, f"segment[{index}] is not a Segment"
            problem = _check_one(seg, index)
            if problem:
                return False, problem
            if seg.name in seen:
                return False, f"duplicate segment name {seg.name!r}"
            seen.add(seg.name)
            if seg.offset < payload_start:
                return False, (f"segment[{index}] {seg.name!r}: offset {seg.offset} "
                               f"begins before the payload at {payload_start}")
            if alignment and seg.offset % alignment:
                return False, (f"segment[{index}] {seg.name!r}: offset {seg.offset} "
                               f"is not {alignment}-byte aligned")
            if seg.end > file_size:
                return False, (f"segment[{index}] {seg.name!r}: extends to {seg.end}, "
                               f"past end of file at {file_size}")
        ordered = sorted(segments, key=lambda s: s.offset)
        for previous, current in zip(ordered, ordered[1:]):
            if current.offset < previous.end:
                return False, (f"segments {previous.name!r} and {current.name!r} overlap: "
                               f"[{previous.offset},{previous.end}) and "
                               f"[{current.offset},{current.end})")
        return True, "ok"
    except Exception as exc:  # validation must never raise on hostile input
        return False, f"segment validation failed: {exc}"


def table_digest(segments: list[Segment] | tuple[Segment, ...]) -> str:
    """Representation digest: identity of the payload's SHAPE, not its bytes.

    Two artifacts holding the same cache in different representations must not collide in
    the store, so this digest participates in the composite artifact key.
    """
    payload = json.dumps(
        {"schema": SEGMENT_SCHEMA,
         "segments": [s.as_dict() for s in sorted(segments, key=lambda s: s.name)]},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_kv(layers: int, kv_heads: int, head_dim: int, tokens: int, dtype: str,
                 offset: int) -> list[Segment]:
    """The standard decoder-only attention case expressed as segments.

    Keeps the previous fixed ``layer,kv,token,head,dim`` tensor available as a single
    segment, so kvx/0.2 artifacts have an exact equivalent under the new table.
    """
    shape = (layers, 2, tokens, kv_heads, head_dim)
    nbytes = dtypes.itemsize(dtype) * prod(shape) if dtypes.is_canonical(dtype) else 0
    return [Segment(
        name="kv", role="kv", layer_start=0, layer_end=layers, dtype=dtype, shape=shape,
        layout="layer,kv,token,head,dim", offset=offset, nbytes=nbytes,
    )]
