"""Canonical schema for hybrid attention + recurrent state. Version 1.

`docs/ggsq-layout.md` showed that a llama.cpp slot state file is not self-describing: it
records dtype and row stride but not the head split, not layer identity, not RoPE state, and
not which recurrent layers were skipped. Those come from the source GGUF at ingestion.

The point of this schema is that they are resolved **once**, at decode time, and written into
the artifact — so a target runtime can reuse the cache without access to the source GGUF.
An artifact that needed the original model file to be interpreted would not be portable, it
would be a pointer to a machine.

Nothing here decodes anything. This is the container contract the decoder must satisfy,
defined first because the current segment table cannot hold a recurrent layer map or a
checkpoint record, and a decoder written against a container that cannot store its output
would have to be rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA = "kvx-hybrid/1"

#: Canonical ordering for attention tensors. Not negotiable per artifact: a source that
#: stored values transposed is normalised at decode time rather than recorded as a variant,
#: so two artifacts of the same cache are byte-comparable.
CANONICAL_LAYOUT = "layer,kv,token,head,dim"

#: RoPE application state. A cache whose keys are post-RoPE cannot be mapped to another
#: model's positions without stripping it first, and the source file never records which
#: it is - so an artifact that does not state this is unusable, not merely underspecified.
ROPE_STATES = frozenset({"applied", "not_applied"})


class SchemaError(ValueError):
    """Raised when hybrid state cannot be represented or verified. Never a warning."""


@dataclass(frozen=True)
class SourceGeometry:
    """Model facts resolved from the exact source GGUF during ingestion.

    Every field here is something the state file does not contain. They are recorded so the
    target never needs the source model to interpret the artifact.
    """

    architecture: str
    n_layer: int
    n_head_kv: tuple[int, ...]        # per layer; GQA groups vary by layer in some models
    n_embd_head_k: int
    n_embd_head_v: int
    n_embd_r: int                     # recurrent R width, 0 when the model has none
    n_embd_s: int
    has_cell_ext: bool                # whether each cell carries a llama_kv_cell_ext
    rope_state: str
    rope_theta: float
    model_weights_sha256: str
    gguf_content_digest: str

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.architecture:
            problems.append("architecture is empty; the layer map cannot be trusted")
        if self.n_layer <= 0:
            problems.append(f"n_layer {self.n_layer} is not positive")
        if len(self.n_head_kv) != self.n_layer:
            problems.append(f"n_head_kv has {len(self.n_head_kv)} entries for "
                            f"{self.n_layer} layers; head geometry is ambiguous")
        if any(h <= 0 for h in self.n_head_kv):
            problems.append("a layer reports zero or negative KV heads")
        if self.n_embd_head_k <= 0 or self.n_embd_head_v <= 0:
            problems.append("head dimension is not positive; the head split is ambiguous")
        if self.rope_state not in ROPE_STATES:
            problems.append(f"rope_state {self.rope_state!r} is not one of "
                            f"{sorted(ROPE_STATES)}; the source file never records it, so "
                            f"it must be resolved at ingestion, not guessed")
        for name in ("model_weights_sha256", "gguf_content_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                problems.append(f"{name} is not a 64-character digest; without it the "
                                f"geometry cannot be tied to the model it came from")
        return problems


@dataclass(frozen=True)
class RecurrentLayerMap:
    """Which source layers carry recurrent tensors.

    The writer skips null recurrent layers with no marker, so the position of a tensor in
    the stream does not identify its layer. Without this map the payloads are unattributable.
    """

    source_layers: tuple[int, ...]

    def validate(self, n_layer: int) -> list[str]:
        problems: list[str] = []
        if len(set(self.source_layers)) != len(self.source_layers):
            problems.append("recurrent layer map repeats a layer")
        if any(not 0 <= i < n_layer for i in self.source_layers):
            problems.append(f"recurrent layer map references a layer outside 0..{n_layer-1}")
        if list(self.source_layers) != sorted(self.source_layers):
            problems.append("recurrent layer map is not in ascending source order")
        return problems


@dataclass(frozen=True)
class CheckpointRecord:
    """One context checkpoint, with its recurrent state in canonical form.

    The source carries `data_tgt` as another native serialization. It is NOT forwarded as an
    opaque blob: an opaque payload inside a canonical artifact would be a portability claim
    the format cannot honour, so it is parsed into recurrent segments like any other state.
    """

    n_tokens: int
    pos_min: int
    pos_max: int
    recurrent_segments: tuple[str, ...]     # names in the artifact's segment table
    has_draft_state: bool = False
    has_speculative_state: bool = False

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.n_tokens <= 0:
            problems.append(f"checkpoint covers {self.n_tokens} tokens")
        if self.pos_min < 0 or self.pos_max < self.pos_min:
            problems.append(f"checkpoint positions {self.pos_min}..{self.pos_max} invalid")
        if not self.recurrent_segments:
            problems.append("checkpoint names no recurrent segments; a hybrid model cannot "
                            "reuse a prefix without them")
        if self.has_draft_state or self.has_speculative_state:
            problems.append("checkpoint carries draft or speculative state, whose "
                            "restoration is not behaviourally proven; refusing rather than "
                            "forwarding state nothing has verified")
        return problems


@dataclass(frozen=True)
class HybridState:
    """The complete canonical description of one hybrid prefix cache."""

    schema: str = SCHEMA
    geometry: SourceGeometry | None = None
    layer_map: RecurrentLayerMap | None = None
    checkpoints: tuple[CheckpointRecord, ...] = ()
    attention_segments: tuple[str, ...] = ()
    recurrent_segments: tuple[str, ...] = ()
    layout: str = CANONICAL_LAYOUT
    notes: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.schema != SCHEMA:
            problems.append(f"schema {self.schema!r} is not {SCHEMA!r}")
        if self.layout != CANONICAL_LAYOUT:
            problems.append(f"layout {self.layout!r} is not canonical {CANONICAL_LAYOUT!r}; "
                            f"a transposed source must be normalised at decode, not "
                            f"recorded as a variant")
        if self.geometry is None:
            problems.append("no source geometry; the artifact would need the original GGUF "
                            "to be interpreted, which is not portability")
        else:
            problems += self.geometry.validate()
        if not self.attention_segments:
            problems.append("no attention segments")
        if self.geometry is not None and self.geometry.n_embd_r > 0:
            if self.layer_map is None:
                problems.append("model has recurrent state but no layer map")
            else:
                problems += self.layer_map.validate(self.geometry.n_layer)
            if not self.recurrent_segments:
                problems.append("model has recurrent state but no recurrent segments")
        if not self.checkpoints:
            problems.append("no checkpoint records; a hybrid prefix cannot be reused "
                            "without at least one")
        for index, checkpoint in enumerate(self.checkpoints):
            problems += [f"checkpoint {index}: {p}" for p in checkpoint.validate()]
        return problems

    def require_valid(self) -> HybridState:
        problems = self.validate()
        if problems:
            raise SchemaError("; ".join(problems))
        return self

    # -- round trip ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "layout": self.layout,
            "geometry": None if self.geometry is None else {
                "architecture": self.geometry.architecture,
                "n_layer": self.geometry.n_layer,
                "n_head_kv": list(self.geometry.n_head_kv),
                "n_embd_head_k": self.geometry.n_embd_head_k,
                "n_embd_head_v": self.geometry.n_embd_head_v,
                "n_embd_r": self.geometry.n_embd_r,
                "n_embd_s": self.geometry.n_embd_s,
                "has_cell_ext": self.geometry.has_cell_ext,
                "rope_state": self.geometry.rope_state,
                "rope_theta": self.geometry.rope_theta,
                "model_weights_sha256": self.geometry.model_weights_sha256,
                "gguf_content_digest": self.geometry.gguf_content_digest,
            },
            "layer_map": None if self.layer_map is None else
                {"source_layers": list(self.layer_map.source_layers)},
            "checkpoints": [{
                "n_tokens": c.n_tokens, "pos_min": c.pos_min, "pos_max": c.pos_max,
                "recurrent_segments": list(c.recurrent_segments),
                "has_draft_state": c.has_draft_state,
                "has_speculative_state": c.has_speculative_state,
            } for c in self.checkpoints],
            "attention_segments": list(self.attention_segments),
            "recurrent_segments": list(self.recurrent_segments),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HybridState:
        if not isinstance(value, dict):
            raise SchemaError(f"expected a mapping, got {type(value).__name__}")
        geometry_raw = value.get("geometry")
        geometry = None
        if geometry_raw is not None:
            if not isinstance(geometry_raw, dict):
                raise SchemaError("geometry is not a mapping")
            try:
                geometry = SourceGeometry(
                    architecture=str(geometry_raw["architecture"]),
                    n_layer=int(geometry_raw["n_layer"]),
                    n_head_kv=tuple(int(x) for x in geometry_raw["n_head_kv"]),
                    n_embd_head_k=int(geometry_raw["n_embd_head_k"]),
                    n_embd_head_v=int(geometry_raw["n_embd_head_v"]),
                    n_embd_r=int(geometry_raw["n_embd_r"]),
                    n_embd_s=int(geometry_raw["n_embd_s"]),
                    has_cell_ext=bool(geometry_raw["has_cell_ext"]),
                    rope_state=str(geometry_raw["rope_state"]),
                    rope_theta=float(geometry_raw["rope_theta"]),
                    model_weights_sha256=str(geometry_raw["model_weights_sha256"]),
                    gguf_content_digest=str(geometry_raw["gguf_content_digest"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SchemaError(f"geometry is not readable: {exc}") from exc
        layer_map_raw = value.get("layer_map")
        layer_map = None
        if layer_map_raw is not None:
            try:
                layer_map = RecurrentLayerMap(
                    tuple(int(x) for x in layer_map_raw["source_layers"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise SchemaError(f"layer map is not readable: {exc}") from exc
        checkpoints = []
        for raw in value.get("checkpoints", []):
            try:
                checkpoints.append(CheckpointRecord(
                    n_tokens=int(raw["n_tokens"]), pos_min=int(raw["pos_min"]),
                    pos_max=int(raw["pos_max"]),
                    recurrent_segments=tuple(str(x) for x in raw["recurrent_segments"]),
                    has_draft_state=bool(raw.get("has_draft_state", False)),
                    has_speculative_state=bool(raw.get("has_speculative_state", False))))
            except (KeyError, TypeError, ValueError) as exc:
                raise SchemaError(f"checkpoint is not readable: {exc}") from exc
        return cls(
            schema=str(value.get("schema", "")),
            geometry=geometry, layer_map=layer_map, checkpoints=tuple(checkpoints),
            attention_segments=tuple(str(x) for x in value.get("attention_segments", [])),
            recurrent_segments=tuple(str(x) for x in value.get("recurrent_segments", [])),
            layout=str(value.get("layout", "")),
            notes=tuple(str(x) for x in value.get("notes", [])),
        )
