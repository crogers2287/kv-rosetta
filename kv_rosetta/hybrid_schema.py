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

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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


# -- evidence for the state classes nothing here has restored ---------------------------
#
# `common_prompt_checkpoint` carries `data_tgt`, `data_dft` and `data_spec`, and the SCKP
# appendix serializes all three. A live server running MTP speculative decoding reports
# `sckp_serializes_speculative_state: true` beside `supports_speculative_checkpoint_state:
# false` - the bytes are in the file and nothing has shown they restore. That is why draft
# and speculative state are refused by default, and the default is not the interesting part:
# the refusal has to be liftable by whoever does the demonstration, without becoming a switch
# that lifts it for everyone.
#
# So the lift is scoped to exactly what was demonstrated - one state class, one build, one
# model - and is supplied by the caller at admission time rather than carried in the
# artifact. An artifact that carried its own proof would be a permission slip signed by the
# thing being admitted.

#: The two checkpoint blob classes whose restoration is unproven here.
DRAFT_STATE = "draft"
SPECULATIVE_STATE = "speculative"
PROOF_STATE_CLASSES = (DRAFT_STATE, SPECULATIVE_STATE)

#: What a run record may report. A proof is not a permission slip: a record of a run that
#: diverged is evidence too, and it is the stronger kind - it refuses, where an absent record
#: merely fails to admit.
PROOF_RESTORED = "restored_and_matched"
PROOF_DIVERGED = "diverged"
PROOF_OUTCOMES = frozenset({PROOF_RESTORED, PROOF_DIVERGED})

#: A proof states its own validity window, and the window is bounded so that no record can
#: admit forever. An unexpiring proof is the "trust me" boolean again, wearing provenance.
MAX_PROOF_VALIDITY = timedelta(days=365)

#: The refusal in force whenever nothing proves otherwise, verbatim as it has always read.
#: It is what an artifact gets with no evidence at all, which is still every artifact.
UNPROVEN_STATE_REFUSAL = (
    "checkpoint carries draft or speculative state, whose restoration is not behaviourally "
    "proven; refusing rather than forwarding state nothing has verified")

#: Keys that would smuggle an admission decision into the artifact payload itself.
EVIDENCE_KEYS = frozenset({"evidence", "proof", "proofs", "restoration_proof",
                           "restoration_proofs", "admission", "admitted_state_classes"})

_DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.match(value) is not None


def _blank(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def _moment(text: Any) -> datetime | None:
    """An ISO-8601 instant, or None when the text is not one.

    A timestamp with no zone is refused rather than assumed local or assumed UTC: the two
    readings are hours apart, and an expiry that depends on where the reader is standing is
    not an expiry.
    """
    if _blank(text):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class RestorationProof:
    """One behavioural demonstration, scoped to exactly what it demonstrated.

    Every field is here because a reader six months from now has to be able to re-derive the
    claim or throw it away: which state class, which build *and which binary*, which model,
    how it was shown, over how many runs, where the record is, and what that record hashed to
    when the proof was written.

    Build string and binary digest are both required because they answered differently once:
    the fleet's fork and a patched tree can report build_info that does not distinguish a
    binary carrying the checkpoint patch from one without it, and the same model reused 0 of
    676 tokens on one and 672 on the other.
    """

    state_class: str
    runtime_build: str          # what the runtime reports as build_info, matched exactly
    binary_sha256: str          # the exact library that was measured
    model_identity: str         # the weights digest it was measured against
    outcome: str
    trials: int
    method: str                 # how it was demonstrated, in enough prose to repeat it
    record_uri: str             # where the run record lives
    record_sha256: str          # what that record hashed to when this was written
    proven_at: str              # ISO-8601 with a timezone
    expires_at: str

    def validate(self) -> list[str]:
        """Everything wrong with the record itself. Scope is checked separately."""
        problems: list[str] = []
        if self.state_class not in PROOF_STATE_CLASSES:
            problems.append(f"proof names state class {self.state_class!r}, not one of "
                            f"{list(PROOF_STATE_CLASSES)}; a proof that does not say what it "
                            f"proved would admit everything or nothing")
        if self.outcome not in PROOF_OUTCOMES:
            problems.append(f"proof outcome {self.outcome!r} is not one of "
                            f"{sorted(PROOF_OUTCOMES)}, so the run's result is unreadable")
        if not isinstance(self.trials, int) or isinstance(self.trials, bool) \
                or self.trials <= 0:
            problems.append(f"proof reports {self.trials!r} trials; a demonstration that was "
                            f"never run is a claim")
        if _blank(self.method):
            problems.append("proof does not say how it was demonstrated, so it can be "
                            "neither re-derived nor contradicted")
        if _blank(self.record_uri):
            problems.append("proof cites no run record, so there is nothing to re-read")
        if _blank(self.runtime_build):
            problems.append("proof names no runtime build, so it is scoped to no build")
        for name in ("binary_sha256", "model_identity", "record_sha256"):
            if not _is_digest(getattr(self, name)):
                problems.append(f"proof {name} is not a 64-character lowercase digest")
        proven, expires = _moment(self.proven_at), _moment(self.expires_at)
        if proven is None:
            problems.append(f"proof proven_at {self.proven_at!r} is not an ISO-8601 instant "
                            f"with a timezone")
        if expires is None:
            problems.append(f"proof expires_at {self.expires_at!r} is not an ISO-8601 "
                            f"instant with a timezone")
        if proven is not None and expires is not None:
            if expires <= proven:
                problems.append("proof expires at or before the run it records")
            elif expires - proven > MAX_PROOF_VALIDITY:
                problems.append(f"proof claims validity for longer than "
                                f"{MAX_PROOF_VALIDITY.days} days; a record that never "
                                f"expires is a standing permission, not a measurement")
        return problems


@dataclass(frozen=True)
class RestorationEvidence:
    """Proofs, plus the facts they have to be checked against.

    Those facts describe the runtime and the model the caller is actually holding. They never
    come from the artifact and never from the proof: a record compared only against its own
    contents shows that it is self-consistent, which is exactly what a forgery is.
    """

    runtime_build: str
    binary_sha256: str
    model_identity: str
    proofs: tuple[RestorationProof, ...] = ()
    #: The digest of each cited run record, hashed by the caller now. A proof whose record
    #: cannot be produced and hashed at admission time is a citation, not a record.
    record_digests: Mapping[str, str] = field(default_factory=dict)
    #: ISO-8601 with a timezone. Empty means "the moment this is being decided".
    as_of: str = ""

    def own_problems(self) -> list[str]:
        problems: list[str] = []
        if _blank(self.runtime_build):
            problems.append("evidence does not name the build it is being checked against")
        for name in ("binary_sha256", "model_identity"):
            if not _is_digest(getattr(self, name)):
                problems.append(f"evidence {name} is not a 64-character lowercase digest, so "
                                f"no proof can be tied to what is actually running")
        if not isinstance(self.record_digests, Mapping):
            problems.append("evidence record_digests is not a mapping of record to digest")
        if self.as_of and _moment(self.as_of) is None:
            problems.append(f"evidence as_of {self.as_of!r} is not an ISO-8601 instant with "
                            f"a timezone, so expiry cannot be decided")
        return problems

    def verdict(self, state_class: str) -> tuple[bool | None, str]:
        """Proven, disproven, or unknown - and why.

        Three states, because unrecorded is not the same answer as recorded-false. This repo
        collapsed them twice, once for hybrid architectures and once for sliding-window ones,
        and both times the collapse read "no" where the honest answer was "nobody looked".
        Here both refuse, and they must still be told apart: a divergence is a finding, and
        an absent record is a gap somebody can go and close.
        """
        problems = self.own_problems()
        if problems:
            return None, "; ".join(problems)
        now = _moment(self.as_of) or datetime.now(timezone.utc)
        restored: list[RestorationProof] = []
        diverged: list[RestorationProof] = []
        misses: list[str] = []
        for index, proof in enumerate(self.proofs):
            why = self._out_of_scope(proof, state_class, now)
            if why:
                misses.append(f"proof {index} does not apply: {why}")
            elif proof.outcome == PROOF_DIVERGED:
                diverged.append(proof)
            else:
                restored.append(proof)
        if diverged:
            found = diverged[0]
            return False, (f"a proof in scope records that {state_class} state did NOT "
                           f"restore on this build: {found.method} ({found.record_uri})")
        if restored:
            found = restored[0]
            return True, (f"{state_class} checkpoint restoration demonstrated on build "
                          f"{found.runtime_build} over {found.trials} trial(s): "
                          f"{found.method} ({found.record_uri})")
        if misses:
            return None, "; ".join(misses)
        return None, f"no proof of {state_class} checkpoint restoration was offered"

    def _out_of_scope(self, proof: RestorationProof, state_class: str,
                      now: datetime) -> str:
        """Why this proof says nothing about this class, build, model or moment."""
        if proof.state_class != state_class:
            return (f"it proves {proof.state_class} state, not {state_class}; a "
                    f"demonstration of one class says nothing about the other")
        problems = proof.validate()
        if problems:
            return "; ".join(problems)
        if proof.runtime_build != self.runtime_build:
            return (f"it was gathered on build {proof.runtime_build!r}, this runtime reports "
                    f"{self.runtime_build!r}")
        if proof.binary_sha256 != self.binary_sha256:
            return (f"it names the same build string but a different binary "
                    f"({proof.binary_sha256[:16]} against {self.binary_sha256[:16]}); a "
                    f"fork's build_info did not distinguish a binary carrying the checkpoint "
                    f"patch from one without it")
        if proof.model_identity != self.model_identity:
            return (f"it was gathered on model {proof.model_identity[:16]}, this is "
                    f"{self.model_identity[:16]}")
        if _moment(proof.expires_at) <= now:
            return (f"it expired at {proof.expires_at}; re-derive it against this build "
                    f"rather than extending it")
        if _moment(proof.proven_at) > now:
            return (f"it is dated {proof.proven_at}, which has not happened yet; a record "
                    f"from the future is not a record")
        observed = self.record_digests.get(proof.record_uri)
        if not observed:
            return (f"its run record {proof.record_uri} was not produced at admission time, "
                    f"so the evidence it cites cannot be read")
        if observed != proof.record_sha256:
            return (f"its run record {proof.record_uri} now hashes to {observed[:16]}, and "
                    f"the proof was written against {proof.record_sha256[:16]}; the record "
                    f"and the claim have drifted apart")
        return ""


def admission_problems(state_class: str,
                       evidence: RestorationEvidence | None) -> list[str]:
    """Why this class of state may not be forwarded. Empty means a proof admits it."""
    if evidence is None:
        return [UNPROVEN_STATE_REFUSAL]
    proven, reason = evidence.verdict(state_class)
    if proven is True:
        return []
    if proven is False:
        return [f"checkpoint carries {state_class} state and {reason}"]
    return [f"{UNPROVEN_STATE_REFUSAL} ({state_class}: {reason})"]


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

    def validate(self, evidence: RestorationEvidence | None = None) -> list[str]:
        """Every reason this checkpoint may not be forwarded.

        `evidence` defaults to None, which is the state every caller is in until somebody
        does the demonstration, and in that state this refuses exactly what it always did.
        A proof is checked per state class, so admitting draft state leaves speculative
        state refused.
        """
        problems: list[str] = []
        if self.n_tokens <= 0:
            problems.append(f"checkpoint covers {self.n_tokens} tokens")
        if self.pos_min < 0 or self.pos_max < self.pos_min:
            problems.append(f"checkpoint positions {self.pos_min}..{self.pos_max} invalid")
        if not self.recurrent_segments:
            problems.append("checkpoint names no recurrent segments; a hybrid model cannot "
                            "reuse a prefix without them")
        for state_class, carried in ((DRAFT_STATE, self.has_draft_state),
                                     (SPECULATIVE_STATE, self.has_speculative_state)):
            if carried:
                problems += admission_problems(state_class, evidence)
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

    def validate(self, evidence: RestorationEvidence | None = None) -> list[str]:
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
        # Evidence is bound to this artifact's own model before it can admit anything in it.
        # A proof gathered on one model would otherwise travel to any artifact it was handed
        # to, which is the blanket override this design exists to avoid.
        bound = evidence
        if evidence is not None:
            if self.geometry is None:
                problems.append("evidence was offered for an artifact with no source "
                                "geometry, so it cannot be tied to the model it would admit")
                bound = None
            elif evidence.model_identity != self.geometry.model_weights_sha256:
                problems.append(
                    f"evidence was gathered on model "
                    f"{str(evidence.model_identity)[:16]} and this "
                    f"artifact was written from "
                    f"{self.geometry.model_weights_sha256[:16]}; a proof for one model does "
                    f"not admit another")
                bound = None
        for index, checkpoint in enumerate(self.checkpoints):
            problems += [f"checkpoint {index}: {p}" for p in checkpoint.validate(bound)]
        return problems

    def require_valid(self, evidence: RestorationEvidence | None = None) -> HybridState:
        problems = self.validate(evidence)
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
        # An artifact does not get to carry its own admission. Evidence is supplied by the
        # caller, out of band, against the runtime and model it is actually holding; a proof
        # travelling inside the payload would be a permission slip signed by the thing asking
        # to be let in, and forging one would be a text edit.
        smuggled = sorted(EVIDENCE_KEYS.intersection(value))
        if smuggled:
            raise SchemaError(
                f"artifact payload carries admission evidence in {smuggled}; evidence is "
                f"supplied by the caller at admission time and is never read from the "
                f"artifact, which cannot vouch for itself")
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
            if isinstance(raw, dict):
                smuggled = sorted(EVIDENCE_KEYS.intersection(raw))
                if smuggled:
                    raise SchemaError(
                        f"checkpoint carries admission evidence in {smuggled}; the record "
                        f"that carries the unproven state does not get to authorise it")
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
