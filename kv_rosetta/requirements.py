"""What a runtime must provide before an artifact may be restored into it.

An artifact is not portable to a runtime that cannot read it, and the way llama.cpp reports
that is the problem. Restoring a hybrid model's state into a build without the context
checkpoint patch returns `n_restored: 259` and then reuses nothing - measured on this host,
252 of 256 tokens reused on a patched build against 0 of 256 on the fleet's own binary, with
both reporting the same n_restored. The failure is silent, downstream, and looks like a
performance problem rather than a compatibility one.

shimquant states the same hazard plainly about its own format: "files built with this patch
only load with this patch". An artifact that needs a patched runtime is not portable, and the
honest thing is to say so in the artifact rather than let a restore discover it.

So an artifact carries its requirements, and they are checked against the target's `/props`
before anything is written to a slot. A missing property is a refusal, not a warning: a
runtime that does not advertise a capability is one that may not have it.

One requirement here cannot be met by a property at all. Draft and speculative checkpoint
state are serialized by the format and restored by nothing anyone has watched - a server
running MTP reports `sckp_serializes_speculative_state` true beside
`supports_speculative_checkpoint_state` false. For those classes the runtime's own claim is
necessary and not sufficient: it is a flag a person compiled in, and this project's standing
rule is that a source-level fact is not a capability. A behavioural proof, scoped to one
build, one binary, one model and one class, is what lifts the refusal - see
`hybrid_schema.RestorationProof`. With no proof supplied, which is the state every caller is
in today, the refusal stands unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hybrid_schema import (
    DRAFT_STATE,
    PROOF_STATE_CLASSES,
    SPECULATIVE_STATE,
    RestorationEvidence,
)

#: The state-file format version llama.cpp will accept. A mismatch is rejected by the reader
#: for the format, which is indistinguishable from a backend failure unless it is caught here
#: - that confusion cost a whole cross-backend result before it was understood.
SEQUENCE_STATE_VERSION = "sequence_state_version"
#: True only on a build carrying the context-checkpoint patch. Hybrid artifacts are unusable
#: without it, and unusable *quietly*.
CHECKPOINT_PERSISTENCE = "slot_checkpoint_persistence"
#: The appendix format, currently "sckp/1".
CHECKPOINT_FORMAT = "slot_checkpoint_format"
#: The build string llama.cpp reports, e.g. "b1-3e73446". Necessary to tie a proof to a
#: build and never sufficient: two binaries built from the same commit, one carrying the
#: checkpoint patch and one not, report it identically. The binary digest settles that.
BUILD_INFO = "build_info"


def support_property(state_class: str) -> str:
    """What the runtime says it has *shown* to restore, for one checkpoint blob class."""
    return f"supports_{state_class}_checkpoint_state"


def serialization_property(state_class: str) -> str:
    """What the format writes, which is a different claim and was once conflated with it."""
    return f"sckp_serializes_{state_class}_state"


class RequirementError(ValueError):
    """Raised when an artifact must not be restored into this runtime."""


@dataclass(frozen=True)
class Requirements:
    """What the artifact needs. Absent means "not required", never "unknown"."""

    #: Exact match required; None if the artifact carries no attention state to version.
    sequence_state_version: int | None = None
    #: True when the artifact carries an SCKP appendix that only a patched build restores.
    needs_checkpoint_persistence: bool = False
    #: Exact match required when the artifact carries checkpoints.
    checkpoint_format: str = ""
    #: The KV cache element types the state was written with. A build running a different KV
    #: quantisation produces rows of a different size and cannot read these bytes at all.
    kv_type_k: str = ""
    kv_type_v: str = ""
    #: Model identity, as llama.cpp's own props report it.
    model_identity: str = ""
    #: Checkpoint blob classes this artifact carries whose restoration nobody has
    #: demonstrated. Not a requirement a runtime can satisfy by advertising something: it
    #: takes the runtime's own claim *and* a behavioural proof scoped to this build, this
    #: binary, this model and this class.
    gated_state_classes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_state_version": self.sequence_state_version,
            "needs_checkpoint_persistence": self.needs_checkpoint_persistence,
            "checkpoint_format": self.checkpoint_format,
            "kv_type_k": self.kv_type_k, "kv_type_v": self.kv_type_v,
            "model_identity": self.model_identity,
            "gated_state_classes": list(self.gated_state_classes),
            "notes": list(self.notes),
        }

    @property
    def portable(self) -> bool:
        """Whether any stock build could restore this, or only a patched one."""
        return not self.needs_checkpoint_persistence and not self.gated_state_classes


def for_artifact(*, hybrid: bool, checkpoints: int, sequence_state_version: int,
                 kv_type_k: str = "f16", kv_type_v: str = "f16",
                 model_identity: str = "", checkpoint_format: str = "sckp/1",
                 draft_state: bool = False, speculative_state: bool = False
                 ) -> Requirements:
    """Derive requirements from what the artifact actually contains.

    A hybrid artifact with **no** checkpoints does not need the patch to be *read* - but it
    will reuse nothing when restored, because the model resumes from a checkpoint it does not
    have. That is recorded as a note rather than a requirement: the restore is legal and
    useless, which is a different fact from illegal.

    `draft_state` and `speculative_state` say which unproven blob classes the checkpoints
    carry. They are recorded per class, because a demonstration of one says nothing about the
    other, and the requirement they raise cannot be met by a property in /props alone.
    """
    notes = []
    if hybrid and not checkpoints:
        notes.append("hybrid model saved without context checkpoints: a restore will be "
                     "accepted and will reuse nothing, because the model resumes from a "
                     "checkpoint this artifact does not carry")
    gated = tuple(name for name, carried in ((DRAFT_STATE, draft_state),
                                             (SPECULATIVE_STATE, speculative_state))
                  if carried)
    if gated:
        notes.append(f"artifact carries {', '.join(gated)} checkpoint state, whose "
                     f"restoration is unproven; it may only be restored into a build that "
                     f"claims the capability and that a behavioural proof names")
    return Requirements(
        sequence_state_version=sequence_state_version,
        needs_checkpoint_persistence=bool(checkpoints),
        checkpoint_format=checkpoint_format if checkpoints else "",
        kv_type_k=kv_type_k, kv_type_v=kv_type_v, model_identity=model_identity,
        gated_state_classes=gated, notes=tuple(notes))


def _gated_state_problems(state_class: str, props: dict[str, Any], *,
                          evidence: RestorationEvidence | None, runtime_build: str,
                          binary_sha256: str, model_identity: str) -> list[str]:
    """Why this class of unproven checkpoint state may not be restored here.

    Two independent things have to hold, and neither substitutes for the other. The runtime
    must claim the capability - a server running MTP speculative decoding reports
    `supports_speculative_checkpoint_state` false beside `sckp_serializes_speculative_state`
    true, and restoring into that is forwarding bytes the runtime says it cannot use. And a
    behavioural proof must exist for this exact build, binary, model and class, because the
    claim is a flag some person compiled in, and this project's rule is that a source-level
    fact is not a capability.
    """
    if state_class not in PROOF_STATE_CLASSES:
        return [f"artifact declares gated state class {state_class!r}, which is not one of "
                f"{list(PROOF_STATE_CLASSES)}; state that cannot be named cannot be proven"]
    problems: list[str] = []
    support = support_property(state_class)
    if not props.get(support):
        problems.append(
            f"this artifact carries {state_class} checkpoint state and the runtime does not "
            f"report {support}. A server running MTP speculative decoding reported it false "
            f"while {serialization_property(state_class)} was true: the format serialising "
            f"the blob is not the runtime restoring it")
    found_build = runtime_build or str(props.get(BUILD_INFO) or "")
    if not found_build:
        problems.append(f"the runtime reports no {BUILD_INFO} and the caller named none, so "
                        f"a proof cannot be tied to the build it was gathered on")
    if evidence is None:
        problems.append(
            f"no behavioural proof of {state_class} checkpoint restoration was supplied. "
            f"Absence of a proof is not a proof that it works: this state class stays "
            f"refused until one exists for this build, binary, model and class")
        return problems
    if found_build and evidence.runtime_build != found_build:
        problems.append(f"the proof was gathered against build {evidence.runtime_build!r} "
                        f"and this runtime reports {found_build!r}")
    if not binary_sha256:
        problems.append(
            "the binary this runtime is running was not identified, so a proof naming one "
            "cannot be tied to it - a fork's build_info did not distinguish a binary "
            "carrying the checkpoint patch from one without it")
    elif evidence.binary_sha256 != binary_sha256:
        problems.append(f"the proof was gathered against binary "
                        f"{str(evidence.binary_sha256)[:16]} and this runtime runs "
                        f"{binary_sha256[:16]}")
    if not model_identity:
        problems.append(f"no model identity is established for this artifact, so a proof of "
                        f"{state_class} restoration cannot be tied to what it would admit")
    elif evidence.model_identity != model_identity:
        problems.append(f"the proof was gathered on model "
                        f"{str(evidence.model_identity)[:16]} and this artifact is "
                        f"{model_identity[:16]}")
    proven, reason = evidence.verdict(state_class)
    if proven is False:
        problems.append(f"{state_class} checkpoint state is refused here: {reason}")
    elif proven is None:
        problems.append(f"{state_class} checkpoint restoration is not proven for this "
                        f"runtime and model: {reason}")
    return problems


def check(requirements: Requirements, props: dict[str, Any], *,
          model_identity: str = "", runtime_identity: str = "", runtime_build: str = "",
          binary_sha256: str = "", evidence: RestorationEvidence | None = None
          ) -> list[str]:
    """Every reason this artifact must not be restored here. Empty means it may be.

    `props` is the target server's own `/props`. A property the artifact requires and the
    runtime does not report is a failure: a build that does not advertise the
    checkpoint-persistence capability is exactly the build that restores hybrid state and
    reuses none of it.

    `runtime_identity` is the loaded model's identity as the **caller** derived it, because
    llama.cpp does not put one in `/props` - it is computed from the weights file. Running
    this end to end without it refused a restore that then succeeded with 508 of 512 tokens
    reused, which is a false refusal: costly in the fail-closed direction rather than the
    dangerous one, but wrong either way.

    `runtime_build`, `binary_sha256` and `evidence` are only consulted for the state classes
    the artifact declares unproven. Passing none of them - which is every existing caller -
    leaves every previous refusal exactly as it was, and refuses the unproven classes.
    """
    problems = []
    if requirements.sequence_state_version is not None:
        found = props.get(SEQUENCE_STATE_VERSION)
        if found is None:
            problems.append(
                f"the runtime does not report {SEQUENCE_STATE_VERSION}, so its state-file "
                f"format cannot be established; a version mismatch is rejected for the "
                f"format and looks exactly like a backend failure")
        elif int(found) != requirements.sequence_state_version:
            problems.append(f"artifact is sequence state version "
                            f"{requirements.sequence_state_version}, runtime reads {found}")
    if requirements.needs_checkpoint_persistence:
        if not props.get(CHECKPOINT_PERSISTENCE):
            problems.append(
                "this artifact carries context checkpoints and the runtime does not report "
                "slot_checkpoint_persistence. Restoring it would be accepted and reuse "
                "nothing - measured here as 252 of 256 tokens on a patched build against 0 "
                "of 256 on an unpatched one, both reporting the same n_restored")
        else:
            # Only meaningful once persistence is reported. A build without the patch also
            # reports no format, and saying so adds nothing to the line above - it is a
            # consequence of the missing capability, not a second fault to fix.
            found = props.get(CHECKPOINT_FORMAT)
            if requirements.checkpoint_format and found != requirements.checkpoint_format:
                problems.append(f"artifact checkpoints are "
                                f"{requirements.checkpoint_format}, runtime reads {found!r}")
    wanted = model_identity or requirements.model_identity
    if wanted:
        found = (runtime_identity or props.get("model_identity")
                 or props.get("l0_sha256") or "")
        if not found:
            problems.append("the runtime reports no model identity, so this cannot be shown "
                            "to be the model the artifact was written from")
        elif found != wanted:
            problems.append(f"artifact was written from {wanted[:16]}, runtime is "
                            f"{found[:16]}")
    for name, wanted_type in (("k", requirements.kv_type_k), ("v", requirements.kv_type_v)):
        if not wanted_type:
            continue
        found = props.get(f"kv_type_{name}") or props.get(f"type_{name}")
        if found and found != wanted_type:
            problems.append(f"artifact {name} cache is {wanted_type}, runtime uses {found}; "
                            f"a different KV quantisation gives rows of a different size and "
                            f"these bytes cannot be read at all")
    for state_class in requirements.gated_state_classes:
        problems += _gated_state_problems(
            state_class, props, evidence=evidence, runtime_build=runtime_build,
            binary_sha256=binary_sha256, model_identity=wanted)
    return problems


def require(requirements: Requirements, props: dict[str, Any], *,
            model_identity: str = "", runtime_identity: str = "", runtime_build: str = "",
            binary_sha256: str = "", evidence: RestorationEvidence | None = None) -> None:
    """check(), as a refusal. Nothing is written to a slot before this passes."""
    problems = check(requirements, props, model_identity=model_identity,
                     runtime_identity=runtime_identity, runtime_build=runtime_build,
                     binary_sha256=binary_sha256, evidence=evidence)
    if problems:
        raise RequirementError("; ".join(problems))
