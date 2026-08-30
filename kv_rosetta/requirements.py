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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The state-file format version llama.cpp will accept. A mismatch is rejected by the reader
#: for the format, which is indistinguishable from a backend failure unless it is caught here
#: - that confusion cost a whole cross-backend result before it was understood.
SEQUENCE_STATE_VERSION = "sequence_state_version"
#: True only on a build carrying the context-checkpoint patch. Hybrid artifacts are unusable
#: without it, and unusable *quietly*.
CHECKPOINT_PERSISTENCE = "slot_checkpoint_persistence"
#: The appendix format, currently "sckp/1".
CHECKPOINT_FORMAT = "slot_checkpoint_format"


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
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_state_version": self.sequence_state_version,
            "needs_checkpoint_persistence": self.needs_checkpoint_persistence,
            "checkpoint_format": self.checkpoint_format,
            "kv_type_k": self.kv_type_k, "kv_type_v": self.kv_type_v,
            "model_identity": self.model_identity, "notes": list(self.notes),
        }

    @property
    def portable(self) -> bool:
        """Whether any stock build could restore this, or only a patched one."""
        return not self.needs_checkpoint_persistence


def for_artifact(*, hybrid: bool, checkpoints: int, sequence_state_version: int,
                 kv_type_k: str = "f16", kv_type_v: str = "f16",
                 model_identity: str = "", checkpoint_format: str = "sckp/1"
                 ) -> Requirements:
    """Derive requirements from what the artifact actually contains.

    A hybrid artifact with **no** checkpoints does not need the patch to be *read* - but it
    will reuse nothing when restored, because the model resumes from a checkpoint it does not
    have. That is recorded as a note rather than a requirement: the restore is legal and
    useless, which is a different fact from illegal.
    """
    notes = []
    if hybrid and not checkpoints:
        notes.append("hybrid model saved without context checkpoints: a restore will be "
                     "accepted and will reuse nothing, because the model resumes from a "
                     "checkpoint this artifact does not carry")
    return Requirements(
        sequence_state_version=sequence_state_version,
        needs_checkpoint_persistence=bool(checkpoints),
        checkpoint_format=checkpoint_format if checkpoints else "",
        kv_type_k=kv_type_k, kv_type_v=kv_type_v, model_identity=model_identity,
        notes=tuple(notes))


def check(requirements: Requirements, props: dict[str, Any], *,
          model_identity: str = "") -> list[str]:
    """Every reason this artifact must not be restored here. Empty means it may be.

    `props` is the target server's own `/props`. A property the artifact requires and the
    runtime does not report is a failure: a build that does not advertise the
    checkpoint-persistence capability is exactly the build that restores hybrid state and
    reuses none of it.
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
        found = props.get("model_identity") or props.get("l0_sha256") or ""
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
    return problems


def require(requirements: Requirements, props: dict[str, Any], *,
            model_identity: str = "") -> None:
    """check(), as a refusal. Nothing is written to a slot before this passes."""
    problems = check(requirements, props, model_identity=model_identity)
    if problems:
        raise RequirementError("; ".join(problems))
