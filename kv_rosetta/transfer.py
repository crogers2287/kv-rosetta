"""Plan and execute a cache transfer: decide, convert, verify, or fall back.

Two rules shape everything here.

**Nothing raises into the caller.** A transfer is an optimisation over prefilling. If any
part of it fails - unknown identity, unsupported conversion, a refused gate, a broken
runtime - the answer is "prefill natively", not an exception. A caller should be able to ask
for a transfer on every request and never need a try block.

**Thresholds are fixed when the plan is made, before anything is measured.** They live on
the frozen plan, and `execute` takes only a plan. Choosing a bound after seeing the numbers
is how a lossy conversion talks its way past a gate, so the type system is used to prevent
it rather than a convention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kv_rosetta.gate import GateBinding, GateResult, GateThresholds, admit


class Mode(str, Enum):
    """How a source cache can reach a target, in increasing order of risk."""

    IDENTITY = "identity"                  # same model, same cache ABI: bytes are reusable
    DTYPE_CONVERT = "dtype_convert"        # same model, different K/V cache dtype
    CROSS_MODEL = "cross_model"            # different weights: needs a learned mapping
    REFUSED = "refused"


#: Conversions this project has actually exercised. A pair outside this set is refused
#: rather than attempted, because an untested conversion that happens to run is worse than
#: one that does not: it produces a cache nobody has checked.
SUPPORTED_DTYPE_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("f16", "q8_0"),
    ("f16", "f16"),
    ("q8_0", "q8_0"),
})


@dataclass(frozen=True)
class Identity:
    """What a cache is bound to. Compared field by field, never summarised to a score."""

    model_digest: str = ""
    cache_abi_digest: str = ""
    cache_dtype_k: str = ""
    cache_dtype_v: str = ""
    architecture: str = ""
    prompt_digest: str = ""


@dataclass(frozen=True)
class TransferPlan:
    """A decision made before any measurement, including the bounds it will be judged by."""

    mode: Mode
    source: Identity
    target: Identity
    thresholds: GateThresholds
    reason: str = ""
    steps: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.mode is not Mode.REFUSED


@dataclass
class TransferReport:
    ok: bool
    mode: Mode
    reason: str = ""
    fallback: str = "native_prefill"
    gate: GateResult | None = None
    seconds: float = 0.0
    phases: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "mode": self.mode.value, "reason": self.reason,
                "fallback": None if self.ok else self.fallback,
                "gate": None if self.gate is None else
                        {"admitted": self.gate.admitted, "reason": self.gate.reason,
                         "metrics": self.gate.metrics},
                "seconds": self.seconds, "phases": self.phases, "notes": list(self.notes)}


def plan(source: Identity, target: Identity,
         thresholds: GateThresholds | None = None) -> TransferPlan:
    """Decide how, or whether, a source cache can reach a target. Never raises."""
    policy = thresholds or GateThresholds()

    def refuse(reason: str) -> TransferPlan:
        return TransferPlan(Mode.REFUSED, source, target, policy, reason)

    if not source.model_digest or not target.model_digest:
        return refuse("model identity is unknown on one side; a cache cannot be matched to "
                      "a model it cannot be compared against")
    if source.prompt_digest and target.prompt_digest and \
            source.prompt_digest != target.prompt_digest:
        return refuse("prompt identity differs; this cache is for another prompt")

    if source.model_digest != target.model_digest:
        # Different weights. The mapping and its calibration do not exist yet, and a gate
        # cannot rescue a transformation that was never built.
        return refuse("cross-model transfer requires a fitted mapping, which is not "
                      "implemented; prefill natively")

    if source.cache_abi_digest and source.cache_abi_digest == target.cache_abi_digest:
        return TransferPlan(Mode.IDENTITY, source, target, policy,
                            "same model and cache ABI", ("restore",))

    pair_k = (source.cache_dtype_k, target.cache_dtype_k)
    pair_v = (source.cache_dtype_v, target.cache_dtype_v)
    if not all((source.cache_dtype_k, source.cache_dtype_v,
                target.cache_dtype_k, target.cache_dtype_v)):
        return refuse("a cache dtype is unadvertised on one side; conversion cannot be "
                      "planned against an unknown format")
    if pair_k not in SUPPORTED_DTYPE_PAIRS or pair_v not in SUPPORTED_DTYPE_PAIRS:
        return refuse(f"conversion {pair_k[0]}->{pair_k[1]} / {pair_v[0]}->{pair_v[1]} is "
                      f"not among the tested pairs {sorted(SUPPORTED_DTYPE_PAIRS)}")
    if source.architecture != target.architecture:
        return refuse("architectures differ; layer geometry cannot be assumed to match")

    return TransferPlan(
        Mode.DTYPE_CONVERT, source, target, policy,
        f"same model, cache dtype {pair_k[0]}->{pair_k[1]}",
        ("decode", "requantise", "encode", "restore", "gate"))


def execute(plan_: TransferPlan, *, run_transfer, reference_logits=None,
            candidate_logits=None, binding: GateBinding | None = None) -> TransferReport:
    """Carry out a planned transfer and judge it. Never raises; always returns a report.

    `run_transfer` is the caller's seam for actually moving the state; this function owns
    the decision, the gate, and the fallback, not the mechanics.

    A lossy mode must be gated. An IDENTITY transfer moves bytes that are already valid for
    the target and is not asked to prove equivalence it trivially has, but anything that
    reinterprets the cache has to be judged on behaviour before it is used.
    """
    started = time.time()
    phases: dict[str, float] = {}

    if not plan_.usable:
        return TransferReport(False, Mode.REFUSED, plan_.reason,
                              seconds=time.time() - started)
    try:
        mark = time.time()
        notes = run_transfer(plan_)
        phases["transfer"] = time.time() - mark
    except Exception as exc:                       # a failed optimisation is not an error
        return TransferReport(False, plan_.mode, f"transfer failed: {type(exc).__name__}: "
                              f"{exc}", seconds=time.time() - started, phases=phases)

    if plan_.mode is Mode.IDENTITY:
        return TransferReport(True, plan_.mode, plan_.reason, seconds=time.time() - started,
                              phases=phases, notes=tuple(notes or ()))

    if reference_logits is None or candidate_logits is None:
        return TransferReport(
            False, plan_.mode,
            "a lossy transfer was not measured against the target's own behaviour; "
            "refusing to use a converted cache on an unverified claim",
            seconds=time.time() - started, phases=phases)

    mark = time.time()
    result = admit(reference_logits, candidate_logits, plan_.thresholds, binding=binding)
    phases["gate"] = time.time() - mark
    return TransferReport(
        result.admitted, plan_.mode,
        plan_.reason if result.admitted else f"gate rejected: {result.reason}",
        gate=result, seconds=time.time() - started, phases=phases,
        notes=tuple(notes or ()))
