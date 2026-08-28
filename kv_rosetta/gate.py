from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from kv_rosetta.metrics import kl_divergence, max_abs_logit_delta, tensor_cosine, top1_agreement


class GateError(ValueError):
    pass


def _finite(value: float, name: str, low: float, high: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise GateError(f"{name} must be finite, got {value}")
    if value < low or (high is not None and value > high):
        bound = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise GateError(f"{name} must be {bound}, got {value}")
    return value


@dataclass(frozen=True)
class GateThresholds:
    """Admission policy.

    ``holdout_tokens`` is a MINIMUM number of positions that must be compared, not
    merely a maximum slice. A one-token comparison must never satisfy a 64-token
    policy: that would let an artifact in on almost no evidence.

    Every threshold is validated on construction. A NaN or infinite threshold would
    silently disable the comparison it guards, which is the opposite of failing closed.
    """

    holdout_tokens: int = 64
    top1_agreement: float = 0.99
    mean_kl: float = 0.01
    max_kl: float = 0.05
    max_logit_delta: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.holdout_tokens, bool) or not isinstance(self.holdout_tokens, int):
            raise GateError("holdout_tokens must be an integer")
        if self.holdout_tokens <= 0:
            raise GateError(f"holdout_tokens must be positive, got {self.holdout_tokens}")
        _finite(self.top1_agreement, "top1_agreement", 0.0, 1.0)
        _finite(self.mean_kl, "mean_kl", 0.0, None)
        _finite(self.max_kl, "max_kl", 0.0, None)
        if self.max_logit_delta is not None:
            _finite(self.max_logit_delta, "max_logit_delta", 0.0, None)


EXACT = GateThresholds(top1_agreement=1.0, mean_kl=0.0, max_kl=0.0, max_logit_delta=0.0)


@dataclass(frozen=True)
class GateBinding:
    """What a gate verdict is valid for.

    A verdict is not a property of two logit arrays; it is a property of a specific
    source artifact translated by a specific mapper into a specific target under a
    specific threshold policy. Recording anything less makes the verdict unreusable
    and, worse, reusable in the wrong place.
    """

    source_model_id: str = ""
    target_model_id: str = ""
    source_artifact_digest: str = ""
    target_representation_digest: str = ""
    mapper_id: str = ""
    calibration_digest: str = ""
    policy_version: str = "kvx-gate/1"

    def as_dict(self) -> dict:
        return {
            "source_model_id": self.source_model_id,
            "target_model_id": self.target_model_id,
            "source_artifact_digest": self.source_artifact_digest,
            "target_representation_digest": self.target_representation_digest,
            "mapper_id": self.mapper_id,
            "calibration_digest": self.calibration_digest,
            "policy_version": self.policy_version,
        }


@dataclass
class GateResult:
    admitted: bool
    metrics: dict
    reason: str
    binding: dict = field(default_factory=dict)


def admit(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    thresholds: GateThresholds = GateThresholds(),
    reference_tensor: np.ndarray | None = None,
    candidate_tensor: np.ndarray | None = None,
    binding: GateBinding | None = None,
) -> GateResult:
    bound = (binding or GateBinding()).as_dict()
    try:
        ref = np.asarray(reference_logits)
        cand = np.asarray(candidate_logits)

        # Step 1: reject on shape/ndim mismatch; never raise.
        if ref.ndim != 2 or cand.ndim != 2 or ref.shape != cand.shape:
            return GateResult(
                False,
                {},
                f"shape mismatch: reference {tuple(ref.shape)} vs candidate {tuple(cand.shape)}",
                bound,
            )

        # Step 2: the policy's holdout count is a MINIMUM. Too few positions is not a
        # small sample to be judged leniently, it is insufficient evidence -> reject.
        holdout = thresholds.holdout_tokens
        available = int(ref.shape[0])
        if available < holdout:
            return GateResult(
                False,
                {"holdout_tokens": available, "required_holdout_tokens": holdout},
                f"insufficient holdout: {available} positions available, policy requires {holdout}",
                bound,
            )

        # Score the tail of the continuation, not its head.
        ref = ref[-holdout:]
        cand = cand[-holdout:]
        held = int(ref.shape[0])

        # Step 4: compute metrics. Cosine is reported only and is never consulted below.
        top1 = top1_agreement(ref, cand)
        kl = kl_divergence(ref, cand)
        mean_kl = float(np.mean(kl))
        max_kl = float(np.max(kl))
        max_delta = max_abs_logit_delta(ref, cand)

        if reference_tensor is None or candidate_tensor is None:
            cosine = float("nan")
        else:
            cosine = float(tensor_cosine(reference_tensor, candidate_tensor))

        metrics = {
            "top1_agreement": top1,
            "mean_kl": mean_kl,
            "max_kl": max_kl,
            "max_logit_delta": max_delta,
            "holdout_tokens": int(held),
            "tensor_cosine": cosine,
        }

        # Step 5: collect failures. The cosine value is deliberately never read here.
        failures: list[str] = []

        if math.isnan(top1):
            failures.append("non-finite metric: top1_agreement")
        elif top1 < thresholds.top1_agreement:
            failures.append(f"top1_agreement {top1} below threshold {thresholds.top1_agreement}")

        if math.isnan(mean_kl):
            failures.append("non-finite metric: mean_kl")
        elif mean_kl > thresholds.mean_kl:
            failures.append(f"mean_kl {mean_kl} above threshold {thresholds.mean_kl}")

        if math.isnan(max_kl):
            failures.append("non-finite metric: max_kl")
        elif max_kl > thresholds.max_kl:
            failures.append(f"max_kl {max_kl} above threshold {thresholds.max_kl}")

        if math.isnan(max_delta):
            failures.append("non-finite metric: max_logit_delta")
        elif thresholds.max_logit_delta is not None and max_delta > thresholds.max_logit_delta:
            failures.append(f"max_logit_delta {max_delta} above threshold {thresholds.max_logit_delta}")

        # Step 6: decide.
        admitted = not failures
        reason = "admitted" if not failures else "; ".join(failures)
        return GateResult(admitted, metrics, reason, bound)
    except Exception as exc:  # the gate must never raise on any input
        return GateResult(False, {}, f"admit failed: {exc}", bound)
