from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from kv_rosetta.metrics import kl_divergence, max_abs_logit_delta, tensor_cosine, top1_agreement


@dataclass(frozen=True)
class GateThresholds:
    holdout_tokens: int = 64
    top1_agreement: float = 0.99
    mean_kl: float = 0.01
    max_kl: float = 0.05
    max_logit_delta: float | None = None


EXACT = GateThresholds(top1_agreement=1.0, mean_kl=0.0, max_kl=0.0, max_logit_delta=0.0)


@dataclass
class GateResult:
    admitted: bool
    metrics: dict
    reason: str


def admit(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    thresholds: GateThresholds = GateThresholds(),
    reference_tensor: np.ndarray | None = None,
    candidate_tensor: np.ndarray | None = None,
) -> GateResult:
    try:
        ref = np.asarray(reference_logits)
        cand = np.asarray(candidate_logits)

        # Step 1: reject on shape/ndim mismatch; never raise.
        if ref.ndim != 2 or cand.ndim != 2 or ref.shape != cand.shape:
            return GateResult(
                False,
                {},
                f"shape mismatch: reference {tuple(ref.shape)} vs candidate {tuple(cand.shape)}",
            )

        # Step 2: keep only the LAST holdout_tokens rows (the tail of the continuation).
        holdout = thresholds.holdout_tokens
        if ref.shape[0] > holdout:
            ref = ref[-holdout:] if holdout > 0 else ref[:0]
            cand = cand[-holdout:] if holdout > 0 else cand[:0]

        # Step 3: nothing held out -> reject.
        held = ref.shape[0]
        if held == 0:
            return GateResult(False, {}, "no holdout tokens")

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
        return GateResult(admitted, metrics, reason)
    except Exception as exc:  # the gate must never raise on any input
        return GateResult(False, {}, f"admit failed: {exc}")
