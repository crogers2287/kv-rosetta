from __future__ import annotations

import numpy as np

__all__ = [
    "MetricsError",
    "log_softmax",
    "kl_divergence",
    "top1_agreement",
    "max_abs_logit_delta",
    "tensor_cosine",
]


class MetricsError(ValueError):
    pass


def log_softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    row_max = np.max(arr, axis=1, keepdims=True)
    shifted = arr - row_max
    exp = np.exp(shifted)
    return shifted - np.log(np.sum(exp, axis=1, keepdims=True))


def kl_divergence(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 2 or cand.ndim != 2:
        raise MetricsError("kl_divergence requires 2-D logits")
    if ref.shape != cand.shape:
        raise MetricsError("kl_divergence requires matching shapes")
    log_p = log_softmax(ref)
    log_q = log_softmax(cand)
    p = np.exp(log_p)
    return np.sum(p * (log_p - log_q), axis=1)


def top1_agreement(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    if ref.shape[0] == 0:
        return 1.0
    agreement = np.sum(np.argmax(ref, axis=1) == np.argmax(cand, axis=1))
    return float(agreement / ref.shape[0])


def max_abs_logit_delta(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    if ref.shape[0] == 0:
        return 0.0
    return float(np.max(np.abs(ref - cand)))


def tensor_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def positionwise_agreement(reference: list[dict], candidate: list[dict]) -> dict:
    """Compare two sequences of next-token distributions position by position.

    This exists because top-1 agreement over a **free** generation is a cliff, not a slope.
    Generation is autoregressive: one wrong token derails every token after it, so the
    statistic reports when the first divergence happened rather than how good the cache was.
    Measured on a real cache pair, a nearly-perfect blend and a fully translated one scored
    identically - 0.042 - which gives nothing to tune a map against.

    Under teacher forcing every position is scored against the same forced prefix, so a
    single early mistake does not contaminate the rest and the result varies smoothly with
    cache quality. That is the difference between a number that admits a cache and a number
    that grades a map.

    Each entry is `{token_id: logprob}` for one position, as `probs()` yields. Comparison
    stops at the shorter sequence and the count is reported, so a short agreement cannot pass
    as a long one.
    """
    compared = min(len(reference), len(candidate))
    if compared == 0:
        return {"positions": 0, "top1_agreement": None, "mean_abs_logprob_delta": None,
                "max_abs_logprob_delta": None, "shared_tokens": 0, "tokens_only_in_one": 0}
    agreed, deltas, only_one, shared = 0, [], 0, 0
    for index in range(compared):
        first, second = reference[index], candidate[index]
        for token in set(first) | set(second):
            if token in first and token in second:
                deltas.append(abs(first[token] - second[token]))
                shared += 1
            else:
                only_one += 1
        top_a = max(first, key=first.get) if first else None
        top_b = max(second, key=second.get) if second else None
        agreed += int(top_a == top_b)
    return {
        "positions": compared,
        "top1_agreement": agreed / compared,
        "mean_abs_logprob_delta": float(np.mean(deltas)) if deltas else None,
        "max_abs_logprob_delta": float(np.max(deltas)) if deltas else None,
        "shared_tokens": shared,
        "tokens_only_in_one": only_one,
    }


def decision_margins(distributions: list[dict]) -> list[float]:
    """Top-1 minus top-2 logprob at each position: how decided the model was.

    A position with a margin near zero is one where the model was nearly indifferent between
    two continuations. Measured on this host, the token at which a perturbed cache first
    diverged was the **least certain position in the whole generation** on three prompts of
    four, and second-least on the fourth - margins of 0.18 to 0.51 against medians of 5.3 to
    7.9.
    """
    out = []
    for position in distributions:
        ordered = sorted(position.values(), reverse=True)
        out.append(float("inf") if len(ordered) < 2 else ordered[0] - ordered[1])
    return out


def confident_agreement(reference: list[dict], candidate: list[dict], *,
                        min_margin: float = 1.0) -> dict:
    """Agreement scored only where the reference model was actually decided.

    Exact-match over a whole generation is hostage to a single near-tied token: any
    disturbance flips it and the cascade condemns everything after. That makes the verdict a
    fact about the prompt rather than about the cache, and it is why the same blend ratio
    passed on one prompt and failed on another.

    Skipping positions the reference itself could not decide removes that hostage without
    weakening the test elsewhere - every confident position is still scored, and there are
    many more of them than undecided ones.

    Refuses when no position clears the margin. "Perfect agreement over nothing" is the
    vacuous pass this whole project is written against.
    """
    if min_margin < 0:
        raise MetricsError(f"min_margin {min_margin} is negative; a margin is a distance")
    compared = min(len(reference), len(candidate))
    margins = decision_margins(reference[:compared])
    scored = [i for i, m in enumerate(margins) if m >= min_margin]
    if not scored:
        raise MetricsError(
            f"no position among {compared} reached a margin of {min_margin}; the reference "
            f"was undecided throughout, so there is nothing this can measure")
    agreed = 0
    for index in scored:
        first, second = reference[index], candidate[index]
        top_a = max(first, key=first.get) if first else None
        top_b = max(second, key=second.get) if second else None
        agreed += int(top_a == top_b)
    return {
        "positions": compared,
        "scored": len(scored),
        "skipped_undecided": compared - len(scored),
        "min_margin": min_margin,
        "agreement": agreed / len(scored),
        "first_disagreement": next((i for i in scored
                                    if max(reference[i], key=reference[i].get)
                                    != max(candidate[i], key=candidate[i].get)), None),
    }
