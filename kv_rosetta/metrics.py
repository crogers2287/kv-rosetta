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
