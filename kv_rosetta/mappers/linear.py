"""Fit a linear map from one model's KV cache to another's. RESEARCH, expected to be refused.

A cache is the activations of specific weights, so there is no format conversion between two
models - only an approximation. This fits the simplest honest one: per layer, a ridge
regression from the source model's K/V at a token to the target model's, on a calibration
corpus both models have seen.

Two things it deliberately does not do.

It does not decide whether the result is usable. That is `gate.py`, on held-out next-token
agreement. A mapper reporting a low fitting residual has shown that it fitted, which is not
the same claim and is exactly the confusion the gate exists to prevent.

It does not travel. A fitted map is bound to the exact pair of models, geometry and
calibration corpus it was fitted from, and refuses to be applied to anything else. A mapper
silently reused across a model pair would produce confident, wrong activations.

Geometry may differ on both axes - the tested pair is 36 layers x 2 heads x 128 against
65 x 4 x 213 - so heads are flattened per layer and the map is rectangular.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class MapperError(ValueError):
    """Raised when a map cannot be fitted, loaded, or legitimately applied."""


@dataclass(frozen=True)
class MapperIdentity:
    """Everything a fitted map is only valid for."""

    source_model_digest: str
    target_model_digest: str
    source_arch: str
    target_arch: str
    source_width: int                 # kv_heads * head_dim, per layer
    target_width: int
    source_layers: int
    target_layers: int
    calibration_sha256: str
    rope_state: str = "not_applied"

    def validate(self) -> list[str]:
        problems = []
        for name in ("source_model_digest", "target_model_digest", "calibration_sha256"):
            if len(getattr(self, name)) != 64:
                problems.append(f"{name} is not a 64-character digest")
        if self.source_model_digest == self.target_model_digest:
            problems.append("source and target are the same model; a fitted map is for "
                            "crossing between models, not for one model to itself")
        for name in ("source_width", "target_width", "source_layers", "target_layers"):
            if getattr(self, name) <= 0:
                problems.append(f"{name} must be positive")
        if self.rope_state != "not_applied":
            problems.append("keys must have RoPE stripped before fitting: a positional "
                            "rotation is not part of what the map should learn")
        return problems

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "source_model_digest", "target_model_digest", "source_arch", "target_arch",
            "source_width", "target_width", "source_layers", "target_layers",
            "calibration_sha256", "rope_state")}


def corpus_digest(prompts: list[list[int]]) -> str:
    """Identity of the calibration set, so a map cannot be reused across corpora."""
    # The JSON brackets already delimit each prompt, so [[1,2],[3]] and [[1],[2,3]] encode
    # differently without an extra separator. A separator was here until a mutation run
    # showed no test could justify it.
    running = hashlib.sha256()
    for tokens in prompts:
        running.update(json.dumps(list(tokens), separators=(",", ":")).encode())
    return running.hexdigest()


def fit_ridge(source: np.ndarray, target: np.ndarray, ridge: float = 1e-2
              ) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with an L2 penalty, returning (weights, bias).

    Solved through the normal equations with an explicit penalty rather than a plain
    least-squares call: the source features are strongly correlated across heads, so the
    unpenalised system is ill-conditioned and produces large weights that fit the
    calibration set and generalise badly.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise MapperError(f"expected 2-D (tokens, width), got {source.shape} and "
                          f"{target.shape}")
    if source.shape[0] != target.shape[0]:
        raise MapperError(f"{source.shape[0]} source tokens against "
                          f"{target.shape[0]} target tokens")
    if source.shape[0] <= source.shape[1]:
        raise MapperError(f"{source.shape[0]} tokens for {source.shape[1]} features: an "
                          f"underdetermined fit would memorise the calibration set")
    if ridge <= 0:
        raise MapperError("ridge penalty must be positive; an unpenalised fit on correlated "
                          "features is what this parameter exists to prevent")

    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    augmented = np.hstack([x, np.ones((x.shape[0], 1))])          # absorb the bias
    penalty = ridge * np.eye(augmented.shape[1])
    penalty[-1, -1] = 0.0                                         # never penalise the bias
    gram = augmented.T @ augmented + penalty
    solution = np.linalg.solve(gram, augmented.T @ y)
    return solution[:-1].astype(np.float32), solution[-1].astype(np.float32)


def residual(source: np.ndarray, target: np.ndarray, weights: np.ndarray,
             bias: np.ndarray) -> float:
    """Relative error of a fit. Reported for layer selection; never an admission criterion."""
    predicted = source @ weights + bias
    denominator = float(np.linalg.norm(target))
    if denominator == 0.0:
        return float(np.linalg.norm(predicted - target))
    return float(np.linalg.norm(predicted - target) / denominator)


@dataclass
class LinearMapper:
    """A fitted map, valid only for the identity it carries."""

    identity: MapperIdentity
    layer_pairs: tuple[tuple[int, int], ...] = ()       # (target_layer, source_layer)
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    biases: dict[str, np.ndarray] = field(default_factory=dict)
    residuals: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def key(target_layer: int, kind: str) -> str:
        return f"{target_layer}:{kind}"

    def require_applicable(self, source_model_digest: str, target_model_digest: str,
                           source_width: int, target_width: int) -> None:
        """Refuse to apply a map to a pair it was not fitted for."""
        mismatches = []
        if source_model_digest != self.identity.source_model_digest:
            mismatches.append("source model")
        if target_model_digest != self.identity.target_model_digest:
            mismatches.append("target model")
        if source_width != self.identity.source_width:
            mismatches.append(f"source width {source_width} != "
                              f"{self.identity.source_width}")
        if target_width != self.identity.target_width:
            mismatches.append(f"target width {target_width} != "
                              f"{self.identity.target_width}")
        if mismatches:
            raise MapperError(
                "this map was fitted for a different pair (" + ", ".join(mismatches) +
                "); applying it would produce confident, wrong activations")

    def apply_layer(self, source: np.ndarray, target_layer: int, kind: str) -> np.ndarray:
        key = self.key(target_layer, kind)
        if key not in self.weights:
            raise MapperError(f"no fitted map for target layer {target_layer} {kind}")
        if source.shape[-1] != self.identity.source_width:
            raise MapperError(f"source width {source.shape[-1]} does not match the fitted "
                              f"{self.identity.source_width}")
        return (source @ self.weights[key] + self.biases[key]).astype(np.float32)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        arrays = {f"w:{k}": v for k, v in self.weights.items()}
        arrays.update({f"b:{k}": v for k, v in self.biases.items()})
        meta = json.dumps({"identity": self.identity.as_dict(),
                           "layer_pairs": [list(p) for p in self.layer_pairs],
                           "residuals": self.residuals})
        np.savez_compressed(path, __meta__=np.frombuffer(meta.encode(), dtype=np.uint8),
                            **arrays)
        return path

    @classmethod
    def load(cls, path: Path | str) -> LinearMapper:
        with np.load(Path(path), allow_pickle=False) as data:
            if "__meta__" not in data:
                raise MapperError("not a kvmap: no metadata")
            meta = json.loads(bytes(data["__meta__"]).decode())
            identity = MapperIdentity(**meta["identity"])
            problems = identity.validate()
            if problems:
                raise MapperError("; ".join(problems))
            weights = {k[2:]: data[k] for k in data.files if k.startswith("w:")}
            biases = {k[2:]: data[k] for k in data.files if k.startswith("b:")}
        return cls(identity=identity,
                   layer_pairs=tuple(tuple(p) for p in meta["layer_pairs"]),
                   weights=weights, biases=biases, residuals=meta.get("residuals", {}))


def select_source_layer(target: np.ndarray, candidates: dict[int, np.ndarray],
                        ridge: float = 1e-2, holdout: float = 0.25
                        ) -> tuple[int, float]:
    """Pick the source layer that predicts this target layer best on held-out tokens.

    Scored on data the fit did not see. Choosing by training residual would pick whichever
    layer overfits hardest, which is the opposite of the intent.
    """
    if not candidates:
        raise MapperError("no candidate source layers")
    if not 0.0 < holdout < 1.0:
        raise MapperError(f"holdout fraction {holdout} must be between 0 and 1")
    best_layer, best_score = None, float("inf")
    for layer, source in sorted(candidates.items()):
        split = int(source.shape[0] * (1.0 - holdout))
        if split <= source.shape[1] or split >= source.shape[0]:
            raise MapperError(f"layer {layer}: not enough tokens to hold any out")
        weights, bias = fit_ridge(source[:split], target[:split], ridge)
        score = residual(source[split:], target[split:], weights, bias)
        if score < best_score:
            best_layer, best_score = layer, score
    return int(best_layer), float(best_score)
