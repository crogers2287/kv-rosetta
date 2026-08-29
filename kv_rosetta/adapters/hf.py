"""transformers reference adapter. NARROW BY DESIGN: an oracle, not a production adapter.

Everything else in this project reads llama.cpp's view of a KV cache. That makes llama.cpp
both the subject and the witness, which is how the 12-versus-16-byte checkpoint record stayed
wrong for a day. transformers is a second, independent implementation of the same maths, so a
claim that survives both is worth more than one that survives either.

Two facts about HF caches that the canonical form has to record rather than assume:

* `DynamicCache` stores keys **after** RoPE, so an artifact exported from here is
  `rope_applied = True`. A cross-model map must strip that rotation first; mapping post-RoPE
  keys would ask the map to learn positional encoding as if it were model identity.
* Tensors are `[batch, kv_heads, seq, head_dim]`, batched. Canonical form is per sequence,
  so anything but a single batch entry is refused rather than silently taking index zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: HF applies rotary embedding before the cache is written, unlike a raw projection dump.
ROPE_STATE = "applied"


class HFAdapterError(ValueError):
    """Raised when a cache cannot be represented faithfully. Never a silent coercion."""


@dataclass(frozen=True)
class HFGeometry:
    n_layer: int
    n_kv_head: int
    head_dim: int
    rope_state: str = ROPE_STATE

    def validate(self) -> list[str]:
        problems = []
        for name in ("n_layer", "n_kv_head", "head_dim"):
            if getattr(self, name) <= 0:
                problems.append(f"{name} must be positive")
        if self.rope_state != ROPE_STATE:
            problems.append(f"an HF cache is always {ROPE_STATE}; {self.rope_state!r} "
                            f"misdescribes it")
        return problems


def _tensors(cache) -> tuple[list, list]:
    """Keys and values out of whichever cache representation this version hands back.

    transformers has three in circulation and a caller should not have to know which one it
    has: a Cache object with key_cache/value_cache, one that only offers to_legacy_cache,
    and the bare legacy tuple of (key, value) pairs that 4.46 returns from a forward pass
    when no Cache was supplied. All three are the same data.
    """
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if keys is None or values is None:
        legacy = getattr(cache, "to_legacy_cache", None)
        if legacy is not None:
            pairs = legacy()
        elif isinstance(cache, (tuple, list)):
            pairs = cache
        else:
            raise HFAdapterError(f"{type(cache).__name__} is not a cache: it exposes "
                                 f"neither key_cache nor to_legacy_cache, and is not a "
                                 f"legacy (key, value) sequence")
        pairs = list(pairs)
        for index, pair in enumerate(pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise HFAdapterError(f"legacy cache entry {index} is not a (key, value) "
                                     f"pair")
        keys = [pair[0] for pair in pairs]
        values = [pair[1] for pair in pairs]
    if len(keys) != len(values):
        raise HFAdapterError(f"{len(keys)} key layers against {len(values)} value layers")
    return list(keys), list(values)


def export_cache(cache) -> tuple[np.ndarray, HFGeometry]:
    """A DynamicCache as canonical (layer, kv, token, head, dim) float32.

    Returns the array and the geometry read from the tensors themselves, so a caller cannot
    describe a cache as something other than what it is.
    """
    keys, values = _tensors(cache)
    if not keys:
        raise HFAdapterError("the cache is empty")

    layers = []
    shape = None
    for index, (k, v) in enumerate(zip(keys, values)):
        k_np = np.asarray(k.detach().to("cpu", copy=False).float().numpy())
        v_np = np.asarray(v.detach().to("cpu", copy=False).float().numpy())
        for name, tensor in (("key", k_np), ("value", v_np)):
            if tensor.ndim != 4:
                raise HFAdapterError(f"layer {index} {name} has {tensor.ndim} dimensions, "
                                     f"expected 4 (batch, heads, seq, dim)")
            if tensor.shape[0] != 1:
                raise HFAdapterError(
                    f"layer {index} {name} has batch {tensor.shape[0]}; canonical form is "
                    f"per sequence, and taking index zero would silently discard the rest")
        if k_np.shape != v_np.shape:
            raise HFAdapterError(f"layer {index}: key shape {k_np.shape} != value "
                                 f"{v_np.shape}")
        if shape is None:
            shape = k_np.shape
        elif k_np.shape != shape:
            raise HFAdapterError(f"layer {index} shape {k_np.shape} differs from layer 0 "
                                 f"{shape}; a ragged cache has no canonical form")
        # (1, heads, seq, dim) -> (seq, heads, dim)
        layers.append(np.stack([k_np[0].transpose(1, 0, 2), v_np[0].transpose(1, 0, 2)]))

    canonical = np.stack(layers).astype(np.float32)      # (layer, kv, token, head, dim)
    geometry = HFGeometry(n_layer=len(layers), n_kv_head=int(shape[1]),
                          head_dim=int(shape[3]))
    problems = geometry.validate()
    if problems:
        raise HFAdapterError("; ".join(problems))
    return canonical, geometry


def import_cache(canonical: np.ndarray, cache_class=None, dtype=None):
    """Canonical tensors back into a transformers cache object.

    The inverse of export_cache. Layer count, ordering and dtype all come from the array and
    the caller; nothing is inferred from a model, because a cache that only round-trips
    against the model it came from would prove nothing about portability.
    """
    try:
        import torch
    except ImportError as exc:                     # pragma: no cover - declared dependency
        raise HFAdapterError("torch is required to build a cache") from exc

    array = np.asarray(canonical)
    if array.ndim != 5 or array.shape[1] != 2:
        raise HFAdapterError(f"expected (layer, kv, token, head, dim) with kv=2, got "
                             f"{array.shape}")
    if cache_class is None:
        from transformers.cache_utils import DynamicCache
        cache_class = DynamicCache

    target_dtype = dtype or torch.float32
    cache = cache_class()
    for layer in range(array.shape[0]):
        # (token, head, dim) -> (1, head, token, dim)
        k = torch.from_numpy(array[layer, 0].transpose(1, 0, 2).copy()).to(target_dtype)[None]
        v = torch.from_numpy(array[layer, 1].transpose(1, 0, 2).copy()).to(target_dtype)[None]
        cache.update(k, v, layer)
    return cache


def describe(geometry: HFGeometry, model_digest: str = "") -> dict[str, Any]:
    """The manifest fields an HF-exported artifact must carry."""
    return {"runtime": "transformers", "rope_state": geometry.rope_state,
            "n_layer": geometry.n_layer, "n_kv_head": geometry.n_kv_head,
            "head_dim": geometry.head_dim, "layout": "layer,kv,token,head,dim",
            "model_digest": model_digest}
