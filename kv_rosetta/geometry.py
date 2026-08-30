"""Converting canonical KV between model geometries, separately from converting meaning.

Two different problems live behind "make the file readable by another model" and keeping them
apart is what makes either tractable.

**Geometry** is mechanical and checkable: a target with sixteen attention layers of four heads
will not read eight layers of four heads, and llama.cpp says so precisely -
`state_read_data: mismatched layer count (16 instead of 8)`. Reshaping to the target's layer
and head counts is arithmetic, and its correctness has a definition: the target runtime
accepts the file.

**Representation** is the hard half. Measured on this host across three model pairs, two
lineage relationships and four fitting strategies, a linear map recovers about half the
variance and the gate rejects the result. Nothing in this module improves that, and nothing
here should be read as claiming to.

The value of separating them is that a geometry conversion can be *proved* correct while the
representation map is still being argued about - and a file that the target accepts is the
precondition for measuring any map at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: How to merge source KV heads when the target has fewer. Neither is obviously right and the
#: gate is the only thing that can decide between them, so both are offered and neither is a
#: default that hides a choice.
HEAD_POLICIES = frozenset({"mean", "select"})


class GeometryError(ValueError):
    """Raised when a conversion cannot be defined. Never a silent reshape."""


@dataclass(frozen=True)
class Geometry:
    """The shape of one model's attention cache, per token."""

    layers: int
    kv_heads: int
    head_dim: int

    def validate(self) -> list[str]:
        return [f"{name} must be positive" for name in ("layers", "kv_heads", "head_dim")
                if getattr(self, name) <= 0]

    @property
    def width(self) -> int:
        return self.kv_heads * self.head_dim


def depth_layer_map(source: Geometry, target: Geometry) -> tuple[tuple[int, int], ...]:
    """(target_layer, source_layer) by normalised depth, when nothing better is known.

    A placeholder for a fitted selection, not a rival to one: a fitted map chose source layers
    1, 3, 5, 6, 7, 10, 8, 11, 12, 15 for ten targets, which is near-monotonic but not this.
    Depth matching is what to use before a fit exists.
    """
    if target.layers == 1:
        return ((0, 0),)
    span = source.layers - 1
    return tuple((t, int(round(t * span / (target.layers - 1))))
                 for t in range(target.layers))


def convert(canonical, source: Geometry, target: Geometry, *,
            layer_map: tuple[tuple[int, int], ...] | None = None,
            head_policy: str = "mean") -> np.ndarray:
    """Reshape canonical `(layer, kv, token, head, dim)` from one geometry into another.

    This changes only shape. Every value that comes out was in the input, possibly averaged
    with its neighbours - there is no fitted map here, and the result is **not** expected to
    be usable as the target's cache. It is expected to be *loadable*, which is a different
    and much weaker claim, and the one that has to hold before quality can be measured.
    """
    array = np.asarray(canonical)
    problems = source.validate() + target.validate()
    if problems:
        raise GeometryError("; ".join(problems))
    if head_policy not in HEAD_POLICIES:
        raise GeometryError(f"head_policy {head_policy!r} is not one of "
                            f"{sorted(HEAD_POLICIES)}")
    if source.head_dim != target.head_dim:
        raise GeometryError(
            f"head_dim {source.head_dim} against {target.head_dim}: this reshapes heads and "
            f"layers and cannot change the width of a head, which is a projection rather "
            f"than a regrouping")
    if array.ndim != 5 or array.shape[1] != 2:
        raise GeometryError(f"expected canonical (layer, kv, token, head, dim) with kv=2, "
                            f"got {array.shape}")
    if array.shape[0] != source.layers or array.shape[3] != source.kv_heads \
            or array.shape[4] != source.head_dim:
        raise GeometryError(
            f"cache is {array.shape[0]}x{array.shape[3]}x{array.shape[4]} but the source "
            f"geometry says {source.layers}x{source.kv_heads}x{source.head_dim}")

    pairs = layer_map if layer_map is not None else depth_layer_map(source, target)
    seen = [t for t, _s in pairs]
    if sorted(seen) != list(range(target.layers)):
        raise GeometryError(
            f"the layer map covers targets {sorted(set(seen))} but the target has "
            f"{target.layers} layers; every target layer needs exactly one source")
    for _t, s in pairs:
        if not 0 <= s < source.layers:
            raise GeometryError(f"layer map reads source layer {s} of {source.layers}")

    tokens = array.shape[2]
    out = np.zeros((target.layers, 2, tokens, target.kv_heads, target.head_dim),
                   dtype=array.dtype)
    for t, s in pairs:
        out[t] = _regroup(array[s], source.kv_heads, target.kv_heads, head_policy)
    return out


def _regroup(layer, source_heads: int, target_heads: int, policy: str):
    """One layer's heads, regrouped. Shape (kv, token, head, dim) in and out."""
    if source_heads == target_heads:
        return layer
    if source_heads > target_heads:
        if source_heads % target_heads:
            raise GeometryError(
                f"{source_heads} source heads do not group evenly into {target_heads}; an "
                f"uneven grouping would weight some source heads more than others without "
                f"saying so")
        group = source_heads // target_heads
        shaped = layer.reshape(layer.shape[0], layer.shape[1], target_heads, group,
                               layer.shape[3])
        # GQA groups adjacent KV heads, so adjacent source heads are the ones to merge.
        return shaped.mean(axis=3) if policy == "mean" else shaped[:, :, :, 0, :]
    if target_heads % source_heads:
        raise GeometryError(
            f"{source_heads} source heads do not expand evenly into {target_heads}")
    return np.repeat(layer, target_heads // source_heads, axis=2)
