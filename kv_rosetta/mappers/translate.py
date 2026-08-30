"""The cross-model path, end to end: one model's cache rendered for another.

Everything here is composition. The parts were built and tested separately - RoPE
apply/strip, byte-span alignment, the per-layer ridge map - and none of them was reachable
from any other. This is the order they have to run in, and the reasons the order is not
negotiable:

1. **Strip RoPE from the keys, at the source positions.** Cached keys are post-rotation. Two
   keys for the same word at different positions are different vectors, so anything that
   averages or maps them while rotated is learning position as if it were meaning.
2. **Pool onto the target tokenization.** Different vocabularies split the same bytes
   differently; the alignment says which source tokens each target token covers and in what
   proportion.
3. **Map each layer.** Layer counts and KV head counts differ, so a fitted linear map takes
   one source layer's flattened `heads x dim` vector to the target's.
4. **Re-apply RoPE at the target positions**, with the target's theta. A token that was third
   in the source prompt may be second in the target tokenization, and the rotation has to
   match where it now sits.

**Values are never rotated.** RoPE is applied to queries and keys only. Stripping or applying
it to values would corrupt them, and because the result stays finite and plausibly scaled,
nothing downstream would notice. Keys and values are therefore handled separately throughout.

What comes out is a **candidate**, not a cache. Nothing here establishes that the result is
usable - only the divergence gate does, by comparing next-token behaviour against the target
model's own prefill. A candidate that looks numerically close and fails the gate is the case
this whole project is built around.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kv_rosetta.mappers.align import Alignment, pool
from kv_rosetta.mappers.rope import apply_rope, default_positions, strip_rope

#: kv axis of the canonical (layer, kv, token, head, dim) form.
KEYS, VALUES = 0, 1


class TranslateError(ValueError):
    """Raised when a translation cannot be justified. The output is never approximate."""


@dataclass(frozen=True)
class TranslationPlan:
    """What a translation would do, stated before it is done."""

    source_layers: int
    target_layers: int
    source_tokens: int
    target_tokens: int
    source_width: int
    target_width: int
    #: (target_layer, source_layer), the same order LinearMapper uses.
    layer_pairs: tuple[tuple[int, int], ...]

    def describe(self) -> str:
        return (f"{self.source_layers}->{self.target_layers} layers, "
                f"{self.source_width}->{self.target_width} per token, "
                f"{self.source_tokens}->{self.target_tokens} tokens")


def _require_canonical(array, name: str) -> np.ndarray:
    found = np.asarray(array)
    if found.ndim != 5 or found.shape[1] != 2:
        raise TranslateError(f"{name} must be canonical (layer, kv, token, head, dim) with "
                             f"kv=2, got {found.shape}")
    if not np.isfinite(found).all():
        raise TranslateError(f"{name} contains non-finite values; a cache with a NaN in it "
                             f"is not a cache")
    return found


def plan_translation(source_kv, mapper, *, alignment: Alignment | None = None
                     ) -> TranslationPlan:
    """Describe the translation without performing it, so it can be refused early."""
    array = _require_canonical(source_kv, "source_kv")
    layers, _kv, tokens, heads, dim = array.shape
    # LinearMapper stores (target_layer, source_layer). Reading it the other way round maps
    # every layer to the wrong one, which produces a full, finite, entirely wrong cache.
    pairs = tuple((int(t), int(s)) for t, s in mapper.layer_pairs)
    if not pairs:
        raise TranslateError("the mapper carries no layer pairs; nothing would be produced")
    for target_layer, source_layer in pairs:
        if not 0 <= source_layer < layers:
            raise TranslateError(f"the mapper reads source layer {source_layer}, but the "
                                 f"cache has {layers}")
        if target_layer < 0:
            raise TranslateError(f"negative target layer {target_layer}")
    if heads * dim != mapper.identity.source_width:
        raise TranslateError(
            f"the cache is {heads}x{dim}={heads * dim} wide per token but the map was fitted "
            f"for {mapper.identity.source_width}; applying it would reinterpret the vectors")
    target_layers = max(t for t, _s in pairs) + 1
    if len({t for t, _s in pairs}) != len(pairs):
        raise TranslateError("two source layers map to one target layer; the later would "
                             "silently overwrite the earlier")
    return TranslationPlan(
        source_layers=layers, target_layers=target_layers, source_tokens=tokens,
        target_tokens=alignment.target_tokens if alignment else tokens,
        source_width=heads * dim, target_width=mapper.identity.target_width,
        layer_pairs=pairs)


def translate(source_kv, mapper, *, target_head_dim: int,
              alignment: Alignment | None = None,
              source_positions=None, target_positions=None,
              source_theta: float = 10000.0, target_theta: float | None = None,
              rope_state: str = "applied") -> np.ndarray:
    """One model's canonical cache rendered into another's geometry. A candidate, not a cache.

    `rope_state` describes the input. "applied" is what any real export carries; the rotation
    is stripped before anything else touches the keys and re-applied at the end. "not_applied"
    is for a source that has already been stripped, and skips both steps rather than
    double-stripping - which would leave the keys rotated backwards and still finite.
    """
    array = _require_canonical(source_kv, "source_kv")
    found = plan_translation(array, mapper, alignment=alignment)
    if rope_state not in ("applied", "not_applied"):
        raise TranslateError(f"rope_state {rope_state!r} is neither 'applied' nor "
                             f"'not_applied'; guessing would rotate the keys the wrong way")
    if target_head_dim <= 0 or target_head_dim % 2:
        raise TranslateError(f"target head_dim {target_head_dim} must be positive and even; "
                             f"RoPE rotates pairs of elements")
    if found.target_width % target_head_dim:
        raise TranslateError(
            f"the map produces {found.target_width} values per token, which is not a whole "
            f"number of {target_head_dim}-wide heads")
    target_heads = found.target_width // target_head_dim
    theta_out = source_theta if target_theta is None else target_theta

    src_pos = (default_positions(found.source_tokens) if source_positions is None
               else np.asarray(source_positions))
    tgt_pos = (default_positions(found.target_tokens) if target_positions is None
               else np.asarray(target_positions))
    if len(src_pos) != found.source_tokens:
        raise TranslateError(f"{len(src_pos)} source positions for "
                             f"{found.source_tokens} tokens")
    if len(tgt_pos) != found.target_tokens:
        raise TranslateError(f"{len(tgt_pos)} target positions for "
                             f"{found.target_tokens} tokens")

    # 1. Strip the rotation from the keys only. Values are never rotated.
    work = array.astype(np.float64, copy=True)
    if rope_state == "applied":
        for layer in range(found.source_layers):
            work[layer, KEYS] = strip_rope(work[layer, KEYS], src_pos, source_theta)

    # 2. Pool onto the target tokenization. Both kv slots move together; only the token axis
    #    changes, and the keys are unrotated by now so the mean is over comparable vectors.
    if alignment is not None:
        work = pool(work, alignment, rope_state="not_applied")

    # 3. Map each target layer from its source layer, keys and values separately.
    out = np.zeros((found.target_layers, 2, found.target_tokens, target_heads,
                    target_head_dim), dtype=np.float32)
    tokens = found.target_tokens
    for target_layer, source_layer in found.layer_pairs:
        for axis, kind in ((KEYS, "k"), (VALUES, "v")):
            flat = work[source_layer, axis].reshape(tokens, found.source_width)
            mapped = mapper.apply_layer(flat, target_layer, kind)
            if mapped.shape != (tokens, found.target_width):
                raise TranslateError(
                    f"the map for target layer {target_layer} {kind} returned "
                    f"{mapped.shape}, not {(tokens, found.target_width)}")
            out[target_layer, axis] = mapped.reshape(tokens, target_heads, target_head_dim)

    # 4. Rotate the keys back, at the positions they now occupy and the target's theta.
    if rope_state == "applied":
        for layer in range(found.target_layers):
            out[layer, KEYS] = apply_rope(out[layer, KEYS], tgt_pos, theta_out)
    if not np.isfinite(out).all():
        raise TranslateError("the translation produced non-finite values; the map or the "
                             "rotation is wrong and the result must not be admitted")
    return out
