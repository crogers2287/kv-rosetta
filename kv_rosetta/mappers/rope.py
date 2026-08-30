"""Rotary position embedding (RoPE) and its exact inverse.

Implements the HuggingFace transformers convention for applying and
removing rotary position embeddings from a KV cache. The forward map
applies a per-position rotation to each head; ``strip_rope`` is the exact
inverse, removing that rotation. This is required for cross-model KV cache
exchange: the source rotation must be removed before a mapping is applied,
then the target rotation applied.
"""

from __future__ import annotations

import numpy as np

from typing import Tuple


class RopeError(ValueError):
    pass


def rope_tables(
    positions: np.ndarray, head_dim: int, theta: float = 10000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cos, sin) tables, each shape (T, 1, head_dim), float64."""
    if head_dim < 2 or head_dim % 2 != 0:
        raise RopeError(f"head_dim must be even and >= 2, got {head_dim}")
    if theta <= 0:
        raise RopeError(f"theta must be positive, got {theta}")
    if positions.ndim != 1:
        raise RopeError(f"positions must be 1-D, got ndim={positions.ndim}")

    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (2.0 * np.arange(half, dtype=np.float64) / head_dim))
    angle = positions.astype(np.float64)[:, None] * inv_freq[None, :]  # (T, half)
    cos_half = np.cos(angle)
    sin_half = np.sin(angle)
    cos = np.concatenate([cos_half, cos_half], axis=1)  # (T, D)
    sin = np.concatenate([sin_half, sin_half], axis=1)  # (T, D)
    return cos[:, None, :], sin[:, None, :]  # (T, 1, D)


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Concatenate -x[..., half:] and x[..., :half] along the last axis."""
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _split_rotary(x: np.ndarray, rope_dims: int | None) -> int:
    """How many leading dimensions of each head the rotation covers.

    Many models rotate only part of each head. Qwen3.5 declares
    `rope.dimension_count` 64 against a head_dim of 256, so three quarters of every head
    vector is never rotated. Rotating all of it corrupts those dimensions, and the result
    stays finite and plausibly scaled - on a real cache pair it drove held-out R2 negative
    for every key while values, which are never rotated, were unaffected. That asymmetry is
    what exposed it.
    """
    width = x.shape[-1]
    if rope_dims is None:
        return width
    if not 2 <= rope_dims <= width or rope_dims % 2:
        raise RopeError(f"rope_dims {rope_dims} must be even and within the head width "
                        f"{width}")
    return rope_dims


def apply_rope(
    x: np.ndarray, positions: np.ndarray, theta: float = 10000.0,
    rope_dims: int | None = None
) -> np.ndarray:
    """Apply RoPE. x shape (T, H, D). Returns same shape and dtype as x.

    `rope_dims` limits the rotation to the leading dimensions of each head, as
    `rope.dimension_count` declares. None rotates the whole head.
    """
    if x.ndim != 3:
        raise RopeError(f"x must be 3-D (T, H, D), got ndim={x.ndim}")
    if x.shape[0] != positions.shape[0]:
        raise RopeError(
            f"position count {positions.shape[0]} does not match T {x.shape[0]}"
        )
    if x.shape[2] % 2 != 0:
        raise RopeError(f"head_dim (x.shape[2]={x.shape[2]}) must be even")

    dtype = x.dtype
    span = _split_rotary(x, rope_dims)
    xf = x.astype(np.float64)
    head = xf[..., :span]
    cos, sin = rope_tables(positions, span, theta)
    rotated = head * cos + rotate_half(head) * sin
    if span == x.shape[2]:
        return rotated.astype(dtype)
    out = xf.copy()
    out[..., :span] = rotated
    return out.astype(dtype)


def strip_rope(
    x: np.ndarray, positions: np.ndarray, theta: float = 10000.0,
    rope_dims: int | None = None
) -> np.ndarray:
    """Exact inverse of apply_rope. x shape (T, H, D). Returns same shape."""
    if x.ndim != 3:
        raise RopeError(f"x must be 3-D (T, H, D), got ndim={x.ndim}")
    if x.shape[0] != positions.shape[0]:
        raise RopeError(
            f"position count {positions.shape[0]} does not match T {x.shape[0]}"
        )
    if x.shape[2] % 2 != 0:
        raise RopeError(f"head_dim (x.shape[2]={x.shape[2]}) must be even")

    dtype = x.dtype
    span = _split_rotary(x, rope_dims)
    xf = x.astype(np.float64)
    head = xf[..., :span]
    cos, sin = rope_tables(positions, span, theta)
    unrotated = head * cos - rotate_half(head) * sin
    if span == x.shape[2]:
        return unrotated.astype(dtype)
    out = xf.copy()
    out[..., :span] = unrotated
    return out.astype(dtype)


def default_positions(n: int, offset: int = 0) -> np.ndarray:
    """np.arange(offset, offset + n, dtype=np.int64)."""
    if n < 0:
        raise RopeError(f"n must be >= 0, got {n}")
    if offset < 0:
        raise RopeError(f"offset must be >= 0, got {offset}")
    return np.arange(offset, offset + n, dtype=np.int64)
