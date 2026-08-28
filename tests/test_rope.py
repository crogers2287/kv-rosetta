"""Unit tests for the RoPE mapper and its exact inverse."""

from __future__ import annotations

import unittest

import numpy as np

from kv_rosetta.mappers.rope import (
    RopeError,
    apply_rope,
    default_positions,
    rotate_half,
    strip_rope,
)


class RopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)

    def test_apply_then_strip_is_identity(self) -> None:
        x = self.rng.standard_normal((17, 4, 64))
        p = default_positions(17)
        out = strip_rope(apply_rope(x, p), p)
        self.assertTrue(np.allclose(out, x, atol=1e-9, rtol=0))

    def test_strip_then_apply_is_identity(self) -> None:
        x = self.rng.standard_normal((17, 4, 64))
        p = default_positions(17)
        out = apply_rope(strip_rope(x, p), p)
        self.assertTrue(np.allclose(out, x, atol=1e-9, rtol=0))

    def test_position_zero_is_a_no_op(self) -> None:
        x = self.rng.standard_normal((17, 4, 64))
        p = np.zeros(17, dtype=np.int64)
        out = apply_rope(x, p)
        self.assertTrue(np.allclose(out, x, atol=1e-12))

    def test_norm_is_preserved(self) -> None:
        x = self.rng.standard_normal((17, 4, 64))
        p = default_positions(17)
        y = apply_rope(x, p)
        norm_x = np.linalg.norm(x, axis=-1)
        norm_y = np.linalg.norm(y, axis=-1)
        self.assertTrue(np.allclose(norm_x, norm_y, atol=1e-9))

    def test_theta_matters(self) -> None:
        x = self.rng.standard_normal((1, 1, 8))
        p = np.array([5], dtype=np.int64)
        y1 = apply_rope(x, p, theta=10000.0)
        y2 = apply_rope(x, p, theta=1000000.0)
        self.assertFalse(np.allclose(y1, y2))

    def test_rotate_half_twice_is_negation(self) -> None:
        x = self.rng.standard_normal((3, 2, 10))
        self.assertTrue(np.array_equal(rotate_half(rotate_half(x)), -x))

    def test_odd_head_dim_rejected(self) -> None:
        x = np.zeros((2, 1, 5))
        p = default_positions(2)
        with self.assertRaises(RopeError):
            apply_rope(x, p)

    def test_shape_mismatch_rejected(self) -> None:
        x = np.zeros((2, 1, 8))
        p = default_positions(3)
        with self.assertRaises(RopeError):
            apply_rope(x, p)

    def test_dtype_preserved(self) -> None:
        x = self.rng.standard_normal((4, 2, 8)).astype(np.float32)
        p = default_positions(4)
        out = apply_rope(x, p)
        self.assertEqual(out.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
