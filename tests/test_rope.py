"""Unit tests for the RoPE mapper and its exact inverse."""

from __future__ import annotations

import unittest

import numpy as np

from kv_rosetta.mappers.rope import (
    RopeError,
    apply_rope,
    default_positions,
    rope_tables,
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


class PartialRotaryTest(unittest.TestCase):
    """Many models rotate only part of each head.

    Qwen3.5 declares rope.dimension_count 64 against a head_dim of 256, so three quarters of
    every head vector is never rotated. Rotating all of it corrupts those dimensions and the
    result stays finite and plausibly scaled. On a real cache pair from this host it drove
    held-out R2 negative for every key, while values - which are never rotated - were
    unaffected. That asymmetry is what exposed it.
    """

    def sample(self, tokens=5, heads=2, dim=256):
        return np.random.default_rng(0).normal(size=(tokens, heads, dim))

    def test_dimensions_past_the_rotary_width_are_untouched(self):
        x = self.sample()
        y = apply_rope(x, default_positions(5), 1e7, rope_dims=64)
        np.testing.assert_array_equal(y[..., 64:], x[..., 64:])

    def test_dimensions_within_it_are_rotated(self):
        x = self.sample()
        y = apply_rope(x, default_positions(5), 1e7, rope_dims=64)
        self.assertFalse(np.allclose(y[..., :64], x[..., :64]))

    def test_a_partial_round_trip_is_exact(self):
        x = self.sample()
        pos = default_positions(5)
        back = strip_rope(apply_rope(x, pos, 1e7, rope_dims=64), pos, 1e7, rope_dims=64)
        np.testing.assert_allclose(back, x, atol=1e-9)

    def test_full_width_is_unchanged_by_the_addition(self):
        x = self.sample()
        pos = default_positions(5)
        np.testing.assert_allclose(strip_rope(apply_rope(x, pos, 1e7), pos, 1e7), x,
                                   atol=1e-9)

    def test_the_whole_head_and_an_explicit_full_width_agree(self):
        x = self.sample(dim=64)
        pos = default_positions(5)
        np.testing.assert_allclose(apply_rope(x, pos, 1e7),
                                   apply_rope(x, pos, 1e7, rope_dims=64))

    def test_partial_and_full_rotation_differ(self):
        """The bug this exists to prevent: they must not be interchangeable."""
        x = self.sample()
        pos = default_positions(5)
        self.assertFalse(np.allclose(apply_rope(x, pos, 1e7),
                                     apply_rope(x, pos, 1e7, rope_dims=64)))

    def test_an_impossible_rotary_width_is_refused(self):
        """The message is asserted because rope_tables raises the same type just after: with
        this guard removed, rope_dims=0 still fails, but for the wrong reason and only by
        luck."""
        x = self.sample()
        for bad in (0, 1, 63, 257, -4):
            with self.subTest(rope_dims=bad):
                with self.assertRaises(RopeError) as caught:
                    apply_rope(x, default_positions(5), 1e7, rope_dims=bad)
                self.assertIn("rope_dims", str(caught.exception))

    def test_positions_must_be_one_dimensional(self):
        """A (T, 1) column would broadcast into the tables and silently change the angles."""
        with self.assertRaises(RopeError) as caught:
            apply_rope(np.zeros((3, 1, 8)), np.zeros((3, 2)), 1e7)
        self.assertIn("1-D", str(caught.exception))


class RemainingGuardTest(unittest.TestCase):
    """Refusals a mutation run found nothing exercising."""

    def test_an_odd_head_width_is_refused_by_the_function_itself(self):
        """RoPE pairs element i with i+half; an odd width has no such pairing.

        The message is asserted, not just the type: rope_tables raises the same RopeError a
        moment later, so a test that only checks "even" passes with this guard removed.
        """
        for fn in (apply_rope, strip_rope):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(RopeError) as caught:
                    fn(np.zeros((3, 1, 7)), default_positions(3), 1e7)
                self.assertIn("x.shape[2]=7", str(caught.exception))

    def test_a_position_count_that_does_not_match_is_refused(self):
        for fn in (apply_rope, strip_rope):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(RopeError) as caught:
                    fn(np.zeros((3, 1, 8)), default_positions(5), 1e7)
                self.assertIn("does not match", str(caught.exception))

    def test_a_negative_count_of_positions_is_refused(self):
        with self.assertRaises(RopeError):
            default_positions(-1)

    def test_a_negative_offset_is_refused(self):
        """Position -1 is not a position; RoPE would rotate it backwards in time."""
        with self.assertRaises(RopeError):
            default_positions(4, offset=-3)

    def test_a_cache_that_is_not_three_dimensional_is_refused(self):
        """(T, H, D) is the contract. A 2-D array would broadcast against the (T,1,D) tables
        and return something finite of the wrong shape."""
        for fn in (apply_rope, strip_rope):
            for bad in (np.zeros((3, 8)), np.zeros((2, 3, 1, 8))):
                with self.subTest(fn=fn.__name__, shape=bad.shape):
                    with self.assertRaises(RopeError) as caught:
                        fn(bad, default_positions(3), 1e7)
                    self.assertIn("3-D", str(caught.exception))


class RopeTableGuardTest(unittest.TestCase):
    """rope_tables is called directly by callers building their own rotation.

    apply_rope and strip_rope check these first, so the guards here are only reachable on the
    direct path - which is exactly why nothing was exercising them.
    """

    def test_a_head_width_it_cannot_pair_is_refused(self):
        for bad in (0, 1, 7, -2):
            with self.subTest(head_dim=bad):
                with self.assertRaises(RopeError) as caught:
                    rope_tables(default_positions(4), bad, 1e7)
                self.assertIn("head_dim", str(caught.exception))

    def test_a_non_positive_theta_is_refused(self):
        """theta 0 divides by zero and a negative one gives complex frequencies."""
        for bad in (0.0, -1.0, -1e7):
            with self.subTest(theta=bad):
                with self.assertRaises(RopeError) as caught:
                    rope_tables(default_positions(4), 64, bad)
                self.assertIn("theta", str(caught.exception))

    def test_the_tables_have_the_documented_shape(self):
        cos, sin = rope_tables(default_positions(5), 64, 1e7)
        self.assertEqual(cos.shape, (5, 1, 64))
        self.assertEqual(sin.shape, (5, 1, 64))
