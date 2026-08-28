from __future__ import annotations

import unittest

import numpy as np

from kv_rosetta.metrics import (
    kl_divergence,
    log_softmax,
    max_abs_logit_delta,
    MetricsError,
    top1_agreement,
    tensor_cosine,
)


class TestLogSoftmax(unittest.TestCase):
    def test_rows_sum_to_one(self):
        rng = np.random.default_rng(1)
        logits = rng.standard_normal((7, 13))
        out = log_softmax(logits)
        self.assertEqual(out.shape, logits.shape)
        self.assertTrue(np.allclose(np.exp(out).sum(axis=-1), 1.0))

    def test_stable_for_large_values(self):
        logits = np.array([[1000.0, 1001.0, 999.0]])
        out = log_softmax(logits)
        self.assertTrue(np.all(np.isfinite(out)))
        # argmax preserved
        self.assertEqual(int(np.argmax(out[0])), 1)


class TestKlDivergence(unittest.TestCase):
    def test_self_is_zero(self):
        rng = np.random.default_rng(3)
        logits = rng.standard_normal((20, 32))
        ref = log_softmax(logits)
        cand = log_softmax(logits)
        kl = kl_divergence(ref, cand)
        self.assertEqual(kl.shape, (20,))
        self.assertTrue(np.allclose(kl, 0.0, atol=1e-12))

    def test_non_negative_on_random(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((50, 64))
        b = rng.standard_normal((50, 64))
        ref = log_softmax(a)
        cand = log_softmax(b)
        kl = kl_divergence(ref, cand)
        self.assertTrue(np.all(kl >= -1e-12))

    def test_shape_mismatch_raises(self):
        ref = log_softmax(np.zeros((5, 10)))
        cand = log_softmax(np.zeros((5, 11)))
        with self.assertRaises(MetricsError):
            kl_divergence(ref, cand)

    def test_ndim_check_raises(self):
        with self.assertRaises(MetricsError):
            kl_divergence(np.zeros(10), np.zeros(10))


class TestTop1Agreement(unittest.TestCase):
    def test_identical_is_one(self):
        ref = np.array([[1.0, 2.0, 3.0], [5.0, 1.0, 1.0]])
        self.assertEqual(top1_agreement(ref, ref), 1.0)

    def test_always_differs_is_zero(self):
        ref = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        cand = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        self.assertEqual(top1_agreement(ref, cand), 0.0)

    def test_empty_returns_one(self):
        self.assertEqual(top1_agreement(np.zeros((0, 5)), np.zeros((0, 5))), 1.0)


class TestMaxAbsLogitDelta(unittest.TestCase):
    def test_true_maximum(self):
        ref = np.array([[1.0, 2.0], [3.0, 4.0]])
        cand = np.array([[1.5, 2.0], [3.0, 4.9]])
        self.assertAlmostEqual(max_abs_logit_delta(ref, cand), 0.9, places=12)

    def test_empty_returns_zero(self):
        self.assertEqual(max_abs_logit_delta(np.zeros((0, 5)), np.zeros((0, 5))), 0.0)


class TestTensorCosine(unittest.TestCase):
    def test_self_is_one(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(tensor_cosine(v, v), 1.0, places=6)

    def test_zero_vector_is_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        z = np.zeros(3)
        self.assertAlmostEqual(tensor_cosine(v, z), 0.0, places=6)
        self.assertAlmostEqual(tensor_cosine(z, v), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
