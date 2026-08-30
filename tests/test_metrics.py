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
    positionwise_agreement,
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


class PositionwiseAgreementTest(unittest.TestCase):
    """A metric that grades a map, as distinct from one that admits a cache.

    Top-1 agreement over a free generation is a cliff: generation is autoregressive, so one
    wrong token derails everything after it and the number reports when the first divergence
    happened rather than how good the cache was. Measured on a real pair, a nearly-perfect
    blend and a fully translated one both scored 0.042 - identical, and useless for tuning.

    Scored position by position against the same forced prefix, the result varies smoothly
    with quality. These tests pin that smoothness, because it is the whole point.
    """

    def dist(self, *pairs):
        return [dict(p) for p in pairs]

    def test_identical_sequences_agree_completely(self):
        ref = self.dist({1: -0.1, 2: -2.0}, {1: -0.5, 2: -1.0})
        found = positionwise_agreement(ref, ref)
        self.assertEqual(found["top1_agreement"], 1.0)
        self.assertEqual(found["mean_abs_logprob_delta"], 0.0)

    def test_the_delta_varies_smoothly_with_perturbation(self):
        """The property a cliff metric does not have: more error, more number."""
        ref = self.dist({1: -0.1, 2: -2.0}, {1: -0.5, 2: -1.0})
        previous = -1.0
        for scale in (0.001, 0.01, 0.1, 1.0):
            candidate = [{k: v + scale for k, v in pos.items()} for pos in ref]
            found = positionwise_agreement(ref, candidate)["mean_abs_logprob_delta"]
            self.assertGreater(found, previous)
            previous = found

    def test_one_wrong_position_does_not_condemn_the_rest(self):
        """Under free generation this would be near zero; here it is 3 of 4."""
        ref = self.dist({1: -0.1, 2: -2.0}, {1: -0.1, 2: -2.0},
                        {1: -0.1, 2: -2.0}, {1: -0.1, 2: -2.0})
        candidate = list(ref)
        candidate[1] = {1: -2.0, 2: -0.1}
        self.assertAlmostEqual(positionwise_agreement(ref, candidate)["top1_agreement"], 0.75)

    def test_the_mean_and_the_max_are_both_reported(self):
        """A single bad position raises the max while barely moving the mean; reporting only
        one of them hides which case you are in."""
        ref = self.dist({1: -0.1}, {1: -0.1}, {1: -0.1}, {1: -0.1})
        candidate = self.dist({1: -0.1}, {1: -0.1}, {1: -0.1}, {1: -4.1})
        found = positionwise_agreement(ref, candidate)
        self.assertAlmostEqual(found["max_abs_logprob_delta"], 4.0)
        self.assertAlmostEqual(found["mean_abs_logprob_delta"], 1.0)

    def test_membership_differences_are_counted_not_folded_in(self):
        found = positionwise_agreement(self.dist({1: -0.1, 2: -0.2}),
                                       self.dist({1: -0.1, 9: -0.2}))
        self.assertEqual((found["shared_tokens"], found["tokens_only_in_one"]), (1, 2))
        self.assertEqual(found["max_abs_logprob_delta"], 0.0)

    def test_comparison_stops_at_the_shorter_and_says_so(self):
        found = positionwise_agreement(self.dist({1: -0.1}),
                                       self.dist({1: -0.1}, {1: -0.2}, {1: -0.3}))
        self.assertEqual(found["positions"], 1)

    def test_nothing_to_compare_is_not_agreement(self):
        found = positionwise_agreement([], [])
        self.assertEqual(found["positions"], 0)
        self.assertIsNone(found["top1_agreement"])
