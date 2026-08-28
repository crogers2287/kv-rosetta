import unittest

import numpy as np

from kv_rosetta.gate import EXACT, GateBinding, GateError, GateThresholds, admit


class GateAdmissionTests(unittest.TestCase):
    def test_identical_logits_admitted(self):
        rng = np.random.default_rng(1234)
        ref = rng.standard_normal((64, 32))
        cand = ref.copy()
        result = admit(ref, cand)
        self.assertTrue(result.admitted)
        self.assertEqual(result.reason, "admitted")

    def test_high_cosine_low_agreement_is_rejected(self):
        T, V = 64, 128
        C = 1000.0
        ref = np.full((T, V), C, dtype=np.float64)
        cand = np.full((T, V), C, dtype=np.float64)
        rows = np.arange(T)
        ref[rows, rows % V] += 1.0
        cand[rows, (rows + 1) % V] += 1.0
        result = admit(ref, cand, reference_tensor=ref, candidate_tensor=cand)
        self.assertGreater(result.metrics["tensor_cosine"], 0.999)
        self.assertFalse(result.admitted)
        n_diff = int(np.sum(np.argmax(ref, axis=1) != np.argmax(cand, axis=1)))
        self.assertGreaterEqual(n_diff, T // 2)

    def test_shape_mismatch_returns_false_without_raising(self):
        ref = np.zeros((10, 20))
        cand = np.zeros((10, 21))
        result = admit(ref, cand)
        self.assertFalse(result.admitted)
        self.assertIn("shape mismatch", result.reason)

    def test_empty_holdout_returns_no_holdout_tokens(self):
        ref = np.zeros((0, 16))
        cand = np.zeros((0, 16))
        result = admit(ref, cand)
        self.assertFalse(result.admitted)
        self.assertIn("insufficient holdout", result.reason)

    def test_holdout_truncation_scores_tail(self):
        T = 100
        ref = np.zeros((T, 16))
        cand = np.zeros((T, 16))
        ref[:50] = 1000.0
        cand[:50] = 0.0
        thresholds = GateThresholds(holdout_tokens=4)
        result = admit(ref, cand, thresholds)
        self.assertEqual(result.metrics["holdout_tokens"], 4)
        self.assertTrue(result.admitted)

    def test_exact_thresholds(self):
        rng = np.random.default_rng(7)
        ref = rng.standard_normal((64, 16))
        identical = admit(ref, ref.copy(), EXACT)
        self.assertTrue(identical.admitted)
        self.assertEqual(identical.reason, "admitted")
        perturbed = admit(ref, ref + 1e-3, EXACT)
        self.assertFalse(perturbed.admitted)

    def test_none_tensors_yield_nan_cosine_without_affecting_admission(self):
        ref = np.zeros((64, 16))
        cand = np.zeros((64, 16))
        result = admit(ref, cand)
        self.assertTrue(np.isnan(result.metrics["tensor_cosine"]))
        self.assertTrue(result.admitted)


class GateFailsClosedTests(unittest.TestCase):
    """Regressions for the fail-closed contract.

    Each of these passed before hardening, which is exactly why they are kept: the
    gate is the last thing standing between a translated cache and a runtime import.
    """

    def test_holdout_is_a_minimum_not_a_slice(self):
        # One position must never satisfy a 64-position policy.
        ref = np.array([[9.0, 0.0, 0.0]])
        result = admit(ref, ref.copy(), GateThresholds(holdout_tokens=64))
        self.assertFalse(result.admitted)
        self.assertIn("insufficient holdout", result.reason)
        self.assertEqual(result.metrics["required_holdout_tokens"], 64)

    def test_short_holdout_rejected_for_every_shortfall(self):
        for available in (1, 8, 63):
            ref = np.zeros((available, 4))
            with self.subTest(available=available):
                self.assertFalse(admit(ref, ref.copy(), GateThresholds(holdout_tokens=64)).admitted)

    def test_exactly_enough_holdout_is_accepted(self):
        ref = np.tile(np.array([[9.0, 0.0, 0.0]]), (64, 1))
        self.assertTrue(admit(ref, ref.copy(), GateThresholds(holdout_tokens=64)).admitted)

    def test_non_finite_thresholds_are_rejected_at_construction(self):
        for kwargs in (
            {"top1_agreement": float("nan")},
            {"top1_agreement": float("inf")},
            {"mean_kl": float("nan")},
            {"mean_kl": float("inf")},
            {"max_kl": float("inf")},
            {"max_logit_delta": float("nan")},
        ):
            with self.subTest(**kwargs), self.assertRaises(GateError):
                GateThresholds(**kwargs)

    def test_out_of_range_thresholds_are_rejected(self):
        for kwargs in (
            {"holdout_tokens": 0},
            {"holdout_tokens": -1},
            {"top1_agreement": -0.1},
            {"top1_agreement": 1.1},
            {"mean_kl": -1.0},
            {"max_kl": -0.001},
            {"max_logit_delta": -1.0},
        ):
            with self.subTest(**kwargs), self.assertRaises(GateError):
                GateThresholds(**kwargs)

    def test_verdict_is_bound_to_its_context(self):
        ref = np.tile(np.array([[9.0, 0.0]]), (64, 1))
        binding = GateBinding(
            source_model_id="src", target_model_id="dst",
            source_artifact_digest="a" * 64, mapper_id="ridge/1",
        )
        result = admit(ref, ref.copy(), binding=binding)
        self.assertEqual(result.binding["source_model_id"], "src")
        self.assertEqual(result.binding["mapper_id"], "ridge/1")
        self.assertEqual(result.binding["policy_version"], "kvx-gate/1")

    def test_admit_never_raises(self):
        ref = np.tile(np.array([[1.0, 2.0]]), (64, 1))
        for bad in (np.zeros((0, 2)), np.zeros((64,)), np.zeros((2, 2, 2)), ref[:, :1]):
            with self.subTest(shape=getattr(bad, "shape", None)):
                self.assertFalse(admit(ref, bad).admitted)


if __name__ == "__main__":
    unittest.main()
