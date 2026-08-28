import unittest

import numpy as np

from kv_rosetta.gate import EXACT, GateThresholds, admit


class GateAdmissionTests(unittest.TestCase):
    def test_identical_logits_admitted(self):
        rng = np.random.default_rng(1234)
        ref = rng.standard_normal((16, 32))
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
        self.assertEqual(result.reason, "no holdout tokens")

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
        ref = rng.standard_normal((8, 16))
        identical = admit(ref, ref.copy(), EXACT)
        self.assertTrue(identical.admitted)
        self.assertEqual(identical.reason, "admitted")
        perturbed = admit(ref, ref + 1e-3, EXACT)
        self.assertFalse(perturbed.admitted)

    def test_none_tensors_yield_nan_cosine_without_affecting_admission(self):
        ref = np.zeros((8, 16))
        cand = np.zeros((8, 16))
        result = admit(ref, cand)
        self.assertTrue(np.isnan(result.metrics["tensor_cosine"]))
        self.assertTrue(result.admitted)


if __name__ == "__main__":
    unittest.main()
