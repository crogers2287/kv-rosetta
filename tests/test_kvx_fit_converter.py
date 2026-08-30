"""The converter experiment's scoring, and the two traps §20 paid for.

A fitted map is only interesting relative to doing nothing, and only honest when scored on
prompts the fit never saw. These pin both.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from kvx_fit_converter import evaluate_layer, plan_split, r_squared
from kvx_diff import DiffError


class SplitPlanning(unittest.TestCase):
    def test_a_valid_split_returns_the_fit_count(self):
        self.assertEqual(plan_split([("a", "/x"), ("b", "/y")], 2, 6), 4)

    def test_more_than_two_models_is_refused(self):
        with self.assertRaises(DiffError) as caught:
            plan_split([("a", "/x"), ("b", "/y"), ("c", "/z")], 2, 6)
        self.assertIn("source then target", str(caught.exception))

    def test_holding_nothing_out_is_refused(self):
        # Scoring a fit on its own fitting data measures memorisation, which is the
        # mistake §20 caught only after the fact.
        with self.assertRaises(DiffError) as caught:
            plan_split([("a", "/x"), ("b", "/y")], 0, 6)
        self.assertIn("memorisation", str(caught.exception))

    def test_holding_everything_out_is_refused(self):
        with self.assertRaises(DiffError):
            plan_split([("a", "/x"), ("b", "/y")], 6, 6)


class RSquared(unittest.TestCase):
    def test_a_perfect_prediction_scores_one(self):
        y = np.random.default_rng(0).normal(size=(20, 4))
        self.assertAlmostEqual(r_squared(y, y), 1.0, places=9)

    def test_predicting_the_mean_scores_zero(self):
        # The floor must be "no better than guessing the average vector", not 0 error.
        y = np.random.default_rng(1).normal(size=(20, 4))
        self.assertAlmostEqual(r_squared(y, np.tile(y.mean(axis=0), (20, 1))), 0.0, places=9)

    def test_a_worse_than_mean_prediction_goes_negative(self):
        y = np.random.default_rng(2).normal(size=(20, 4))
        self.assertLess(r_squared(y, -y), 0.0)

    def test_a_constant_target_reports_nan_rather_than_a_perfect_score(self):
        # Zero spread means R^2 is undefined; returning 1.0 would read as success.
        y = np.ones((10, 4))
        self.assertTrue(np.isnan(r_squared(y, y)))


def _decoded(tensors):
    return {"tensors": tensors, "token_ids": (1, 2), "cell_count": tensors[(0, "k")].shape[0]}


class LayerEvaluation(unittest.TestCase):
    def _pair(self, transform, seed=0, tokens=64, dim=8):
        rng = np.random.default_rng(seed)
        out_source, out_target = [], []
        for prompt in range(4):
            x = rng.normal(size=(tokens, 1, dim))
            y = transform(x)
            out_source.append(_decoded({(0, "k"): x.astype(np.float32)}))
            out_target.append(_decoded({(0, "k"): y.astype(np.float32)}))
        return out_source, out_target

    def test_a_genuinely_linear_difference_is_recovered_above_baseline(self):
        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(8, 8))
        src, tgt = self._pair(lambda x: x @ matrix)
        result = evaluate_layer(src[:3], tgt[:3], src[3:], tgt[3:], 0, "k", 1, 1e-6)
        self.assertGreater(result["fitted_r2_mean"], 0.99)
        self.assertGreater(result["gain"], 0.5)

    def test_an_identical_pair_shows_no_gain_over_doing_nothing(self):
        # The case that matters most: if the source already IS the target, a fit that
        # scores 0.99 has bought nothing, and only the baseline reveals that.
        src, tgt = self._pair(lambda x: x)
        result = evaluate_layer(src[:3], tgt[:3], src[3:], tgt[3:], 0, "k", 1, 1e-6)
        self.assertGreater(result["baseline_r2_mean"], 0.99)
        self.assertLess(abs(result["gain"]), 0.05)

    def test_unrelated_tensors_cannot_be_fitted(self):
        rng = np.random.default_rng(11)
        src, tgt = self._pair(lambda x: rng.normal(size=x.shape))
        result = evaluate_layer(src[:3], tgt[:3], src[3:], tgt[3:], 0, "k", 1, 1e-2)
        self.assertLess(result["fitted_r2_mean"], 0.5)

    def test_per_head_scores_are_retained(self):
        src, tgt = self._pair(lambda x: x, dim=8)
        result = evaluate_layer(src[:3], tgt[:3], src[3:], tgt[3:], 0, "k", 1, 1e-6)
        self.assertEqual(len(result["per_head_fitted"]), 1)
        self.assertEqual(len(result["per_head_baseline"]), 1)


if __name__ == "__main__":
    unittest.main()
