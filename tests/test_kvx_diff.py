"""The cache-difference analyser's refusals and its decomposition.

This runner exists to tell a difference a converter could undo from one that destroyed
information. Direction, scale and combined error are reported separately for that reason,
and these tests pin that they actually separate.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from kvx_diff import DiffError, build_parser, compare_tensor, parse_models, require_same_input


def _decoded(token_ids=(1, 2, 3), cells=3):
    return {"token_ids": tuple(token_ids), "cell_count": cells}


class InputEquality(unittest.TestCase):
    def test_identical_input_is_accepted(self):
        require_same_input(_decoded(), _decoded(), "a", "b")

    def test_different_token_ids_are_refused(self):
        # Two models given different tokens differ for a reason that has nothing to do
        # with their weights, and every per-layer number would be measuring that instead.
        with self.assertRaises(DiffError) as caught:
            require_same_input(_decoded((1, 2, 3)), _decoded((1, 2, 9)), "a", "b")
        self.assertIn("different token ids", str(caught.exception))

    def test_the_refusal_reports_the_shared_prefix(self):
        with self.assertRaises(DiffError) as caught:
            require_same_input(_decoded((1, 2, 3)), _decoded((1, 2, 9)), "a", "b")
        self.assertIn("2 shared prefix", str(caught.exception))

    def test_differing_cell_counts_are_refused(self):
        with self.assertRaises(DiffError):
            require_same_input(_decoded((1, 2, 3), 3), _decoded((1, 2, 3), 4), "a", "b")


class ModelArguments(unittest.TestCase):
    def test_name_and_path_are_split(self):
        self.assertEqual(parse_models(["base=/a.gguf", "tuned=/b.gguf"]),
                         [("base", "/a.gguf"), ("tuned", "/b.gguf")])

    def test_a_bare_path_is_refused(self):
        with self.assertRaises(DiffError) as caught:
            parse_models(["/a.gguf", "tuned=/b.gguf"])
        self.assertIn("NAME=PATH", str(caught.exception))

    def test_duplicate_names_are_refused(self):
        # Each name is a state filename; a duplicate silently overwrites the first
        # capture and the comparison would then be a model against itself.
        with self.assertRaises(DiffError) as caught:
            parse_models(["a=/x.gguf", "a=/y.gguf"])
        self.assertIn("duplicate", str(caught.exception))

    def test_one_model_cannot_be_compared(self):
        with self.assertRaises(DiffError):
            parse_models(["a=/x.gguf"])

    def test_an_empty_side_is_refused(self):
        with self.assertRaises(DiffError):
            parse_models(["=/x.gguf", "b=/y.gguf"])


class Decomposition(unittest.TestCase):
    def _t(self, seed=0, shape=(8, 2, 4)):
        return np.random.default_rng(seed).normal(size=shape).astype(np.float32)

    def test_identical_tensors_report_no_difference(self):
        x = self._t()
        result = compare_tensor(x, x)
        self.assertAlmostEqual(result["cosine"], 1.0, places=6)
        self.assertAlmostEqual(result["norm_ratio"], 1.0, places=6)
        self.assertAlmostEqual(result["relative_error"], 0.0, places=6)

    def test_a_pure_scale_difference_keeps_cosine_at_one(self):
        # This is the case a converter could undo with one number per layer, and it must
        # be visible as scale rather than hidden inside a combined error figure.
        x = self._t()
        result = compare_tensor(x, x * 2.5)
        self.assertAlmostEqual(result["cosine"], 1.0, places=6)
        self.assertAlmostEqual(result["norm_ratio"], 2.5, places=5)
        self.assertGreater(result["relative_error"], 1.0)

    def test_a_direction_difference_drops_cosine(self):
        # Unrelated tensors of equal scale: the norm ratio alone would call this a match.
        x, y = self._t(0), self._t(1)
        y = y * (np.linalg.norm(x) / np.linalg.norm(y))
        result = compare_tensor(x, y)
        self.assertLess(abs(result["cosine"]), 0.5)
        self.assertAlmostEqual(result["norm_ratio"], 1.0, places=5)

    def test_per_head_figures_are_reported_separately(self):
        # A difference concentrated in one head is a different problem from one spread
        # across all of them, and the pooled number cannot distinguish those.
        x = self._t()
        y = x.copy()
        y[:, 0, :] *= 4.0
        result = compare_tensor(x, y)
        self.assertEqual(len(result["per_head"]), 2)
        self.assertAlmostEqual(result["per_head"][0]["norm_ratio"], 4.0, places=5)
        self.assertAlmostEqual(result["per_head"][1]["norm_ratio"], 1.0, places=5)

    def test_a_zero_tensor_reports_nan_rather_than_a_perfect_score(self):
        # 0/0 must never come back as agreement.
        zeros = np.zeros((4, 2, 4), dtype=np.float32)
        result = compare_tensor(zeros, zeros)
        self.assertTrue(np.isnan(result["cosine"]))


if __name__ == "__main__":
    unittest.main()


class ExtraArgsTests(unittest.TestCase):
    """Both captures must receive identical launch flags or the diff is meaningless."""

    def _base(self):
        return ["--binary", "b", "--model", "a=x", "--n-head-kv", "2",
                "--head-dim", "256", "--slots", "s", "--out", "o.json"]

    def test_extra_defaults_to_empty(self):
        args = build_parser().parse_args(self._base())
        self.assertEqual(args.extra, [])

    def test_extra_accumulates_in_order(self):
        args = build_parser().parse_args(
            self._base() + ["--extra=--device", "--extra", "CUDA0"])
        self.assertEqual(args.extra, ["--device", "CUDA0"])

    def test_flag_valued_extra_needs_equals_form(self):
        """argparse reads a bare leading dash as the next option, so --extra=-x is required."""
        with self.assertRaises(SystemExit):
            build_parser().parse_args(self._base() + ["--extra", "--device"])
