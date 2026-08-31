"""Fitting is checked against constructed ground truth, and a fit is not an admission.

A mapper that reports a low residual has shown it fitted the calibration set. Whether the
result is usable is `gate.py`'s question, on held-out next-token agreement. These tests keep
those apart, and pin the refusals that stop a fitted map being applied where it does not
belong - a map reused across a model pair produces confident, wrong activations, which is
worse than no map at all.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kv_rosetta.mappers.linear import (
    LinearMapper,
    MapperError,
    MapperIdentity,
    corpus_digest,
    fit_ridge,
    residual,
    select_source_layer,
    select_source_layers,
    stack_sources,
)


def identity(**overrides) -> MapperIdentity:
    base = dict(source_model_digest="a" * 64, target_model_digest="b" * 64,
                source_arch="qwen2", target_arch="qwen35", source_width=16,
                target_width=24, source_layers=36, target_layers=65,
                calibration_sha256="c" * 64, rope_state="not_applied")
    base.update(overrides)
    return MapperIdentity(**base)


class FitTest(unittest.TestCase):
    """The oracle is a relation I constructed, not the fitter's own output."""

    def setUp(self):
        self.rng = np.random.default_rng(3)

    def known_problem(self, tokens=2000, src=16, dst=24, noise=0.01):
        x = self.rng.normal(0, 1, (tokens, src))
        w = self.rng.normal(0, 1, (src, dst))
        b = self.rng.normal(0, 1, dst)
        y = x @ w + b + self.rng.normal(0, noise, (tokens, dst))
        return x, y, w, b

    def test_a_known_linear_relation_is_recovered(self):
        x, y, w, b = self.known_problem()
        weights, bias = fit_ridge(x, y, ridge=1e-3)
        self.assertLess(float(np.abs(weights - w).max()), 0.01)
        self.assertLess(float(np.abs(bias - b).max()), 0.01)

    def test_a_rectangular_map_handles_differing_geometry(self):
        """The real pair is 2x128 against 4x213, so the map is not square."""
        x, y, _w, _b = self.known_problem(src=256, dst=852, tokens=3000)
        weights, bias = fit_ridge(x, y, ridge=1e-2)
        self.assertEqual(weights.shape, (256, 852))
        self.assertLess(residual(x, y, weights, bias), 0.05)

    def test_the_ridge_penalty_shrinks_the_weights(self):
        x, y, _w, _b = self.known_problem(noise=1.0)
        small, _ = fit_ridge(x, y, ridge=1e-4)
        large, _ = fit_ridge(x, y, ridge=1e2)
        self.assertLess(float(np.abs(large).max()), float(np.abs(small).max()))

    def test_an_unpenalised_fit_is_refused(self):
        x, y, _w, _b = self.known_problem()
        for bad in (0.0, -1.0):
            with self.subTest(ridge=bad):
                with self.assertRaises(MapperError) as caught:
                    fit_ridge(x, y, ridge=bad)
                self.assertIn("correlated features", str(caught.exception))

    def test_an_underdetermined_fit_is_refused(self):
        x, y, _w, _b = self.known_problem(tokens=10, src=16)
        with self.assertRaises(MapperError) as caught:
            fit_ridge(x, y)
        self.assertIn("memorise the calibration set", str(caught.exception))

    def test_mismatched_token_counts_are_refused(self):
        x, y, _w, _b = self.known_problem()
        with self.assertRaises(MapperError):
            fit_ridge(x, y[:-5])

    def test_the_bias_is_not_penalised(self):
        """A penalised bias would drag predictions toward zero for offset data."""
        x = self.rng.normal(0, 1, (2000, 8))
        y = x @ self.rng.normal(0, 1, (8, 4)) + 50.0
        _weights, bias = fit_ridge(x, y, ridge=1e3)
        self.assertGreater(float(bias.min()), 40.0)


class LayerSelectionTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5)

    def test_the_layer_that_actually_predicts_is_chosen(self):
        x_true = self.rng.normal(0, 1, (1200, 12))
        target = x_true @ self.rng.normal(0, 1, (12, 10))
        candidates = {0: self.rng.normal(0, 1, (1200, 12)),
                      1: x_true,
                      2: self.rng.normal(0, 1, (1200, 12))}
        chosen, score = select_source_layer(target, candidates)
        self.assertEqual(chosen, 1)
        self.assertLess(score, 0.1)

    def test_selection_scores_on_held_out_tokens(self):
        """Scoring on the fit itself would pick whichever layer overfits hardest."""
        import inspect
        source = inspect.getsource(select_source_layer)
        self.assertIn("split:", source)
        self.assertIn("holdout", source)

    def test_no_candidates_is_refused(self):
        with self.assertRaises(MapperError):
            select_source_layer(self.rng.normal(0, 1, (100, 4)), {})

    def test_an_impossible_holdout_is_refused(self):
        candidates = {0: self.rng.normal(0, 1, (500, 4))}
        for bad in (0.0, 1.0, -0.5):
            with self.subTest(holdout=bad):
                with self.assertRaises(MapperError) as caught:
                    select_source_layer(self.rng.normal(0, 1, (500, 4)), candidates,
                                        holdout=bad)
                # Assert the specific refusal: a bad fraction also trips the
                # not-enough-tokens check downstream, so without this the range guard
                # broke no test.
                self.assertIn("must be between 0 and 1", str(caught.exception))


class IdentityBindingTest(unittest.TestCase):
    """A fitted map must not travel to a pair it was not fitted for."""

    def mapper(self, **overrides):
        return LinearMapper(identity=identity(**overrides),
                            weights={"0:k": np.zeros((16, 24), dtype=np.float32)},
                            biases={"0:k": np.zeros(24, dtype=np.float32)})

    def test_the_fitted_pair_is_accepted(self):
        self.mapper().require_applicable("a" * 64, "b" * 64, 16, 24)

    def test_another_source_model_is_refused(self):
        with self.assertRaises(MapperError) as caught:
            self.mapper().require_applicable("z" * 64, "b" * 64, 16, 24)
        self.assertIn("source model", str(caught.exception))

    def test_another_target_model_is_refused(self):
        with self.assertRaises(MapperError):
            self.mapper().require_applicable("a" * 64, "z" * 64, 16, 24)

    def test_differing_geometry_is_refused(self):
        with self.assertRaises(MapperError):
            self.mapper().require_applicable("a" * 64, "b" * 64, 99, 24)
        with self.assertRaises(MapperError):
            self.mapper().require_applicable("a" * 64, "b" * 64, 16, 99)

    def test_a_map_from_a_model_to_itself_is_refused(self):
        problems = identity(target_model_digest="a" * 64).validate()
        self.assertTrue(any("same model" in p for p in problems))

    def test_keys_must_have_rope_stripped(self):
        problems = identity(rope_state="applied").validate()
        self.assertTrue(any("RoPE stripped" in p for p in problems))

    def test_a_short_digest_is_refused(self):
        self.assertTrue(identity(calibration_sha256="short").validate())

    def test_applying_an_unfitted_layer_is_refused(self):
        with self.assertRaises(MapperError) as caught:
            self.mapper().apply_layer(np.zeros((4, 16), dtype=np.float32), 7, "k")
        self.assertIn("no fitted map", str(caught.exception))

    def test_applying_to_the_wrong_width_is_refused(self):
        with self.assertRaises(MapperError):
            self.mapper().apply_layer(np.zeros((4, 99), dtype=np.float32), 0, "k")


class PersistenceTest(unittest.TestCase):
    def test_a_map_round_trips_through_a_file(self):
        rng = np.random.default_rng(9)
        original = LinearMapper(
            identity=identity(), layer_pairs=((0, 3), (1, 5)),
            weights={"0:k": rng.normal(0, 1, (16, 24)).astype(np.float32)},
            biases={"0:k": rng.normal(0, 1, 24).astype(np.float32)},
            residuals={"0:k": 0.12})
        with tempfile.TemporaryDirectory() as directory:
            path = original.save(Path(directory) / "m.kvmap.npz")
            loaded = LinearMapper.load(path)
        self.assertEqual(loaded.identity, original.identity)
        self.assertEqual(loaded.layer_pairs, original.layer_pairs)
        np.testing.assert_array_equal(loaded.weights["0:k"], original.weights["0:k"])
        self.assertEqual(loaded.residuals["0:k"], 0.12)

    def test_a_file_without_metadata_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bare.npz"
            np.savez(path, junk=np.zeros(4))
            with self.assertRaises(MapperError) as caught:
                LinearMapper.load(path)
            self.assertIn("not a kvmap", str(caught.exception))

    def test_a_loaded_map_with_invalid_identity_is_refused(self):
        original = LinearMapper(identity=identity(), weights={}, biases={})
        with tempfile.TemporaryDirectory() as directory:
            path = original.save(Path(directory) / "m.npz")
            import json
            with np.load(path) as data:
                arrays = {k: data[k] for k in data.files}
            meta = json.loads(bytes(arrays["__meta__"]).decode())
            meta["identity"]["rope_state"] = "applied"
            arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
            np.savez(path, **arrays)
            with self.assertRaises(MapperError):
                LinearMapper.load(path)


class CorpusDigestTest(unittest.TestCase):
    def test_the_digest_depends_on_content_and_order(self):
        a = corpus_digest([[1, 2], [3]])
        self.assertNotEqual(a, corpus_digest([[3], [1, 2]]))
        self.assertNotEqual(a, corpus_digest([[1, 2], [4]]))
        self.assertEqual(a, corpus_digest([[1, 2], [3]]))

    def test_prompt_boundaries_change_the_digest(self):
        """Regrouping the same tokens is a different corpus and must digest differently."""
        self.assertNotEqual(corpus_digest([[1, 2], [3]]), corpus_digest([[1], [2, 3]]))
        self.assertNotEqual(corpus_digest([[1, 2, 3]]), corpus_digest([[1, 2], [3]]))


if __name__ == "__main__":
    unittest.main()


class MultiSourceSelectionTests(unittest.TestCase):
    """Concatenating several source layers is the published recipe; one layer is weaker.

    Each refusal is asserted on its message: a mis-stacked block still fits and still
    produces a mapper, so these guards are the only thing between a wrong column order
    and a silently useless map.
    """

    def setUp(self):
        rng = np.random.default_rng(7)
        self.target = rng.normal(size=(600, 8))
        mix = rng.normal(size=(8, 8))
        self.candidates = {
            0: rng.normal(size=(600, 8)),
            1: self.target @ mix + 0.01 * rng.normal(size=(600, 8)),   # the real signal
            2: rng.normal(size=(600, 8)),
        }

    def test_stack_preserves_caller_order_not_sorted_order(self):
        stacked = stack_sources(self.candidates, (2, 0))
        self.assertTrue(np.array_equal(stacked[:, :8], self.candidates[2]))
        self.assertTrue(np.array_equal(stacked[:, 8:], self.candidates[0]))

    def test_stack_refuses_empty_selection(self):
        with self.assertRaises(MapperError) as cm:
            stack_sources(self.candidates, ())
        self.assertIn("no source layers to stack", str(cm.exception))

    def test_stack_refuses_unknown_layer(self):
        with self.assertRaises(MapperError) as cm:
            stack_sources(self.candidates, (0, 99))
        self.assertIn("not among the candidates", str(cm.exception))

    def test_stack_refuses_mismatched_token_counts(self):
        bad = dict(self.candidates)
        bad[3] = np.zeros((17, 8))
        with self.assertRaises(MapperError) as cm:
            stack_sources(bad, (0, 3))
        self.assertIn("disagree on token count", str(cm.exception))

    def test_selection_finds_the_predictive_layer_first(self):
        chosen, _ = select_source_layers(self.target, self.candidates, k=3)
        self.assertEqual(chosen[0], 1)

    def test_selection_stops_early_rather_than_padding_to_k(self):
        chosen, _ = select_source_layers(self.target, self.candidates, k=3)
        self.assertLess(len(chosen), 3)

    def test_multi_source_is_never_worse_than_the_single_best(self):
        one, score_one = select_source_layers(self.target, self.candidates, k=1)
        many, score_many = select_source_layers(self.target, self.candidates, k=3)
        self.assertEqual(one[0], many[0])
        self.assertLessEqual(score_many, score_one + 1e-12)

    def test_agrees_with_the_single_layer_selector_at_k_of_one(self):
        chosen, _ = select_source_layers(self.target, self.candidates, k=1)
        single, _ = select_source_layer(self.target, self.candidates)
        self.assertEqual(chosen, (single,))

    def test_refuses_no_candidates(self):
        with self.assertRaises(MapperError) as cm:
            select_source_layers(self.target, {}, k=1)
        self.assertIn("no candidate source layers", str(cm.exception))

    def test_refuses_k_below_one(self):
        with self.assertRaises(MapperError) as cm:
            select_source_layers(self.target, self.candidates, k=0)
        self.assertIn("k must be at least 1", str(cm.exception))

    def test_refuses_k_above_candidate_count(self):
        with self.assertRaises(MapperError) as cm:
            select_source_layers(self.target, self.candidates, k=9)
        self.assertIn("only 3 exist", str(cm.exception))

    def test_refuses_out_of_range_holdout(self):
        with self.assertRaises(MapperError) as cm:
            select_source_layers(self.target, self.candidates, k=1, holdout=1.5)
        self.assertIn("must be between 0 and 1", str(cm.exception))

    def test_refuses_when_concatenation_outruns_the_token_budget(self):
        """Every candidate must carry signal, or greedy stops before the budget bites."""
        # 20 tokens, holdout 0.25 -> 15 training rows. One layer of 8 features fits;
        # two concatenated layers is 16 features and cannot be determined by 15 rows,
        # so the guard trips on the second round regardless of what greedy prefers.
        rng = np.random.default_rng(3)
        cands = {i: rng.normal(size=(20, 8)) for i in range(3)}
        target = cands[0] @ rng.normal(size=(8, 8)) + 0.01 * rng.normal(size=(20, 8))
        with self.assertRaises(MapperError) as cm:
            select_source_layers(target, cands, k=3)
        self.assertIn("cannot determine", str(cm.exception))

    def test_refuses_calibration_data_that_scores_as_nan(self):
        """A NaN residual compares false against everything, which would otherwise
        leave the selection silently empty rather than reporting bad input."""
        rng = np.random.default_rng(11)
        cands = {0: rng.normal(size=(600, 8)), 1: rng.normal(size=(600, 8))}
        target = rng.normal(size=(600, 8))
        target[3, 2] = np.nan
        with self.assertRaises(MapperError) as cm:
            select_source_layers(target, cands, k=2)
        self.assertIn("no source layer improved", str(cm.exception))


class RidgeAndSingleLayerGuardTests(unittest.TestCase):
    def test_fit_ridge_refuses_non_matrix_input(self):
        with self.assertRaises(MapperError) as cm:
            fit_ridge(np.zeros(10), np.zeros((10, 2)))
        self.assertIn("expected 2-D", str(cm.exception))

    def test_single_layer_selector_refuses_too_few_tokens_to_hold_out(self):
        rng = np.random.default_rng(5)
        target = rng.normal(size=(9, 8))
        with self.assertRaises(MapperError) as cm:
            select_source_layer(target, {0: rng.normal(size=(9, 8))})
        self.assertIn("not enough tokens to hold any out", str(cm.exception))
