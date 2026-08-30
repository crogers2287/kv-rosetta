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


class ScoringProtocolTest(unittest.TestCase):
    """A verdict is only evidence under the protocol that produced it.

    Free generation and teacher forcing measure different things. Under free generation the
    candidate is conditioned on its own tokens after the first divergence, and because
    generation is autoregressive one wrong token condemns the rest - measured here, a
    nearly-perfect cache and a hopeless one both scored 0.042 top-1. Under teacher forcing
    every position is scored against the same forced prefix and the number tracks cache
    quality instead of tracking when the first slip happened.

    Conflating them cost this project eight iterations, which is why the protocol is recorded
    in the binding rather than assumed.
    """

    def test_the_protocol_is_recorded_in_the_binding(self):
        self.assertIn("scoring_protocol", GateBinding().as_dict())

    def test_teacher_forcing_is_the_default(self):
        """The stricter and more informative of the two, so an unstated protocol is not the
        one that flatters a candidate."""
        self.assertEqual(GateBinding().scoring_protocol, "teacher_forced")

    def test_both_protocols_are_accepted(self):
        for protocol in ("free", "teacher_forced"):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    GateBinding(scoring_protocol=protocol).as_dict()["scoring_protocol"],
                    protocol)

    def test_an_unrecorded_protocol_is_refused(self):
        """Blank is not a protocol. A verdict whose provenance is unstated cannot be told
        apart from one taken under the other."""
        for bad in ("", "eyeballed", "free-ish", "TEACHER_FORCED"):
            with self.subTest(protocol=bad):
                with self.assertRaises(GateError):
                    GateBinding(scoring_protocol=bad)

    def test_two_verdicts_under_different_protocols_do_not_compare_equal(self):
        """The point of recording it: a stored verdict must not be reused across protocols."""
        self.assertNotEqual(GateBinding(scoring_protocol="free").as_dict(),
                            GateBinding(scoring_protocol="teacher_forced").as_dict())

    def test_admit_carries_the_protocol_into_its_result(self):
        import numpy as np
        reference = np.zeros((64, 8))
        found = admit(reference, reference.copy(),
                      binding=GateBinding(scoring_protocol="free"))
        self.assertEqual(found.binding["scoring_protocol"], "free")


class ThresholdTypeTest(unittest.TestCase):
    """Thresholds must be real numbers, and a bool is not one.

    Python makes True == 1, so a threshold of True would pass every numeric check and read as
    a top1_agreement of 1.0 - a policy that looks maximally strict and was never written by
    anyone. A string threshold would raise deep inside a comparison instead of at
    construction, far from the mistake.
    """

    def test_a_bool_threshold_is_refused(self):
        for field in ("top1_agreement", "mean_kl", "max_kl", "max_logit_delta"):
            with self.subTest(field=field):
                with self.assertRaises(GateError):
                    GateThresholds(**{field: True})

    def test_a_non_numeric_threshold_is_refused_at_construction(self):
        for value in ("0.99", None, [0.99], {"v": 1}):
            with self.subTest(value=value):
                with self.assertRaises(GateError):
                    GateThresholds(top1_agreement=value)

    def test_a_bool_holdout_is_refused(self):
        """True would become a one-position holdout, which is the opposite of a policy."""
        with self.assertRaises(GateError):
            GateThresholds(holdout_tokens=True)

    def test_a_non_integer_holdout_is_refused(self):
        for value in (64.0, "64", None):
            with self.subTest(value=value):
                with self.assertRaises(GateError):
                    GateThresholds(holdout_tokens=value)
