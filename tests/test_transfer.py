"""Planning refuses what it cannot verify, and execution never raises into the caller.

A transfer is an optimisation over prefilling. If any part fails, the answer is "prefill
natively", not an exception - a caller should be able to ask on every request without a try
block. And the bounds a lossy conversion is judged by are fixed when the plan is made,
because choosing a threshold after seeing the numbers is how a bad conversion gets admitted.
"""

import dataclasses
import unittest

import numpy as np

from kv_rosetta.gate import GateThresholds
from kv_rosetta.transfer import (
    SUPPORTED_DTYPE_PAIRS,
    Identity,
    Mode,
    TransferPlan,
    execute,
    plan,
)

MODEL = "m" * 64


def ident(**overrides) -> Identity:
    base = dict(model_digest=MODEL, cache_abi_digest="a" * 64, cache_dtype_k="f16",
                cache_dtype_v="f16", architecture="qwen35", prompt_digest="p" * 64)
    base.update(overrides)
    return Identity(**base)


def logits(rows=64, cols=8, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (rows, cols)) + shift


class PlanningTest(unittest.TestCase):
    def test_the_same_abi_is_an_identity_transfer(self):
        found = plan(ident(), ident())
        self.assertIs(found.mode, Mode.IDENTITY)
        self.assertEqual(found.steps, ("restore",))

    def test_a_different_cache_dtype_is_a_conversion(self):
        found = plan(ident(), ident(cache_abi_digest="b" * 64, cache_dtype_k="q8_0",
                                    cache_dtype_v="q8_0"))
        self.assertIs(found.mode, Mode.DTYPE_CONVERT)
        self.assertIn("gate", found.steps)

    def test_a_different_model_is_refused_not_attempted(self):
        found = plan(ident(), ident(model_digest="z" * 64))
        self.assertIs(found.mode, Mode.REFUSED)
        self.assertIn("not implemented", found.reason)

    def test_an_untested_dtype_pair_is_refused(self):
        found = plan(ident(cache_dtype_k="q4_0", cache_dtype_v="q4_0"),
                     ident(cache_abi_digest="b" * 64))
        self.assertIs(found.mode, Mode.REFUSED)
        self.assertIn("not among the tested pairs", found.reason)
        self.assertNotIn(("q4_0", "f16"), SUPPORTED_DTYPE_PAIRS)

    def test_an_unadvertised_dtype_is_refused(self):
        for field in ("cache_dtype_k", "cache_dtype_v"):
            with self.subTest(field=field):
                found = plan(ident(), ident(cache_abi_digest="b" * 64, **{field: ""}))
                self.assertIs(found.mode, Mode.REFUSED)
                self.assertIn("unadvertised", found.reason)

    def test_an_unknown_model_identity_is_refused(self):
        """Both sides empty, so the cross-model check cannot shadow this one.

        With one side empty and the other set, the digests differ and the cross-model
        refusal fires first - the unknown-identity guard then broke no test, which a
        mutation run caught. Two empty digests compare equal, so only this guard can refuse.
        """
        found = plan(ident(model_digest=""), ident(model_digest=""))
        self.assertIs(found.mode, Mode.REFUSED)
        self.assertIn("model identity is unknown", found.reason)

    def test_one_side_missing_an_identity_is_also_refused(self):
        self.assertIs(plan(ident(model_digest=""), ident()).mode, Mode.REFUSED)
        self.assertIs(plan(ident(), ident(model_digest="")).mode, Mode.REFUSED)

    def test_a_different_prompt_is_refused(self):
        found = plan(ident(), ident(prompt_digest="q" * 64))
        self.assertIs(found.mode, Mode.REFUSED)
        self.assertIn("another prompt", found.reason)

    def test_a_different_architecture_is_refused(self):
        found = plan(ident(), ident(cache_abi_digest="b" * 64, cache_dtype_k="q8_0",
                                    cache_dtype_v="q8_0", architecture="llama"))
        self.assertIs(found.mode, Mode.REFUSED)
        self.assertIn("layer geometry", found.reason)

    def test_planning_never_raises_on_junk(self):
        for bad in (Identity(), ident(model_digest=""), ident(architecture="")):
            with self.subTest(source=bad):
                self.assertIsInstance(plan(bad, ident()), TransferPlan)


class ThresholdsAreFixedTest(unittest.TestCase):
    """The bounds cannot move after the numbers are seen."""

    def test_a_plan_is_frozen(self):
        found = plan(ident(), ident())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            found.thresholds = GateThresholds(top1_agreement=0.0)

    def test_thresholds_are_frozen(self):
        found = plan(ident(), ident())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            found.thresholds.top1_agreement = 0.0

    def test_execute_accepts_no_threshold_argument(self):
        """A separate thresholds argument would let a caller re-judge after measuring."""
        import inspect
        self.assertNotIn("thresholds", inspect.signature(execute).parameters)


class ExecutionTest(unittest.TestCase):
    def test_a_refused_plan_reports_the_fallback_without_running_anything(self):
        ran = []
        report = execute(plan(ident(), ident(model_digest="z" * 64)),
                         run_transfer=lambda p: ran.append(p))
        self.assertFalse(report.ok)
        self.assertEqual(report.fallback, "native_prefill")
        self.assertEqual(ran, [], "a refused plan still moved state")

    def test_an_identity_transfer_needs_no_gate(self):
        report = execute(plan(ident(), ident()), run_transfer=lambda p: ("moved",))
        self.assertTrue(report.ok)
        self.assertIsNone(report.gate)

    def test_a_lossy_transfer_without_measurement_is_refused(self):
        report = execute(plan(ident(), ident(cache_abi_digest="b" * 64,
                                             cache_dtype_k="q8_0", cache_dtype_v="q8_0")),
                         run_transfer=lambda p: ())
        self.assertFalse(report.ok)
        self.assertIn("unverified claim", report.reason)

    def test_a_lossy_transfer_that_matches_is_admitted(self):
        reference = logits(seed=1)
        report = execute(
            plan(ident(), ident(cache_abi_digest="b" * 64, cache_dtype_k="q8_0",
                                cache_dtype_v="q8_0")),
            run_transfer=lambda p: (), reference_logits=reference,
            candidate_logits=reference.copy())
        self.assertTrue(report.ok, report.reason)
        self.assertTrue(report.gate.admitted)

    def test_a_lossy_transfer_that_diverges_is_rejected(self):
        report = execute(
            plan(ident(), ident(cache_abi_digest="b" * 64, cache_dtype_k="q8_0",
                                cache_dtype_v="q8_0")),
            run_transfer=lambda p: (), reference_logits=logits(seed=1),
            candidate_logits=logits(seed=2))
        self.assertFalse(report.ok)
        self.assertIn("gate rejected", report.reason)
        self.assertEqual(report.fallback, "native_prefill")

    def test_an_exception_in_the_transfer_becomes_a_fallback(self):
        def boom(_plan):
            raise RuntimeError("the runtime went away")

        report = execute(plan(ident(), ident()), run_transfer=boom)
        self.assertFalse(report.ok)
        self.assertIn("the runtime went away", report.reason)
        self.assertEqual(report.fallback, "native_prefill")

    def test_no_execution_path_raises(self):
        """The property a caller depends on: ask on every request, never guard it."""
        def boom(_plan):
            raise MemoryError("allocation failed")

        cases = [
            (plan(ident(), ident(model_digest="z" * 64)), None, None),
            (plan(ident(), ident()), None, None),
            (plan(ident(), ident(cache_abi_digest="b" * 64, cache_dtype_k="q8_0",
                                 cache_dtype_v="q8_0")), logits(), logits(rows=8)),
        ]
        for planned, ref, cand in cases:
            with self.subTest(mode=planned.mode):
                report = execute(planned, run_transfer=lambda p: (),
                                 reference_logits=ref, candidate_logits=cand)
                self.assertIsNotNone(report)
        self.assertFalse(execute(plan(ident(), ident()), run_transfer=boom).ok)

    def test_a_shape_mismatch_is_a_rejection_not_a_crash(self):
        report = execute(
            plan(ident(), ident(cache_abi_digest="b" * 64, cache_dtype_k="q8_0",
                                cache_dtype_v="q8_0")),
            run_transfer=lambda p: (), reference_logits=logits(rows=64),
            candidate_logits=logits(rows=8))
        self.assertFalse(report.ok)
        self.assertIn("gate rejected", report.reason)

    def test_the_report_serialises_without_the_gate_object(self):
        report = execute(plan(ident(), ident()), run_transfer=lambda p: ())
        import json
        json.dumps(report.as_dict())


if __name__ == "__main__":
    unittest.main()
