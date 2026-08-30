"""What a runtime must provide before an artifact may be restored into it.

The failure this prevents is silent. Restoring a hybrid model's state into a build without the
context-checkpoint patch returns a large n_restored and then reuses nothing - measured on this
host as 252 of 256 tokens on a patched build against 0 of 256 on the fleet's own binary, both
reporting n_restored 259. Discovered downstream it looks like a performance problem rather
than a compatibility one.
"""

import unittest

from kv_rosetta.requirements import (
    RequirementError,
    Requirements,
    check,
    for_artifact,
    require,
)

PATCHED = {"sequence_state_version": 3, "slot_checkpoint_persistence": True,
           "slot_checkpoint_format": "sckp/1"}
STOCK = {"sequence_state_version": 3}


class DerivationTest(unittest.TestCase):
    def test_checkpoints_make_an_artifact_non_portable(self):
        """shimquant says it of its own format: files built with the patch need the patch."""
        found = for_artifact(hybrid=True, checkpoints=2, sequence_state_version=3)
        self.assertTrue(found.needs_checkpoint_persistence)
        self.assertFalse(found.portable)

    def test_a_plain_artifact_needs_no_patch(self):
        found = for_artifact(hybrid=False, checkpoints=0, sequence_state_version=3)
        self.assertTrue(found.portable)
        self.assertEqual(found.notes, ())

    def test_a_hybrid_without_checkpoints_is_legal_and_useless(self):
        """Different from illegal, and recorded as a note rather than a requirement: the
        restore is accepted, and reuses nothing, because the model resumes from a checkpoint
        the artifact does not carry."""
        found = for_artifact(hybrid=True, checkpoints=0, sequence_state_version=3)
        self.assertTrue(found.portable)
        self.assertIn("reuse nothing", found.notes[0])


class CheckTest(unittest.TestCase):
    def hybrid(self):
        return for_artifact(hybrid=True, checkpoints=1, sequence_state_version=3)

    def test_a_patched_runtime_accepts_a_checkpointed_artifact(self):
        self.assertEqual(check(self.hybrid(), PATCHED), [])

    def test_a_stock_runtime_is_refused_with_the_measured_reason(self):
        """One problem, not two: a build without the patch also reports no checkpoint
        format, and saying so adds nothing - it is a consequence of the missing capability
        rather than a second thing to fix."""
        problems = check(self.hybrid(), STOCK)
        self.assertEqual(len(problems), 1)
        self.assertIn("252 of 256", problems[0])

    def test_a_runtime_that_reports_no_version_is_refused_not_assumed(self):
        """A version mismatch is rejected for the format and looks exactly like a backend
        failure; that confusion already cost one cross-backend result."""
        problems = check(self.hybrid(), {"slot_checkpoint_persistence": True,
                                         "slot_checkpoint_format": "sckp/1"})
        self.assertTrue(any("cannot be established" in p for p in problems))

    def test_a_version_mismatch_is_refused(self):
        problems = check(self.hybrid(), {**PATCHED, "sequence_state_version": 2})
        self.assertTrue(any("runtime reads 2" in p for p in problems))

    def test_a_different_checkpoint_format_is_refused(self):
        problems = check(self.hybrid(), {**PATCHED, "slot_checkpoint_format": "sckp/2"})
        self.assertTrue(any("sckp/2" in p for p in problems))

    def test_a_different_kv_quantisation_is_refused(self):
        """The fleet runs q4_0 KV; an f16 artifact's rows are a different size entirely."""
        wanted = for_artifact(hybrid=False, checkpoints=0, sequence_state_version=3,
                              kv_type_k="f16", kv_type_v="f16")
        problems = check(wanted, {**STOCK, "kv_type_k": "q4_0", "kv_type_v": "q4_0"})
        self.assertEqual(len(problems), 2)
        self.assertIn("cannot be read at all", problems[0])

    def test_a_different_model_is_refused(self):
        wanted = for_artifact(hybrid=False, checkpoints=0, sequence_state_version=3,
                              model_identity="a" * 64)
        problems = check(wanted, {**STOCK, "model_identity": "b" * 64})
        self.assertTrue(any("written from" in p for p in problems))

    def test_a_runtime_with_no_identity_cannot_be_shown_to_match(self):
        wanted = for_artifact(hybrid=False, checkpoints=0, sequence_state_version=3,
                              model_identity="a" * 64)
        problems = check(wanted, STOCK)
        self.assertTrue(any("reports no model identity" in p for p in problems))

    def test_every_reason_is_reported_not_just_the_first(self):
        """An operator fixing one problem should not have to run again to find the next."""
        wanted = for_artifact(hybrid=True, checkpoints=1, sequence_state_version=3,
                              model_identity="a" * 64)
        self.assertGreaterEqual(len(check(wanted, {"sequence_state_version": 2})), 3)


class RequireTest(unittest.TestCase):
    def test_it_raises_rather_than_returning(self):
        with self.assertRaises(RequirementError) as caught:
            require(for_artifact(hybrid=True, checkpoints=1, sequence_state_version=3),
                    STOCK)
        self.assertIn("slot_checkpoint_persistence", str(caught.exception))

    def test_it_is_silent_when_everything_matches(self):
        require(for_artifact(hybrid=True, checkpoints=1, sequence_state_version=3), PATCHED)

    def test_an_override_identity_is_honoured(self):
        wanted = for_artifact(hybrid=False, checkpoints=0, sequence_state_version=3)
        with self.assertRaises(RequirementError):
            require(wanted, {**STOCK, "model_identity": "b" * 64}, model_identity="a" * 64)


if __name__ == "__main__":
    unittest.main()
