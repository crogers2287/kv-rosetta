"""Draft and speculative checkpoint state is refused until something demonstrates otherwise.

The format serializes all three checkpoint blobs - a live server running MTP reports
`sckp_serializes_speculative_state` true beside `supports_speculative_checkpoint_state`
false - so the bytes being present says nothing about the restore working. The refusal that
follows is right, and an absolute refusal makes the capability unreachable even after someone
does the work.

These tests pin both halves. The default, with no evidence anywhere, refuses in exactly the
words it always did. A proof lifts the refusal only for the state class, build, binary and
model it was gathered on, only while it has not expired, and only when the run record it
cites can still be produced and hashes to what the proof was written against. Everything
else - a proof for another build, a record that drifted, a record dated tomorrow, a proof
smuggled inside the artifact - lands back on the refusal.

Each test asserts on the message rather than the exception class. The recurring failure in
this repo is a test that passes whether or not its guard exists, because a different path
raises the same class; a message is what tells the guard from its absence.
"""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from kv_rosetta.hybrid_schema import (
    DRAFT_STATE,
    EVIDENCE_KEYS,
    MAX_PROOF_VALIDITY,
    PROOF_DIVERGED,
    PROOF_RESTORED,
    SPECULATIVE_STATE,
    CheckpointRecord,
    HybridState,
    RecurrentLayerMap,
    RestorationEvidence,
    RestorationProof,
    SchemaError,
    SourceGeometry,
    UNPROVEN_STATE_REFUSAL,
)
from kv_rosetta.requirements import (
    RequirementError,
    check,
    for_artifact,
    require,
)

MODEL = "a" * 64
BINARY = "c" * 64
RECORD = "d" * 64
BUILD = "b1-3e73446"
NOW = "2026-08-30T00:00:00Z"
RECORD_URI = "docs/records/proofs/spec-b1-3e73446.json"

#: A runtime that claims both unproven classes and reports the build it is.
CLAIMS = {"sequence_state_version": 3, "slot_checkpoint_persistence": True,
          "slot_checkpoint_format": "sckp/1", "build_info": BUILD,
          "supports_draft_checkpoint_state": True,
          "supports_speculative_checkpoint_state": True,
          "sckp_serializes_draft_state": True,
          "sckp_serializes_speculative_state": True}


def proof(**overrides) -> RestorationProof:
    base = dict(
        state_class=SPECULATIVE_STATE, runtime_build=BUILD, binary_sha256=BINARY,
        model_identity=MODEL, outcome=PROOF_RESTORED, trials=3,
        method=("restored a speculative checkpoint into a fresh slot and compared 512 "
                "greedy tokens against a cold run; identical on every trial"),
        record_uri=RECORD_URI, record_sha256=RECORD,
        proven_at="2026-08-01T00:00:00Z", expires_at="2026-10-01T00:00:00Z")
    base.update(overrides)
    return RestorationProof(**base)


def evidence(**overrides) -> RestorationEvidence:
    base = dict(runtime_build=BUILD, binary_sha256=BINARY, model_identity=MODEL,
                proofs=(proof(),), record_digests={RECORD_URI: RECORD}, as_of=NOW)
    base.update(overrides)
    return RestorationEvidence(**base)


def checkpoint(**overrides) -> CheckpointRecord:
    base = dict(n_tokens=252, pos_min=0, pos_max=251, recurrent_segments=("ckpt0.r",),
                has_speculative_state=True)
    base.update(overrides)
    return CheckpointRecord(**base)


def state(**overrides) -> HybridState:
    base = dict(
        geometry=SourceGeometry(
            architecture="qwen35", n_layer=4, n_head_kv=(4, 4, 4, 4), n_embd_head_k=128,
            n_embd_head_v=128, n_embd_r=512, n_embd_s=128, has_cell_ext=False,
            rope_state="applied", rope_theta=1000000.0, model_weights_sha256=MODEL,
            gguf_content_digest="b" * 64),
        layer_map=RecurrentLayerMap((0, 2, 3)),
        checkpoints=(checkpoint(),),
        attention_segments=("k", "v"), recurrent_segments=("r", "s"))
    base.update(overrides)
    return HybridState(**base)


def only(problems: list[str]) -> str:
    """The single refusal, asserting there was exactly one. Two would hide which fired."""
    if len(problems) != 1:                                    # pragma: no cover - a failure
        raise AssertionError(f"expected one problem, got {problems}")
    return problems[0]


class TheDefaultIsUnchanged(unittest.TestCase):
    """No evidence anywhere is the state every caller is in, and it must not have moved."""

    def test_speculative_state_is_refused_in_the_original_words(self):
        problem = only(state().validate())
        self.assertEqual(problem, f"checkpoint 0: {UNPROVEN_STATE_REFUSAL}")

    def test_draft_state_is_refused_in_the_original_words(self):
        record = checkpoint(has_speculative_state=False, has_draft_state=True)
        problem = only(state(checkpoints=(record,)).validate())
        self.assertEqual(problem, f"checkpoint 0: {UNPROVEN_STATE_REFUSAL}")

    def test_both_classes_are_refused_separately(self):
        record = checkpoint(has_draft_state=True, has_speculative_state=True)
        self.assertEqual(len(state(checkpoints=(record,)).validate()), 2)

    def test_require_valid_still_raises_with_that_message(self):
        with self.assertRaises(SchemaError) as caught:
            state().require_valid()
        self.assertIn("nothing has verified", str(caught.exception))

    def test_a_checkpoint_without_unproven_state_is_untouched(self):
        self.assertEqual(state(checkpoints=(checkpoint(
            has_speculative_state=False),)).validate(), [])


class AProofAdmitsExactlyWhatItProves(unittest.TestCase):
    def test_a_scoped_proof_admits_that_class(self):
        self.assertEqual(state().validate(evidence()), [])
        self.assertIs(state().require_valid(evidence()).geometry.architecture, "qwen35")

    def test_a_draft_proof_does_not_admit_speculative_state(self):
        held = evidence(proofs=(proof(state_class=DRAFT_STATE),))
        problem = only(state().validate(held))
        self.assertIn("it proves draft state, not speculative", problem)
        self.assertIn("says nothing about the other", problem)

    def test_a_speculative_proof_does_not_admit_draft_state(self):
        record = checkpoint(has_draft_state=True, has_speculative_state=True)
        problems = state(checkpoints=(record,)).validate(evidence())
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("it proves speculative state, not draft", problems[0])

    def test_a_proof_of_one_class_each_admits_both(self):
        record = checkpoint(has_draft_state=True, has_speculative_state=True)
        held = evidence(proofs=(proof(), proof(state_class=DRAFT_STATE)))
        self.assertEqual(state(checkpoints=(record,)).validate(held), [])

    def test_an_unknown_state_class_proves_nothing(self):
        held = evidence(proofs=(proof(state_class="target"),))
        self.assertIn("it proves target state, not speculative", only(state().validate(held)))


class AProofIsScopedToOneRuntimeAndModel(unittest.TestCase):
    def test_a_proof_for_another_build_does_not_apply(self):
        held = evidence(proofs=(proof(runtime_build="b1-deadbee"),))
        self.assertIn("it was gathered on build 'b1-deadbee', this runtime reports",
                      only(state().validate(held)))

    def test_the_same_build_string_on_a_different_binary_does_not_apply(self):
        """REQ-077's lesson: the fleet's fork and a patched tree report a build string that
        does not say whether the checkpoint patch is in the binary."""
        held = evidence(proofs=(proof(binary_sha256="e" * 64),))
        problem = only(state().validate(held))
        self.assertIn("same build string but a different binary", problem)
        self.assertIn("did not distinguish a binary carrying the checkpoint patch", problem)

    def test_a_proof_gathered_on_another_model_does_not_apply(self):
        """Evidence naming the right model, carrying a proof about a different one. The
        outer binding matches, so only the per-proof check refuses this."""
        held = evidence(proofs=(proof(model_identity="f" * 64),))
        problem = only(state().validate(held))
        self.assertIn("it was gathered on model ffffffffffffffff", problem)

    def test_a_proof_for_another_model_does_not_apply(self):
        held = evidence(model_identity="f" * 64,
                        proofs=(proof(model_identity="f" * 64),))
        problems = state().validate(held)
        self.assertTrue(any("a proof for one model does not admit another" in p
                            for p in problems), problems)

    def test_evidence_bound_to_another_model_admits_nothing_in_this_artifact(self):
        """The binding is what stops one demonstration travelling to every artifact."""
        held = evidence(model_identity="f" * 64,
                        proofs=(proof(model_identity="f" * 64),))
        self.assertTrue(any(UNPROVEN_STATE_REFUSAL in p for p in state().validate(held)))

    def test_evidence_offered_for_an_artifact_with_no_geometry_is_dropped(self):
        problems = state(geometry=None).validate(evidence())
        self.assertTrue(any("cannot be tied to the model it would admit" in p
                            for p in problems), problems)
        self.assertTrue(any(UNPROVEN_STATE_REFUSAL in p for p in problems))

    def test_evidence_that_names_no_build_admits_nothing(self):
        held = evidence(runtime_build="   ")
        self.assertIn("does not name the build it is being checked against",
                      only(state().validate(held)))

    def test_evidence_whose_binary_is_not_a_digest_admits_nothing(self):
        for bad in ("", "cc", "C" * 64, None):
            with self.subTest(binary=bad):
                self.assertIn("binary_sha256 is not a 64-character lowercase digest",
                              only(state().validate(evidence(binary_sha256=bad))))

    def test_evidence_whose_model_is_not_a_digest_admits_nothing(self):
        """Checked on the evidence itself: the artifact binding refuses it first, and a
        digest that is not one must not become admitting for an artifact that had none."""
        held = evidence(model_identity="short", proofs=(proof(model_identity="short"),))
        verdict, reason = held.verdict(SPECULATIVE_STATE)
        self.assertIsNone(verdict)
        self.assertIn("model_identity is not a 64-character lowercase digest", reason)

    def test_record_digests_must_be_a_mapping(self):
        self.assertIn("record_digests is not a mapping",
                      only(state().validate(evidence(record_digests=[RECORD]))))


class AProofGoesStale(unittest.TestCase):
    def test_an_expired_proof_does_not_apply(self):
        held = evidence(as_of="2026-12-01T00:00:00Z")
        problem = only(state().validate(held))
        self.assertIn("it expired at 2026-10-01T00:00:00Z", problem)
        self.assertIn("re-derive it against this build", problem)

    def test_a_proof_dated_in_the_future_does_not_apply(self):
        held = evidence(proofs=(proof(proven_at="2026-09-15T00:00:00Z"),))
        self.assertIn("which has not happened yet", only(state().validate(held)))

    def test_a_proof_that_never_expires_is_refused(self):
        far = proof(expires_at="2030-01-01T00:00:00Z")
        problem = only(state().validate(evidence(proofs=(far,))))
        self.assertIn(f"longer than {MAX_PROOF_VALIDITY.days} days", problem)
        self.assertIn("standing permission, not a measurement", problem)

    def test_an_inverted_validity_window_is_refused(self):
        held = evidence(proofs=(proof(expires_at="2026-07-01T00:00:00Z"),))
        self.assertIn("expires at or before the run it records",
                      only(state().validate(held)))

    def test_timestamps_without_a_zone_are_not_moments(self):
        for field_name in ("proven_at", "expires_at"):
            with self.subTest(field=field_name):
                held = evidence(proofs=(proof(**{field_name: "2026-08-01T00:00:00"}),))
                self.assertIn(f"{field_name} '2026-08-01T00:00:00' is not an ISO-8601",
                              only(state().validate(held)))

    def test_an_unreadable_as_of_decides_nothing(self):
        self.assertIn("as_of 'yesterday' is not an ISO-8601",
                      only(state().validate(evidence(as_of="yesterday"))))

    def test_an_absent_as_of_is_the_present_moment(self):
        now = datetime.now(timezone.utc)
        fresh = proof(proven_at=(now - timedelta(days=1)).isoformat(),
                      expires_at=(now + timedelta(days=1)).isoformat())
        self.assertEqual(state().validate(evidence(proofs=(fresh,), as_of="")), [])
        stale = proof(proven_at=(now - timedelta(days=10)).isoformat(),
                      expires_at=(now - timedelta(days=1)).isoformat())
        self.assertIn("it expired at",
                      only(state().validate(evidence(proofs=(stale,), as_of=""))))


class TheRunRecordHasToStillSayIt(unittest.TestCase):
    def test_a_record_that_cannot_be_produced_now_is_a_citation_not_a_record(self):
        held = evidence(record_digests={})
        problem = only(state().validate(held))
        self.assertIn("was not produced at admission time", problem)
        self.assertIn(RECORD_URI, problem)

    def test_a_record_that_no_longer_hashes_the_same_has_drifted(self):
        held = evidence(record_digests={RECORD_URI: "9" * 64})
        problem = only(state().validate(held))
        self.assertIn("the record and the claim have drifted apart", problem)

    def test_a_proof_citing_no_record_is_refused(self):
        held = evidence(proofs=(proof(record_uri="  "),))
        self.assertIn("cites no run record", only(state().validate(held)))

    def test_a_proof_whose_record_digest_is_not_a_digest_is_refused(self):
        held = evidence(proofs=(proof(record_sha256="not-a-digest"),))
        self.assertIn("record_sha256 is not a 64-character lowercase digest",
                      only(state().validate(held)))


class AProofHasToBeFalsifiable(unittest.TestCase):
    def test_a_proof_of_zero_runs_is_a_claim(self):
        for bad in (0, -1, True, "3"):
            with self.subTest(trials=bad):
                held = evidence(proofs=(proof(trials=bad),))
                self.assertIn("a demonstration that was never run is a claim",
                              only(state().validate(held)))

    def test_a_proof_that_does_not_say_how_is_refused(self):
        held = evidence(proofs=(proof(method="   "),))
        self.assertIn("neither re-derived nor contradicted", only(state().validate(held)))

    def test_an_unreadable_outcome_is_refused(self):
        held = evidence(proofs=(proof(outcome="looked fine"),))
        self.assertIn("outcome 'looked fine' is not one of", only(state().validate(held)))

    def test_a_proof_naming_no_build_is_refused(self):
        held = evidence(proofs=(proof(runtime_build=""),))
        self.assertIn("names no runtime build", only(state().validate(held)))

    def test_the_record_reports_its_own_malformation_directly(self):
        """Asserted on `validate()` rather than through an admission, because the scope
        check refuses an unknown class first and would pass either way - which is the
        vacuous shape this repo keeps finding in its own tests."""
        problems = proof(state_class="target").validate()
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("would admit everything or nothing", problems[0])

    def test_a_timestamp_that_is_not_text_is_refused_rather_than_crashing(self):
        for bad in (None, 17540000, ["2026-08-01T00:00:00Z"]):
            with self.subTest(proven_at=bad):
                held = evidence(proofs=(proof(proven_at=bad),))
                self.assertIn("is not an ISO-8601 instant with a timezone",
                              only(state().validate(held)))


class GuardsThisAuditSurfacedInPassing(unittest.TestCase):
    """Pre-existing refusals in the same two files that no test noticed the removal of.

    Found by widening the audit from if-guarded raises to every condition, because almost
    everything in this schema refuses by appending to a list rather than raising. Each was
    shadowed by a second message the existing test happened to match on instead.
    """

    def test_a_non_positive_layer_count_says_so(self):
        broken = replace(state().geometry, n_layer=0, n_head_kv=())
        problems = state(geometry=broken).validate()
        self.assertTrue(any("n_layer 0 is not positive" in p for p in problems), problems)

    def test_a_layer_with_no_kv_heads_says_so(self):
        broken = replace(state().geometry, n_head_kv=(4, 0, 4, 4))
        problems = state(geometry=broken).validate()
        self.assertTrue(any("zero or negative KV heads" in p for p in problems), problems)

    def test_a_checkpoint_covering_no_tokens_says_so(self):
        problems = state(checkpoints=(checkpoint(n_tokens=0),)).validate()
        self.assertTrue(any("checkpoint covers 0 tokens" in p for p in problems), problems)

    def test_an_artifact_with_no_attention_segments_says_so(self):
        problems = state(attention_segments=()).validate()
        self.assertTrue(any(p.endswith("no attention segments") for p in problems), problems)

    def test_an_artifact_declaring_no_kv_types_does_not_compare_them(self):
        """The same shape in requirements.check(): an artifact that names no cache dtype
        must not be reported as mismatching whatever the runtime happens to be running."""
        plain = for_artifact(hybrid=False, checkpoints=0, sequence_state_version=3,
                             kv_type_k="", kv_type_v="")
        self.assertEqual(
            check(plain, {**CLAIMS, "kv_type_k": "q4_0", "kv_type_v": "q4_0"}), [])

    def test_a_recurrent_model_with_no_recurrent_segments_says_so(self):
        problems = state(recurrent_segments=()).validate()
        self.assertTrue(any("has recurrent state but no recurrent segments" in p
                            for p in problems), problems)


class UnknownIsNotTheSameAnswerAsNo(unittest.TestCase):
    """Three states, the shape `expected_reuse` arrived at after getting this wrong twice."""

    def test_nothing_offered_is_unknown(self):
        verdict, reason = evidence(proofs=()).verdict(SPECULATIVE_STATE)
        self.assertIsNone(verdict)
        self.assertEqual(reason,
                         "no proof of speculative checkpoint restoration was offered")

    def test_a_demonstration_that_diverged_is_a_finding_not_a_gap(self):
        verdict, reason = evidence(
            proofs=(proof(outcome=PROOF_DIVERGED),)).verdict(SPECULATIVE_STATE)
        self.assertIs(verdict, False)
        self.assertIn("records that speculative state did NOT restore", reason)

    def test_a_passing_demonstration_is_proven(self):
        verdict, reason = evidence().verdict(SPECULATIVE_STATE)
        self.assertIs(verdict, True)
        self.assertIn("over 3 trial(s)", reason)

    def test_a_divergence_outranks_a_pass_in_the_same_scope(self):
        """Fail closed: two in-scope records disagreeing is not a majority vote."""
        held = evidence(proofs=(proof(), proof(outcome=PROOF_DIVERGED)))
        self.assertIs(held.verdict(SPECULATIVE_STATE)[0], False)
        self.assertIn("did NOT restore", only(state().validate(held)))

    def test_a_recorded_divergence_reads_differently_from_an_absent_record(self):
        diverged = only(state().validate(evidence(proofs=(proof(outcome=PROOF_DIVERGED),))))
        absent = only(state().validate(evidence(proofs=())))
        self.assertNotIn(UNPROVEN_STATE_REFUSAL, diverged)
        self.assertIn(UNPROVEN_STATE_REFUSAL, absent)


class AnArtifactCannotVouchForItself(unittest.TestCase):
    def test_a_payload_carrying_its_own_proof_is_refused(self):
        for key in sorted(EVIDENCE_KEYS):
            with self.subTest(key=key):
                payload = state().to_dict()
                payload[key] = [{"state_class": SPECULATIVE_STATE}]
                with self.assertRaises(SchemaError) as caught:
                    HybridState.from_dict(payload)
                self.assertIn("artifact payload carries admission evidence",
                              str(caught.exception))

    def test_a_checkpoint_carrying_its_own_proof_is_refused(self):
        payload = state().to_dict()
        payload["checkpoints"][0]["proof"] = {"outcome": PROOF_RESTORED}
        with self.assertRaises(SchemaError) as caught:
            HybridState.from_dict(payload)
        self.assertIn("does not get to authorise it", str(caught.exception))

    def test_the_serialized_form_carries_no_evidence_at_all(self):
        payload = state().to_dict()
        self.assertEqual(EVIDENCE_KEYS.intersection(payload), set())
        self.assertEqual(EVIDENCE_KEYS.intersection(payload["checkpoints"][0]), set())

    def test_a_round_trip_still_arrives_refused(self):
        restored = HybridState.from_dict(state().to_dict())
        self.assertEqual(only(restored.validate()), f"checkpoint 0: {UNPROVEN_STATE_REFUSAL}")


class TheRuntimeGate(unittest.TestCase):
    """requirements.check(): the other half, about the runtime rather than the artifact."""

    def artifact(self):
        return for_artifact(hybrid=True, checkpoints=1, sequence_state_version=3,
                            model_identity=MODEL, speculative_state=True)

    def passing(self, **overrides):
        base = dict(model_identity=MODEL, runtime_identity=MODEL, runtime_build=BUILD,
                    binary_sha256=BINARY, evidence=evidence())
        base.update(overrides)
        return check(self.artifact(), CLAIMS, **base)

    def test_for_artifact_records_the_class_and_stops_calling_it_portable(self):
        wanted = self.artifact()
        self.assertEqual(wanted.gated_state_classes, (SPECULATIVE_STATE,))
        self.assertFalse(wanted.portable)
        self.assertIn("whose restoration is unproven", wanted.notes[-1])
        self.assertIn("gated_state_classes", wanted.as_dict())

    def test_an_artifact_without_unproven_state_is_unaffected(self):
        plain = for_artifact(hybrid=True, checkpoints=1, sequence_state_version=3)
        self.assertEqual(plain.gated_state_classes, ())
        self.assertEqual(check(plain, CLAIMS), [])

    def test_a_full_holding_is_admitted(self):
        self.assertEqual(self.passing(), [])
        require(self.artifact(), CLAIMS, model_identity=MODEL, runtime_identity=MODEL,
                runtime_build=BUILD, binary_sha256=BINARY, evidence=evidence())

    def test_no_evidence_is_refused_and_says_absence_is_not_proof(self):
        problems = self.passing(evidence=None)
        self.assertTrue(any("Absence of a proof is not a proof" in p for p in problems),
                        problems)

    def test_it_raises_rather_than_returning(self):
        with self.assertRaises(RequirementError) as caught:
            require(self.artifact(), CLAIMS, model_identity=MODEL, runtime_identity=MODEL,
                    runtime_build=BUILD, binary_sha256=BINARY)
        self.assertIn("Absence of a proof is not a proof", str(caught.exception))

    def test_a_runtime_that_does_not_claim_the_capability_is_refused(self):
        """The measured shape: the format serializes it, the runtime says it cannot."""
        props = {**CLAIMS, "supports_speculative_checkpoint_state": False}
        problems = check(self.artifact(), props, model_identity=MODEL,
                         runtime_identity=MODEL, runtime_build=BUILD,
                         binary_sha256=BINARY, evidence=evidence())
        self.assertTrue(any("the format serialising the blob is not the runtime restoring "
                            "it" in p for p in problems), problems)

    def test_a_runtime_that_cannot_be_named_is_refused(self):
        props = {k: v for k, v in CLAIMS.items() if k != "build_info"}
        problems = check(self.artifact(), props, model_identity=MODEL,
                         runtime_identity=MODEL, binary_sha256=BINARY, evidence=evidence())
        self.assertTrue(any("cannot be tied to the build it was gathered on" in p
                            for p in problems), problems)

    def test_a_proof_for_another_build_than_the_target_is_refused(self):
        problems = self.passing(evidence=evidence(runtime_build="b1-deadbee",
                                                  proofs=(proof(runtime_build="b1-deadbee"),)))
        self.assertTrue(any("this runtime reports 'b1-3e73446'" in p for p in problems),
                        problems)

    def test_an_unidentified_binary_is_refused_rather_than_assumed(self):
        problems = self.passing(binary_sha256="")
        self.assertTrue(any("binary this runtime is running was not identified" in p
                            for p in problems), problems)

    def test_a_proof_against_another_binary_is_refused(self):
        problems = self.passing(binary_sha256="e" * 64)
        self.assertTrue(any("this runtime runs eeeeeeeeeeeeeeee" in p for p in problems),
                        problems)

    def test_a_proof_for_another_model_is_refused(self):
        held = evidence(model_identity="f" * 64, proofs=(proof(model_identity="f" * 64),))
        problems = self.passing(evidence=held)
        self.assertTrue(any("this artifact is aaaaaaaaaaaaaaaa" in p for p in problems),
                        problems)

    def test_an_artifact_with_no_model_identity_cannot_bind_a_proof(self):
        anonymous = for_artifact(hybrid=True, checkpoints=1, sequence_state_version=3,
                                 speculative_state=True)
        problems = check(anonymous, CLAIMS, runtime_build=BUILD, binary_sha256=BINARY,
                         evidence=evidence())
        self.assertTrue(any("cannot be tied to what it would admit" in p for p in problems),
                        problems)

    def test_a_recorded_divergence_is_refused_in_its_own_words(self):
        problems = self.passing(evidence=evidence(proofs=(proof(outcome=PROOF_DIVERGED),)))
        self.assertTrue(any("speculative checkpoint state is refused here" in p
                            for p in problems), problems)

    def test_an_expired_proof_is_refused_as_unproven(self):
        problems = self.passing(evidence=evidence(as_of="2026-12-01T00:00:00Z"))
        self.assertTrue(any("is not proven for this runtime and model" in p
                            for p in problems), problems)

    def test_a_state_class_that_cannot_be_named_cannot_be_proven(self):
        wanted = self.artifact()
        smuggled = type(wanted)(**{**wanted.as_dict(),
                                   "gated_state_classes": ("everything",)})
        problems = check(smuggled, CLAIMS, model_identity=MODEL, runtime_identity=MODEL,
                         runtime_build=BUILD, binary_sha256=BINARY, evidence=evidence())
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("state that cannot be named cannot be proven", problems[0])

    def test_the_draft_class_is_gated_on_its_own_property(self):
        wanted = for_artifact(hybrid=True, checkpoints=1, sequence_state_version=3,
                              model_identity=MODEL, draft_state=True,
                              speculative_state=True)
        self.assertEqual(wanted.gated_state_classes, (DRAFT_STATE, SPECULATIVE_STATE))
        problems = check(wanted, {**CLAIMS, "supports_draft_checkpoint_state": False},
                         model_identity=MODEL, runtime_identity=MODEL, runtime_build=BUILD,
                         binary_sha256=BINARY,
                         evidence=evidence(proofs=(proof(),
                                                   proof(state_class=DRAFT_STATE))))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("supports_draft_checkpoint_state", problems[0])


if __name__ == "__main__":
    unittest.main()
