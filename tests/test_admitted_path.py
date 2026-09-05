"""The admitted restore path must refuse everything the KVX import would, before restoring.

It skips reading the payload on the request path, so every check that a full verification
would have made has to happen at admission or at resolution. These tests pin the refusals and
measure the read count, because "no payload copy" is a claim that has to be substantiated
rather than asserted.
"""

import struct
import tempfile
import unittest

from kv_rosetta.adapters.admitted_path import (
    MAX_UNCOVERED_TAIL, uncovered_allowance)
from pathlib import Path

from kv_rosetta.admitted_store import AdmissionError, AdmittedStore
from kv_rosetta.adapters.admitted_path import AdmittedPath
from tests.test_hybrid_export_gate import (
    StubAdapter,
    ggsq_body,
    props,
    save_with_appendix,
    sckp_appendix,
)
from tests.test_hybrid_support_predicate import TARGET_ONLY

TOKENS = [11, 22, 33, 44]


class RecordingAdapter(StubAdapter):
    """Records endpoint calls and answers restore/completion for the direct path."""

    def __init__(self, *args, restore=None, probe=None, pristine_fails=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._restore = restore or {}
        self._probe = probe or {"timings": {"cache_n": 3, "prompt_n": 1}}
        self._pristine_fails = pristine_fails
        self._restores = 0

    def _post(self, path, payload, timeout=None):
        self.posts.append((path, dict(payload)))
        if "action=restore" in path:
            self._restores += 1
            if self._pristine_fails and self._restores > 1:
                from kv_rosetta.adapters.base import AdapterError
                raise AdapterError("simulated pristine failure")
            return dict(self._restore)
        if path == "/completion":
            return dict(self._probe)
        return {}

    def endpoints(self):
        return [p for p, _ in self.posts]


def raw_state(directory: Path, tokens=TOKENS, version=3):
    appendix = sckp_appendix(n_tokens=3, pos_min=0, pos_max=2)
    body = ggsq_body(tokens=tokens, version=version, trailer=appendix)
    path = directory / "raw.bin"
    path.write_bytes(body)
    save = {"n_written": len(body), "checkpoint_bytes": len(appendix),
            "checkpoint_n_tokens": 3, "checkpoint_pos_min": 0, "checkpoint_pos_max": 2,
            "n_checkpoints_saved": 1, "n_saved": len(tokens)}
    return path, save


def restore_response(save):
    return {"n_restored": len(TOKENS),
            "n_checkpoints_restored": save["n_checkpoints_saved"],
            "checkpoint_bytes": save["checkpoint_bytes"],
            "checkpoint_n_tokens": save["checkpoint_n_tokens"],
            "checkpoint_pos_min": save["checkpoint_pos_min"],
            "checkpoint_pos_max": save["checkpoint_pos_max"]}


class AdmittedPathTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.raw, self.save = raw_state(self.dir)
        self.store = AdmittedStore(self.dir / "store")

    def path_for(self, runtime=None, **kwargs):
        slots = self.dir / f"slots{len(list(self.dir.iterdir()))}"
        slots.mkdir()
        adapter = RecordingAdapter(runtime or props(**TARGET_ONLY), slots,
                                   save=dict(self.save), body=self.raw.read_bytes(),
                                   restore=restore_response(self.save), **kwargs)
        return AdmittedPath(adapter, self.store), adapter

    def admitted(self):
        path, _ = self.path_for()
        return path.admit(self.raw, model="", token_ids=TOKENS,
                          save_response=self.save)

    # -- the happy path --------------------------------------------------------------

    def test_admit_then_restore_succeeds(self):
        obj = self.admitted()
        path, adapter = self.path_for()
        report = path.restore(obj.digest, model="", token_ids=TOKENS)
        self.assertTrue(report.ok, report.reason)
        self.assertEqual(report.cache_n, 3)
        self.assertEqual(report.prompt_n, 1)

    # -- the request path must not read the payload -----------------------------------

    def test_the_request_path_reads_no_payload_bytes(self):
        obj = self.admitted()
        path, _ = self.path_for()
        report = path.restore(obj.digest, model="", token_ids=TOKENS)
        self.assertTrue(report.ok, report.reason)
        self.assertEqual(report.reads.payload_bytes, 0,
                         "kv-rosetta read payload bytes on the request path")
        self.assertLess(report.reads.metadata_bytes, 8192,
                        "request-path metadata reads are not bounded")

    def test_the_admitted_object_is_named_not_copied(self):
        obj = self.admitted()
        path, adapter = self.path_for()
        before = sorted(p.name for p in self.store.root.iterdir())
        report = path.restore(obj.digest, model="", token_ids=TOKENS)
        self.assertTrue(report.ok, report.reason)
        self.assertEqual(sorted(p.name for p in self.store.root.iterdir()), before,
                         "the restore created or removed files in the store")
        for _, payload in adapter.posts:
            if "filename" in payload:
                self.assertEqual(payload["filename"], obj.basename)

    # -- refusals before the runtime is touched ---------------------------------------

    def assertRefusedBeforeRestore(self, report, adapter):
        self.assertFalse(report.ok)
        self.assertEqual([e for e in adapter.endpoints() if "restore" in e], [],
                         f"restored anyway: {adapter.endpoints()}")

    def test_an_unpatched_runtime_refuses_before_opening_state(self):
        obj = self.admitted()
        path, adapter = self.path_for(
            {"default_generation_settings": {"type_k": "f16"}, "build_info": "b"})
        report = path.restore(obj.digest, model="", token_ids=TOKENS)
        self.assertRefusedBeforeRestore(report, adapter)
        self.assertEqual(adapter.endpoints(), [])

    def test_a_missing_cache_dtype_refuses_before_restore(self):
        obj = self.admitted()
        runtime = props(**TARGET_ONLY)
        del runtime["target_cache_type_k"]
        path, adapter = self.path_for(runtime)
        self.assertRefusedBeforeRestore(
            path.restore(obj.digest, model="", token_ids=TOKENS), adapter)

    def test_an_active_draft_configuration_refuses_before_restore(self):
        obj = self.admitted()
        path, adapter = self.path_for(
            props(active_checkpoint_state_classes=["target", "draft"]))
        self.assertRefusedBeforeRestore(
            path.restore(obj.digest, model="", token_ids=TOKENS), adapter)

    def test_a_different_cache_dtype_refuses_before_restore(self):
        obj = self.admitted()
        path, adapter = self.path_for(props(target_cache_type_k="q8_0", **TARGET_ONLY))
        report = path.restore(obj.digest, model="", token_ids=TOKENS)
        self.assertRefusedBeforeRestore(report, adapter)
        self.assertIn("cache ABI mismatch", report.reason)

    def test_different_prompt_tokens_refuse_before_restore(self):
        obj = self.admitted()
        path, adapter = self.path_for()
        report = path.restore(obj.digest, model="", token_ids=[99, 98, 97, 96])
        self.assertRefusedBeforeRestore(report, adapter)
        self.assertIn("prompt identity", report.reason)

    def test_an_unadmitted_file_cannot_enter_the_restore_path(self):
        import hashlib
        rogue = self.dir / "rogue.bin"
        rogue.write_bytes(b"arbitrary bytes")
        digest = hashlib.sha256(rogue.read_bytes()).hexdigest()
        (self.store.root / f"{digest}.state").write_bytes(rogue.read_bytes())
        path, adapter = self.path_for()
        self.assertRefusedBeforeRestore(
            path.restore(digest, model="", token_ids=TOKENS), adapter)

    def test_a_replaced_admitted_object_refuses_before_restore(self):
        obj = self.admitted()
        import time as _t
        _t.sleep(0.01)
        (self.store.root / f"{obj.digest}.state").write_bytes(b"different bytes entirely")
        path, adapter = self.path_for()
        report = path.restore(obj.digest, model="", token_ids=TOKENS)
        self.assertRefusedBeforeRestore(report, adapter)
        self.assertIn("no longer matches", report.reason)

    # -- admission-time refusals --------------------------------------------------------

    def test_admission_refuses_an_unpatched_runtime(self):
        path, adapter = self.path_for(
            {"default_generation_settings": {"type_k": "f16"}, "build_info": "b"})
        with self.assertRaises(AdmissionError):
            path.admit(self.raw, model="", token_ids=TOKENS, save_response=self.save)

    def test_admission_refuses_a_truncated_state(self):
        truncated = self.dir / "short.bin"
        truncated.write_bytes(self.raw.read_bytes()[:-40])
        path, _ = self.path_for()
        with self.assertRaises(AdmissionError):
            path.admit(truncated, model="", token_ids=TOKENS, save_response=self.save)

    def test_admission_refuses_when_the_declared_bytes_disagree(self):
        path, _ = self.path_for()
        with self.assertRaises(AdmissionError) as caught:
            path.admit(self.raw, model="", token_ids=TOKENS,
                       save_response=dict(self.save, n_written=self.save["n_written"] + 8))
        self.assertIn("bytes but the file is", str(caught.exception))

    def test_admission_refuses_tokens_the_state_does_not_carry(self):
        path, _ = self.path_for()
        with self.assertRaises(AdmissionError) as caught:
            path.admit(self.raw, model="", token_ids=[1, 2, 3, 4, 5, 6],
                       save_response=self.save)
        self.assertIn("tokens", str(caught.exception))

    def test_admission_refuses_an_untested_sequence_version(self):
        v2 = self.dir / "v2"
        v2.mkdir()
        raw, save = raw_state(v2, version=2)
        path, _ = self.path_for()
        with self.assertRaises(AdmissionError) as caught:
            path.admit(raw, model="", token_ids=TOKENS, save_response=save)
        self.assertIn("untested compound tuple", str(caught.exception))

    def test_admission_refuses_missing_checkpoint_coverage(self):
        path, _ = self.path_for()
        with self.assertRaises(AdmissionError):
            path.admit(self.raw, model="", token_ids=TOKENS,
                       save_response=dict(self.save, checkpoint_n_tokens=0))

    # -- verification failures leave the slot safe --------------------------------------

    def test_metadata_disagreement_erases_and_refuses(self):
        obj = self.admitted()
        path, adapter = self.path_for()
        adapter._restore = dict(restore_response(self.save), checkpoint_n_tokens=99)
        report = path.restore(obj.digest, model="", token_ids=TOKENS)
        self.assertFalse(report.ok)
        self.assertIn("does not match the admitted state", report.reason)
        self.assertIn("/slots/0?action=erase", adapter.endpoints())

    def test_no_reuse_erases_and_refuses(self):
        obj = self.admitted()
        path, adapter = self.path_for(probe={"timings": {"cache_n": 0, "prompt_n": 4}})
        report = path.restore(obj.digest, model="", token_ids=TOKENS)
        self.assertFalse(report.ok)
        self.assertIn("/slots/0?action=erase", adapter.endpoints())

    def test_a_tail_that_does_not_match_the_uncovered_count_refuses(self):
        """Isolates the tail contract from the coverage-equality check.

        test_no_reuse_erases_and_refuses uses cache_n=0, which coverage equality rejects
        first, so it proved nothing about the tail. A mutation run confirmed that. Here
        cache_n equals the declared coverage and only prompt_n is wrong.
        """
        obj = self.admitted()
        path, adapter = self.path_for(probe={"timings": {"cache_n": 3, "prompt_n": 2}})
        report = path.restore(obj.digest, model="", token_ids=TOKENS)
        self.assertFalse(report.ok)
        self.assertIn("tail contract violated", report.reason)
        self.assertIn("/slots/0?action=erase", adapter.endpoints())

    def test_admission_refuses_a_damaged_appendix_at_the_declared_offset(self):
        """Isolates the appendix check from the byte-count check.

        Truncating the file fails the n_written comparison first. Here the size is exactly
        as declared and only the appendix magic is damaged, so the offset-bound appendix
        validation is the sole remaining guard.
        """
        data = bytearray(self.raw.read_bytes())
        offset = self.save["n_written"] - self.save["checkpoint_bytes"]
        data[offset] ^= 0xFF                      # break the SCKP magic in place
        damaged = self.dir / "damaged.bin"
        damaged.write_bytes(bytes(data))
        self.assertEqual(damaged.stat().st_size, self.save["n_written"])
        path, _ = self.path_for()
        with self.assertRaises(AdmissionError) as caught:
            path.admit(damaged, model="", token_ids=TOKENS, save_response=self.save)
        self.assertIn("appendix at the declared offset", str(caught.exception))

    def test_a_failed_pristine_restore_refuses(self):
        obj = self.admitted()
        path, adapter = self.path_for(pristine_fails=True)
        report = path.restore(obj.digest, model="", token_ids=TOKENS)
        self.assertFalse(report.ok)
        self.assertIn("pristine", report.reason)
        self.assertIn("/slots/0?action=erase", adapter.endpoints())


if __name__ == "__main__":
    unittest.main()


class RecordedRequirementsTest(unittest.TestCase):
    """Admission records what a runtime will need to restore these bytes.

    Recorded at admission because it is a fact about the bytes just written. Checked at
    restore, it turns a silent uselessness - a hybrid state accepted by an unpatched build
    that then reuses nothing - into an explicit refusal.
    """

    #: A runtime that satisfies everything except the patch. The identity has to be here:
    #: admission records the model it captured from, and a runtime that reports none cannot
    #: be shown to be that model - which is a refusal in its own right.
    STOCK = {"sequence_state_version": 3, "model_identity": "a" * 64}
    PATCHED = {**STOCK, "slot_checkpoint_persistence": True,
               "slot_checkpoint_format": "sckp/1"}

    def test_a_checkpointed_artifact_is_marked_as_needing_the_patch(self):
        from kv_rosetta.requirements import Requirements, check
        recorded = requirements_for(checkpoints=2, version=3)
        self.assertTrue(recorded["needs_checkpoint_persistence"])
        self.assertTrue(check(Requirements(**recorded), self.STOCK))
        self.assertEqual(check(Requirements(**recorded), self.PATCHED), [])

    def test_an_artifact_without_checkpoints_needs_no_patch(self):
        from kv_rosetta.requirements import Requirements, check
        recorded = requirements_for(checkpoints=0, version=3)
        self.assertFalse(recorded["needs_checkpoint_persistence"])
        self.assertEqual(check(Requirements(**recorded), self.STOCK), [])

    def test_the_recorded_form_round_trips_through_the_checker(self):
        """The manifest stores a plain dict; it has to reconstruct a Requirements exactly."""
        from kv_rosetta.requirements import Requirements
        recorded = requirements_for(checkpoints=1, version=3)
        self.assertEqual(Requirements(**recorded).as_dict(), recorded)


def requirements_for(*, checkpoints, version):
    from kv_rosetta import requirements
    return requirements.for_artifact(
        hybrid=bool(checkpoints), checkpoints=checkpoints,
        sequence_state_version=version, kv_type_k="f16", kv_type_v="f16",
        model_identity="a" * 64).as_dict()


class ModelNamedByAlias(AdmittedPathTest):
    """A runtime named by its llama-swap alias, not by a path to its weights.

    Every other test here passes model="", so this path was never exercised: admit()
    resolved model_path from /props and then digested the raw `model` argument instead.
    Against a live fleet the sidecar names models by alias, so admission failed outright
    with WeightsError: cannot stat 'tiel-kvx-w6800'.
    """

    def test_an_alias_is_admitted_using_the_runtime_reported_weights_path(self):
        import struct
        def s_(text):
            raw = text.encode()
            return struct.pack("<Q", len(raw)) + raw
        blob = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
        blob += s_("general.architecture") + struct.pack("<I", 8) + s_("qwen35moe")
        weights = self.dir / "weights.gguf"
        weights.write_bytes(bytes(blob))
        runtime = props(**TARGET_ONLY)
        runtime["model_path"] = str(weights)
        path, _ = self.path_for(runtime=runtime)
        obj = path.admit(self.raw, model="tiel-kvx-w6800", token_ids=TOKENS,
                         save_response=self.save)
        self.assertTrue(obj.manifest["model_content_digest"])

    def test_an_alias_with_no_resolvable_weights_records_no_content_digest(self):
        # Refusing here would block admission on a runtime that simply does not report
        # model_path; recording an empty digest keeps the artifact honest instead, and
        # every identity check that matters runs off model_weights_sha256.
        runtime = props(**TARGET_ONLY)
        runtime["model_path"] = ""
        path, _ = self.path_for(runtime=runtime)
        obj = path.admit(self.raw, model="tiel-kvx-w6800", token_ids=TOKENS,
                         save_response=self.save)
        self.assertEqual(obj.manifest["model_content_digest"], "")


class RuntimeCannotSupportTheState(AdmittedPathTest):
    """The refusal that fires on a real MTP launch.

    A tiel instance running draft-MTP reports active_checkpoint_state_classes
    ['target','speculative'], and admission stops there rather than publishing an artifact
    whose restoration the runtime does not claim to support. Worth pinning because the
    message is what an operator acts on: §36 showed the runtime's own labels are wrong in
    both directions, so a refusal that names the wrong class sends the fix to the wrong
    code path.
    """

    def test_a_runtime_declaring_unproven_state_classes_is_refused(self):
        runtime = props(active_checkpoint_state_classes=["target", "speculative"])
        path, _ = self.path_for(runtime=runtime)
        with self.assertRaises(AdmissionError) as caught:
            path.admit(self.raw, model="", token_ids=TOKENS, save_response=self.save)
        self.assertIn("runtime cannot support this state", str(caught.exception))

    def test_a_target_only_runtime_is_admitted(self):
        # The control: without it the test above would pass on any refusal at all.
        self.assertTrue(self.admitted().digest)


class RestoreUnderAnAlias(AdmittedPathTest):
    """The restore side had the same alias-as-path bug as admit().

    Fixing admit() alone left every sidecar restore returning
    "cannot stat tiel-kvx-w6800", because ensure() names models by llama-swap alias.
    """

    def _gguf(self, arch="qwen35moe"):
        import struct
        def s_(t):
            raw = t.encode()
            return struct.pack("<Q", len(raw)) + raw
        blob = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
        blob += s_("general.architecture") + struct.pack("<I", 8) + s_(arch)
        path = self.dir / "w.gguf"
        path.write_bytes(bytes(blob))
        return path

    def test_an_aliased_restore_resolves_identity_from_the_runtime(self):
        runtime = props(**TARGET_ONLY)
        runtime["model_path"] = str(self._gguf())
        path, _ = self.path_for(runtime=runtime)
        obj = path.admit(self.raw, model="tiel-kvx-w6800", token_ids=TOKENS,
                         save_response=self.save)
        report = path.restore(obj.digest, model="tiel-kvx-w6800", token_ids=TOKENS, slot=0)
        self.assertTrue(report.ok, report.reason)

    def test_an_unresolvable_identity_is_refused_not_waved_through(self):
        # The artifact recorded a content digest; if the live one cannot be resolved the
        # restore is refused rather than trusted on the record alone.
        runtime = props(**TARGET_ONLY)
        runtime["model_path"] = str(self._gguf())
        path, adapter = self.path_for(runtime=runtime)
        obj = path.admit(self.raw, model="tiel-kvx-w6800", token_ids=TOKENS,
                         save_response=self.save)
        adapter._props_value = dict(runtime, model_path="/nonexistent/gone.gguf")
        report = path.restore(obj.digest, model="tiel-kvx-w6800", token_ids=TOKENS, slot=0)
        self.assertFalse(report.ok)
        self.assertIn("model identity mismatch", report.reason)


class TokenMismatchMessage(AdmittedPathTest):
    """Equal-length, different-content token lists must not report as equal counts.

    The check compares ids; the message reported lengths. On a real admission that printed
    "state carries 6169 tokens, not the 6169 under test", which reads as a broken checker
    rather than as the caller passing the wrong tokens.
    """

    def test_a_different_length_still_reports_the_two_lengths(self):
        path, _ = self.path_for()
        with self.assertRaises(AdmissionError) as caught:
            path.admit(self.raw, model="", token_ids=TOKENS[:2], save_response=self.save)
        self.assertIn("not the 2 under test", str(caught.exception))

    def test_the_same_length_reports_where_they_diverge(self):
        path, _ = self.path_for()
        wrong = list(TOKENS)
        wrong[1] = wrong[1] + 1
        with self.assertRaises(AdmissionError) as caught:
            path.admit(self.raw, model="", token_ids=wrong, save_response=self.save)
        message = str(caught.exception)
        self.assertIn("first difference at position 1", message)
        self.assertIn(str(TOKENS[1]), message)


class UncoveredAllowanceTests(unittest.TestCase):
    """A hybrid model resumes from a checkpoint boundary, so its tail scales with the
    artifact. A flat ceiling read a 99.8%-complete restore as a failure."""

    def test_small_artifacts_keep_the_measured_floor(self):
        self.assertEqual(uncovered_allowance(100), MAX_UNCOVERED_TAIL)
        self.assertEqual(uncovered_allowance(1), MAX_UNCOVERED_TAIL)

    def test_large_artifacts_scale_with_size(self):
        self.assertEqual(uncovered_allowance(9146), 92)

    def test_the_flash_next_tail_that_was_refused_now_passes(self):
        self.assertLessEqual(19, uncovered_allowance(9146))

    def test_a_restore_that_covered_almost_nothing_is_still_refused(self):
        # 2,000 of 9,146 covered leaves 7,146 uncovered, far beyond any allowance.
        self.assertGreater(9146 - 2000, uncovered_allowance(9146))

    def test_refuses_a_non_positive_token_count(self):
        with self.assertRaises(ValueError) as cm:
            uncovered_allowance(0)
        self.assertIn("is not positive", str(cm.exception))


class ExactTailAllowanceTests(unittest.TestCase):
    """The allowance is the artifact's own tail, not a fraction guessed at it."""

    def test_the_live_refusal_is_now_allowed(self):
        # cache_n=70718 prompt_n=1747 uncovered=1747 allowance=725 -- refused on every load
        # (REQ-100). The manifest declared coverage 70,718 of 72,465, so 1,747 IS the tail.
        self.assertGreaterEqual(uncovered_allowance(72_465, 70_718), 1_747)

    def test_a_checkpoint_at_the_end_keeps_the_floor(self):
        # pos_max at the last token: nothing past the checkpoint, floor of 8 still applies
        self.assertEqual(uncovered_allowance(29_470, 29_470), 8)

    def test_a_restore_that_covered_almost_nothing_is_still_refused_exactly(self):
        # Declared coverage is what the manifest promised; a restore reusing far less than
        # that fails the coverage-equality check before the tail is ever consulted, and the
        # tail allowance must not be so loose that it would pass on its own.
        self.assertLess(uncovered_allowance(9_146, 9_100), 9_146 - 2_000)

    def test_declared_coverage_outside_the_artifact_is_refused(self):
        with self.assertRaises(ValueError):
            uncovered_allowance(100, 101)
        with self.assertRaises(ValueError):
            uncovered_allowance(100, -1)

    def test_without_a_declaration_the_proportional_bound_remains(self):
        self.assertEqual(uncovered_allowance(9_146), max(8, 92))


class LongTailRestoreTest(AdmittedPathTest):
    """End to end: a restore whose tail is longer than 1% but exactly what the manifest
    declared must pass -- this is the shape of the live refusal, in miniature."""

    def _long_state(self, n_tokens=1000, covered=950):
        tokens = list(range(1, n_tokens + 1))
        appendix = sckp_appendix(n_tokens=covered, pos_min=0, pos_max=covered - 1)
        body = ggsq_body(tokens=tokens, version=3, trailer=appendix)
        raw = self.dir / "long.bin"
        raw.write_bytes(body)
        save = {"n_written": len(body), "checkpoint_bytes": len(appendix),
                "checkpoint_n_tokens": covered, "checkpoint_pos_min": 0,
                "checkpoint_pos_max": covered - 1, "n_checkpoints_saved": 1,
                "n_saved": n_tokens}
        return raw, save, tokens

    def test_a_tail_past_one_percent_but_equal_to_the_declared_gap_passes(self):
        raw, save, tokens = self._long_state(1000, 950)
        self.raw, self.save = raw, save
        path, _ = self.path_for()
        obj = path.admit(raw, model="", token_ids=tokens, save_response=save)
        # 50 uncovered of 1,000: the old 1% rule allowed 10 and refused this.
        path, adapter = self.path_for(probe={"timings": {"cache_n": 950, "prompt_n": 50}})
        report = path.restore(obj.digest, model="", token_ids=tokens)
        self.assertTrue(report.ok, report.reason)
        self.assertEqual((report.cache_n, report.prompt_n), (950, 50))

    def test_a_tail_longer_than_the_declared_gap_still_refuses(self):
        raw, save, tokens = self._long_state(1000, 950)
        self.raw, self.save = raw, save
        path, _ = self.path_for()
        obj = path.admit(raw, model="", token_ids=tokens, save_response=save)
        # runtime reports it reused 950 but wants to reprocess 60: more than exists past
        # the checkpoint, so the numbers do not describe this artifact.
        path, adapter = self.path_for(probe={"timings": {"cache_n": 950, "prompt_n": 60}})
        report = path.restore(obj.digest, model="", token_ids=tokens)
        self.assertFalse(report.ok)
        self.assertIn("tail contract violated", report.reason)


class CancelledRestoreTest(AdmittedPathTest):
    """REQ-114: a caller that hung up gets no more work done on its behalf."""

    def test_gone_before_the_probe_refuses_and_probes_nothing(self):
        obj = self.admitted()
        path, adapter = self.path_for()
        report = path.restore(obj.digest, model="", token_ids=TOKENS, cancelled=lambda: True)
        self.assertFalse(report.ok)
        self.assertIn("caller gone", report.reason)
        self.assertNotIn("/completion", report.calls)
        self.assertEqual(report.phases.get("aborted_before"), "reuse_probe")

    def test_gone_after_the_probe_keeps_the_verified_prefix_and_skips_the_pristine_restore(self):
        obj = self.admitted()
        path, adapter = self.path_for()
        polls = []
        def gone():                      # first poll (before probe) no, second (before pristine) yes
            polls.append(1); return len(polls) >= 2
        report = path.restore(obj.digest, model="", token_ids=TOKENS, cancelled=gone)
        self.assertTrue(report.ok, report.reason)          # reuse WAS verified
        self.assertEqual(report.cache_n, 3)
        self.assertIn("/completion", report.calls)
        self.assertEqual(report.calls.count("/slots/0?action=restore"), 1,
                         "no second (pristine) restore after the caller left")
        self.assertNotIn("/slots/0?action=erase", report.calls)
        self.assertEqual(report.phases.get("aborted_before"), "pristine_restore")
