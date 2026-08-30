"""The admitted restore path must refuse everything the KVX import would, before restoring.

It skips reading the payload on the request path, so every check that a full verification
would have made has to happen at admission or at resolution. These tests pin the refusals and
measure the read count, because "no payload copy" is a claim that has to be substantiated
rather than asserted.
"""

import struct
import tempfile
import unittest
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

    def test_a_checkpointed_artifact_is_marked_as_needing_the_patch(self):
        from kv_rosetta.requirements import Requirements, check
        recorded = requirements_for(checkpoints=2, version=3)
        self.assertTrue(recorded["needs_checkpoint_persistence"])
        self.assertTrue(check(Requirements(**recorded), {"sequence_state_version": 3}))

    def test_an_artifact_without_checkpoints_needs_no_patch(self):
        from kv_rosetta.requirements import Requirements, check
        recorded = requirements_for(checkpoints=0, version=3)
        self.assertFalse(recorded["needs_checkpoint_persistence"])
        self.assertEqual(check(Requirements(**recorded), {"sequence_state_version": 3}), [])

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
