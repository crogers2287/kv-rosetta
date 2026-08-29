"""capabilities(), export() and import_() must agree on one support decision.

Three surfaces answered the question independently, so they disagreed: capabilities()
advertised OPAQUE whenever a checkpoint protocol was complete, while export() additionally
required a tested compound tuple and a provably target-only configuration. A caller who
trusted capabilities() and then called export() got a refusal; a caller who called import_()
got neither check. These tests pin all three to the same predicate.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from kv_rosetta.adapters.base import (
    AdapterError,
    ExportRequest,
    ImportRequest,
    Representation,
)
from tests.test_hybrid_export_gate import (
    StubAdapter,
    ggsq_body,
    props,
    save_with_appendix,
    sckp_appendix,
)


class RestoreStub(StubAdapter):
    """Records restore/erase POSTs and answers with configurable metadata."""

    def __init__(self, *args, restore=None, probe=None, pristine_fails=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._restore = restore
        self._probe = probe or {"timings": {"cache_n": 3, "prompt_n": 1}}
        self._pristine_fails = pristine_fails
        self._restores = 0

    def _post(self, path, payload, timeout=None):
        if "action=restore" in path:
            self.posts.append((path, dict(payload)))
            self._restores += 1
            if self._pristine_fails and self._restores > 1:
                raise AdapterError("simulated failed re-restore")
            return dict(self._restore or {})
        if path == "/completion":
            self.posts.append((path, {"n_predict": payload.get("n_predict")}))
            return dict(self._probe)
        return super()._post(path, payload, timeout)

    def restore_posts(self):
        return [p for p, _ in self.posts if "action=restore" in p]


def build(runtime_props, cls=StubAdapter, **kwargs):
    directory = Path(tempfile.mkdtemp())
    slots = directory / "slots"
    slots.mkdir()
    return cls(runtime_props, slots, **kwargs), directory / "artifact.kvx"


TARGET_ONLY = dict(active_checkpoint_state_classes=["target"])


class SupportPredicateTest(unittest.TestCase):
    def caps(self, runtime_props, **kwargs):
        adapter, _ = build(runtime_props, **kwargs)
        return adapter.capabilities()

    def exports(self, runtime_props, **kwargs) -> bool:
        """Whether export() accepts, so capability and export can be compared."""
        adapter, out = build(runtime_props, **kwargs)
        try:
            adapter.export(ExportRequest(model="", out_path=out,
                                         representation=Representation.OPAQUE))
            return True
        except AdapterError:
            return False

    # -- 1. active draft/speculative yields empty capabilities ----------------------

    def test_active_draft_state_yields_no_capabilities(self):
        caps = self.caps(props(active_checkpoint_state_classes=["target", "draft"]))
        self.assertEqual(caps.export, frozenset())
        self.assertEqual(caps.import_, frozenset())

    def test_active_speculative_state_yields_no_capabilities(self):
        caps = self.caps(props(active_checkpoint_state_classes=["target", "speculative"]))
        self.assertEqual(caps.export, frozenset())
        self.assertEqual(caps.import_, frozenset())

    def test_unreported_active_classes_yield_no_capabilities(self):
        # A runtime that does not say what it requires cannot be proven target-only.
        caps = self.caps(props())
        self.assertEqual(caps.export, frozenset())

    def test_target_only_configuration_is_advertised(self):
        caps = self.caps(props(**TARGET_ONLY))
        self.assertIn(Representation.OPAQUE, caps.export)
        self.assertIn(Representation.OPAQUE, caps.import_)

    # -- 3. the tuple allowlist reaches capabilities too ---------------------------

    def test_untested_compound_tuple_is_not_advertised(self):
        caps = self.caps(props(sequence_state_version=2, **TARGET_ONLY))
        self.assertEqual(caps.export, frozenset())

    # -- capability and export never disagree --------------------------------------

    def test_capability_and_export_agree(self):
        cases = {
            "target-only": props(**TARGET_ONLY),
            "draft-active": props(active_checkpoint_state_classes=["target", "draft"]),
            "spec-active": props(active_checkpoint_state_classes=["target", "speculative"]),
            "classes-unreported": props(),
            "untested-tuple": props(sequence_state_version=2, **TARGET_ONLY),
            "no-protocol": {"default_generation_settings": {"type_k": "f16"},
                            "build_info": "bstub-0000000"},
        }
        for name, runtime_props in cases.items():
            with self.subTest(case=name):
                save, body = save_with_appendix()
                advertised = Representation.OPAQUE in self.caps(runtime_props).export
                accepted = self.exports(runtime_props, save=save, body=body)
                self.assertEqual(advertised, accepted,
                                 f"{name}: capabilities said {advertised}, export said "
                                 f"{accepted}")

    # -- the identity binds the configuration --------------------------------------

    def real_abi(self, runtime_props):
        from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter
        return LlamaCppHTTPAdapter.cache_abi_identity(build(runtime_props)[0])

    def test_active_classes_are_bound_into_the_cache_abi(self):
        target = self.real_abi(props(**TARGET_ONLY))
        draft = self.real_abi(props(active_checkpoint_state_classes=["target", "draft"]))
        self.assertNotEqual(target.digest(), draft.digest(),
                            "a launch requiring draft state must not share a cache ABI "
                            "identity with a target-only launch")
        self.assertIn("active:target", target.flags)

    def test_compound_tuple_is_bound_into_the_cache_abi(self):
        v3 = self.real_abi(props(**TARGET_ONLY))
        v2 = self.real_abi(props(sequence_state_version=2, **TARGET_ONLY))
        self.assertNotEqual(v3.digest(), v2.digest())


if __name__ == "__main__":
    unittest.main()


def make_artifact(directory: Path, **save_overrides) -> tuple[Path, dict]:
    """Export a real compound artifact through the adapter, so import sees a real file."""
    save, body = save_with_appendix(**save_overrides)
    slots = directory / "export-slots"
    slots.mkdir()
    adapter = StubAdapter(props(**TARGET_ONLY), slots, save=save, body=body)
    out = directory / "artifact.kvx"
    adapter.export(ExportRequest(model="", out_path=out,
                                 representation=Representation.OPAQUE))
    return out, save


class HybridImportGateTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.artifact, self.save = make_artifact(
            self.directory, checkpoint_n_tokens=3, checkpoint_pos_max=2)

    def importer(self, runtime_props, **kwargs):
        slots = self.directory / f"slots{len(list(self.directory.iterdir()))}"
        slots.mkdir()
        restore = kwargs.pop("restore", None)
        if restore is None:
            restore = {"n_restored": 263,
                       "n_checkpoints_restored": self.save["n_checkpoints_saved"],
                       "checkpoint_bytes": self.save["checkpoint_bytes"],
                       "checkpoint_n_tokens": self.save["checkpoint_n_tokens"],
                       "checkpoint_pos_min": self.save["checkpoint_pos_min"],
                       "checkpoint_pos_max": self.save["checkpoint_pos_max"]}
        return RestoreStub(runtime_props, slots, save=dict(self.save),
                           body=ggsq_body(trailer=sckp_appendix()),
                           restore=restore, **kwargs)

    def run_import(self, adapter, **import_kwargs):
        return adapter.import_(self.artifact, ImportRequest(model="", slot=0),
                               **import_kwargs)

    # -- 2. an unsupported configuration refuses before any restore POST -----------

    def test_active_draft_configuration_refuses_before_restore(self):
        adapter = self.importer(props(active_checkpoint_state_classes=["target", "draft"]))
        report = self.run_import(adapter)
        self.assertFalse(report.ok)
        self.assertEqual(adapter.restore_posts(), [],
                         "refused only after staging state into the runtime")

    def test_untested_tuple_refuses_before_restore(self):
        adapter = self.importer(props(sequence_state_version=2, **TARGET_ONLY))
        report = self.run_import(adapter)
        self.assertFalse(report.ok)
        self.assertEqual(adapter.restore_posts(), [])

    def test_unpatched_runtime_refuses_before_restore(self):
        adapter = self.importer({"default_generation_settings": {"type_k": "f16"},
                                 "build_info": "bstub-0000000"})
        report = self.run_import(adapter)
        self.assertFalse(report.ok)
        self.assertEqual(adapter.restore_posts(), [])

    # -- 4. every restore-metadata field must match, independently -----------------

    def test_each_restore_metadata_mismatch_is_refused(self):
        fields = {"n_checkpoints_restored": 99, "checkpoint_bytes": 1,
                  "checkpoint_n_tokens": 7, "checkpoint_pos_min": 5,
                  "checkpoint_pos_max": 9}
        for field, wrong in fields.items():
            with self.subTest(field=field):
                adapter = self.importer(props(**TARGET_ONLY))
                base = {"n_restored": 263,
                        "n_checkpoints_restored": self.save["n_checkpoints_saved"],
                        "checkpoint_bytes": self.save["checkpoint_bytes"],
                        "checkpoint_n_tokens": self.save["checkpoint_n_tokens"],
                        "checkpoint_pos_min": self.save["checkpoint_pos_min"],
                        "checkpoint_pos_max": self.save["checkpoint_pos_max"]}
                adapter._restore = dict(base, **{field: wrong})
                report = self.run_import(adapter)
                self.assertFalse(report.ok, f"{field} mismatch was accepted")
                self.assertIn("does not match the manifest", report.reason)
                self.assertIn(str(wrong), report.reason)

    # -- 6. verify_reuse=False cannot bless a compound import ----------------------

    def test_verify_reuse_false_cannot_bypass_the_proof(self):
        adapter = self.importer(props(**TARGET_ONLY),
                                probe={"timings": {"cache_n": 0, "prompt_n": 4}})
        report = self.run_import(adapter, verify_reuse=False)
        self.assertFalse(report.ok,
                         "verify_reuse=False blessed a compound import that reused nothing")

    # -- 7. a failed pristine re-restore cannot return ok=true ---------------------

    def test_failed_pristine_restore_is_not_reported_ok(self):
        adapter = self.importer(props(**TARGET_ONLY), pristine_fails=True)
        report = self.run_import(adapter)
        self.assertFalse(report.ok)
        self.assertIn("pristine", report.reason)
