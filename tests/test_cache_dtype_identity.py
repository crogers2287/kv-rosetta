"""The K/V cache dtype must be bound into cache identity, and withheld when unknown.

Two launches that differ only in K/V cache type produce caches that are not interchangeable,
yet /props reported no cache dtype at all, so CacheABIIdentity carried empty strings and both
launches hashed to the same digest. Weight quantization is not a substitute: a Q4 model can
run an f16, q8_0, or quantized KV cache.

An empty value is safer than a confident wrong one - a wrong value compares equal across
genuinely different caches - so an unadvertised dtype withholds capability rather than
falling back.
"""

import tempfile
import unittest
from pathlib import Path

from kv_rosetta.adapters.base import (
    AdapterError,
    ExportRequest,
    ImportRequest,
    Representation,
)
from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter
from tests.test_hybrid_export_gate import StubAdapter, ggsq_body, props, save_with_appendix
from tests.test_hybrid_support_predicate import RestoreStub, TARGET_ONLY


def build(runtime_props, cls=StubAdapter, **kwargs):
    directory = Path(tempfile.mkdtemp())
    slots = directory / "slots"
    slots.mkdir()
    return cls(runtime_props, slots, **kwargs), directory


def real_abi(runtime_props):
    return LlamaCppHTTPAdapter.cache_abi_identity(build(runtime_props)[0])


class CacheDtypeIdentityTest(unittest.TestCase):
    # -- 1 and 2: each axis independently changes the ABI --------------------------

    def test_differing_k_dtype_changes_the_cache_abi(self):
        f16 = real_abi(props(**TARGET_ONLY))
        q8 = real_abi(props(target_cache_type_k="q8_0", **TARGET_ONLY))
        self.assertNotEqual(f16.digest(), q8.digest())
        self.assertEqual(f16.k_dtype, "f16")
        self.assertEqual(q8.k_dtype, "q8_0")

    def test_differing_v_dtype_changes_the_cache_abi(self):
        f16 = real_abi(props(**TARGET_ONLY))
        q8 = real_abi(props(target_cache_type_v="q8_0", **TARGET_ONLY))
        self.assertNotEqual(f16.digest(), q8.digest())
        self.assertEqual(q8.v_dtype, "q8_0")

    # -- 3: an unadvertised dtype withholds capability ------------------------------

    def test_missing_dtype_yields_empty_capabilities(self):
        for missing in ("target_cache_type_k", "target_cache_type_v"):
            with self.subTest(missing=missing):
                runtime = props(**TARGET_ONLY)
                del runtime[missing]
                caps = build(runtime)[0].capabilities()
                self.assertEqual(caps.export, frozenset())
                self.assertEqual(caps.import_, frozenset())

    def test_empty_dtype_string_is_treated_as_missing(self):
        caps = build(props(target_cache_type_k="", **TARGET_ONLY))[0].capabilities()
        self.assertEqual(caps.export, frozenset())

    # -- 4: refusal happens before any state endpoint -------------------------------

    def test_missing_dtype_refuses_export_before_any_save(self):
        save, body = save_with_appendix()
        runtime = props(**TARGET_ONLY)
        del runtime["target_cache_type_k"]
        adapter, directory = build(runtime, save=save, body=body)
        adapter.posts.clear()
        with self.assertRaises(AdapterError) as caught:
            adapter.export(ExportRequest(model="", out_path=directory / "x.kvx",
                                         representation=Representation.OPAQUE))
        self.assertIn("cache type", str(caught.exception).lower())
        self.assertEqual([p for p, _ in adapter.posts], [])

    def test_missing_dtype_refuses_import_before_any_restore(self):
        from tests.test_compound_format_agreement import genuine_artifact

        directory = Path(tempfile.mkdtemp())
        artifact, save = genuine_artifact(directory)
        runtime = props(**TARGET_ONLY)
        del runtime["target_cache_type_v"]
        slots = directory / "import-slots"
        slots.mkdir()
        adapter = RestoreStub(runtime, slots, save=dict(save),
                              body=ggsq_body(trailer=b""), restore={})
        report = adapter.import_(artifact, ImportRequest(model="", slot=0))
        self.assertFalse(report.ok)
        self.assertEqual(adapter.restore_posts(), [])

    # -- 5: weight quantization must not populate cache dtype -----------------------

    def test_weight_quantization_cannot_fill_the_cache_dtype(self):
        runtime = props(**TARGET_ONLY)
        del runtime["target_cache_type_k"]
        del runtime["target_cache_type_v"]
        runtime["model_ftype"] = "Q4_K - Medium"
        runtime["default_generation_settings"] = dict(
            runtime["default_generation_settings"], type_k="Q4_K")
        abi = real_abi(runtime)
        self.assertEqual(abi.k_dtype, "")
        self.assertEqual(abi.v_dtype, "")
        self.assertEqual(build(runtime)[0].capabilities().export, frozenset())

    # -- a differing runtime refuses the artifact on identity -----------------------

    def test_a_runtime_with_another_k_dtype_refuses_the_artifact(self):
        from tests.test_compound_format_agreement import genuine_artifact

        directory = Path(tempfile.mkdtemp())
        artifact, save = genuine_artifact(directory)
        slots = directory / "other-slots"
        slots.mkdir()
        adapter = RestoreStub(props(target_cache_type_k="q8_0", **TARGET_ONLY), slots,
                              save=dict(save), body=ggsq_body(trailer=b""), restore={})
        report = adapter.import_(artifact, ImportRequest(model="", slot=0))
        self.assertFalse(report.ok)
        self.assertIn("cache ABI", report.reason)
        self.assertEqual(adapter.restore_posts(), [])

    def test_the_same_dtypes_still_import(self):
        from tests.test_compound_format_agreement import genuine_artifact

        directory = Path(tempfile.mkdtemp())
        artifact, save = genuine_artifact(directory)
        slots = directory / "same-slots"
        slots.mkdir()
        restore = {"n_restored": 263,
                   "n_checkpoints_restored": save["n_checkpoints_saved"],
                   "checkpoint_bytes": save["checkpoint_bytes"],
                   "checkpoint_n_tokens": save["checkpoint_n_tokens"],
                   "checkpoint_pos_min": save["checkpoint_pos_min"],
                   "checkpoint_pos_max": save["checkpoint_pos_max"]}
        adapter = RestoreStub(props(**TARGET_ONLY), slots, save=dict(save),
                              body=ggsq_body(trailer=b""), restore=restore)
        report = adapter.import_(artifact, ImportRequest(model="", slot=0))
        self.assertTrue(report.ok, report.reason)


if __name__ == "__main__":
    unittest.main()
