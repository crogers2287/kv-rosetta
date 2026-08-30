"""Guards that a mutation audit found no test would notice the removal of.

Each test here disappeared into a green suite when its guard was deleted, which means the
guard was documentation rather than behaviour. The path-containment and file-type cases are
the ones that matter: they are the store's defence against a symlink or a directory
substituted for an object it is about to hand to a server.
"""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kv_rosetta import dtypes, segments
from kv_rosetta.admitted_store import AdmissionError, AdmittedStore
from kv_rosetta.hybrid_schema import HybridState, SchemaError
from kv_rosetta.identity import (ArtifactKey, CacheABIIdentity, IdentityError,
                                 ModelIdentity, PromptIdentity, require_digest)
# store.IdentityError is a StoreError subclass, distinct from identity.IdentityError;
# the containment guards raise the store one.
from kv_rosetta.store import IdentityError as StoreIdentityError
from kv_rosetta.manifest import ManifestError, ModelABI, load
from kv_rosetta.mappers.linear import MapperError, fit_ridge, select_source_layer
from kv_rosetta.store import ArtifactStore, Store


def _manifest(**over):
    value = {
        "schema": "kvx/0.1",
        "model": {"architecture": "qwen", "weights_id": "qwen-a"},
        "prompt": {"tokenizer_id": "tok", "token_ids_sha256": "1" * 64, "token_count": 8},
        "kv": {"layers": 4, "heads": 2, "head_dim": 8, "dtype": "f16",
               "layout": "layer,kv,token,head,dim", "byte_order": "little"},
        "position": {"kind": "rope", "theta": 1000000},
    }
    value.update(over)
    return value


class ManifestFieldGuards(unittest.TestCase):
    """Malformed manifest fields are refused with a message naming the field."""

    def test_section_that_is_not_an_object_is_refused(self):
        # A JSON list where a section belongs must not be indexed as a mapping.
        with self.assertRaises(ManifestError) as caught:
            ModelABI.from_manifest(_manifest(kv=["layers", 4]))
        self.assertIn("kv", str(caught.exception))

    def test_blank_text_field_is_refused(self):
        # "" and "   " are not identities; accepting them would fingerprint two different
        # models identically.
        for blank in ("", "   "):
            m = _manifest()
            m["model"]["architecture"] = blank
            with self.assertRaises(ManifestError):
                ModelABI.from_manifest(m)

    def test_non_positive_and_boolean_token_count_are_refused(self):
        # bool is a subclass of int in Python: True must not pass as a token count.
        for bad in (0, -1, True, 1.5, "8"):
            m = _manifest()
            m["prompt"]["token_count"] = bad
            with self.assertRaises(ManifestError):
                ModelABI.from_manifest(m)

    def test_unknown_kv_dtype_is_refused(self):
        m = _manifest()
        m["kv"]["dtype"] = "int4"
        with self.assertRaises(ManifestError) as caught:
            ModelABI.from_manifest(m)
        self.assertIn("dtype", str(caught.exception))

    def test_big_endian_byte_order_is_refused(self):
        m = _manifest()
        m["kv"]["byte_order"] = "big"
        with self.assertRaises(ManifestError) as caught:
            ModelABI.from_manifest(m)
        self.assertIn("byte_order", str(caught.exception))

    def test_manifest_file_whose_root_is_not_an_object_is_refused(self):
        path = Path(tempfile.mkdtemp()) / "m.json"
        path.write_text(json.dumps([_manifest()]))
        with self.assertRaises(ManifestError) as caught:
            load(path)
        self.assertIn("object", str(caught.exception))


class DtypeGuards(unittest.TestCase):
    def test_to_numpy_refuses_an_unregistered_dtype(self):
        # Quantised types are opaque-only; decoding one as raw would produce garbage
        # tensors rather than an error.
        with self.assertRaises(dtypes.DtypeError):
            dtypes.to_numpy(b"\x00" * 8, "q8_0", (4,))


class SegmentGuards(unittest.TestCase):
    def test_segment_from_a_non_object_is_refused(self):
        # The message matters: without the isinstance check the list still fails, but in
        # the field-extraction except clause, which reports a missing field rather than a
        # malformed segment. Asserting the class alone would pass either way.
        with self.assertRaises(segments.SegmentError) as caught:
            segments.Segment.from_dict(["name", "k"])
        self.assertIn("JSON object", str(caught.exception))


class IdentityGuards(unittest.TestCase):
    def test_require_digest_refuses_anything_but_64_lowercase_hex(self):
        for bad in ("a" * 63, "a" * 65, "A" * 64, "g" * 64, "../" + "a" * 61, ""):
            with self.assertRaises(IdentityError):
                require_digest(bad)
        self.assertEqual(require_digest("a" * 64), "a" * 64)


class HybridSchemaGuards(unittest.TestCase):
    def test_geometry_that_is_not_a_mapping_is_refused(self):
        # Same shape as the segment case: the fallback except also raises SchemaError, so
        # the message is what distinguishes the guard from its absence.
        with self.assertRaises(SchemaError) as caught:
            HybridState.from_dict({"geometry": ["qwen35", 48]})
        self.assertIn("not a mapping", str(caught.exception))


class LinearMapperGuards(unittest.TestCase):
    def test_ridge_refuses_tensors_that_are_not_two_dimensional(self):
        # (tokens, width) is the contract; a stray layer axis would silently broadcast.
        # Without the rank check these inputs still raise MapperError, but from the
        # underdetermined-fit check further down, so the message is the discriminator.
        three_d = np.zeros((2, 3, 4), dtype=np.float32)
        flat = np.zeros((2, 4), dtype=np.float32)
        for bad_source, bad_target in ((three_d, flat), (flat, three_d)):
            with self.assertRaises(MapperError) as caught:
                fit_ridge(bad_source, bad_target, 1.0)
            self.assertIn("2-D", str(caught.exception))

    def test_layer_selection_refuses_a_holdout_that_leaves_nothing_to_fit(self):
        # 6 tokens of width 8: any split either has fewer rows than columns or holds
        # nothing out. Fitting here would report a residual measured on the training set.
        candidates = {0: np.zeros((6, 8), dtype=np.float32)}
        target = np.zeros((6, 8), dtype=np.float32)
        with self.assertRaises(MapperError) as caught:
            select_source_layer(target, candidates, ridge=1.0, holdout=0.2)
        self.assertIn("hold any out", str(caught.exception))


def _key():
    return ArtifactKey(
        PromptIdentity(tokenizer_id="tok", token_ids_sha256="d" * 64, token_count=10),
        ModelIdentity(architecture="qwen", weights_sha256="a" * 64,
                      tokenizer_sha256="b" * 64, chat_template_sha256="c" * 64),
        CacheABIIdentity(runtime="llama.cpp", runtime_revision="ca3d5a3",
                         state_format="ggsq/3", k_dtype="f16", v_dtype="f16",
                         rope_kind="normal", rope_base=1e6))


class StoreDigestGuard(unittest.TestCase):
    def test_store_require_digest_refuses_a_non_string_and_bad_hex(self):
        # A fingerprint arrives from a manifest written by another process; ``../../etc``
        # must not reach a path join.
        from kv_rosetta.store import require_digest as store_require_digest
        for bad in (None, 123, ["a" * 64], "../../etc", "A" * 64, "a" * 63):
            with self.assertRaises(StoreIdentityError):
                store_require_digest(bad)
        self.assertEqual(store_require_digest("f" * 64), "f" * 64)


class StoreContainmentGuards(unittest.TestCase):
    """A symlink planted inside the store must not redirect a write out of it.

    Digest validation alone does not cover this: every path component is well-formed hex,
    and the escape happens during resolution, not in the name.
    """

    def test_store_refuses_a_model_directory_symlinked_out_of_the_root(self):
        root = Path(tempfile.mkdtemp()) / "store"
        outside = Path(tempfile.mkdtemp()) / "outside"
        outside.mkdir()
        store = Store(root)
        from kv_rosetta.store import model_key
        (root / model_key("p", "m")).symlink_to(outside, target_is_directory=True)
        with self.assertRaises(StoreIdentityError) as caught:
            store.path_for("p", "m", "e" * 64)
        self.assertIn("escapes", str(caught.exception))
        store.close()

    def test_artifact_store_refuses_a_model_directory_symlinked_out_of_the_root(self):
        root = Path(tempfile.mkdtemp()) / "store"
        outside = Path(tempfile.mkdtemp()) / "outside"
        outside.mkdir()
        store = ArtifactStore(root)
        key = _key()
        (root / require_digest(key.model.digest(), "model_digest")[:16]).symlink_to(
            outside, target_is_directory=True)
        with self.assertRaises(StoreIdentityError) as caught:
            store.path_for(key)
        self.assertIn("escapes", str(caught.exception))
        store.close()


class AdmittedStoreGuards(unittest.TestCase):
    def _admit(self, root, digest, *, make):
        """Write a manifest bound to ``digest`` and build the object with ``make``."""
        obj = root / f"{digest}.state"
        make(obj)
        (root / f"{digest}.manifest.json").write_text(json.dumps(
            {"raw_sha256": digest, "file_facts": {}}))
        return obj

    def test_store_pointed_at_a_regular_file_is_refused(self):
        path = Path(tempfile.mkdtemp()) / "not-a-dir"
        path.write_bytes(b"")
        with self.assertRaises(AdmissionError) as caught:
            AdmittedStore(path, create=False)
        self.assertIn("not a directory", str(caught.exception))

    def test_object_symlinked_out_of_the_store_is_refused(self):
        root = Path(tempfile.mkdtemp()) / "admitted"
        store = AdmittedStore(root)
        outside = Path(tempfile.mkdtemp()) / "elsewhere.state"
        outside.write_bytes(b"stolen")
        digest = "b" * 64
        self._admit(root, digest, make=lambda p: p.symlink_to(outside))
        with self.assertRaises(AdmissionError) as caught:
            store.resolve(digest)
        self.assertIn("resolves outside", str(caught.exception))

    def test_object_that_is_a_directory_is_refused(self):
        # os.open(dir, O_RDONLY|O_NOFOLLOW) succeeds on Linux, so O_NOFOLLOW does not
        # cover this: only the S_ISREG check on the descriptor does.
        root = Path(tempfile.mkdtemp()) / "admitted"
        store = AdmittedStore(root)
        digest = "c" * 64
        self._admit(root, digest, make=lambda p: p.mkdir())
        with self.assertRaises(AdmissionError) as caught:
            store.resolve(digest)
        self.assertIn("not a regular file", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
