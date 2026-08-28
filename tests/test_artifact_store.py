"""Composite artifact identity — the steer's P0 #1 acceptance criteria.

A prompt fingerprint identifies a reusable prefix, not an artifact. These tests pin the
consequences of that distinction.
"""

import os
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from kv_rosetta.identity import ArtifactKey, CacheABIIdentity, ModelIdentity, PromptIdentity
from kv_rosetta.store import ArtifactStore, IdentityError, StoreError


def _prompt():
    return PromptIdentity(tokenizer_id="tok", token_ids_sha256="d" * 64, token_count=10)


def _model(weights="a" * 64):
    return ModelIdentity(architecture="qwen", weights_sha256=weights,
                         tokenizer_sha256="b" * 64, chat_template_sha256="c" * 64)


def _abi(**kw):
    base = dict(runtime="llama.cpp", runtime_revision="ca3d5a3", state_format="ggsq/3",
                k_dtype="f16", v_dtype="f16", rope_kind="normal", rope_base=1e6)
    base.update(kw)
    return CacheABIIdentity(**base)


class CompositeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "store"
        self.store = ArtifactStore(self.root)
        self.src = Path(tempfile.mkdtemp()) / "a.kvx"
        self.src.write_bytes(b"KVX1" + b"\x00" * 60)

    def _keys(self):
        prompt, model = _prompt(), _model()
        return {
            "cuda_opaque": ArtifactKey(prompt, model, _abi(), encoding="opaque"),
            "hip_opaque": ArtifactKey(prompt, model, _abi(runtime_revision="hip"), encoding="opaque"),
            "canonical": ArtifactKey(prompt, model, _abi(state_format=""), encoding="raw"),
            "translated": ArtifactKey(prompt, model, _abi(state_format=""), encoding="raw",
                                      mapper_id="ridge/1"),
        }

    def test_four_representations_of_one_prompt_coexist(self):
        keys = self._keys()
        for key in keys.values():
            self.store.put(key, self.src)
        # All four share a prompt but must be independently retrievable.
        self.assertEqual(len({k.digest() for k in keys.values()}), 4)
        for name, key in keys.items():
            with self.subTest(representation=name):
                self.assertIsNotNone(self.store.get(key))
        found = self.store.find(prompt_digest=_prompt().digest())
        self.assertEqual(len(found), 4)

    def test_lookup_narrows_by_encoding_and_mapper(self):
        keys = self._keys()
        for key in keys.values():
            self.store.put(key, self.src)
        self.assertEqual(len(self.store.find(encoding="opaque")), 2)
        self.assertEqual(len(self.store.find(mapper_id="ridge/1")), 1)
        self.assertEqual(len(self.store.find(mapper_id="")), 3)

    def test_different_weights_behind_the_same_name_are_a_different_identity(self):
        prompt = _prompt()
        a = ArtifactKey(prompt, _model("a" * 64), _abi())
        b = ArtifactKey(prompt, _model("f" * 64), _abi())
        self.assertNotEqual(a.digest(), b.digest())

    def test_identical_weights_at_another_path_are_the_same_identity(self):
        # ModelIdentity carries no path, so relocation cannot change identity.
        self.assertNotIn("path", ModelIdentity.__dataclass_fields__)
        self.assertEqual(_model().digest(), _model().digest())

    def test_cache_abi_difference_alone_creates_a_separate_artifact(self):
        prompt, model = _prompt(), _model()
        f16 = ArtifactKey(prompt, model, _abi(k_dtype="f16"), encoding="opaque")
        q8 = ArtifactKey(prompt, model, _abi(k_dtype="q8_0"), encoding="opaque")
        self.store.put(f16, self.src)
        self.store.put(q8, self.src)
        self.assertNotEqual(f16.digest(), q8.digest())
        self.assertEqual(len(self.store.find(prompt_digest=prompt.digest())), 2)

    def test_artifact_digest_is_never_truncated_in_the_path(self):
        key = self._keys()["canonical"]
        record = self.store.put(key, self.src)
        self.assertEqual(Path(record.path).stem, key.digest())
        self.assertEqual(len(Path(record.path).stem), 64)

    def test_paths_stay_inside_the_store_root(self):
        key = self._keys()["canonical"]
        self.assertTrue(str(self.store.path_for(key)).startswith(str(self.root.resolve())))

    def test_a_malformed_digest_cannot_escape_the_root(self):
        class Evil:
            prompt = _prompt()
            model = _model()
            cache_abi = _abi()
            encoding = "raw"
            format_version = ""
            representation_digest = ""
            mapper_id = ""

            def digest(self):
                return "../../../../tmp/pwned"

        with self.assertRaises(IdentityError):
            self.store.path_for(Evil())

    def test_put_twice_preserves_created(self):
        key = self._keys()["canonical"]
        first = self.store.put(key, self.src, now=1000.0)
        second = self.store.put(key, self.src, now=2000.0)
        self.assertEqual(second.created, first.created)
        self.assertGreater(second.last_used, first.last_used)

    def test_record_for_a_missing_file_is_dropped_not_served(self):
        key = self._keys()["canonical"]
        record = self.store.put(key, self.src)
        os.unlink(record.path)
        self.assertIsNone(self.store.get(key))
        self.assertEqual(self.store.find(prompt_digest=_prompt().digest()), ())

    def test_a_failed_deletion_keeps_the_record(self):
        """Dropping the row after a failed unlink orphans the file permanently."""
        key = self._keys()["canonical"]
        record = self.store.put(key, self.src)
        parent = Path(record.path).parent
        mode = parent.stat().st_mode
        os.chmod(parent, stat.S_IRUSR | stat.S_IXUSR)   # read-only dir: unlink must fail
        try:
            with self.assertRaises(StoreError):
                self.store.delete(key)
        finally:
            os.chmod(parent, mode)
        self.assertIsNotNone(self.store.get(key))

    def test_delete_removes_row_and_file(self):
        key = self._keys()["canonical"]
        record = self.store.put(key, self.src)
        self.assertTrue(self.store.delete(key))
        self.assertFalse(Path(record.path).exists())
        self.assertFalse(self.store.delete(key))

    def test_missing_source_is_refused(self):
        with self.assertRaises(StoreError):
            self.store.put(self._keys()["canonical"], self.src.parent / "absent.kvx")

    def test_a_second_store_on_the_same_root_sees_the_rows(self):
        key = self._keys()["canonical"]
        self.store.put(key, self.src)
        with ArtifactStore(self.root) as other:
            self.assertIsNotNone(other.get(key.digest()))


if __name__ == "__main__":
    unittest.main()
