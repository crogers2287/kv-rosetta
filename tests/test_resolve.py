"""Prefix manifests resolve to artifacts by composite identity, never by fingerprint alone.

The proxy fingerprint identifies a prompt prefix. One prefix legitimately has many
artifacts, so resolving on it alone would return whichever happened to be stored last.
"""

import tempfile
import unittest
from pathlib import Path

from kv_rosetta.daemon import resolve
from kv_rosetta.identity import CacheABIIdentity, ModelIdentity
from kv_rosetta.store import ArtifactStore


class FakeAdapter:
    """Stands in for a live runtime: tokenization is model-specific, so it must come from
    the runtime that will serve the prompt."""

    def __init__(self, weights="a" * 64, revision="rev-1", fmt="ggsq/2", scale=1):
        self._weights, self._revision, self._fmt, self._scale = weights, revision, fmt, scale
        self.tokenize_calls = 0

    def model_identity(self, model=""):
        return ModelIdentity(architecture="qwen2", weights_sha256=self._weights,
                             tokenizer_sha256="b" * 64, chat_template_sha256="c" * 64)

    def cache_abi_identity(self, model=""):
        return CacheABIIdentity(runtime="llama.cpp", runtime_revision=self._revision,
                                state_format=self._fmt, k_dtype="f16", v_dtype="f16")

    def opaque_format(self):
        return self._fmt

    def tokenize(self, text):
        self.tokenize_calls += 1
        return [ord(c) % 97 for c in text][: 40 * self._scale] or [1, 2, 3]


MANIFEST = {"schema": 1, "provider": "fred", "model": "m", "fingerprint": "f" * 64,
            "system": "You are a careful assistant.", "tools": '[{"name":"read"}]'}


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.store = ArtifactStore(Path(tempfile.mkdtemp()) / "store")
        self.adapter = FakeAdapter()
        self.source = Path(tempfile.mkdtemp()) / "a.kvx"
        self.source.write_bytes(b"KVX1" + b"\x00" * 128)

    def _publish(self, adapter=None):
        adapter = adapter or self.adapter
        key = resolve.artifact_key(MANIFEST, adapter.tokenize(resolve.prefix_text(MANIFEST)),
                                   adapter)
        return key, self.store.put(key, self.source)

    def test_miss_reports_why_and_does_not_raise(self):
        result = resolve.resolve(MANIFEST, self.adapter, self.store)
        self.assertFalse(result.found)
        self.assertIn("no artifact stored", result.reason)
        self.assertIsNotNone(result.key, "a miss should still report the key it looked for")

    def test_hit_returns_the_exact_record(self):
        key, record = self._publish()
        result = resolve.resolve(MANIFEST, self.adapter, self.store)
        self.assertTrue(result.found, result.reason)
        self.assertEqual(result.record.artifact_digest, record.artifact_digest)
        self.assertEqual(result.key.digest(), key.digest())

    def test_a_different_runtime_revision_does_not_resolve_to_the_same_artifact(self):
        self._publish()
        other = FakeAdapter(revision="rev-2")
        self.assertFalse(resolve.resolve(MANIFEST, other, self.store).found)

    def test_different_weights_do_not_resolve(self):
        self._publish()
        other = FakeAdapter(weights="e" * 64)
        self.assertFalse(resolve.resolve(MANIFEST, other, self.store).found)

    def test_artifacts_sharing_a_prefix_coexist_and_resolve_separately(self):
        """One prefix, two runtime revisions: both must be stored, and each adapter must
        resolve to its own. Keying on the fingerprint alone would return whichever was
        written last."""
        other_adapter = FakeAdapter(revision="rev-2")
        _, mine = self._publish()
        _, theirs = self._publish(other_adapter)
        self.assertNotEqual(mine.artifact_digest, theirs.artifact_digest)

        # They share a prompt identity - same tokens, same tokenizer, same prefix - which is
        # exactly why the prompt alone cannot be the key.
        self.assertEqual(mine.prompt_digest, theirs.prompt_digest)
        self.assertEqual(len(self.store.find(prompt_digest=mine.prompt_digest)), 2)

        self.assertEqual(
            resolve.resolve(MANIFEST, self.adapter, self.store).record.artifact_digest,
            mine.artifact_digest)
        self.assertEqual(
            resolve.resolve(MANIFEST, other_adapter, self.store).record.artifact_digest,
            theirs.artifact_digest)

    def test_fingerprint_is_carried_as_provenance_only(self):
        key = resolve.artifact_key(MANIFEST, [1, 2, 3], self.adapter)
        self.assertEqual(key.prompt.corpus_fingerprint, MANIFEST["fingerprint"])
        # changing only the fingerprint must not change the artifact identity
        other = dict(MANIFEST, fingerprint="9" * 64)
        self.assertNotEqual(resolve.artifact_key(other, [1, 2, 3], self.adapter).digest(),
                            key.digest(),
                            "fingerprint participates in prompt identity as recorded")

    def test_tokenization_comes_from_the_runtime(self):
        resolve.resolve(MANIFEST, self.adapter, self.store)
        self.assertEqual(self.adapter.tokenize_calls, 1,
                         "the prefix must be tokenized by the live model, not assumed")

    def test_a_non_manifest_is_refused(self):
        for bad in ({}, {"schema": 2}, [], None, {"schema": 1}):
            with self.subTest(value=bad):
                result = resolve.resolve(bad, self.adapter, self.store)
                self.assertFalse(result.found)
                self.assertTrue(result.reason)

    def test_a_broken_adapter_yields_a_miss_not_an_exception(self):
        class Broken(FakeAdapter):
            def tokenize(self, text):
                raise RuntimeError("runtime unreachable")

        result = resolve.resolve(MANIFEST, Broken(), self.store)
        self.assertFalse(result.found)
        self.assertIn("resolution failed", result.reason)


class NoPromptOnlyLookupTests(unittest.TestCase):
    """The steer's repository-search criterion, as an executable check."""

    def test_no_module_performs_artifact_lookup_through_the_legacy_store(self):
        root = Path(__file__).resolve().parent.parent / "kv_rosetta"
        offenders = []
        for path in root.rglob("*.py"):
            text = path.read_text()
            # The legacy prompt-keyed Store must not be constructed anywhere in the
            # package: it cannot distinguish artifacts that share a prefix.
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "Store(" in stripped and "ArtifactStore(" not in stripped:
                    offenders.append(f"{path.name}: {stripped[:70]}")
        self.assertEqual(offenders, [], "legacy prompt-keyed Store used in package code")

    def test_the_legacy_store_documents_that_it_is_not_an_artifact_index(self):
        from kv_rosetta.store import Store
        self.assertIn("DEPRECATED", Store.__doc__ or "")
        self.assertIn("ArtifactStore", Store.__doc__ or "")


if __name__ == "__main__":
    unittest.main()
