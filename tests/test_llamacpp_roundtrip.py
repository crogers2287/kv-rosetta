"""Live llama.cpp round trip: 256 tokens, exact token IDs, token-for-token parity.

Skipped unless a llama-server started with --slot-save-path is reachable:

    KVX_LLAMA_URL=http://127.0.0.1:8781 \
    KVX_LLAMA_SLOTS=/path/to/slot-save-dir \
    python3 -m unittest tests.test_llamacpp_roundtrip

This is the first test in the project that proves a real runtime cache path rather than a
property of the file format. A one-time check does not count: it is kept so that a change
to the container, the identity formula or the adapter cannot quietly break the only thing
the system exists to do.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import urllib.request

from kv_rosetta import container

from kv_rosetta import weights
from kv_rosetta.store import ArtifactStore
from kv_rosetta.adapters.base import AdapterError
from kv_rosetta.adapters.base import ExportRequest, ImportRequest, Representation
from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter

_URL = os.environ.get("KVX_LLAMA_URL", "")
_SLOTS = os.environ.get("KVX_LLAMA_SLOTS", "")
_PROMPT_TOKENS = 256
_PREDICT = 16


@unittest.skipUnless(_URL and _SLOTS, "set KVX_LLAMA_URL and KVX_LLAMA_SLOTS to run")
class LlamaCppSameBackendRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = LlamaCppHTTPAdapter(_URL, _SLOTS)
        text = "In the year 1892, the naturalist recorded the following observations. " * 40
        ids = cls.adapter._post("/tokenize", {"content": text})["tokens"][:_PROMPT_TOKENS]
        cls.ids = ids
        # Pin the slot: on a multi-slot server an unpinned completion can land on a slot
        # that still holds a cache, and a "cold" control run would silently be warm.
        cls.request = {"prompt": ids, "n_predict": _PREDICT, "temperature": 0.0,
                       "top_k": 1, "n_probs": 5, "cache_prompt": True, "seed": 1,
                       "id_slot": 0}

    def _tokens(self, response):
        return [c["id"] for c in response.get("completion_probabilities", [])]

    def _top_probs(self, response):
        out = []
        for entry in response.get("completion_probabilities", []):
            top = entry.get("top_probs", entry.get("probs", []))
            out.append({t["id"]: t.get("prob", t.get("logprob")) for t in top})
        return out

    def test_prompt_is_exactly_256_tokens(self):
        self.assertEqual(len(self.ids), _PROMPT_TOKENS)

    def test_capabilities_are_probed_from_the_live_server(self):
        caps = self.adapter.capabilities()
        self.assertEqual(caps.runtime, "llama.cpp")
        self.assertTrue(caps.runtime_revision, "build info must come from the server")
        self.assertIn(Representation.OPAQUE, caps.export)
        self.assertIn(Representation.OPAQUE, caps.import_)
        # The format must be the version the runtime emits, not a constant. This server
        # writes 2 while the checked-out header declares 3.
        self.assertEqual(caps.opaque_formats, frozenset({f"ggsq/{self.adapter.state_version()}"}))
        self.assertRegex(caps.cache_abi_digest, r"\A[0-9a-f]{64}\Z")

    def test_round_trip_restores_the_cache_with_token_for_token_parity(self):
        adapter = self.adapter
        out = Path(tempfile.mkdtemp()) / "roundtrip.kvx"

        adapter.erase()
        cold = adapter.complete(self.request)
        self.assertEqual(cold["timings"]["cache_n"], 0, "cold run must not reuse a cache")
        reference_tokens = self._tokens(cold)
        reference_probs = self._top_probs(cold)
        self.assertEqual(len(reference_tokens), _PREDICT)

        artifact = adapter.export(ExportRequest(model="", out_path=out,
                                                representation=Representation.OPAQUE))
        self.assertTrue(Path(artifact).is_file())
        self.assertEqual(Path(artifact).read_bytes()[:4], b"KVX1")

        # Erase, then confirm the cache really is gone before claiming a restore worked.
        adapter.erase()
        cold_again = adapter.complete(self.request)
        self.assertEqual(cold_again["timings"]["cache_n"], 0)

        adapter.erase()
        report = adapter.import_(artifact, ImportRequest(model=""))
        self.assertTrue(report.ok, report.reason)
        self.assertGreater(report.tokens_restored, 0)

        warm = adapter.complete(self.request)
        self.assertGreater(warm["timings"]["cache_n"], 0, "restore did not populate the cache")
        self.assertLess(warm["timings"]["prompt_n"], _PROMPT_TOKENS,
                        "a restored cache must avoid re-prefilling the prompt")

        # The invariant is that a restored cache is indistinguishable from the runtime's
        # OWN prompt-cache reuse - not that it is bit-identical to a cold prefill. With
        # quantized KV those differ: a cold prefill quantizes values it just computed while
        # any reused cache reads back already-quantized ones, so llama.cpp diverges from
        # itself. Demanding absolute exactness would fail the adapter for the runtime's
        # behaviour; demanding parity with native reuse holds for every KV type.
        native = adapter.complete(self.request)      # runtime's own cache, no artifact
        self.assertGreater(native["timings"]["cache_n"], 0)
        self.assertEqual(warm["content"], native["content"],
                         "restored cache differs from the runtime's own cache reuse")
        self.assertEqual(self._tokens(warm), self._tokens(native), "token IDs diverged")
        for index, (a, b) in enumerate(zip(self._top_probs(native), self._top_probs(warm))):
            for token_id in set(a) | set(b):
                self.assertAlmostEqual(
                    a.get(token_id, 0.0) or 0.0, b.get(token_id, 0.0) or 0.0, places=9,
                    msg=f"probability diverged at generated position {index}")
        if cold["content"] == native["content"]:
            # Exact KV: then the restore must also match the cold prefill exactly.
            self.assertEqual(warm["content"], cold["content"])
            self.assertEqual(self._tokens(warm), reference_tokens)

    def test_import_refuses_a_mismatched_cache_abi(self):
        out = Path(tempfile.mkdtemp()) / "abi.kvx"
        artifact = self.adapter.export(ExportRequest(model="", out_path=out,
                                                     representation=Representation.OPAQUE))
        report = self.adapter.import_(
            artifact, ImportRequest(model="", expected_cache_abi_digest="0" * 64))
        self.assertFalse(report.ok)
        self.assertIn("cache ABI mismatch", report.reason)

    def test_import_refuses_a_corrupted_artifact(self):
        out = Path(tempfile.mkdtemp()) / "corrupt.kvx"
        artifact = Path(self.adapter.export(ExportRequest(model="", out_path=out,
                                                          representation=Representation.OPAQUE)))
        raw = bytearray(artifact.read_bytes())
        header_len = int.from_bytes(raw[8:12], "little")
        offset = json.loads(raw[12:12 + header_len].decode())["blob"]["offset"]
        raw[offset] ^= 0xFF
        broken = artifact.with_name("broken.kvx")
        broken.write_bytes(bytes(raw))
        report = self.adapter.import_(broken, ImportRequest(model=""))
        self.assertFalse(report.ok)
        self.assertIn("verification", report.reason)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_URL and _SLOTS, "set KVX_LLAMA_URL and KVX_LLAMA_SLOTS to run")
class SlotBindingAndFullPrefixReuse(unittest.TestCase):
    """A restore is proven only when the intended slot reuses the COMPLETE prefix.

    The measured contract on llama.cpp: after restoring an L-token prefix, probing that
    slot with those L tokens yields cache_n == L-1 and prompt_n == 1, because the final
    token is always reprocessed. cache_n > 0 is not sufficient - probing a 200-token cache
    with a 100-token prefix returns cache_n=99, which is a different prompt that merely
    shares a prefix.
    """

    @classmethod
    def setUpClass(cls):
        cls.adapter = LlamaCppHTTPAdapter(_URL, _SLOTS)
        text = "In the year 1892, the naturalist recorded observations. " * 40
        cls.ids = cls.adapter._post("/tokenize", {"content": text})["tokens"][:200]

    def _artifact(self, slot=0):
        self.adapter.erase(slot)
        self.adapter._post("/completion", {"prompt": self.ids, "n_predict": 2,
                                           "temperature": 0.0, "cache_prompt": True,
                                           "id_slot": slot})
        out = Path(tempfile.mkdtemp()) / "slotbind.kvx"
        return Path(self.adapter.export(ExportRequest(
            model="", out_path=out, representation=Representation.OPAQUE, slot=slot)))

    def _probe(self, tokens, slot):
        return self.adapter._post("/completion", {
            "prompt": list(tokens), "n_predict": 1, "temperature": 0.0,
            "cache_prompt": True, "id_slot": slot})["timings"]

    def test_restore_and_verification_are_bound_to_the_same_slot(self):
        artifact = self._artifact(slot=0)
        self.adapter.erase(0)
        report = self.adapter.import_(artifact, ImportRequest(model="", slot=0))
        self.assertTrue(report.ok, report.reason)
        self.assertIn("slot 0", report.reason)

    def _slot_count(self) -> int:
        return len(json.loads(
            urllib.request.urlopen(_URL.rstrip("/") + "/slots", timeout=30).read()))

    def test_a_different_slot_does_not_satisfy_the_restore(self):
        """Slot B must not inherit slot A's proof.

        Needs a genuinely multi-slot server: with --parallel 1 a request naming slot 1 is
        served by slot 0, which would make this assert the opposite of what it means.
        """
        if self._slot_count() < 2:
            self.skipTest("server has one slot; cross-slot isolation is not observable")
        artifact = self._artifact(slot=0)
        self.adapter.erase(0)
        self.adapter.erase(1)
        report = self.adapter.import_(artifact, ImportRequest(model="", slot=0))
        self.assertTrue(report.ok, report.reason)
        timings = self._probe(self.ids, slot=1)
        self.assertEqual(timings["cache_n"], 0,
                         "an unrestored slot reported reuse; the proof is not slot-bound")

    def test_partial_prefix_reuse_is_not_full_reuse(self):
        """The exact reason cache_n > 0 is an unsafe acceptance test."""
        artifact = self._artifact(slot=0)
        self.adapter.erase(0)
        self.assertTrue(self.adapter.import_(artifact, ImportRequest(model="", slot=0)).ok)
        half = self.ids[:100]
        timings = self._probe(half, slot=0)
        self.assertGreater(timings["cache_n"], 0, "a shared prefix does reuse some cells")
        self.assertNotEqual(timings["cache_n"], len(self.ids) - 1,
                            "partial reuse must be distinguishable from full reuse")

    def test_verification_leaves_the_slot_at_the_imported_prefix(self):
        """The probe generates a token; the slot must not be handed back mutated."""
        artifact = self._artifact(slot=0)
        expected = len(self.adapter._artifact_token_ids(artifact))
        self.adapter.erase(0)
        self.assertTrue(self.adapter.import_(artifact, ImportRequest(model="", slot=0)).ok)
        slots = json.loads(
            urllib.request.urlopen(_URL.rstrip("/") + "/slots", timeout=30).read())
        self.assertEqual(int(slots[0]["n_prompt_tokens"]), expected,
                         "slot holds more than the imported prefix after verification")

    def test_parity_survives_a_verified_import(self):
        artifact = self._artifact(slot=0)
        self.adapter.erase(0)
        cold = self.adapter._post("/completion", {
            "prompt": self.ids, "n_predict": 8, "temperature": 0.0, "top_k": 1,
            "n_probs": 5, "cache_prompt": True, "id_slot": 0})
        self.adapter.erase(0)
        self.assertTrue(self.adapter.import_(artifact, ImportRequest(model="", slot=0)).ok)
        warm = self.adapter._post("/completion", {
            "prompt": self.ids, "n_predict": 8, "temperature": 0.0, "top_k": 1,
            "n_probs": 5, "cache_prompt": True, "id_slot": 0})
        self.assertEqual(warm["content"], cold["content"])
        self.assertEqual([c["id"] for c in warm["completion_probabilities"]],
                         [c["id"] for c in cold["completion_probabilities"]])

    def test_artifact_is_labelled_with_the_emitted_state_version(self):
        artifact = self._artifact(slot=0)
        header = json.loads(
            artifact.read_bytes()[12:12 + int.from_bytes(artifact.read_bytes()[8:12], "little")])
        self.assertEqual(header["blob"]["opaque_format"],
                         f"ggsq/{self.adapter.state_version()}")
        self.assertIn(header["blob"]["opaque_format"],
                      self.adapter.capabilities().opaque_formats)


@unittest.skipUnless(_URL and _SLOTS, "set KVX_LLAMA_URL and KVX_LLAMA_SLOTS to run")
class IdentityBindingTests(unittest.TestCase):
    """Artifacts must carry, and reproduce, the identity they are stored under."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = LlamaCppHTTPAdapter(_URL, _SLOTS)
        text = "In the year 1892 the naturalist wrote. " * 30
        cls.ids = cls.adapter._post("/tokenize", {"content": text})["tokens"][:150]

    def _artifact(self):
        self.adapter.erase(0)
        self.adapter._post("/completion", {"prompt": self.ids, "n_predict": 1,
                                           "temperature": 0.0, "cache_prompt": True,
                                           "id_slot": 0})
        out = Path(tempfile.mkdtemp()) / "identity.kvx"
        return Path(self.adapter.export(ExportRequest(
            model="", out_path=out, representation=Representation.OPAQUE, slot=0)))

    def test_prompt_identity_is_not_a_stand_in(self):
        header = container.read_header(self._artifact())["prompt"]
        self.assertTrue(header["tokenizer_id"], "empty tokenizer id")
        self.assertRegex(header["token_ids_sha256"], r"\A[0-9a-f]{64}\Z")
        self.assertEqual(header["token_count"], len(self.ids))

    def test_model_identity_is_content_derived(self):
        identity = container.read_header(self._artifact())["identity"]
        self.assertRegex(identity["weights_sha256"], r"\A[0-9a-f]{64}\Z")
        self.assertRegex(identity["model_digest"], r"\A[0-9a-f]{64}\Z")
        # The weights digest must be the file's contents, not anything path-derived.
        props = self.adapter.props()
        expected = weights.model_content_digest(props["model_path"])
        self.assertEqual(identity["weights_sha256"], expected)

    def test_artifact_carries_a_reproducible_composite_key(self):
        artifact = self._artifact()
        recorded = container.read_header(artifact)["artifact_key"]
        self.assertRegex(recorded["digest"], r"\A[0-9a-f]{64}\Z")
        store = ArtifactStore(Path(tempfile.mkdtemp()) / "store")
        record = self.adapter.publish(artifact, store)
        self.assertEqual(record.artifact_digest, recorded["digest"])

    def test_published_artifact_is_keyed_on_the_full_composite_digest(self):
        store = ArtifactStore(Path(tempfile.mkdtemp()) / "store")
        record = self.adapter.publish(self._artifact(), store)
        self.assertEqual(Path(record.path).stem, record.artifact_digest)
        self.assertEqual(len(record.artifact_digest), 64)
        self.assertIsNotNone(store.get(record.artifact_digest))
        self.assertEqual(len(store.find(prompt_digest=record.prompt_digest)), 1)

    def test_publishing_an_unverifiable_artifact_is_refused(self):
        store = ArtifactStore(Path(tempfile.mkdtemp()) / "store")
        with self.assertRaises(AdapterError):
            self.adapter.publish(Path(tempfile.mkdtemp()) / "absent.kvx", store)

    def test_publishing_a_corrupted_artifact_is_refused(self):
        artifact = self._artifact()
        raw = bytearray(artifact.read_bytes())
        offset = container.read_header(artifact)["blob"]["offset"]
        raw[offset] ^= 0xFF
        broken = artifact.with_name("broken.kvx")
        broken.write_bytes(bytes(raw))
        store = ArtifactStore(Path(tempfile.mkdtemp()) / "store")
        with self.assertRaises(AdapterError):
            self.adapter.publish(broken, store)
