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
        cls.request = {"prompt": ids, "n_predict": _PREDICT, "temperature": 0.0,
                       "top_k": 1, "n_probs": 5, "cache_prompt": True, "seed": 1}

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
        self.assertIn("ggsq/3", caps.opaque_formats)
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

        self.assertEqual(warm["content"], cold["content"])
        self.assertEqual(self._tokens(warm), reference_tokens, "token IDs diverged")
        for index, (a, b) in enumerate(zip(reference_probs, self._top_probs(warm))):
            for token_id in set(a) | set(b):
                self.assertAlmostEqual(
                    a.get(token_id, 0.0) or 0.0, b.get(token_id, 0.0) or 0.0, places=9,
                    msg=f"probability diverged at generated position {index}")

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
