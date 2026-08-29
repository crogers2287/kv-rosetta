"""The unpatched runtime's hybrid failure, retained as a live negative control.

This test must keep PASSING while llama.cpp is unpatched. It is not a bug report waiting to
be fixed; it is the control that the checkpoint-persistence work is measured against, and it
is what proves a future patched build actually changed something.

The failure is narrower than "hybrid models cannot restore". A recurrent state captured after
an exact token sequence is deterministic and restorable. What current llama.cpp omits is the
context-checkpoint state that hybrid and recurrent models need in order to resume - the save
path serialises only prompt.tokens, and the restore path calls prompt.clear(), which drops
checkpoints. See docs/hybrid-checkpoint-research.md for the pinned source map.

    KVX_HYBRID_URL=http://127.0.0.1:8785 KVX_HYBRID_SLOTS=/path/to/slots \
      python3 -m unittest tests.test_hybrid_negative_control -v
"""

import json
import os
import unittest
import urllib.request

from kv_rosetta import gguf
from kv_rosetta.adapters.base import Representation
from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter

_URL = os.environ.get("KVX_HYBRID_URL", "")
_SLOTS = os.environ.get("KVX_HYBRID_SLOTS", "")
_PROMPT_TOKENS = 256


@unittest.skipUnless(_URL and _SLOTS, "set KVX_HYBRID_URL and KVX_HYBRID_SLOTS to run")
class UnpatchedHybridFailsClosed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = LlamaCppHTTPAdapter(_URL, _SLOTS)
        cls.arch = gguf.architecture(cls.adapter.props().get("model_path", ""))
        text = "In the year 1892 the naturalist recorded. " * 30
        cls.ids = cls.adapter._post("/tokenize", {"content": text})["tokens"][:_PROMPT_TOKENS]

    def _complete(self, slot=0):
        return self.adapter._post("/completion", {
            "prompt": self.ids, "n_predict": 1, "temperature": 0.0,
            "cache_prompt": True, "id_slot": slot})

    def test_the_model_under_test_is_actually_hybrid(self):
        """Without this, the rest of the file proves nothing about hybrid models."""
        self.assertIn(self.arch, gguf.HYBRID_ARCHITECTURES | gguf.RECURRENT_ARCHITECTURES,
                      f"{self.arch} is not hybrid or recurrent; point this at a hybrid model")

    def test_save_restore_reports_success_but_reuse_never_happens(self):
        """The whole negative control, in the order that makes it meaningful."""
        adapter = self.adapter

        adapter.erase(0)
        cold = self._complete()
        self.assertEqual(cold["timings"]["cache_n"], 0, "cold run must not reuse a cache")
        self.assertEqual(cold["timings"]["prompt_n"], _PROMPT_TOKENS)

        saved = adapter._post("/slots/0?action=save", {"filename": "negative-control.bin"})
        self.assertGreater(saved["n_saved"], 0)
        self.assertGreater(saved["n_written"], 0)

        erased = adapter._post("/slots/0?action=erase", {})
        self.assertGreater(erased["n_erased"], 0)

        # Control: prove the cache is really gone, so a later "restore worked" cannot be
        # confused with a cache that was never dropped.
        control = self._complete()
        self.assertEqual(control["timings"]["cache_n"], 0)

        adapter.erase(0)
        restored = adapter._post("/slots/0?action=restore", {"filename": "negative-control.bin"})
        # The runtime reports complete success here. That is the trap.
        self.assertEqual(restored["n_restored"], saved["n_saved"],
                         "the runtime reports every cell restored")
        self.assertEqual(restored["n_read"], saved["n_written"])

        slots = json.loads(urllib.request.urlopen(_URL.rstrip("/") + "/slots", timeout=30).read())
        self.assertEqual(int(slots[0]["n_prompt_tokens"]), _PROMPT_TOKENS,
                         "the slot even knows how many tokens it holds")

        warm = self._complete()
        self.assertEqual(warm["timings"]["cache_n"], 0,
                         "if this is nonzero the runtime has been patched; update the "
                         "capability rule and retire this control deliberately")
        self.assertEqual(warm["timings"]["prompt_n"], _PROMPT_TOKENS,
                         "every token is re-prefilled despite a fully successful restore")

    def test_n_restored_is_not_evidence_of_a_usable_cache(self):
        """The single sentence this control exists to keep true."""
        self.adapter.erase(0)
        self._complete()
        saved = self.adapter._post("/slots/0?action=save", {"filename": "evidence.bin"})
        self.adapter.erase(0)
        restored = self.adapter._post("/slots/0?action=restore", {"filename": "evidence.bin"})
        self.assertGreater(restored["n_restored"], 0)
        self.assertEqual(self._complete()["timings"]["cache_n"], 0)

    def test_the_adapter_withholds_the_capability_on_this_runtime(self):
        caps = self.adapter.capabilities()
        self.assertEqual(caps.export, frozenset(),
                         "opaque export must not be advertised for an unpatched hybrid")
        self.assertEqual(caps.import_, frozenset())
        self.assertTrue(any("withheld" in note for note in caps.notes), caps.notes)

    def test_export_is_refused_rather_than_producing_a_useless_artifact(self):
        from pathlib import Path
        import tempfile
        from kv_rosetta.adapters.base import AdapterError, ExportRequest
        with self.assertRaises(AdapterError):
            self.adapter.export(ExportRequest(
                model="", out_path=Path(tempfile.mkdtemp()) / "nope.kvx",
                representation=Representation.CANONICAL))


if __name__ == "__main__":
    unittest.main()
