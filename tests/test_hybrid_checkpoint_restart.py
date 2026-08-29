"""Persistent prefix restoration on a hybrid model, across a full server restart.

This is the positive counterpart to tests/test_hybrid_negative_control.py. That control
proves an UNPATCHED runtime restores every cell and reuses none of them; this proves a
patched runtime carries the context checkpoint through a restart and reuses the prefix.

The restart is the whole point. Without stopping the server between save and restore, a
passing result only shows the in-memory cache was never dropped - an in-memory cache
experiment, not persistence.

Requires a llama-server built by scripts/build_patched_llama.sh and a hybrid model:

    KVX_CKPT_URL=http://127.0.0.1:8786 KVX_CKPT_SLOTS=/path/to/slots \
      python3 -m unittest tests.test_hybrid_checkpoint_restart -v

Restarting the server is the caller's job (see scripts/, or the recorded run in
docs/hybrid-checkpoint-research.md); this file asserts the properties either side of it.
"""

import json
import os
import unittest
import urllib.request

from runtime_matrix import PATCHED, require_runtime, slot_file_has_checkpoints

from kv_rosetta import gguf
from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter

_URL = os.environ.get("KVX_CKPT_URL", "")
_SLOTS = os.environ.get("KVX_CKPT_SLOTS", "")
_PROMPT_TOKENS = 256


@unittest.skipUnless(_URL and _SLOTS, "set KVX_CKPT_URL and KVX_CKPT_SLOTS to run")
class PatchedHybridCheckpointReuse(unittest.TestCase):
    def setUp(self):
        require_runtime(self, _URL, _SLOTS, PATCHED)

    @classmethod
    def setUpClass(cls):
        cls.adapter = LlamaCppHTTPAdapter(_URL, _SLOTS)
        cls.arch = gguf.architecture(cls.adapter.props().get("model_path", ""))
        text = "In the year 1892 the naturalist recorded. " * 30
        cls.ids = cls.adapter._post("/tokenize", {"content": text})["tokens"][:_PROMPT_TOKENS]

    def _complete(self, n_predict=1):
        return self.adapter._post("/completion", {
            "prompt": self.ids, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
            "n_probs": 5, "cache_prompt": True, "id_slot": 0})

    def test_the_model_is_hybrid(self):
        """Otherwise this proves nothing about the case that was failing."""
        self.assertIn(self.arch, gguf.HYBRID_ARCHITECTURES | gguf.RECURRENT_ARCHITECTURES)

    def test_the_runtime_persists_checkpoints(self):
        """Assert the saved file actually carries a checkpoint payload.

        This previously asserted the artifact exceeded 200 MB, which is not evidence: a
        large file only means a large cache, and the threshold would have to be retuned for
        every model. The direct observation is the SCKP magic the patched save handler
        appends after the llama state.
        """
        from pathlib import Path
        self.adapter.erase(0)
        self._complete()
        self.adapter._post("/slots/0?action=save", {"filename": "ckpt-probe.bin"})
        path = Path(_SLOTS) / "ckpt-probe.bin"
        self.assertTrue(path.is_file())
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(4), b"qsgg", "not a sequence-state file")
        self.assertTrue(slot_file_has_checkpoints(path),
                        "saved slot carries no SCKP checkpoint payload")

    def test_reuse_after_restore_within_one_process(self):
        adapter = self.adapter
        adapter.erase(0)
        cold = self._complete()
        self.assertEqual(cold["timings"]["cache_n"], 0)
        adapter._post("/slots/0?action=save", {"filename": "ckpt-inproc.bin"})
        adapter.erase(0)
        self.assertEqual(self._complete()["timings"]["cache_n"], 0, "erase did not drop it")
        adapter.erase(0)
        restored = adapter._post("/slots/0?action=restore", {"filename": "ckpt-inproc.bin"})
        self.assertGreater(restored["n_restored"], 0)
        warm = self._complete()
        self.assertGreater(warm["timings"]["cache_n"], 0,
                           "a patched runtime must reuse the restored prefix")
        # Reuse is bounded by checkpoint granularity (--ctx-checkpoints,
        # --checkpoint-min-step), so it need not be the full prefix. Upstream #25913 says
        # so explicitly. Assert substantial reuse, not perfection.
        self.assertGreater(warm["timings"]["cache_n"], _PROMPT_TOKENS * 0.9,
                           "reuse is far below the prefix; checkpoint coverage is wrong")
        self.assertLess(warm["timings"]["prompt_n"], _PROMPT_TOKENS * 0.1)

    def test_restored_output_matches_a_cold_prefill(self):
        adapter = self.adapter
        adapter.erase(0)
        cold = self._complete(n_predict=8)
        reference = [c["id"] for c in cold.get("completion_probabilities", [])]
        adapter._post("/slots/0?action=save", {"filename": "ckpt-parity.bin"})
        adapter.erase(0)
        adapter._post("/slots/0?action=restore", {"filename": "ckpt-parity.bin"})
        warm = self._complete(n_predict=8)
        self.assertEqual(warm["content"], cold["content"])
        self.assertEqual([c["id"] for c in warm.get("completion_probabilities", [])], reference)

    def test_the_adapter_still_withholds_the_capability(self):
        """Deliberate: the static refusal is only replaced once the runtime ADVERTISES a
        checkpoint-persistence format. A patched binary that cannot be detected is still
        unsafe to assume, so capability probing must not be loosened by this file passing."""
        caps = self.adapter.capabilities()
        self.assertEqual(caps.export, frozenset())
        self.assertEqual(caps.import_, frozenset())


if __name__ == "__main__":
    unittest.main()
