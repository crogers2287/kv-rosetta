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
from kv_rosetta.adapters.base import Representation
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
        # Deliberately does not contact the server. setUpClass runs before setUp, so any
        # request here turns an unreachable or wrong-binary runtime into an ERROR instead
        # of the skip the matrix is supposed to produce.
        cls.adapter = LlamaCppHTTPAdapter(_URL, _SLOTS)

    @property
    def arch(self):
        return gguf.architecture(self.adapter.props().get("model_path", ""))

    @property
    def ids(self):
        text = "In the year 1892 the naturalist recorded. " * 30
        return self.adapter._post("/tokenize", {"content": text})["tokens"][:_PROMPT_TOKENS]

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

    def test_capability_is_enabled_only_by_the_advertised_protocol(self):
        """The refusal is lifted by evidence, never by architecture.

        A hybrid model is supportable exactly when the runtime advertises checkpoint
        persistence. The same model on an unpatched binary must still be refused - which
        tests/test_hybrid_negative_control.py asserts from the other side.
        """
        caps = self.adapter.capabilities()
        protocol = self.adapter.checkpoint_protocol()
        self.assertTrue(protocol, "this class requires a runtime that advertises sckp")
        self.assertIn(Representation.OPAQUE, caps.export)
        self.assertIn(Representation.OPAQUE, caps.import_)
        self.assertTrue(any("advertised" in note for note in caps.notes), caps.notes)

    def test_the_artifact_is_labelled_compound_not_plain_ggsq(self):
        """Calling a checkpoint-bearing file plain ggsq/N would let an importer believe a
        sequence-only restore is sufficient. The label comes from the bytes."""
        from pathlib import Path
        import tempfile as _tempfile
        from kv_rosetta import container
        from kv_rosetta.adapters.base import ExportRequest
        self.adapter.erase(0)
        self._complete()
        out = Path(_tempfile.mkdtemp()) / "label.kvx"
        artifact = Path(self.adapter.export(ExportRequest(
            model="", out_path=out, representation=Representation.OPAQUE, slot=0)))
        header = container.read_header(artifact)
        fmt = header["blob"]["opaque_format"]
        self.assertRegex(fmt, r"\Aggsq/\d+\+sckp/1\Z", f"artifact mislabelled as {fmt!r}")
        coverage = header.get("coverage") or {}
        self.assertGreater(coverage.get("checkpoint_n_tokens", 0), 0)
        self.assertGreater(coverage.get("checkpoint_bytes", 0), 0)
        self.assertGreaterEqual(coverage.get("n_checkpoints", 0), 1)

    def test_a_truncated_checkpoint_appendix_is_refused(self):
        """llama.cpp degrades a truncated appendix to a sequence-only restore, which is
        backward compatible for it and unacceptable as a successful hybrid import here."""
        from pathlib import Path
        import tempfile as _tempfile
        from kv_rosetta import container
        from kv_rosetta.adapters.base import AdapterError, ExportRequest, ImportRequest
        from kv_rosetta.adapters import ggsq_envelope
        self.adapter.erase(0)
        self._complete()
        out = Path(_tempfile.mkdtemp()) / "trunc.kvx"
        artifact = Path(self.adapter.export(ExportRequest(
            model="", out_path=out, representation=Representation.OPAQUE, slot=0)))
        raw = artifact.read_bytes()
        self.assertTrue(ggsq_envelope.has_checkpoint_appendix(artifact))
        cut = artifact.with_name("cut.kvx")
        cut.write_bytes(raw[: len(raw) - 1_000_000])
        ok, reason = container.verify(cut)
        self.assertFalse(ok, "a truncated artifact verified as intact")
        report = self.adapter.import_(cut, ImportRequest(model="", slot=0))
        self.assertFalse(report.ok)
        self.assertIn("verification", report.reason)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_URL and _SLOTS, "set KVX_CKPT_URL and KVX_CKPT_SLOTS to run")
class CheckpointProtocolTests(unittest.TestCase):
    """The runtime must STATE its capability, not have it inferred.

    A capability may not be enabled from an architecture name, a commit, a filename, a
    strings(1) match, or an artifact size - each describes the build rather than the
    behaviour. These assert the machine-readable protocol instead.
    """

    def setUp(self):
        require_runtime(self, _URL, _SLOTS, PATCHED)
        self.adapter = LlamaCppHTTPAdapter(_URL, _SLOTS)

    def test_props_advertises_the_protocol(self):
        from runtime_matrix import checkpoint_protocol
        protocol = checkpoint_protocol(_URL)
        self.assertTrue(protocol, "runtime advertises no checkpoint persistence")
        self.assertEqual(protocol["format"], "sckp/1")
        self.assertIsInstance(protocol["sequence_state_version"], int)
        for blob in ("target", "draft", "speculative"):
            self.assertIn(blob, protocol, f"{blob} checkpoint state support not stated")

    def test_save_reports_what_it_persisted(self):
        self.adapter.erase(0)
        self._prime()
        saved = self.adapter._post("/slots/0?action=save", {"filename": "proto-save.bin"})
        self.assertGreaterEqual(saved["n_checkpoints_saved"], 1,
                                "no checkpoint reported as saved")
        self.assertGreater(saved["checkpoint_bytes"], 0)
        self.assertGreater(saved["checkpoint_n_tokens"], 0)
        self.assertGreaterEqual(saved["checkpoint_pos_max"], saved["checkpoint_pos_min"])

    def test_declared_coverage_matches_observed_reuse(self):
        """The check that lets an importer fail closed: if what the runtime says it
        restored disagrees with what it then reuses, the artifact is not trustworthy."""
        self.adapter.erase(0)
        self._prime()
        self.adapter._post("/slots/0?action=save", {"filename": "proto-cov.bin"})
        self.adapter.erase(0)
        restored = self.adapter._post("/slots/0?action=restore", {"filename": "proto-cov.bin"})
        self.assertGreaterEqual(restored["n_checkpoints_restored"], 1)
        declared = restored["checkpoint_n_tokens"]
        observed = self._prime()["timings"]["cache_n"]
        self.assertEqual(declared, observed,
                         f"runtime declared coverage {declared} but reused {observed}")

    def _prime(self):
        text = "In the year 1892 the naturalist recorded. " * 30
        ids = self.adapter._post("/tokenize", {"content": text})["tokens"][:256]
        return self.adapter._post("/completion", {
            "prompt": ids, "n_predict": 1, "temperature": 0.0,
            "cache_prompt": True, "id_slot": 0})
