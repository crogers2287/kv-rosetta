"""Which architectures can have a restored cache reused.

Classified from measurement, then confirmed against llama.cpp's own source: two qwen35
models - one MTP, one not - each restored successfully (n_restored equal to the saved cell
count) and then reported cache_n=0 on the next completion, re-prefilling every token. A
qwen2 model under identical flags reused the full prefix.

The cause is architectural, not a bug: LLM_ARCH_QWEN35 appears in llm_arch_is_hybrid() in
src/llama-arch.cpp. A recurrent state is a function of the entire processed sequence, so it
has no common-prefix semantics.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from kv_rosetta.gguf import (
    GGUFError,
    HYBRID_ARCHITECTURES,
    RECURRENT_ARCHITECTURES,
    architecture,
    read_metadata,
    supports_prefix_reuse,
)


def _gguf(arch: str, extra_arrays: bool = False) -> Path:
    """Build a minimal GGUF header declaring one architecture."""
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    n_kv = 2 if extra_arrays else 1
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, n_kv))
    if extra_arrays:
        # An array before the key under test: if the reader skips elements instead of
        # consuming them it desynchronises and every later key is garbage.
        out += s("tokenizer.ggml.tokens") + struct.pack("<I", 9)
        out += struct.pack("<IQ", 8, 3) + s("a") + s("bb") + s("ccc")
    out += s("general.architecture") + struct.pack("<I", 8) + s(arch)
    path = Path(tempfile.mkdtemp()) / "model.gguf"
    path.write_bytes(bytes(out))
    return path


class ArchitectureReadTests(unittest.TestCase):
    def test_reads_architecture(self):
        self.assertEqual(architecture(_gguf("qwen2")), "qwen2")

    def test_arrays_are_fully_consumed(self):
        self.assertEqual(architecture(_gguf("qwen35", extra_arrays=True)), "qwen35")

    def test_non_gguf_is_refused(self):
        bad = Path(tempfile.mkdtemp()) / "not.gguf"
        bad.write_bytes(b"NOTGGUF" + b"\x00" * 64)
        with self.assertRaises(GGUFError):
            architecture(bad)

    def test_missing_file_is_refused(self):
        with self.assertRaises(GGUFError):
            architecture(Path(tempfile.mkdtemp()) / "absent.gguf")

    def test_key_filter_limits_the_result(self):
        meta = read_metadata(_gguf("qwen2", extra_arrays=True), ("general.architecture",))
        self.assertEqual(list(meta), ["general.architecture"])


class PrefixReuseSupportTests(unittest.TestCase):
    def test_the_architecture_that_reuses_is_allowed(self):
        ok, reason = supports_prefix_reuse("qwen2")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_the_measured_failing_architecture_is_refused(self):
        ok, reason = supports_prefix_reuse("qwen35")
        self.assertFalse(ok)
        self.assertIn("hybrid", reason)

    def test_every_hybrid_architecture_is_refused(self):
        for arch in sorted(HYBRID_ARCHITECTURES):
            with self.subTest(arch=arch):
                self.assertFalse(supports_prefix_reuse(arch)[0])

    def test_every_recurrent_architecture_is_refused(self):
        for arch in sorted(RECURRENT_ARCHITECTURES):
            with self.subTest(arch=arch):
                self.assertFalse(supports_prefix_reuse(arch)[0])

    def test_an_unknown_architecture_is_refused_rather_than_assumed(self):
        ok, reason = supports_prefix_reuse("")
        self.assertFalse(ok)
        self.assertIn("unknown", reason)

    def test_case_and_whitespace_do_not_defeat_the_check(self):
        self.assertFalse(supports_prefix_reuse("  QWEN35 ")[0])

    def test_an_unlisted_attention_architecture_is_permitted(self):
        # Only architectures llama.cpp classifies as hybrid or recurrent are withheld;
        # an ordinary attention model must not be blocked by an over-broad rule.
        self.assertTrue(supports_prefix_reuse("llama")[0])


if __name__ == "__main__":
    unittest.main()
