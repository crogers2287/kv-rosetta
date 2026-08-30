"""The size predictor, checked against artifacts that were actually written.

The numbers below are file sizes from REQ-036/037/038 - real state files produced by
llama-server on this host and recorded in bench/. They are the point of the exercise: a size
law fitted to two points reproduces those two points by construction and proves nothing, so
what matters is that terms derived from the writer predict a file four times outside the range
they were checked on.

The 4-byte-per-token header term is the one this exists to protect. It is invisible at 128
tokens and 128 KB adrift at 32,000, which is exactly the size of mistake a space guard makes
silently.
"""

import unittest
from pathlib import Path
from unittest import mock

from kv_rosetta.sizing import (
    KVGeometry,
    SizingError,
    bytes_per_token,
    geometry_of,
    row_size,
    state_bytes,
)

MODEL = Path("/mnt/storage/pre1940_finetune/library_of_alexandria_Q4_K_M.gguf")
PER_LAYER_KV = Path("/mnt/storage/local-gguf/gemma4-31b/gemma-4-31B-it-MTP-Q8_0.gguf")
HYBRID = Path("/mnt/storage/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")

#: qwen2 Q4_K_M as read from its GGUF: 36 layers, 2 KV heads, head_dim 128.
QWEN2 = KVGeometry(n_layer=36, n_kv_head=2, head_dim=128, architecture="qwen2",
                   value_head_dim=128)

#: cells -> file size, measured. The saved slots carried four more header token ids than
#: cache cells, which is worth 16 bytes and is why header_tokens is passed explicitly.
MEASURED = {128: 4_721_548, 129: 4_758_428, 8192: 302_121_868, 32000: 1_180_160_908}


class MeasuredArtifactTest(unittest.TestCase):
    def test_it_predicts_every_recorded_artifact_exactly(self):
        for cells, actual in MEASURED.items():
            with self.subTest(cells=cells):
                self.assertEqual(state_bytes(QWEN2, cells, header_tokens=cells + 4), actual)

    def test_the_marginal_token_matches_two_adjacent_artifacts(self):
        """128 and 129 tokens were saved back to back precisely to measure this."""
        self.assertEqual(MEASURED[129] - MEASURED[128], bytes_per_token(QWEN2))

    def test_the_marginal_cost_is_consistent_with_the_total(self):
        for cells in (0, 1, 977, 32000):
            with self.subTest(cells=cells):
                step = state_bytes(QWEN2, cells + 1) - state_bytes(QWEN2, cells)
                self.assertEqual(step, bytes_per_token(QWEN2))

    def test_the_header_token_term_is_not_optional(self):
        """Dropping it is invisible at 128 tokens and 128 KB adrift at 32,000."""
        without = state_bytes(QWEN2, 32000, header_tokens=0)
        self.assertEqual(MEASURED[32000] - without, 4 * 32004)


class GeometryTest(unittest.TestCase):
    @unittest.skipUnless(MODEL.is_file(), "the qwen2 test model is not on this host")
    def test_geometry_comes_from_the_gguf(self):
        self.assertEqual(geometry_of(MODEL), QWEN2)

    @unittest.skipUnless(PER_LAYER_KV.is_file(), "the gemma4 model is not on this host")
    def test_a_model_with_per_layer_kv_heads_is_refused(self):
        """This model declares head_count_kv [16, 16, 16, 4].

        One row size cannot describe that. Quietly taking the first entry would overstate
        the last layer by four times, and the resulting number would look entirely ordinary.
        """
        with self.assertRaises(SizingError) as caught:
            geometry_of(PER_LAYER_KV)
        self.assertIn("per-layer head_count_kv", str(caught.exception))

    def test_nonsense_geometry_is_refused(self):
        for geometry in (KVGeometry(0, 2, 128), KVGeometry(36, 0, 128),
                         KVGeometry(36, 2, 0), KVGeometry(-1, 2, 128)):
            with self.subTest(geometry=geometry):
                with self.assertRaises(SizingError):
                    state_bytes(geometry, 128)

    def test_negative_counts_are_refused(self):
        """Each count needs its own case: header_tokens defaults to cells, so passing a
        negative cell count alone is caught by the header check and the cell check is never
        reached. A mutation run found exactly that shadowing."""
        with self.assertRaises(SizingError):
            state_bytes(QWEN2, -1, header_tokens=5)
        with self.assertRaises(SizingError):
            state_bytes(QWEN2, 128, header_tokens=-1)

    def test_a_gguf_missing_a_geometry_key_is_refused(self):
        """Defaulting a missing block_count would produce a confident, wrong number."""
        for absent in ("block_count", "attention.head_count", "attention.head_count_kv",
                       "embedding_length"):
            full = {"qwen2.block_count": 36, "qwen2.attention.head_count": 16,
                    "qwen2.attention.head_count_kv": 2, "qwen2.embedding_length": 2048}
            full.pop(f"qwen2.{absent}")
            with self.subTest(absent=absent):
                with mock.patch("kv_rosetta.sizing.gguf.architecture", return_value="qwen2"), \
                     mock.patch("kv_rosetta.sizing.gguf.read_metadata", return_value=full):
                    with self.assertRaises(SizingError) as caught:
                        geometry_of("ignored.gguf")
                self.assertIn(absent, str(caught.exception))

    def test_a_gguf_declaring_zero_heads_is_refused_not_divided_by(self):
        declared = {"qwen2.block_count": 36, "qwen2.attention.head_count": 0,
                    "qwen2.attention.head_count_kv": 2, "qwen2.embedding_length": 2048}
        with mock.patch("kv_rosetta.sizing.gguf.architecture", return_value="qwen2"), \
             mock.patch("kv_rosetta.sizing.gguf.read_metadata", return_value=declared):
            with self.assertRaises(SizingError) as caught:
                geometry_of("ignored.gguf")
        self.assertIn("head_count 0", str(caught.exception))


class RowSizeTest(unittest.TestCase):
    def test_f16_is_two_bytes_an_element(self):
        self.assertEqual(row_size("f16", 256), 512)

    def test_a_quantised_kv_type_is_sized_by_its_block(self):
        """q8_0 packs 32 elements into 34 bytes, so a row is not elements times a width."""
        self.assertEqual(row_size("q8_0", 256), 256 // 32 * 34)

    def test_a_row_that_is_not_whole_blocks_is_refused(self):
        with self.assertRaises(SizingError) as caught:
            row_size("q8_0", 100)
        self.assertIn("whole number", str(caught.exception))

    def test_an_unknown_type_is_refused_rather_than_assumed_two_bytes(self):
        with self.assertRaises(SizingError):
            row_size("q3_k", 256)

    def test_quantised_kv_predicts_a_smaller_file(self):
        self.assertLess(state_bytes(QWEN2, 8192, kv_type="q8_0"),
                        state_bytes(QWEN2, 8192, kv_type="f16"))


if __name__ == "__main__":
    unittest.main()


class HeadDimensionTest(unittest.TestCase):
    """Where the head dimension comes from, and when it must not be derived.

    Deriving it as embedding_length / head_count is only correct when a model does not say
    otherwise. qwen35 says otherwise: it declares key_length 256, while the derivation gives
    5120 / 24 = 213.33, which floored to 213 and produced a file-size estimate wrong by a
    fifth - plausible enough to have been believed.
    """

    def gguf(self, **declared):
        base = {"qwen2.block_count": 36, "qwen2.attention.head_count": 16,
                "qwen2.attention.head_count_kv": 2, "qwen2.embedding_length": 2048}
        base.update({f"qwen2.{k}": v for k, v in declared.items()})
        return mock.patch("kv_rosetta.sizing.gguf.read_metadata", return_value=base), \
            mock.patch("kv_rosetta.sizing.gguf.architecture", return_value="qwen2")

    def test_a_declared_key_length_wins_over_the_derivation(self):
        meta, arch = self.gguf(**{"attention.key_length": 256})
        with meta, arch:
            self.assertEqual(geometry_of("ignored.gguf").head_dim, 256)

    def test_the_derivation_is_used_only_when_nothing_is_declared(self):
        meta, arch = self.gguf()
        with meta, arch:
            self.assertEqual(geometry_of("ignored.gguf").head_dim, 128)

    def test_an_indivisible_embedding_length_is_refused_not_floored(self):
        """This is the qwen35 shape with its key_length removed - 5120 over 24."""
        meta, arch = self.gguf(**{"embedding_length": 5120, "attention.head_count": 24})
        with meta, arch:
            with self.assertRaises(SizingError) as caught:
                geometry_of("ignored.gguf")
        self.assertIn("would invent one", str(caught.exception))

    def test_a_declared_value_length_sizes_values_separately(self):
        meta, arch = self.gguf(**{"attention.key_length": 128,
                                  "attention.value_length": 64})
        with meta, arch:
            geometry = geometry_of("ignored.gguf")
        self.assertEqual((geometry.head_dim, geometry.value_dim), (128, 64))
        self.assertEqual(geometry.n_embd_k_gqa, 256)
        self.assertEqual(geometry.n_embd_v_gqa, 128)

    def test_asymmetric_keys_and_values_change_the_size(self):
        symmetric = KVGeometry(n_layer=36, n_kv_head=2, head_dim=128)
        lopsided = KVGeometry(n_layer=36, n_kv_head=2, head_dim=128, value_head_dim=64)
        self.assertLess(state_bytes(lopsided, 1024), state_bytes(symmetric, 1024))

    def test_a_nonsense_value_dimension_is_refused(self):
        with self.assertRaises(SizingError) as caught:
            state_bytes(KVGeometry(36, 2, 128, value_head_dim=0), 128)
        self.assertIn("value_head_dim", str(caught.exception))

    @unittest.skipUnless(HYBRID.is_file(), "the qwen35 model is not on this host")
    def test_the_live_hybrid_model_reads_its_declared_key_length(self):
        geometry = geometry_of(HYBRID)
        self.assertEqual(geometry.head_dim, 256)
        self.assertNotEqual(geometry.head_dim, 5120 // 24)


class MarginalCostTest(unittest.TestCase):
    def test_the_marginal_cost_follows_the_kv_type(self):
        """It is derived from state_bytes, and the type has to reach both calls.

        A first version passed neither and silently returned the f16 figure for every
        quantised type - the same number, plausible, and wrong by nearly half.
        """
        self.assertLess(bytes_per_token(QWEN2, kv_type="q8_0"),
                        bytes_per_token(QWEN2, kv_type="f16"))
        self.assertEqual(bytes_per_token(QWEN2, kv_type="q8_0"),
                         state_bytes(QWEN2, 1, kv_type="q8_0")
                         - state_bytes(QWEN2, 0, kv_type="q8_0"))

    def test_it_inherits_the_refusals(self):
        with self.assertRaises(SizingError):
            bytes_per_token(KVGeometry(0, 2, 128))
        with self.assertRaises(SizingError):
            bytes_per_token(KVGeometry(36, 2, 128, architecture="qwen35"))


class HybridRefusalTest(unittest.TestCase):
    """Hybrid and recurrent models are refused, and now actually checked.

    The first version said so in the docstring and did not check, which is worse than
    silence: a caller reading it would have taken an attention-only figure for a whole file.
    Applying it to qwen35 left a NEGATIVE remainder against a measured artifact, which is
    how the gap surfaced.
    """

    def test_a_hybrid_architecture_is_refused(self):
        for arch in ("qwen35", "qwen35moe", "jamba", "falcon-h1"):
            with self.subTest(arch=arch):
                with self.assertRaises(SizingError) as caught:
                    state_bytes(KVGeometry(65, 4, 256, architecture=arch), 2048)
                self.assertIn("hybrid", str(caught.exception))

    def test_a_recurrent_architecture_is_refused(self):
        with self.assertRaises(SizingError) as caught:
            state_bytes(KVGeometry(24, 4, 128, architecture="mamba2"), 2048)
        self.assertIn("no attention KV", str(caught.exception))

    def test_an_ordinary_architecture_still_passes(self):
        self.assertGreater(state_bytes(QWEN2, 128), 0)
        self.assertGreater(state_bytes(KVGeometry(28, 8, 64, architecture="qwen3"), 128), 0)

    @unittest.skipUnless(HYBRID.is_file(), "the qwen35 model is not on this host")
    def test_the_live_hybrid_model_is_refused_end_to_end(self):
        with self.assertRaises(SizingError):
            state_bytes(geometry_of(HYBRID), 2048)
