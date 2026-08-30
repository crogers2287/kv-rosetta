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
    HybridGeometry,
    KVGeometry,
    SizingError,
    bytes_per_token,
    geometry_of,
    hybrid_geometry_of,
    hybrid_state_bytes,
    row_size,
    state_bytes,
    writes_cell_ext,
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


class SecondModelTest(unittest.TestCase):
    """The same terms against a differently shaped model, predicted before it was run.

    Everything else here is qwen2: 36 layers, 2 KV heads, head_dim 128. This is qwen3 with 28
    layers and 8 KV heads - three times the cost per token, a different architecture string,
    and a declared key_length of 128 where embedding_length over head_count would have given
    64. The numbers below were printed by the predictor first and the artifacts written
    afterwards.
    """

    #: qwen3, as read from Qwen3-Embedding-0.6B-Q8_0.gguf.
    QWEN3 = KVGeometry(n_layer=28, n_kv_head=8, head_dim=128, architecture="qwen3",
                       value_head_dim=128)

    #: cells -> file size, measured on this host with the HIP build at ca3d5a3e1.
    MEASURED = {128: 14_682_828, 129: 14_797_532}

    def test_it_predicts_a_second_model_exactly(self):
        for cells, actual in self.MEASURED.items():
            with self.subTest(cells=cells):
                self.assertEqual(state_bytes(self.QWEN3, cells, header_tokens=cells + 4),
                                 actual)

    def test_the_second_model_costs_three_times_as_much_per_token(self):
        """Geometry, not a shared constant, is what the prediction rests on."""
        self.assertEqual(self.MEASURED[129] - self.MEASURED[128],
                         bytes_per_token(self.QWEN3))
        self.assertEqual(bytes_per_token(self.QWEN3), 114_704)
        self.assertEqual(bytes_per_token(QWEN2), 36_880)

    def test_deriving_the_head_dimension_here_would_have_halved_it(self):
        """This model declares key_length 128; embedding_length / head_count is 1024/16."""
        derived = KVGeometry(n_layer=28, n_kv_head=8, head_dim=1024 // 16,
                             architecture="qwen3")
        self.assertNotEqual(state_bytes(derived, 128, header_tokens=132),
                            self.MEASURED[128])


class HybridSizingTest(unittest.TestCase):
    """A hybrid artifact's two halves, each derived, checked against measured files.

    RA-003 argued from two fitted points that the hybrid's large fixed term is recurrent
    state. These are the terms themselves, and the fixed part is now a number rather than a
    residual: 156,894,356 bytes for this model, identical between a 256- and a 257-token
    save, and 90.3% of the file at 256 tokens.
    """

    #: Qwen3.8-27B-UD-Q4_K_XL, as read from its GGUF.
    QWEN35 = HybridGeometry(attention_layers=16, recurrent_layers=48, n_kv_head=4,
                            head_dim=256, conv_row_bytes=122_880, ssm_row_bytes=3_145_728,
                            architecture="qwen35", value_head_dim=256)

    #: cells -> file size, measured on this host with the HIP build at ca3d5a3e1.
    MEASURED = {256: 173_679_168, 257: 173_744_732}

    def test_it_predicts_the_measured_hybrid_artifacts_exactly(self):
        for cells, actual in self.MEASURED.items():
            with self.subTest(cells=cells):
                self.assertEqual(
                    hybrid_state_bytes(self.QWEN35, cells, header_tokens=cells + 4,
                                       checkpoints=0), actual)

    def test_the_recurrent_tail_does_not_grow_with_tokens(self):
        """Measured: the two files' recurrent sections are byte-identical."""
        marginal = (hybrid_state_bytes(self.QWEN35, 257, header_tokens=261, checkpoints=0)
                    - hybrid_state_bytes(self.QWEN35, 256, header_tokens=260,
                                         checkpoints=0))
        self.assertEqual(marginal, 65_564)
        # 4 header id + 24 cell meta with ext + 16 layers of f16 keys and values.
        self.assertEqual(marginal, 4 + 24 + 16 * 2 * (4 * 256 * 2))

    def test_the_fixed_tail_dominates_a_short_prefix(self):
        total = hybrid_state_bytes(self.QWEN35, 256, header_tokens=260, checkpoints=0)
        tail = total - hybrid_state_bytes(self.QWEN35, 0, header_tokens=0, checkpoints=0) * 0
        fixed = total - 256 * 65_564
        self.assertGreater(fixed / total, 0.90)

    @unittest.skipUnless(HYBRID.is_file(), "the qwen35 model is not on this host")
    def test_the_geometry_is_read_from_the_gguf(self):
        self.assertEqual(hybrid_geometry_of(HYBRID), self.QWEN35)

    @unittest.skipUnless(MODEL.is_file(), "the qwen2 test model is not on this host")
    def test_a_non_hybrid_model_is_sent_to_the_other_function(self):
        with self.assertRaises(SizingError) as caught:
            hybrid_geometry_of(MODEL)
        self.assertIn("not hybrid", str(caught.exception))

    def test_nonsense_recurrent_rows_are_refused(self):
        for bad in ({"conv_row_bytes": 0}, {"ssm_row_bytes": 0}):
            with self.subTest(bad=bad):
                geometry = HybridGeometry(**{**self.QWEN35.__dict__, **bad})
                with self.assertRaises(SizingError):
                    hybrid_state_bytes(geometry, 256, checkpoints=0)

    def test_zero_recurrent_cells_is_refused(self):
        """A saved slot always holds one recurrent cell; zero would silently drop the tail."""
        with self.assertRaises(SizingError):
            hybrid_state_bytes(self.QWEN35, 256, recurrent_cells=0, checkpoints=0)


class CellExtTest(unittest.TestCase):
    """Whether a 12-byte cell_ext is written per cell - decided by architecture, not GGUF."""

    def test_an_mrope_architecture_writes_it(self):
        for arch in ("qwen35", "qwen35moe", "qwen2vl", "qwen3vl", "paddleocr"):
            with self.subTest(arch=arch):
                self.assertTrue(writes_cell_ext(arch))

    def test_an_ordinary_architecture_does_not(self):
        for arch in ("qwen2", "qwen3", "llama"):
            with self.subTest(arch=arch):
                self.assertFalse(writes_cell_ext(arch))

    def test_an_architecture_that_decides_at_runtime_is_refused(self):
        """glm4 picks its rope type from model state; 12 bytes a cell is 3 MB at 256K cells
        and looks entirely reasonable at 128."""
        for arch in ("glm4", "glm4moe", "hunyuan_vl", "dflash"):
            with self.subTest(arch=arch):
                with self.assertRaises(SizingError) as caught:
                    writes_cell_ext(arch)
                self.assertIn("cannot be read off the architecture", str(caught.exception))

    def test_the_qwen2_prediction_depends_on_it_being_absent(self):
        """The four exact qwen2 results all assume no cell_ext; this pins that assumption."""
        self.assertFalse(writes_cell_ext("qwen2"))


class HybridGeometryRefusalTest(unittest.TestCase):
    """What hybrid_geometry_of refuses to read out of a GGUF."""

    BASE = {"qwen35.block_count": 65, "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256, "qwen35.ssm.inner_size": 6144,
            "qwen35.ssm.state_size": 128, "qwen35.ssm.conv_kernel": 4,
            "qwen35.ssm.group_count": 16, "qwen35.full_attention_interval": 4,
            "qwen35.nextn_predict_layers": 1}

    def read(self, **overrides):
        declared = {**self.BASE}
        for key, value in overrides.items():
            full = f"qwen35.{key}"
            if value is None:
                declared.pop(full, None)
            else:
                declared[full] = value
        return (mock.patch("kv_rosetta.sizing.gguf.architecture", return_value="qwen35"),
                mock.patch("kv_rosetta.sizing.gguf.read_metadata", return_value=declared))

    def geometry(self, **overrides):
        arch, meta = self.read(**overrides)
        with arch, meta:
            return hybrid_geometry_of("ignored.gguf")

    def test_the_baseline_reads_the_known_split(self):
        found = self.geometry()
        self.assertEqual((found.attention_layers, found.recurrent_layers), (16, 48))

    def test_a_missing_required_key_is_refused(self):
        for absent in ("block_count", "attention.head_count_kv", "attention.key_length",
                       "ssm.inner_size", "ssm.state_size", "ssm.conv_kernel",
                       "ssm.group_count"):
            with self.subTest(absent=absent):
                with self.assertRaises(SizingError) as caught:
                    self.geometry(**{absent: None})
                self.assertIn(absent, str(caught.exception))

    def test_an_explicit_recurrent_layer_array_is_refused_not_ignored(self):
        """The split is derived from the interval, which would silently contradict it."""
        with self.assertRaises(SizingError) as caught:
            self.geometry(**{"attention.recurrent_layers": [1, 1, 0, 1]})
        self.assertIn("would ignore it", str(caught.exception))

    def test_a_declared_zero_interval_is_refused_not_defaulted(self):
        """`or 4` would read a declared 0 as llama.cpp's default and hide it."""
        with self.assertRaises(SizingError) as caught:
            self.geometry(full_attention_interval=0)
        self.assertIn("not positive", str(caught.exception))

    def test_an_absent_interval_takes_the_default(self):
        self.assertEqual(self.geometry(full_attention_interval=None).attention_layers, 16)

    def test_more_nextn_layers_than_blocks_is_refused(self):
        with self.assertRaises(SizingError) as caught:
            self.geometry(nextn_predict_layers=65)
        self.assertIn("leaves", str(caught.exception))

    def test_a_non_positive_ssm_dimension_is_refused(self):
        for key in ("ssm.inner_size", "ssm.state_size", "ssm.conv_kernel",
                    "ssm.group_count"):
            with self.subTest(key=key):
                with self.assertRaises(SizingError) as caught:
                    self.geometry(**{key: 0})
                self.assertIn("SSM dimension", str(caught.exception))

    def test_negative_counts_are_refused_each_on_its_own(self):
        """header_tokens defaults to cells, so a negative cell count alone never reaches the
        cell check - the same shadowing the non-hybrid tests found."""
        with self.assertRaises(SizingError):
            hybrid_state_bytes(HybridSizingTest.QWEN35, -1, header_tokens=5, checkpoints=0)
        with self.assertRaises(SizingError):
            hybrid_state_bytes(HybridSizingTest.QWEN35, 256, header_tokens=-1, checkpoints=0)


class CheckpointAppendixSizingTest(unittest.TestCase):
    """The SCKP appendix a patched build appends, now derived rather than inferred.

    Its terms are read off save_slot_checkpoints in the patched server: magic, version and
    count, then per checkpoint an int64 n_tokens and two int32 positions, then data_tgt,
    data_dft and data_spec each written length-prefixed and interleaved with its payload, so
    an empty buffer still costs its uint64 length.

    Confirmed on two independently produced files: a 256-token one-checkpoint artifact
    written here on CPU, and the 2,048-token two-checkpoint artifact behind RA-003, made
    weeks earlier by a different build. Both come out exact.
    """

    G = HybridSizingTest.QWEN35
    NO_APPENDIX_256 = 173_679_168
    ONE_CHECKPOINT_256 = 330_573_584
    TWO_CHECKPOINTS_2048 = 604_958_676

    def test_it_predicts_the_file_written_here(self):
        self.assertEqual(
            hybrid_state_bytes(self.G, 256, header_tokens=260, checkpoints=1),
            self.ONE_CHECKPOINT_256)

    def test_it_predicts_the_ra003_artifact_it_never_saw(self):
        """A different build, a different prefix length, twice the checkpoints, weeks apart.

        This is the one that makes the terms worth anything: the appendix arithmetic was
        settled on a one-checkpoint file and this file has two.
        """
        self.assertEqual(
            hybrid_state_bytes(self.G, 2048, header_tokens=2052, checkpoints=2),
            self.TWO_CHECKPOINTS_2048)

    def test_one_checkpoint_roughly_doubles_a_short_artifact(self):
        """Each checkpoint carries a recurrent-only state, so the fixed tail is paid again.

        Defaulting checkpoints to zero would have under-predicted such a file by half - a
        space guard failing open, which is the direction that runs a disk out mid-admission.
        """
        without = hybrid_state_bytes(self.G, 256, header_tokens=260, checkpoints=0)
        self.assertEqual(without, self.NO_APPENDIX_256)
        self.assertGreater(self.ONE_CHECKPOINT_256 / without, 1.9)

    def test_each_further_checkpoint_costs_the_same(self):
        step = [hybrid_state_bytes(self.G, 256, header_tokens=260, checkpoints=n)
                for n in range(4)]
        # 16-byte record, three uint64 buffer lengths, an 8-byte preamble, the recurrent tail.
        per_checkpoint = 16 + 24 + 8 + 156_894_356
        self.assertEqual({step[i + 1] - step[i] for i in (1, 2)}, {per_checkpoint})

    def test_the_appendix_header_is_paid_once_not_per_checkpoint(self):
        """magic, version and count are written before the first record, not with each."""
        none = hybrid_state_bytes(self.G, 256, header_tokens=260, checkpoints=0)
        one = hybrid_state_bytes(self.G, 256, header_tokens=260, checkpoints=1)
        two = hybrid_state_bytes(self.G, 256, header_tokens=260, checkpoints=2)
        self.assertEqual((one - none) - (two - one), 12)

    def test_checkpoints_must_be_stated_and_has_no_default(self):
        with self.assertRaises(TypeError):
            hybrid_state_bytes(self.G, 2048)

    def test_a_negative_checkpoint_count_is_refused(self):
        with self.assertRaises(SizingError) as caught:
            hybrid_state_bytes(self.G, 2048, checkpoints=-1)
        self.assertIn("is negative", str(caught.exception))
