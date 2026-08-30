"""Sliding-window models: per-layer geometry, two sections, and the size law that lands.

A gemma4 model breaks every scalar this project had. `attention.head_count_kv` is one value
per layer, the head dimension differs between sliding and full-attention layers, and the
state file is not one attention section but two - `llama_kv_cache_iswa::state_write` emits
the base cache then the SWA cache back to back, each declaring only its own filtered layers.

The proof that the layout is understood is byte-exact sizing, and it is available offline:
docs/records/payoff/payoff-gemma-{stock,patched}.json recorded the saved file sizes of two
real gemma4-12b artifacts on this host. Nothing below is fitted to them - every term comes
from the writer in the pinned tree - and they are reproduced to the byte.

The reader bug they exposed is worth stating on its own. read_metadata() summarised an array
of more than eight elements as the string "[48 items]", which is the same string for every
48-element array in every model, so two different gemma4 models compared **equal** on the
field that most distinguishes them.
"""

import io
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kv_rosetta import gguf
from kv_rosetta.adapters.llamacpp_ggsq import (
    AttentionSection,
    GGSQError,
    LayerHeads,
    TensorSpan,
    materialise_section,
    read_attention_section,
    row_size as ggml_row_size,
)
from kv_rosetta.sizing import (
    KVGeometry,
    LayerKV,
    LayeredGeometry,
    SizingError,
    geometry_of,
    layered_geometry_of,
    layered_state_bytes,
    sliding_cells,
    state_bytes,
)

GEMMA4 = Path("/mnt/storage/models/gemma-4-12b-qat/gemma-4-12b-it-qat-q4_0.gguf")

#: gemma4-12b as its GGUF declares it: 48 layers in a repeating five-sliding/one-full
#: pattern, 8 KV heads of 256 on the sliding layers and 1 head of 512 on the others.
GEMMA4_SLIDING = tuple(bool((index + 1) % 6) for index in range(48))
GEMMA4_HEADS = tuple(8 if s else 1 for s in GEMMA4_SLIDING)
GEMMA4_GEOMETRY = LayeredGeometry(
    layers=tuple(LayerKV(index=index, n_kv_head=GEMMA4_HEADS[index],
                         head_dim=256 if GEMMA4_SLIDING[index] else 512,
                         value_head_dim=256 if GEMMA4_SLIDING[index] else 512,
                         sliding=GEMMA4_SLIDING[index])
                 for index in range(48)),
    architecture="gemma4", sliding_window=1024)

#: Saved slot sizes from docs/records/payoff/payoff-gemma-{stock,patched}.json. Both were
#: written by llama-server on this host over the same 7,371-cell sequence; the patched build
#: appends two SCKP checkpoints and the stock build appends none.
GEMMA4_CELLS = 7371
GEMMA4_STOCK_BYTES = 456_442_220
GEMMA4_PATCHED_BYTES = 1_127_557_496

QWEN2 = KVGeometry(n_layer=36, n_kv_head=2, head_dim=128, architecture="qwen2",
                   value_head_dim=128)
QWEN2_MEASURED = {128: 4_721_548, 8192: 302_121_868, 32000: 1_180_160_908}

F16 = 1


def gguf_file(pairs: list[tuple[str, int, object]]) -> Path:
    """A GGUF header declaring exactly these keys. Type ids are GGUF's own."""
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    def value(kind: int, item) -> bytes:
        if kind == 4:
            return struct.pack("<I", item)
        if kind == 6:
            return struct.pack("<f", item)
        if kind == 7:
            return struct.pack("<?", item)
        if kind == 8:
            return s(item)
        if kind == 9:
            element, items = item
            out = struct.pack("<IQ", element, len(items))
            return out + b"".join(value(element, one) for one in items)
        raise AssertionError(f"the fixture builder has no type {kind}")

    body = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs)))
    for key, kind, item in pairs:
        body += s(key) + struct.pack("<I", kind) + value(kind, item)
    path = Path(tempfile.mkdtemp()) / "model.gguf"
    path.write_bytes(bytes(body))
    return path


def gemma4_like(heads: tuple[int, ...], sliding: tuple[bool, ...], *, window: int = 1024,
                key_length: int = 512, key_length_swa: int = 256,
                arch: str = "gemma4", block_count: int | None = None) -> Path:
    pairs = [
        ("general.architecture", 8, arch),
        (f"{arch}.block_count", 4, len(heads) if block_count is None else block_count),
        (f"{arch}.attention.head_count", 4, 16),
        (f"{arch}.attention.head_count_kv", 9, (4, list(heads))),
        (f"{arch}.attention.sliding_window_pattern", 9, (7, list(sliding))),
        (f"{arch}.attention.sliding_window", 4, window),
        (f"{arch}.attention.key_length", 4, key_length),
        (f"{arch}.attention.value_length", 4, key_length),
        (f"{arch}.attention.key_length_swa", 4, key_length_swa),
        (f"{arch}.attention.value_length_swa", 4, key_length_swa),
        (f"{arch}.embedding_length", 4, 3840),
        ("rope.freq_base", 6, 1000000.0),
    ]
    return gguf_file(pairs)


def build_section(widths: list[int], cells: int, *, type_id: int = F16) -> bytes:
    """One attention section, one layer per entry in `widths`, from the writer's field order.

    Values carry the layer's own width, which is the point: a decoder handed one width for
    every layer reshapes the wrong number of heads and never notices.
    """
    body = struct.pack("<I", 1) + struct.pack("<I", cells)
    if cells == 0:
        return body
    for cell in range(cells):
        body += struct.pack("<i", cell) + struct.pack("<I", 1) + struct.pack("<i", 0)
    body += struct.pack("<I", 0)                       # v_trans
    body += struct.pack("<I", len(widths))
    for kind in ("k", "v"):
        for position, width in enumerate(widths):
            stride = ggml_row_size(type_id, width)
            body += struct.pack("<i", type_id) + struct.pack("<Q", stride)
            fill = position + (0 if kind == "k" else 100)
            body += struct.pack(f"<{width * cells}e",
                                *[float(fill)] * (width * cells))
    return body


class MetadataArrayTests(unittest.TestCase):
    """The summary that compared equal to a different model's summary."""

    def test_a_long_array_is_summarised_rather_than_materialised(self):
        path = gemma4_like(GEMMA4_HEADS, GEMMA4_SLIDING)
        value = gguf.read_metadata(path)["gemma4.attention.head_count_kv"]
        self.assertIsInstance(value, gguf.TruncatedArray)
        self.assertEqual(len(value), 48)
        self.assertEqual(str(value), "[48 items]")

    def test_a_summarised_array_refuses_to_be_compared(self):
        """The bug in one line: a summary carries no elements, so it can answer neither
        "same" nor "different", and both answers were wrong for a real pair of models."""
        left = gguf.TruncatedArray(48, 4)
        right = gguf.TruncatedArray(48, 4)
        with self.assertRaises(gguf.GGUFError) as caught:
            left == right                                     # noqa: B015
        self.assertIn("summarised rather than read", str(caught.exception))
        with self.assertRaises(gguf.GGUFError):
            left != right                                     # noqa: B015

    def test_full_arrays_returns_the_elements(self):
        path = gemma4_like(GEMMA4_HEADS, GEMMA4_SLIDING)
        read = gguf.read_metadata(path, full_arrays=("attention.head_count_kv",))
        self.assertEqual(tuple(read["gemma4.attention.head_count_kv"]), GEMMA4_HEADS)
        # Only the named array is materialised; the rest still summarise.
        self.assertIsInstance(read["gemma4.attention.sliding_window_pattern"],
                              gguf.TruncatedArray)

    def test_a_short_array_is_still_returned_whole(self):
        """Under nine elements needs no summary, and that is why the four-layer gemma4
        was caught by the old refusal and the forty-eight-layer one was not."""
        path = gemma4_like((16, 16, 16, 4), (True, True, True, False))
        value = gguf.read_metadata(path)["gemma4.attention.head_count_kv"]
        self.assertEqual(value, [16, 16, 16, 4])


class CrossModelGateGeometryTests(unittest.TestCase):
    """The latent bug: two different models admitted as the same geometry."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import cross_model_gate
        self.gate = cross_model_gate

    def test_two_models_differing_only_per_layer_are_refused(self):
        """Both declare 48 layers, 48 KV head counts and the same head dimensions. Read as
        summaries they were identical strings and the pair was admitted."""
        a = gemma4_like(GEMMA4_HEADS, GEMMA4_SLIDING)
        other = tuple(4 if head == 8 else head for head in GEMMA4_HEADS)
        b = gemma4_like(other, GEMMA4_SLIDING)
        geo_a, geo_b = self.gate.geometry_of(str(a)), self.gate.geometry_of(str(b))
        self.assertEqual(tuple(geo_a["n_head_kv"]), GEMMA4_HEADS)
        with self.assertRaises(self.gate.GateError) as caught:
            self.gate.require_same_geometry(geo_a, geo_b)
        self.assertIn("n_head_kv", str(caught.exception))

    def test_two_models_differing_only_in_their_window_pattern_are_refused(self):
        a = gemma4_like(GEMMA4_HEADS, GEMMA4_SLIDING)
        flipped = tuple(bool(index % 6) for index in range(48))
        b = gemma4_like(GEMMA4_HEADS, flipped)
        with self.assertRaises(self.gate.GateError) as caught:
            self.gate.require_same_geometry(self.gate.geometry_of(str(a)),
                                            self.gate.geometry_of(str(b)))
        self.assertIn("sliding_window_pattern", str(caught.exception))

    def test_the_swa_head_dimension_is_compared(self):
        a = gemma4_like(GEMMA4_HEADS, GEMMA4_SLIDING, key_length_swa=256)
        b = gemma4_like(GEMMA4_HEADS, GEMMA4_SLIDING, key_length_swa=128)
        with self.assertRaises(self.gate.GateError) as caught:
            self.gate.require_same_geometry(self.gate.geometry_of(str(a)),
                                            self.gate.geometry_of(str(b)))
        self.assertIn("key_length_swa", str(caught.exception))

    def test_a_model_still_matches_itself(self):
        """Refusing everything would pass the tests above and break the gate. A per-layer
        model must still compare equal to a model declaring the same layers."""
        a = gemma4_like(GEMMA4_HEADS, GEMMA4_SLIDING)
        b = gemma4_like(GEMMA4_HEADS, GEMMA4_SLIDING)
        self.gate.require_same_geometry(self.gate.geometry_of(str(a)),
                                        self.gate.geometry_of(str(b)))

    def test_an_array_left_summarised_is_refused_rather_than_compared(self):
        """The defence against the next per-layer key nobody thought to read in full."""
        path = gemma4_like(GEMMA4_HEADS, GEMMA4_SLIDING)
        real = gguf.read_metadata

        def summarise_everything(*args, **kwargs):
            kwargs["full_arrays"] = ()
            return real(*args, **kwargs)

        with mock.patch.object(self.gate.gguf, "read_metadata", summarise_everything):
            with self.assertRaises(self.gate.GateError) as caught:
                self.gate.geometry_of(str(path))
        self.assertIn("summarised rather than read", str(caught.exception))


class LayeredGeometryReadTests(unittest.TestCase):
    @unittest.skipUnless(GEMMA4.is_file(), "the gemma4-12b model is not on this host")
    def test_the_real_model_reads_as_forty_eight_differing_layers(self):
        geometry = layered_geometry_of(GEMMA4)
        self.assertEqual(geometry, GEMMA4_GEOMETRY)
        self.assertFalse(geometry.is_uniform)
        self.assertEqual(len(geometry.base_layers), 8)
        self.assertEqual(len(geometry.sliding_layers), 40)

    @unittest.skipUnless(GEMMA4.is_file(), "the gemma4-12b model is not on this host")
    def test_the_scalar_reader_refuses_it_by_name(self):
        """geometry_of used to reach int('[48 items]') and raise a bare ValueError about
        string parsing - a refusal by accident, from the wrong type, saying nothing."""
        with self.assertRaises(SizingError) as caught:
            geometry_of(GEMMA4)
        self.assertIn("per-layer head_count_kv", str(caught.exception))

    def test_a_summarised_head_count_is_refused_not_broadcast(self):
        declared = {"gemma4.block_count": 48,
                    "gemma4.attention.head_count_kv": gguf.TruncatedArray(48, 4),
                    "gemma4.attention.key_length": 512}
        with mock.patch("kv_rosetta.sizing.gguf.architecture", return_value="gemma4"), \
             mock.patch("kv_rosetta.sizing.gguf.read_metadata", return_value=declared):
            with self.assertRaises(SizingError) as caught:
                layered_geometry_of("ignored.gguf")
        self.assertIn("summarised rather than read", str(caught.exception))

    def test_a_head_count_array_of_the_wrong_length_is_refused(self):
        path = gemma4_like(GEMMA4_HEADS[:40], GEMMA4_SLIDING, block_count=48)
        with self.assertRaises(SizingError) as caught:
            layered_geometry_of(path)
        self.assertIn("cannot be matched up", str(caught.exception))

    def test_a_pattern_of_the_wrong_length_is_refused(self):
        path = gemma4_like(GEMMA4_HEADS, GEMMA4_SLIDING[:40], block_count=48)
        with self.assertRaises(SizingError) as caught:
            layered_geometry_of(path)
        self.assertIn("sliding_window_pattern entries", str(caught.exception))

    def test_a_scalar_pattern_is_refused_because_it_has_meant_two_things(self):
        pairs = [("general.architecture", 8, "gemma4"), ("gemma4.block_count", 4, 48),
                 ("gemma4.attention.head_count_kv", 9, (4, list(GEMMA4_HEADS))),
                 ("gemma4.attention.sliding_window_pattern", 4, 6),
                 ("gemma4.attention.sliding_window", 4, 1024),
                 ("gemma4.attention.key_length", 4, 512),
                 ("gemma4.attention.key_length_swa", 4, 256)]
        with self.assertRaises(SizingError) as caught:
            layered_geometry_of(gguf_file(pairs))
        self.assertIn("repeat period", str(caught.exception))

    def test_a_missing_block_count_or_head_count_is_refused(self):
        for absent in ("gemma4.block_count", "gemma4.attention.head_count_kv"):
            declared = {"gemma4.block_count": 48,
                        "gemma4.attention.head_count_kv": [1] * 48,
                        "gemma4.attention.key_length": 512}
            declared.pop(absent)
            with self.subTest(absent=absent):
                with mock.patch("kv_rosetta.sizing.gguf.architecture",
                                return_value="gemma4"), \
                     mock.patch("kv_rosetta.sizing.gguf.read_metadata",
                                return_value=declared):
                    with self.assertRaises(SizingError) as caught:
                        layered_geometry_of("ignored.gguf")
                self.assertIn(absent, str(caught.exception))

    def test_zero_blocks_is_refused_by_name(self):
        """Deleting this guard lands on the empty-geometry refusal instead, which is a
        different sentence about a different mistake."""
        declared = {"gemma4.block_count": 0, "gemma4.attention.head_count_kv": 1,
                    "gemma4.attention.key_length": 512}
        with mock.patch("kv_rosetta.sizing.gguf.architecture", return_value="gemma4"), \
             mock.patch("kv_rosetta.sizing.gguf.read_metadata", return_value=declared):
            with self.assertRaises(SizingError) as caught:
                layered_geometry_of("ignored.gguf")
        self.assertIn("declares 0 blocks", str(caught.exception))

    def test_a_model_declaring_no_key_length_is_refused(self):
        declared = {"gemma4.block_count": 48, "gemma4.attention.head_count_kv": [1] * 48}
        with mock.patch("kv_rosetta.sizing.gguf.architecture", return_value="gemma4"), \
             mock.patch("kv_rosetta.sizing.gguf.read_metadata", return_value=declared):
            with self.assertRaises(SizingError) as caught:
                layered_geometry_of("ignored.gguf")
        self.assertIn("no single embedding_length", str(caught.exception))

    def test_sliding_layers_without_a_swa_head_dimension_are_refused(self):
        declared = {"gemma4.block_count": 4, "gemma4.attention.head_count_kv": [8] * 4,
                    "gemma4.attention.key_length": 512,
                    "gemma4.attention.sliding_window": 1024,
                    "gemma4.attention.sliding_window_pattern": [True, True, False, False]}
        with mock.patch("kv_rosetta.sizing.gguf.architecture", return_value="gemma4"), \
             mock.patch("kv_rosetta.sizing.gguf.read_metadata", return_value=declared):
            with self.assertRaises(SizingError) as caught:
                layered_geometry_of("ignored.gguf")
        self.assertIn("key_length_swa", str(caught.exception))

    def test_a_nonsense_head_count_is_refused_before_it_becomes_a_geometry(self):
        path = gemma4_like(tuple(0 for _ in range(48)), GEMMA4_SLIDING)
        with self.assertRaises(SizingError) as caught:
            layered_geometry_of(path)
        self.assertIn("n_kv_head must be positive", str(caught.exception))

    def test_a_model_with_no_sliding_layers_reads_as_uniform(self):
        path = gemma4_like(tuple([2] * 8), tuple([False] * 8), window=0)
        geometry = layered_geometry_of(path)
        self.assertTrue(geometry.is_uniform)
        self.assertEqual(geometry.sliding_layers, ())
        self.assertEqual(len(geometry.base_layers), 8)


class SlidingCellCountTests(unittest.TestCase):
    def test_the_window_bounds_the_sliding_section(self):
        """From the writer: a cell is skipped when pos_max - pos >= n_swa, so over
        contiguous positions exactly the last n_swa survive."""
        self.assertEqual(sliding_cells(GEMMA4_GEOMETRY, 7371), 1024)
        self.assertEqual(sliding_cells(GEMMA4_GEOMETRY, 512), 512)
        self.assertEqual(sliding_cells(GEMMA4_GEOMETRY, 1024), 1024)

    def test_a_model_without_a_window_has_no_count_to_take(self):
        with self.assertRaises(SizingError) as caught:
            sliding_cells(QWEN2.layered(), 128)
        self.assertIn("no sliding-window layers", str(caught.exception))

    def test_an_unverified_window_shape_is_refused_rather_than_assumed_standard(self):
        """swa_type is assigned in C++, not declared in the GGUF. A chunked window keeps
        whole chunks and a symmetric one keeps half a window either side."""
        elsewhere = LayeredGeometry(layers=GEMMA4_GEOMETRY.layers, architecture="gemma3",
                                    sliding_window=1024)
        with self.assertRaises(SizingError) as caught:
            sliding_cells(elsewhere, 7371)
        self.assertIn("set in llama.cpp", str(caught.exception))

    def test_a_negative_cell_count_is_refused(self):
        with self.assertRaises(SizingError) as caught:
            sliding_cells(GEMMA4_GEOMETRY, -1)
        self.assertIn("negative", str(caught.exception))


class MeasuredGemma4ArtifactTest(unittest.TestCase):
    """The strongest evidence available offline: two real files, to the byte."""

    def test_it_predicts_the_stock_build_artifact_exactly(self):
        predicted = layered_state_bytes(
            GEMMA4_GEOMETRY, GEMMA4_CELLS,
            sliding_cell_count=sliding_cells(GEMMA4_GEOMETRY, GEMMA4_CELLS),
            checkpoint_cells=(), header_tokens=GEMMA4_CELLS + 4)
        self.assertEqual(predicted, GEMMA4_STOCK_BYTES)

    def test_it_predicts_the_checkpoint_persisting_artifact_exactly(self):
        """The patched build appended two SCKP records. Each is written with PARTIAL_ONLY,
        which llama_kv_cache_iswa honours by skipping the base cache, so each carries a
        sliding section and nothing else - 60% of this file."""
        window = sliding_cells(GEMMA4_GEOMETRY, GEMMA4_CELLS)
        predicted = layered_state_bytes(
            GEMMA4_GEOMETRY, GEMMA4_CELLS, sliding_cell_count=window,
            checkpoint_cells=(window, window), header_tokens=GEMMA4_CELLS + 4)
        self.assertEqual(predicted, GEMMA4_PATCHED_BYTES)

    def test_one_section_would_not_have_predicted_either(self):
        """Reading the file as a single 48-layer section is the shape this replaces. It is
        not a rounding error: it is 2.3x the real file."""
        single = LayeredGeometry(
            layers=tuple(LayerKV(index=layer.index, n_kv_head=layer.n_kv_head,
                                 head_dim=layer.head_dim,
                                 value_head_dim=layer.value_head_dim)
                         for layer in GEMMA4_GEOMETRY.layers),
            architecture="gemma4")
        naive = layered_state_bytes(single, GEMMA4_CELLS, checkpoint_cells=(),
                                    header_tokens=GEMMA4_CELLS + 4)
        self.assertGreater(naive, 2 * GEMMA4_STOCK_BYTES)

    def test_the_sliding_section_stops_growing_past_the_window(self):
        """Past 1,024 cells only the base section grows, so a per-token rate taken from a
        short gemma4 artifact over-predicts a long one."""
        def size(cells):
            return layered_state_bytes(
                GEMMA4_GEOMETRY, cells,
                sliding_cell_count=sliding_cells(GEMMA4_GEOMETRY, cells),
                checkpoint_cells=(), header_tokens=cells)
        short = size(513) - size(512)
        long = size(2001) - size(2000)
        self.assertGreater(short, long)
        self.assertEqual(long, 4 + 12 + 8 * 2 * 1024)


class UniformModelsUseTheSameLawTest(unittest.TestCase):
    def test_the_layered_law_reproduces_the_qwen2_artifacts(self):
        """A second implementation of the same arithmetic is a place for the two to drift
        apart, and only one of them has artifacts."""
        for cells, actual in QWEN2_MEASURED.items():
            with self.subTest(cells=cells):
                self.assertEqual(
                    layered_state_bytes(QWEN2.layered(), cells, checkpoint_cells=(),
                                        header_tokens=cells + 4), actual)
                self.assertEqual(state_bytes(QWEN2, cells, header_tokens=cells + 4), actual)


class LayeredSizingRefusalTests(unittest.TestCase):
    def test_a_hybrid_architecture_is_refused(self):
        hybrid = LayeredGeometry(layers=QWEN2.layered().layers, architecture="qwen35")
        with self.assertRaises(SizingError) as caught:
            layered_state_bytes(hybrid, 128, checkpoint_cells=())
        self.assertIn("hybrid", str(caught.exception))

    def test_a_recurrent_architecture_is_refused(self):
        recurrent = LayeredGeometry(layers=QWEN2.layered().layers, architecture="mamba")
        with self.assertRaises(SizingError) as caught:
            layered_state_bytes(recurrent, 128, checkpoint_cells=())
        self.assertIn("recurrent", str(caught.exception))

    def test_a_nonsense_layer_is_refused(self):
        broken = LayeredGeometry(layers=(LayerKV(0, 0, 128, 128),), architecture="llama")
        with self.assertRaises(SizingError) as caught:
            layered_state_bytes(broken, 128, checkpoint_cells=())
        self.assertIn("n_kv_head must be positive", str(caught.exception))

    def test_a_negative_cell_count_is_refused(self):
        """header_tokens is passed so the header check does not shadow this one, and a
        model with no sliding layers is used so the window check does not either."""
        with self.assertRaises(SizingError) as caught:
            layered_state_bytes(QWEN2.layered(), -1, checkpoint_cells=(), header_tokens=5)
        self.assertIn("cell count -1 is negative", str(caught.exception))

    def test_a_negative_header_token_count_is_refused(self):
        with self.assertRaises(SizingError) as caught:
            layered_state_bytes(QWEN2.layered(), 128, checkpoint_cells=(),
                                header_tokens=-1)
        self.assertIn("header token count", str(caught.exception))

    def test_a_sliding_model_will_not_size_without_its_sliding_cell_count(self):
        """Defaulting it to the sequence length would over-predict gemma4-12b at 7,371
        cells by 2.3 GB; defaulting it to zero would under-predict by 74%."""
        with self.assertRaises(SizingError) as caught:
            layered_state_bytes(GEMMA4_GEOMETRY, GEMMA4_CELLS, checkpoint_cells=())
        self.assertIn("state that count", str(caught.exception))

    def test_a_sliding_cell_count_on_a_model_without_a_window_is_refused(self):
        with self.assertRaises(SizingError) as caught:
            layered_state_bytes(QWEN2.layered(), 128, sliding_cell_count=64,
                                checkpoint_cells=())
        self.assertIn("describes nothing", str(caught.exception))

    def test_a_negative_sliding_cell_count_is_refused(self):
        with self.assertRaises(SizingError) as caught:
            layered_state_bytes(GEMMA4_GEOMETRY, 2048, sliding_cell_count=-1,
                                checkpoint_cells=())
        self.assertIn("sliding cell count -1 is negative", str(caught.exception))

    def test_a_sliding_section_larger_than_the_sequence_is_refused(self):
        with self.assertRaises(SizingError) as caught:
            layered_state_bytes(GEMMA4_GEOMETRY, 10, sliding_cell_count=20,
                                checkpoint_cells=())
        self.assertIn("cannot hold 20 cells", str(caught.exception))

    def test_a_checkpoint_of_a_model_with_no_window_is_refused_by_name(self):
        """Deleting this guard lands on the checkpoint-extent check instead, which says
        something true about a different mistake. PARTIAL_ONLY on a model with one cache
        writes the whole attention section, and that payload has never been measured."""
        with self.assertRaises(SizingError) as caught:
            layered_state_bytes(QWEN2.layered(), 128, checkpoint_cells=(4,))
        self.assertIn("PARTIAL_ONLY", str(caught.exception))

    def test_a_checkpoint_wider_than_the_window_is_refused(self):
        for counts in ((-1,), (2048,)):
            with self.subTest(counts=counts):
                with self.assertRaises(SizingError) as caught:
                    layered_state_bytes(GEMMA4_GEOMETRY, 4096, sliding_cell_count=1024,
                                        checkpoint_cells=counts)
                self.assertIn("do not all lie", str(caught.exception))

    def test_an_empty_section_carries_only_its_two_counts(self):
        """The writer emits n_stream and cell_count and then `continue`s - no meta, no
        v_trans, no per-layer headers."""
        empty = layered_state_bytes(QWEN2.layered(), 0, checkpoint_cells=(),
                                    header_tokens=0)
        self.assertEqual(empty, 12 + 8)


class PerLayerTensorTests(unittest.TestCase):
    """Spans must carry the model's layer number, and be read at that layer's width."""

    #: Two "sliding" layers of 8 heads x 4 and one full-attention layer of 1 head x 32.
    #: All three rows are the same 32 elements wide **on purpose**: that is gemma4's shape,
    #: where 8x256 and 1x2048 would both be 2048, and it is why the width check inside
    #: materialise cannot catch a wrong head geometry.
    WIDTHS = [8 * 4, 8 * 4, 1 * 32]

    def section(self, indices):
        body = build_section(self.WIDTHS, cells=2)
        handle = io.BytesIO(body)
        return handle, read_attention_section(handle, 0, len(body), has_cell_ext=False,
                                              layer_indices=indices)

    def test_a_filtered_section_reports_the_model_layers_it_was_given(self):
        """gemma4-12b's SWA section holds layers 0,1,2,3,4,6,... Read as 0..39 every
        tensor after the fifth is attributed to a layer that also exists, so nothing
        downstream notices."""
        _handle, section = self.section((0, 1, 5))
        self.assertEqual([span.layer_index for span in section.spans if span.kind == "k"],
                         [0, 1, 5])

    def test_the_default_map_is_the_section_position(self):
        _handle, section = self.section(None)
        self.assertEqual([span.layer_index for span in section.spans if span.kind == "k"],
                         [0, 1, 2])

    def test_a_layer_map_of_the_wrong_length_is_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.section((0, 1))
        self.assertIn("named 2 model layers", str(caught.exception))

    def test_a_layer_map_naming_a_layer_twice_is_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.section((0, 1, 1))
        self.assertIn("names a model layer twice", str(caught.exception))

    def test_a_negative_model_layer_is_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.section((0, 1, -1))
        self.assertIn("negative model layer", str(caught.exception))

    def test_every_layer_is_read_at_its_own_width(self):
        handle, section = self.section((0, 1, 5))
        layout = {0: LayerHeads(8, 4, 4), 1: LayerHeads(8, 4, 4), 5: LayerHeads(1, 32, 32)}
        tensors = materialise_section(handle, section, layout)
        self.assertEqual(sorted(tensors), [(0, "k"), (0, "v"), (1, "k"), (1, "v"),
                                           (5, "k"), (5, "v")])
        self.assertEqual(tensors[(0, "k")].shape, (2, 8, 4))
        self.assertEqual(tensors[(5, "k")].shape, (2, 1, 32))
        # The fill identifies the span, so a layer read at the wrong offset shows up here.
        self.assertEqual(tensors[(1, "k")].reshape(-1)[0], 1.0)
        self.assertEqual(tensors[(5, "v")].reshape(-1)[0], 102.0)

    def test_one_head_geometry_for_every_layer_returns_heads_that_were_never_heads(self):
        """This is what the per-layer layout replaces, and why it is not optional. 8x4 and
        1x32 are both 32 wide, so materialise's width check passes and the reshape succeeds
        - it just splits one head into eight. Nothing raises, and every later comparison is
        between arrays of the wrong shape."""
        handle, section = self.section((0, 1, 5))
        uniform = {index: LayerHeads(8, 4, 4) for index in (0, 1, 5)}
        wrong = materialise_section(handle, section, uniform)
        self.assertEqual(wrong[(5, "k")].shape, (2, 8, 4))

        handle.seek(0)
        layout = {0: LayerHeads(8, 4, 4), 1: LayerHeads(8, 4, 4), 5: LayerHeads(1, 32, 32)}
        self.assertEqual(materialise_section(handle, section, layout)[(5, "k")].shape,
                         (2, 1, 32))

    def test_a_layer_the_layout_does_not_describe_is_refused(self):
        """No fallback: the layer that would be defaulted here is exactly the one whose
        geometry differs, and a default is a shape this decoder invented."""
        handle, section = self.section((0, 1, 5))
        with self.assertRaises(GGSQError) as caught:
            materialise_section(handle, section, {0: LayerHeads(8, 4, 4)})
        self.assertIn("refusing to pick a geometry", str(caught.exception))

    def test_a_span_that_is_not_attention_kv_is_refused(self):
        """A recurrent span has no head geometry, and picking the value dimension for it
        would return an array shaped like something it is not."""
        handle, _section = self.section((0, 1, 5))
        span = TensorSpan(layer_index=0, kind="r", type_id=F16, type_name="f16",
                          row_size=64, n_elements=32, offset=0, nbytes=128)
        fake = AttentionSection(1, 2, False, 1, spans=(span,))
        with self.assertRaises(GGSQError) as caught:
            materialise_section(handle, fake, {0: LayerHeads(8, 4, 4)})
        self.assertIn("not attention K or V", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
