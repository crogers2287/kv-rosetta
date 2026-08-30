"""Section bounds must be exact, and anything inexact must be refused.

The SCKP record was assumed to be 12 bytes when the writer emits 16, and every test passed
against that assumption for a day because the fixtures shared it. So these fixtures are built
from the writer's field order in docs/ggsq-layout.md, and the oracle - expected offsets and
sizes - is computed independently of the parser under test.
"""

import io
import struct
import unittest

from kv_rosetta.adapters.llamacpp_ggsq import (
    GGML_TYPES,
    GGSQError,
    Reader,
    elements_from_row_size,
    read_attention_section,
    row_size,
)

F16, Q8_0 = 1, 8


def build_attention(cell_count=4, n_layer=2, k_type=F16, k_elems=256, v_trans=False,
                    n_seq_id=1, cell_ext=b"", trailing=b"", declared_k_type=None,
                    declared_size_el=None):
    """Corrupt cases are constructed, not byte-patched.

    Searching a built body for a pattern to overwrite is unreliable - the same bytes occur
    in the meta section - so the builder writes the wrong value directly.
    """
    """An attention section built from the writer's field order, not from the parser."""
    body = struct.pack("<I", 1)                    # n_stream
    body += struct.pack("<I", cell_count)
    for cell in range(cell_count):
        body += struct.pack("<i", cell)            # pos
        body += struct.pack("<I", n_seq_id)
        body += cell_ext
        body += struct.pack(f"<{n_seq_id}i", *range(n_seq_id))
    body += struct.pack("<I", 1 if v_trans else 0)
    body += struct.pack("<I", n_layer)
    k_stride = row_size(k_type, k_elems)
    for _ in range(n_layer):
        body += struct.pack("<i", k_type if declared_k_type is None else declared_k_type)
        body += struct.pack("<Q", k_stride)
        body += bytes(cell_count * k_stride)
    for _ in range(n_layer):
        body += struct.pack("<i", k_type)
        if not v_trans:
            body += struct.pack("<Q", k_stride) + bytes(cell_count * k_stride)
        else:
            size_el = GGML_TYPES[k_type][2] // GGML_TYPES[k_type][1]
            written = size_el if declared_size_el is None else declared_size_el
            body += struct.pack("<I", written) + struct.pack("<I", k_elems)
            body += bytes(k_elems * cell_count * size_el)
    return body + trailing


def section(body):
    handle = io.BytesIO(body)
    return read_attention_section(handle, 0, len(body), has_cell_ext=False)


class RowSizeTest(unittest.TestCase):
    def test_row_size_accounts_for_block_quantisation(self):
        self.assertEqual(row_size(F16, 256), 512)          # 1 elem/block, 2 bytes
        self.assertEqual(row_size(Q8_0, 256), 256 // 32 * 34)

    def test_row_size_round_trips_through_elements(self):
        for type_id in (F16, Q8_0, 0, 30):
            with self.subTest(type_id=type_id):
                self.assertEqual(elements_from_row_size(type_id, row_size(type_id, 256)), 256)

    def test_an_unsupported_type_is_refused(self):
        with self.assertRaises(GGSQError):
            row_size(999, 256)
        with self.assertRaises(GGSQError):
            elements_from_row_size(999, 512)

    def test_a_width_not_matching_the_block_is_refused(self):
        with self.assertRaises(GGSQError) as caught:
            row_size(Q8_0, 250)
        self.assertIn("not a multiple", str(caught.exception))

    def test_a_stride_not_matching_the_block_is_refused(self):
        with self.assertRaises(GGSQError):
            elements_from_row_size(Q8_0, 33)


class ReaderBoundsTest(unittest.TestCase):
    def reader(self, payload=b"\x00" * 16, end=None):
        return Reader(io.BytesIO(payload), 0, len(payload) if end is None else end)

    def test_a_read_past_the_section_end_is_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.reader(end=8).take(9)
        self.assertIn("passes the end of the section", str(caught.exception))

    def test_a_skip_past_the_section_end_is_refused(self):
        with self.assertRaises(GGSQError):
            self.reader(end=8).skip(9)

    def test_a_short_file_is_refused_even_inside_the_declared_section(self):
        reader = Reader(io.BytesIO(b"\x00" * 4), 0, 64)
        with self.assertRaises(GGSQError) as caught:
            reader.take(16)
        self.assertIn("file ended", str(caught.exception))

    def test_negative_reads_and_skips_are_refused(self):
        with self.assertRaises(GGSQError):
            self.reader().take(-1)
        with self.assertRaises(GGSQError):
            self.reader().skip(-1)


class AttentionSectionTest(unittest.TestCase):
    def test_a_well_formed_section_ends_exactly_at_the_body_end(self):
        body = build_attention()
        found = section(body)
        self.assertEqual(found.end_offset, len(body),
                         "the parse did not consume the section exactly")
        self.assertEqual(found.cell_count, 4)
        self.assertEqual(found.n_layer, 2)
        self.assertEqual(len(found.spans), 4)          # 2 layers x (k, v)

    def test_spans_locate_payloads_at_independently_computed_offsets(self):
        cell_count, n_layer, k_elems = 4, 2, 256
        body = build_attention(cell_count=cell_count, n_layer=n_layer, k_elems=k_elems)
        found = section(body)
        stride = row_size(F16, k_elems)
        # Oracle: recompute where each payload must start, from the writer's field order.
        offset = 4 + 4 + cell_count * (4 + 4 + 4)      # header + per-cell meta
        offset += 4 + 4                                # v_trans + n_layer
        for layer in range(n_layer):
            offset += 4 + 8                            # k_type + k_size_row
            span = found.spans[layer]
            self.assertEqual(span.offset, offset, f"k layer {layer} offset")
            self.assertEqual(span.nbytes, cell_count * stride)
            self.assertEqual(span.n_elements, k_elems)
            offset += cell_count * stride
        for layer in range(n_layer):
            offset += 4 + 8
            span = found.spans[n_layer + layer]
            self.assertEqual(span.offset, offset, f"v layer {layer} offset")
            offset += cell_count * stride

    def test_positions_are_recovered_in_order(self):
        self.assertEqual(section(build_attention(cell_count=5)).positions, (0, 1, 2, 3, 4))

    def test_a_variable_sequence_id_count_shifts_the_payload_correctly(self):
        body = build_attention(n_seq_id=3)
        self.assertEqual(section(body).end_offset, len(body))

    def test_a_cell_extension_must_be_supplied_not_guessed(self):
        body = build_attention(cell_ext=b"\x00" * 8)
        # Parsed without being told about the extension, the meta walk desynchronises.
        with self.assertRaises(GGSQError):
            section(body)
        handle = io.BytesIO(body)
        told = read_attention_section(handle, 0, len(body), has_cell_ext=True,
                                      cell_ext_size=8)
        self.assertEqual(told.end_offset, len(body))

    def test_the_transposed_value_layout_is_parsed_and_marked(self):
        body = build_attention(v_trans=True)
        found = section(body)
        self.assertEqual(found.end_offset, len(body))
        self.assertTrue(found.v_trans)
        transposed = [s for s in found.spans if s.kind == "v"]
        self.assertTrue(all(s.transposed for s in transposed))

    def test_a_declared_element_size_contradicting_the_type_is_refused(self):
        # f16 is 2 bytes per element; a header claiming 7 contradicts its own type.
        body = build_attention(v_trans=True, declared_size_el=7)
        with self.assertRaises(GGSQError) as caught:
            section(body)
        self.assertIn("contradicts itself", str(caught.exception))

    def test_a_truncated_section_is_refused(self):
        body = build_attention()
        with self.assertRaises(GGSQError):
            section(body[:-32])

    def test_trailing_unexplained_bytes_are_detectable(self):
        body = build_attention(trailing=b"\xff" * 16)
        found = section(body)
        self.assertLess(found.end_offset, len(body),
                        "trailing bytes must remain visible to the caller, not absorbed")

    def test_an_unsupported_ggml_type_is_refused(self):
        body = build_attention(declared_k_type=999)
        with self.assertRaises(GGSQError) as caught:
            section(body)
        self.assertIn("unsupported ggml type", str(caught.exception))

    def test_an_implausible_layer_count_is_refused(self):
        body = build_attention()
        broken = bytearray(body)
        marker = 4 + 4 + 4 * (4 + 4 + 4)
        broken[marker + 4:marker + 8] = struct.pack("<I", 100000)
        with self.assertRaises(GGSQError) as caught:
            section(bytes(broken))
        # Assert the specific refusal: without this the test passes on the bounds check
        # instead, and the plausibility guard proves nothing. A mutation run caught that.
        self.assertIn("implausible n_layer", str(caught.exception))

    def test_a_multi_stream_body_is_refused_rather_than_half_parsed(self):
        body = bytearray(build_attention())
        body[0:4] = struct.pack("<I", 2)
        with self.assertRaises(GGSQError) as caught:
            section(bytes(body))
        self.assertIn("only a single", str(caught.exception))

    def test_an_empty_stream_consumes_only_its_header(self):
        body = struct.pack("<I", 1) + struct.pack("<I", 0)
        found = section(body)
        self.assertEqual(found.cell_count, 0)
        self.assertEqual(found.end_offset, 8)


if __name__ == "__main__":
    unittest.main()


def build_recurrent(cell_count=3, n_layer=4, layers=(0, 2), r_type=F16, r_elems=512,
                    n_seq_id=0, trailing=b""):
    """A recurrent section built from the writer's field order.

    Note n_seq_id defaults to 0: a per-sequence save writes none, unlike the attention meta.

    There is **no leading n_stream** here. `llama_memory_recurrent::state_write` writes
    cell_count first, unlike the attention writer. This builder emitted one anyway and the
    parser read one, so every test below passed while both were wrong together - the same
    shape of fault as the 12-versus-16-byte checkpoint record. A real qwen35 artifact settled
    it: the section begins with cell_count = 1.
    """
    body = struct.pack("<I", cell_count)
    for cell in range(cell_count):
        body += struct.pack("<i", cell) + struct.pack("<I", n_seq_id)
        if n_seq_id:
            body += struct.pack(f"<{n_seq_id}i", *range(n_seq_id))
    body += struct.pack("<I", 0)                       # s_trans
    body += struct.pack("<I", n_layer)
    stride = row_size(r_type, r_elems)
    for _kind in ("r", "s"):
        for _layer in layers:
            body += struct.pack("<i", r_type) + struct.pack("<Q", stride)
            body += bytes(cell_count * stride)
    return body + trailing


def build_sckp(checkpoints=1, layers=(0, 2), draft=b"", spec=b""):
    target = build_recurrent(layers=layers)
    body = b"SCKP" + struct.pack("<II", 1, checkpoints)
    for index in range(checkpoints):
        body += struct.pack("<qii", 252, 0, 251)
        body += struct.pack("<Q", len(target)) + target
        body += struct.pack("<Q", len(draft)) + draft
        body += struct.pack("<Q", len(spec)) + spec
    return body


class RecurrentSectionTest(unittest.TestCase):
    def parse(self, body, layers=(0, 2), **kwargs):
        from kv_rosetta.adapters.llamacpp_ggsq import read_recurrent_section
        return read_recurrent_section(io.BytesIO(body), 0, len(body),
                                      recurrent_layers=layers, **kwargs)

    def test_a_well_formed_section_consumes_exactly(self):
        body = build_recurrent()
        found = self.parse(body)
        self.assertEqual(found.end_offset, len(body))
        self.assertEqual(len(found.spans), 4)          # 2 layers x (r, s)
        self.assertEqual([s.layer_index for s in found.spans], [0, 2, 0, 2])
        self.assertEqual([s.kind for s in found.spans], ["r", "r", "s", "s"])

    def test_the_layer_map_attributes_payloads_to_source_layers(self):
        found = self.parse(build_recurrent(layers=(1, 3)), layers=(1, 3))
        self.assertEqual([s.layer_index for s in found.spans], [1, 3, 1, 3])

    def test_a_wrong_layer_map_does_not_silently_succeed(self):
        # Three layers claimed where two were written: the parse must not fit.
        with self.assertRaises(GGSQError):
            self.parse(build_recurrent(layers=(0, 2)), layers=(0, 1, 2))

    def test_a_layer_map_outside_the_layer_count_is_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.parse(build_recurrent(), layers=(0, 99))
        self.assertIn("outside", str(caught.exception))

    def test_a_repeated_layer_in_the_map_is_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.parse(build_recurrent(), layers=(0, 0))
        self.assertIn("repeats", str(caught.exception))

    def test_sequence_ids_in_a_per_sequence_save_are_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.parse(build_recurrent(n_seq_id=2))
        self.assertIn("per-sequence save writes none", str(caught.exception))

    def test_a_transposed_recurrent_state_is_refused_not_guessed(self):
        body = bytearray(build_recurrent())
        # cell_count, then three cells of (pos, n_seq_id). No n_stream precedes it.
        marker = 4 + 3 * (4 + 4)
        body[marker:marker + 4] = struct.pack("<I", 1)     # s_trans = 1
        with self.assertRaises(GGSQError) as caught:
            self.parse(bytes(body))
        self.assertIn("never been exercised", str(caught.exception))

    def test_an_empty_recurrent_stream_consumes_only_its_header(self):
        """Four bytes, not eight: the header is cell_count alone."""
        self.assertEqual(self.parse(struct.pack("<I", 0)).end_offset, 4)


class CheckpointAppendixTest(unittest.TestCase):
    def parse(self, body, layers=(0, 2)):
        from kv_rosetta.adapters.llamacpp_ggsq import read_checkpoint_appendix
        return read_checkpoint_appendix(io.BytesIO(body), 0, len(body),
                                        recurrent_layers=layers)

    def test_the_target_payload_is_decoded_not_forwarded(self):
        blobs = self.parse(build_sckp())
        self.assertEqual(len(blobs), 1)
        blob = blobs[0]
        self.assertEqual((blob.n_tokens, blob.pos_min, blob.pos_max), (252, 0, 251))
        self.assertEqual(len(blob.target.spans), 4,
                         "the target state was not parsed into recurrent spans")
        self.assertEqual(blob.target.end_offset, blob.target_offset + blob.target_nbytes)

    def test_several_checkpoints_are_each_decoded(self):
        blobs = self.parse(build_sckp(checkpoints=3))
        self.assertEqual(len(blobs), 3)
        self.assertTrue(all(b.target.spans for b in blobs))

    def test_a_draft_payload_is_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.parse(build_sckp(draft=b"\x00" * 32))
        self.assertIn("not behaviourally proven", str(caught.exception))

    def test_a_speculative_payload_is_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.parse(build_sckp(spec=b"\x00" * 32))
        self.assertIn("not behaviourally proven", str(caught.exception))

    def test_a_missing_magic_is_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.parse(b"XXXX" + build_sckp()[4:])
        self.assertIn("no SCKP magic", str(caught.exception))

    def test_a_future_version_is_refused(self):
        body = bytearray(build_sckp())
        body[4:8] = struct.pack("<I", 2)
        with self.assertRaises(GGSQError):
            self.parse(bytes(body))

    def test_a_count_above_the_writers_cap_is_refused(self):
        body = bytearray(build_sckp())
        body[8:12] = struct.pack("<I", 5000)
        with self.assertRaises(GGSQError) as caught:
            self.parse(bytes(body))
        self.assertIn("cap of 1024", str(caught.exception))

    def test_trailing_bytes_after_the_appendix_are_refused(self):
        with self.assertRaises(GGSQError) as caught:
            self.parse(build_sckp() + b"\xff" * 8)
        self.assertIn("ends at", str(caught.exception))

    def test_a_target_that_does_not_fill_its_declared_length_is_refused(self):
        target = build_recurrent()
        body = b"SCKP" + struct.pack("<II", 1, 1)
        body += struct.pack("<qii", 252, 0, 251)
        body += struct.pack("<Q", len(target) + 8) + target + b"\x00" * 8
        body += struct.pack("<Q", 0) + struct.pack("<Q", 0)
        with self.assertRaises(GGSQError):
            self.parse(body)

    def test_an_empty_target_is_refused(self):
        body = b"SCKP" + struct.pack("<II", 1, 1)
        body += struct.pack("<qii", 252, 0, 251)
        body += struct.pack("<Q", 0) * 3
        with self.assertRaises(GGSQError) as caught:
            self.parse(body)
        self.assertIn("no target state", str(caught.exception))

    def test_an_impossible_extent_is_refused(self):
        body = bytearray(build_sckp())
        body[12:28] = struct.pack("<qii", 252, 100, 5)     # pos_max < pos_min
        with self.assertRaises(GGSQError) as caught:
            self.parse(bytes(body))
        self.assertIn("impossible extent", str(caught.exception))


class DequantisationTest(unittest.TestCase):
    """The oracle is constructed, not taken from the decoder.

    Known float values are packed exactly as ggml packs them, then decoded and compared to
    the values we started from. A round trip through the decoder alone would only prove it
    is self-consistent, which is what the 12-byte checkpoint record was.
    """

    def setUp(self):
        import numpy as np
        self.np = np

    def test_f16_recovers_exactly(self):
        values = self.np.array([1.0, -2.5, 3.75, 0.0], dtype=self.np.float16)
        from kv_rosetta.adapters.llamacpp_ggsq import dequantise
        out = dequantise(values.tobytes(), F16, 4)
        self.np.testing.assert_array_equal(out, values.astype(self.np.float32))

    def test_f32_recovers_exactly(self):
        values = self.np.array([1.0, -2.5, 1e-8, 12345.678], dtype=self.np.float32)
        from kv_rosetta.adapters.llamacpp_ggsq import dequantise
        self.np.testing.assert_array_equal(dequantise(values.tobytes(), 0, 4), values)

    def test_bf16_recovers_its_representable_values(self):
        from kv_rosetta.adapters.llamacpp_ggsq import dequantise
        originals = self.np.array([1.0, -2.0, 4.0, 0.5], dtype=self.np.float32)
        packed = (originals.view(self.np.uint32) >> 16).astype("<u2").tobytes()
        self.np.testing.assert_array_equal(dequantise(packed, 30, 4), originals)

    def test_q8_0_recovers_within_its_own_quantisation_step(self):
        from kv_rosetta.adapters.llamacpp_ggsq import dequantise
        values = self.np.array([1.5, -2.0, 0.0, 3.25] * 8, dtype=self.np.float32)
        scale = float(self.np.abs(values).max() / 127.0)
        quantised = self.np.round(values / scale).astype(self.np.int8)
        raw = self.np.float16(scale).tobytes() + quantised.tobytes()
        out = dequantise(raw, Q8_0, 32)
        # The bound is the format's own step, not a tolerance chosen to make it pass.
        self.assertLessEqual(float(self.np.abs(out - values).max()), scale)

    def test_q4_0_recovers_within_its_own_quantisation_step(self):
        from kv_rosetta.adapters.llamacpp_ggsq import dequantise
        scale = 0.5
        nibbles = self.np.arange(32) % 16
        expected = scale * (nibbles.astype(self.np.float32) - 8.0)
        low, high = nibbles[:16], nibbles[16:]
        packed = (low | (high << 4)).astype(self.np.uint8)
        raw = self.np.float16(scale).tobytes() + packed.tobytes()
        out = dequantise(raw, 2, 32)
        self.np.testing.assert_allclose(out, expected, atol=1e-3)

    def test_a_payload_of_the_wrong_length_is_refused(self):
        from kv_rosetta.adapters.llamacpp_ggsq import dequantise
        with self.assertRaises(GGSQError) as caught:
            dequantise(b"\x00" * 10, Q8_0, 32)
        self.assertIn("expected", str(caught.exception))

    def test_an_element_count_off_the_block_boundary_is_refused(self):
        """Isolates the block-multiple check from the payload-length check.

        33 q8_0 elements truncate to one block, so a 34-byte payload satisfies the length
        check and only the block check can catch it. Without this the guard broke no test.
        """
        from kv_rosetta.adapters.llamacpp_ggsq import dequantise
        with self.assertRaises(GGSQError) as caught:
            dequantise(b"\x00" * 34, Q8_0, 33)
        self.assertIn("not a multiple", str(caught.exception))

    def test_an_unimplemented_quantisation_is_refused_not_approximated(self):
        from kv_rosetta.adapters.llamacpp_ggsq import dequantise
        with self.assertRaises(GGSQError) as caught:
            dequantise(b"\x00" * (32 // 32 * 20), 3, 32)      # q4_1
        self.assertIn("refusing to approximate", str(caught.exception))


class MaterialiseTest(unittest.TestCase):
    """Canonical ordering, including normalisation of the transposed value layout."""

    def setUp(self):
        import numpy as np
        self.np = np

    def span(self, offset, nbytes, n_elements, transposed=False):
        from kv_rosetta.adapters.llamacpp_ggsq import TensorSpan
        return TensorSpan(layer_index=0, kind="k", type_id=F16, type_name="f16",
                          row_size=nbytes, n_elements=n_elements, offset=offset,
                          nbytes=nbytes, transposed=transposed)

    def test_row_major_values_land_in_token_head_dim_order(self):
        from kv_rosetta.adapters.llamacpp_ggsq import materialise
        cells, heads, dim = 3, 2, 4
        values = self.np.arange(cells * heads * dim, dtype=self.np.float16)
        handle = io.BytesIO(values.tobytes())
        out = materialise(handle, self.span(0, values.nbytes, heads * dim),
                          n_head=heads, head_dim=dim, cell_count=cells)
        self.assertEqual(out.shape, (cells, heads, dim))
        # Oracle: element (t, h, d) is the (t*heads*dim + h*dim + d)-th written value.
        for t in range(cells):
            for h in range(heads):
                for d in range(dim):
                    self.assertEqual(out[t, h, d], t * heads * dim + h * dim + d)

    def test_a_transposed_value_span_is_normalised_to_the_same_order(self):
        from kv_rosetta.adapters.llamacpp_ggsq import materialise
        cells, heads, dim = 3, 2, 4
        width = heads * dim
        logical = self.np.arange(cells * width, dtype=self.np.float16).reshape(cells, width)
        # The writer stores transposed values as width rows of cell_count elements.
        transposed = logical.T.copy()
        handle = io.BytesIO(transposed.tobytes())
        out = materialise(handle, self.span(0, transposed.nbytes, width, transposed=True),
                          n_head=heads, head_dim=dim, cell_count=cells)
        self.assertEqual(out.shape, (cells, heads, dim))
        self.np.testing.assert_array_equal(
            out.reshape(cells, width), logical.astype(self.np.float32),
            "the transposed layout was not normalised back to token-major order")

    def test_geometry_that_does_not_match_the_span_is_refused(self):
        from kv_rosetta.adapters.llamacpp_ggsq import materialise
        values = self.np.zeros(24, dtype=self.np.float16)
        handle = io.BytesIO(values.tobytes())
        with self.assertRaises(GGSQError) as caught:
            materialise(handle, self.span(0, values.nbytes, 8),
                        n_head=3, head_dim=4, cell_count=3)     # 12 != 8
        self.assertIn("does not match the declared geometry", str(caught.exception))

    def test_a_short_payload_is_refused(self):
        from kv_rosetta.adapters.llamacpp_ggsq import materialise
        handle = io.BytesIO(b"\x00" * 8)
        with self.assertRaises(GGSQError) as caught:
            materialise(handle, self.span(0, 64, 8), n_head=2, head_dim=4, cell_count=4)
        self.assertIn("short by", str(caught.exception))

    def test_invalid_head_geometry_is_refused(self):
        from kv_rosetta.adapters.llamacpp_ggsq import materialise
        handle = io.BytesIO(b"\x00" * 16)
        for heads, dim in ((0, 4), (2, 0), (-1, 4)):
            with self.subTest(heads=heads, dim=dim):
                with self.assertRaises(GGSQError) as caught:
                    materialise(handle, self.span(0, 16, 8), n_head=heads, head_dim=dim,
                                cell_count=1)
                # Assert the specific refusal: the geometry-match check fires first for
                # these shapes, so without this the head-geometry guard proved nothing.
                self.assertIn("invalid head geometry", str(caught.exception))


class RealHybridArtifactTest(unittest.TestCase):
    """The structure of an actual qwen35 state file, pinned as numbers.

    Every other recurrent test here builds its own body, and for a long time both the builder
    and the parser emitted a leading n_stream that the writer never writes - so they agreed
    with each other and with nothing else. These constants come from decoding a 256-token
    artifact saved by llama-server at ca3d5a3e1 on a W6800, and they close the file to zero
    leftover bytes.
    """

    #: Qwen3.8-27B-UD-Q4_K_XL: 65 blocks, one of them NextN/MTP, full_attention_interval 4.
    N_LAYER_ALL, N_LAYER, INTERVAL = 65, 64, 4
    #: SSM metadata from the same GGUF.
    D_INNER, D_STATE, D_CONV, N_GROUP = 6144, 128, 4, 16
    #: Attention: 4 KV heads x key_length 256, f16.
    KV_ROW = 4 * 256 * 2

    FILE_BYTES = 173_679_168
    HEADER_TOKENS, CELLS = 260, 256
    ATTENTION_BYTES = 16_783_760
    RECURRENT_BYTES = 156_894_356

    def layers(self):
        recurrent = tuple(i for i in range(self.N_LAYER) if (i + 1) % self.INTERVAL != 0)
        attention = tuple(i for i in range(self.N_LAYER) if (i + 1) % self.INTERVAL == 0)
        return recurrent, attention

    def test_the_layer_split_follows_the_full_attention_interval(self):
        recurrent, attention = self.layers()
        self.assertEqual((len(recurrent), len(attention)), (48, 16))

    def test_the_mtp_layer_carries_no_attention_kv(self):
        """The rule marks the NextN layer non-recurrent, so it looks like a 17th attention
        layer. The decoded file says n_layer = 16: it is not in the cache at all."""
        _recurrent, attention = self.layers()
        self.assertEqual(len(attention), 16)
        self.assertNotIn(self.N_LAYER, attention)

    def test_the_attention_section_size_is_accounted_for(self):
        cells, layers = self.CELLS, 16
        # 8 for n_stream+cell_count, 24 per cell (pos, n_seq_id, 12-byte cell_ext, seq_id),
        # 8 for v_trans+n_layer, then per layer a 12-byte header and rows for keys and again
        # for values.
        predicted = 8 + 24 * cells + 8 + layers * (2 * 12 + cells * 2 * self.KV_ROW)
        self.assertEqual(predicted, self.ATTENTION_BYTES)

    def test_the_recurrent_rows_come_from_the_ssm_metadata(self):
        conv = (self.D_CONV - 1) * (self.D_INNER + 2 * self.N_GROUP * self.D_STATE) * 4
        ssm = self.D_INNER * self.D_STATE * 4
        self.assertEqual((conv, ssm), (122_880, 3_145_728))

    def test_the_recurrent_section_size_is_accounted_for(self):
        conv, ssm = 122_880, 3_145_728
        # cell_count, one cell of (pos, n_seq_id), s_trans+n_layer, then per present layer a
        # 12-byte header and one row, for R and again for S.
        predicted = 4 + 8 + 8 + 48 * (2 * 12 + conv + ssm)
        self.assertEqual(predicted, self.RECURRENT_BYTES)

    def test_the_two_sections_close_the_file_exactly(self):
        total = (12 + 4 * self.HEADER_TOKENS + self.ATTENTION_BYTES + self.RECURRENT_BYTES)
        self.assertEqual(total, self.FILE_BYTES)

    def test_the_recurrent_state_dominates_at_this_length(self):
        """90% of a 256-token hybrid artifact is state that does not grow with tokens.

        This is why a per-token rate taken from a hybrid artifact over-predicts so badly, and
        why sizing refuses hybrids rather than scaling one.
        """
        self.assertGreater(self.RECURRENT_BYTES / self.FILE_BYTES, 0.90)

    def test_the_marginal_token_is_attention_only(self):
        """Measured: a 257-token file is 65,564 bytes larger, and its recurrent section is
        byte-identical."""
        marginal = 4 + 24 + 16 * 2 * self.KV_ROW
        self.assertEqual(marginal, 65_564)
