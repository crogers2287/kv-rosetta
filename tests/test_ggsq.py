from __future__ import annotations

import struct
import unittest

from kv_rosetta.adapters import ggsq




def build_blob(stream_specs, *, has_cell_ext=False, cell_ext_size=0, data_fill=0xAB):
    """Construct a synthetic blob following state_write's exact byte order.

    Returns (blob_bytes, expected) where expected is a list of
    (index, kind, ggml_type, row_size, n_embd_v_gqa, offset, nbytes) tuples
    mirroring what the parser will report for each layer entry.
    """
    out = bytearray()
    expected = []

    out += struct.pack("<I", len(stream_specs))

    for s, spec in enumerate(stream_specs):
        cell_count = spec["cell_count"]
        out += struct.pack("<I", cell_count)

        if cell_count == 0:
            continue

        positions = spec["positions"]
        seq_ids = spec["seq_ids"]
        assert len(positions) == cell_count
        assert len(seq_ids) == cell_count

        for c in range(cell_count):
            out += struct.pack("<i", positions[c])
            n_seq_id = len(seq_ids[c])
            out += struct.pack("<I", n_seq_id)
            if has_cell_ext:
                out += bytes([data_fill]) * cell_ext_size
            for q in range(n_seq_id):
                out += struct.pack("<i", seq_ids[c][q])

        v_trans = spec["v_trans"]
        n_layer = spec["n_layer"]
        out += struct.pack("<I", v_trans)
        out += struct.pack("<I", n_layer)

        # state_write_data writes ALL keys for ALL layers first, then all values in a
        # second pass. Emitting them interleaved per layer would describe a format that
        # llama.cpp never produces, and any parser validated against it would be wrong.
        keys = [l for l in spec["layers"] if l["kind"] == "k"]
        values = [l for l in spec["layers"] if l["kind"] == "v"]

        for li, layer in enumerate(keys):
            out += struct.pack("<i", layer["ggml_type"])
            out += struct.pack("<Q", layer["row_size"])
            nbytes = cell_count * layer["row_size"]
            data_off = len(out)
            out += bytes([data_fill]) * nbytes
            expected.append((li, "k", layer["ggml_type"], layer["row_size"], 0, data_off, nbytes))

        for li, layer in enumerate(values):
            ggml_type = layer["ggml_type"]
            row_size = layer["row_size"]
            n_embd_v_gqa = layer.get("n_embd_v_gqa", 0)
            if v_trans == 0:
                out += struct.pack("<i", ggml_type)
                out += struct.pack("<Q", row_size)
                nbytes = cell_count * row_size
                data_off = len(out)
                out += bytes([data_fill]) * nbytes
                expected.append((li, "v", ggml_type, row_size, 0, data_off, nbytes))
            else:
                # transposed values: uint32 ELEMENT size, then uint32 n_embd_v_gqa
                out += struct.pack("<i", ggml_type)
                out += struct.pack("<I", row_size)
                out += struct.pack("<I", n_embd_v_gqa)
                nbytes = n_embd_v_gqa * cell_count * row_size
                data_off = len(out)
                out += bytes([data_fill]) * nbytes
                expected.append((li, "v", ggml_type, row_size, n_embd_v_gqa, data_off, nbytes))

    return bytes(out), expected


class TestGGSQ(unittest.TestCase):
    def _assert_entries(self, layers, expected):
        self.assertEqual(len(layers), len(expected))
        for got, exp in zip(layers, expected):
            self.assertEqual(got.index, exp[0])
            self.assertEqual(got.kind, exp[1])
            self.assertEqual(got.ggml_type, exp[2])
            self.assertEqual(got.row_size, exp[3])
            self.assertEqual(got.n_embd_v_gqa, exp[4])
            self.assertEqual(got.offset, exp[5])
            self.assertEqual(got.nbytes, exp[6])

    def test_single_stream_f16(self):
        stream = {
            "cell_count": 3,
            "positions": [0, 1, 2],
            "seq_ids": [(0,), (0,), (0,)],
            "v_trans": 0,
            "n_layer": 2,
            "layers": [
                {"kind": "k", "ggml_type": 1, "row_size": 4},
                {"kind": "v", "ggml_type": 1, "row_size": 4},
                {"kind": "k", "ggml_type": 1, "row_size": 4},
                {"kind": "v", "ggml_type": 1, "row_size": 4},
            ],
        }
        data_fill = 0xAB
        blob_bytes, expected = build_blob([stream], data_fill=data_fill)
        ggsq_blob = ggsq.parse(blob_bytes)

        self.assertEqual(ggsq_blob.n_stream, 1)
        self.assertEqual(ggsq_blob.token_count, 3)
        self.assertEqual(len(ggsq_blob.streams), 1)

        s = ggsq_blob.streams[0]
        self.assertEqual(s.cell_count, 3)
        self.assertEqual(s.positions, (0, 1, 2))
        self.assertEqual(s.seq_ids, ((0,), (0,), (0,)))
        self.assertFalse(s.v_trans)
        self.assertEqual(s.n_layer, 2)
        self.assertEqual(len(s.layers), 4)
        self.assertEqual([e.kind for e in s.layers], ["k", "k", "v", "v"])

        self._assert_entries(s.layers, expected)
        for got in s.layers:
            self.assertEqual(blob_bytes[got.offset], data_fill)
            self.assertLessEqual(got.offset + got.nbytes, len(blob_bytes))

    def test_transposed_value(self):
        stream = {
            "cell_count": 2,
            "positions": [0, 1],
            "seq_ids": [(0,), (0,)],
            "v_trans": 1,
            "n_layer": 1,
            "layers": [
                {"kind": "k", "ggml_type": 1, "row_size": 8},
                {"kind": "v", "ggml_type": 1, "row_size": 2, "n_embd_v_gqa": 3},
            ],
        }
        blob_bytes, expected = build_blob([stream])
        ggsq_blob = ggsq.parse(blob_bytes)

        s = ggsq_blob.streams[0]
        self.assertTrue(s.v_trans)
        self.assertEqual(len(s.layers), 2)

        k, v = s.layers
        self.assertEqual(k.kind, "k")
        self.assertEqual(v.kind, "v")
        self.assertEqual(v.row_size, 2)
        self.assertEqual(v.n_embd_v_gqa, 3)
        self.assertEqual(v.nbytes, 3 * 2 * 2)

        self._assert_entries(s.layers, expected)

    def test_multi_seq(self):
        stream = {
            "cell_count": 2,
            "positions": [0, 1],
            "seq_ids": [(0, 1), (0,)],
            "v_trans": 0,
            "n_layer": 1,
            "layers": [
                {"kind": "k", "ggml_type": 1, "row_size": 4},
                {"kind": "v", "ggml_type": 1, "row_size": 4},
            ],
        }
        blob_bytes, _ = build_blob([stream])
        ggsq_blob = ggsq.parse(blob_bytes)

        s = ggsq_blob.streams[0]
        self.assertEqual(s.positions, (0, 1))
        self.assertEqual(s.seq_ids, ((0, 1), (0,)))

    def test_empty_stream_skipped(self):
        empty = {
            "cell_count": 0,
            "positions": [],
            "seq_ids": [],
            "v_trans": 0,
            "n_layer": 0,
            "layers": [],
        }
        live = {
            "cell_count": 1,
            "positions": [5],
            "seq_ids": [(0,)],
            "v_trans": 0,
            "n_layer": 1,
            "layers": [
                {"kind": "k", "ggml_type": 1, "row_size": 4},
                {"kind": "v", "ggml_type": 1, "row_size": 4},
            ],
        }
        blob_bytes, expected = build_blob([empty, live])
        ggsq_blob = ggsq.parse(blob_bytes)

        self.assertEqual(ggsq_blob.n_stream, 2)
        self.assertEqual(len(ggsq_blob.streams), 1)
        self.assertEqual(ggsq_blob.streams[0].index, 1)
        self.assertEqual(ggsq_blob.token_count, 1)
        self._assert_entries(ggsq_blob.streams[0].layers, expected)

    def test_truncated_raises(self):
        stream = {
            "cell_count": 3,
            "positions": [0, 1, 2],
            "seq_ids": [(0,), (0,), (0,)],
            "v_trans": 0,
            "n_layer": 2,
            "layers": [
                {"kind": "k", "ggml_type": 1, "row_size": 4},
                {"kind": "v", "ggml_type": 1, "row_size": 4},
                {"kind": "k", "ggml_type": 1, "row_size": 4},
                {"kind": "v", "ggml_type": 1, "row_size": 4},
            ],
        }
        blob_bytes, _ = build_blob([stream])
        with self.assertRaises(ggsq.GGSQError):
            ggsq.parse(blob_bytes[:-1])

    def test_oversized_row_raises(self):
        out = bytearray()
        out += struct.pack("<I", 1)   # n_stream
        out += struct.pack("<I", 1)   # cell_count
        out += struct.pack("<i", 0)   # pos
        out += struct.pack("<I", 1)   # n_seq_id
        out += struct.pack("<i", 0)   # seq_id
        out += struct.pack("<I", 0)   # v_trans
        out += struct.pack("<I", 1)   # n_layer
        out += struct.pack("<i", 1)   # k_type
        out += struct.pack("<Q", 1 << 20)  # k_size_row, but no data follows
        with self.assertRaises(ggsq.GGSQError):
            ggsq.parse(bytes(out))

    def test_can_decode(self):
        f16_stream = {
            "cell_count": 1,
            "positions": [0],
            "seq_ids": [(0,)],
            "v_trans": 0,
            "n_layer": 1,
            "layers": [
                {"kind": "k", "ggml_type": 1, "row_size": 4},
                {"kind": "v", "ggml_type": 1, "row_size": 4},
            ],
        }
        blob_bytes, _ = build_blob([f16_stream])
        ggsq_blob = ggsq.parse(blob_bytes)
        ok, msg = ggsq.can_decode(ggsq_blob)
        self.assertTrue(ok)
        self.assertEqual(msg, "ok")

        q8_stream = {
            "cell_count": 1,
            "positions": [0],
            "seq_ids": [(0,)],
            "v_trans": 0,
            "n_layer": 1,
            "layers": [
                {"kind": "k", "ggml_type": 8, "row_size": 4},
                {"kind": "v", "ggml_type": 1, "row_size": 4},
            ],
        }
        blob_bytes2, _ = build_blob([q8_stream])
        ggsq_blob2 = ggsq.parse(blob_bytes2)
        ok2, msg2 = ggsq.can_decode(ggsq_blob2)
        self.assertFalse(ok2)
        self.assertIn("Q8_0", msg2)

    def test_has_cell_ext(self):
        stream = {
            "cell_count": 2,
            "positions": [0, 1],
            "seq_ids": [(0,), (0,)],
            "v_trans": 0,
            "n_layer": 1,
            "layers": [
                {"kind": "k", "ggml_type": 1, "row_size": 4},
                {"kind": "v", "ggml_type": 1, "row_size": 4},
            ],
        }
        blob_bytes, expected = build_blob(
            [stream], has_cell_ext=True, cell_ext_size=8
        )
        ggsq_blob = ggsq.parse(blob_bytes, has_cell_ext=True, cell_ext_size=8)

        s = ggsq_blob.streams[0]
        self.assertEqual(s.positions, (0, 1))
        self.assertEqual(s.seq_ids, ((0,), (0,)))
        self._assert_entries(s.layers, expected)


if __name__ == "__main__":
    unittest.main()
