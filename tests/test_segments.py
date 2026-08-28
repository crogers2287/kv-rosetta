"""Segment table and segmented container (KVX v2).

The single homogeneous layer,kv,token,head,dim tensor cannot describe differing K and V
dtypes, quantization scales, compressed latent state, or recurrent state. These tests pin
the replacement, including the cases the old format could not express at all.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kv_rosetta import container, segments
from kv_rosetta.segments import Segment


def _seg(name, role="k", dtype="f16", shape=(4,), offset=0, nbytes=8, **kw):
    return Segment(name=name, role=role, layer_start=0, layer_end=1, dtype=dtype,
                   shape=shape, layout="x", offset=offset, nbytes=nbytes, **kw)


class SegmentValidationTests(unittest.TestCase):
    def test_valid_table_passes(self):
        table = [_seg("a", offset=0), _seg("b", offset=64)]
        self.assertEqual(segments.validate(table, payload_start=0, file_size=256), (True, "ok"))

    def test_empty_table_rejected(self):
        ok, reason = segments.validate([], payload_start=0, file_size=64)
        self.assertFalse(ok)
        self.assertIn("empty", reason)

    def test_overlapping_segments_rejected(self):
        # Both 64-byte aligned so alignment cannot mask the overlap.
        table = [_seg("a", offset=0, shape=(64,), nbytes=128), _seg("b", offset=64)]
        ok, reason = segments.validate(table, payload_start=0, file_size=512)
        self.assertFalse(ok)
        self.assertIn("overlap", reason)

    def test_unknown_role_rejected(self):
        ok, reason = segments.validate([_seg("a", role="bogus")], payload_start=0, file_size=64)
        self.assertFalse(ok)
        self.assertIn("unknown role", reason)

    def test_size_must_match_dtype_times_shape(self):
        ok, reason = segments.validate([_seg("a", nbytes=999)], payload_start=0, file_size=4096)
        self.assertFalse(ok)
        self.assertIn("does not match dtype", reason)

    def test_misaligned_offset_rejected(self):
        ok, reason = segments.validate([_seg("a", offset=8)], payload_start=0, file_size=256)
        self.assertFalse(ok)
        self.assertIn("aligned", reason)

    def test_segment_before_payload_rejected(self):
        ok, reason = segments.validate([_seg("a", offset=0)], payload_start=64, file_size=256)
        self.assertFalse(ok)
        self.assertIn("begins before the payload", reason)

    def test_segment_past_end_of_file_rejected(self):
        ok, reason = segments.validate([_seg("a", offset=0)], payload_start=0, file_size=4)
        self.assertFalse(ok)
        self.assertIn("past end of file", reason)

    def test_duplicate_names_rejected(self):
        ok, reason = segments.validate([_seg("a", offset=0), _seg("a", offset=64)],
                                       payload_start=0, file_size=256)
        self.assertFalse(ok)
        self.assertIn("duplicate", reason)

    def test_negative_values_rejected(self):
        ok, _ = segments.validate([_seg("a", offset=-64)], payload_start=0, file_size=256)
        self.assertFalse(ok)

    def test_validate_never_raises(self):
        for bad in ([None], ["not a segment"], [{"name": "a"}], [123]):
            with self.subTest(value=bad):
                ok, reason = segments.validate(bad, payload_start=0, file_size=64)
                self.assertFalse(ok)
                self.assertIsInstance(reason, str)

    def test_table_digest_is_stable_and_order_independent(self):
        a, b = _seg("a", offset=0), _seg("b", offset=64)
        self.assertEqual(segments.table_digest([a, b]), segments.table_digest([b, a]))
        self.assertNotEqual(segments.table_digest([a, b]), segments.table_digest([a]))

    def test_round_trip_through_dict(self):
        seg = _seg("a", quant={"scheme": "per-layer"}, position={"kind": "rope"})
        self.assertEqual(Segment.from_dict(seg.as_dict()), seg)


class SegmentedContainerTests(unittest.TestCase):
    """The heterogeneous case: f16 keys, bf16 values, and a separate scales segment."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.manifest = {"schema": "kvx/0.3", "kv": {
            "layers": 2, "heads": 2, "head_dim": 4, "dtype": "f16",
            "layout": "per-segment", "byte_order": "little", "tokens": 3}}
        self.k = np.arange(2 * 3 * 2 * 4, dtype=np.float16).reshape(2, 3, 2, 4)
        self.v = np.arange(2 * 3 * 2 * 4, dtype=np.uint16).reshape(2, 3, 2, 4)
        self.scales = np.ones((2,), dtype=np.float32)
        self.parts = [
            (Segment("k", "k", 0, 2, "f16", self.k.shape, "layer,token,head,dim", 0, self.k.nbytes), self.k.tobytes()),
            (Segment("v", "v", 0, 2, "bf16", self.v.shape, "layer,token,head,dim", 0, self.v.nbytes), self.v.tobytes()),
            (Segment("v_scales", "scales", 0, 2, "f32", self.scales.shape, "layer", 0, self.scales.nbytes,
                     quant={"scheme": "per-layer"}), self.scales.tobytes()),
        ]
        self.path = container.write_segments(self.dir / "seg.kvx", self.manifest, self.parts)

    def test_written_as_container_version_two(self):
        self.assertEqual(self.path.read_bytes()[4], container.SEGMENTED_VERSION)

    def test_verifies_clean(self):
        self.assertEqual(container.verify(self.path), (True, "ok"))

    def test_each_segment_round_trips_with_its_own_dtype(self):
        art = container.read(self.path)
        np.testing.assert_array_equal(art.segment("k"), self.k)
        np.testing.assert_array_equal(art.segment("v"), self.v)
        np.testing.assert_array_equal(art.segment("v_scales"), self.scales)

    def test_segments_are_aligned_and_non_overlapping(self):
        segs = sorted(container.read(self.path).segments, key=lambda s: s.offset)
        for seg in segs:
            self.assertEqual(seg.offset % container.ALIGNMENT, 0)
        for previous, current in zip(segs, segs[1:]):
            self.assertGreaterEqual(current.offset, previous.end)

    def test_representation_digest_recorded(self):
        self.assertEqual(len(container.read(self.path).blob["representation_digest"]), 64)

    def test_unknown_segment_name_raises(self):
        with self.assertRaises(container.ContainerError):
            container.read(self.path).segment("nope")

    def test_tampering_with_a_segment_is_detected(self):
        raw = bytearray(self.path.read_bytes())
        raw[container.read(self.path).blob["offset"]] ^= 0xFF
        out = self.dir / "tampered.kvx"
        out.write_bytes(bytes(raw))
        ok, reason = container.verify(out)
        self.assertFalse(ok)
        self.assertIn("sha256", reason)

    def test_declared_size_must_match_supplied_bytes(self):
        bad = [(Segment("k", "k", 0, 1, "f16", (4,), "x", 0, 8), b"only-3")]
        with self.assertRaises(container.ContainerError):
            container.write_segments(self.dir / "bad.kvx", self.manifest, bad)

    def test_empty_part_list_rejected(self):
        with self.assertRaises(container.ContainerError):
            container.write_segments(self.dir / "none.kvx", self.manifest, [])

    def test_v1_artifacts_still_read(self):
        tensor = np.zeros((2, 2, 3, 2, 4), dtype=np.float16)
        old = container.write_raw(self.dir / "old.kvx", {"schema": "kvx/0.2", "kv": {
            "layers": 2, "heads": 2, "head_dim": 4, "dtype": "f16",
            "layout": "layer,kv,token,head,dim", "byte_order": "little", "tokens": 3}}, tensor)
        self.assertEqual(old.read_bytes()[4], container.CONTAINER_VERSION)
        self.assertEqual(container.verify(old), (True, "ok"))
        np.testing.assert_array_equal(container.read(old).tensor, tensor)

    def test_segments_property_refused_on_non_segmented(self):
        old = container.write_raw(self.dir / "raw.kvx", {"schema": "kvx/0.2", "kv": {
            "layers": 1, "heads": 1, "head_dim": 2, "dtype": "f16",
            "layout": "layer,kv,token,head,dim", "byte_order": "little", "tokens": 1}},
            np.zeros((1, 2, 1, 1, 2), dtype=np.float16))
        with self.assertRaises(container.ContainerError):
            container.read(old).segments


if __name__ == "__main__":
    unittest.main()
