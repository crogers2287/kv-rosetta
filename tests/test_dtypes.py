from __future__ import annotations

import unittest

import numpy as np

from kv_rosetta.dtypes import DtypeError, convert, itemsize


class TestItemsize(unittest.TestCase):
    def test_known_itemsizes(self) -> None:
        self.assertEqual(itemsize("f32"), 4)
        self.assertEqual(itemsize("f16"), 2)
        self.assertEqual(itemsize("bf16"), 2)

    def test_unknown_raises(self) -> None:
        with self.assertRaises(DtypeError):
            itemsize("fp8_e4m3")


class TestConvert(unittest.TestCase):
    def test_f32_bf16_f32_roundtrip(self) -> None:
        original = np.array([1.0, -2.5, 0.0], dtype=np.float32)
        as_bf16 = convert(original, "f32", "bf16")
        self.assertEqual(as_bf16.dtype, np.uint16)
        back = convert(as_bf16, "bf16", "f32")
        np.testing.assert_array_equal(back, original)

    def test_identity(self) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = convert(arr, "f32", "f32")
        np.testing.assert_array_equal(out, arr)

    def test_unknown_source_raises(self) -> None:
        arr = np.zeros((2, 2), dtype=np.uint16)
        with self.assertRaises(DtypeError):
            convert(arr, "q8_0", "f32")

    def test_unknown_dest_raises(self) -> None:
        arr = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(DtypeError):
            convert(arr, "f32", "q4_0")


if __name__ == "__main__":
    unittest.main()
