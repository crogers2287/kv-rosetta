"""Compare our quantisers against ggml's own, byte for byte.

Every other check of this code is a round trip through code I wrote, which proves
self-consistency and nothing else - the same shape as the 12-versus-16-byte checkpoint
record that stayed wrong for a day because its fixtures shared the error.

ggml exports quantize_row_q8_0_ref and ggml_quantize_chunk. Calling them gives bytes
llama.cpp would actually produce, which can falsify ours. Skipped when the library is
absent so CI stays green without a build.
"""

import ctypes
import os
import unittest
from pathlib import Path

from kv_rosetta.adapters.llamacpp_ggsq import dequantise, quantise

DEFAULT_LIB = "/mnt/storage/llama-kvx-patched/build/bin/libggml-base.so"
LIB = os.environ.get("KVX_GGML_LIB", DEFAULT_LIB)

GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 = 2, 8


def _load():
    if not Path(LIB).is_file():
        return None
    try:
        lib = ctypes.CDLL(LIB)
        lib.quantize_row_q8_0_ref.argtypes = [ctypes.POINTER(ctypes.c_float),
                                              ctypes.c_void_p, ctypes.c_int64]
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_void_p, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_int64,
                                            ctypes.POINTER(ctypes.c_float)]
        return lib
    except (OSError, AttributeError):
        return None


GGML = _load()


@unittest.skipUnless(GGML is not None, f"ggml library not available at {LIB}")
class GgmlOracleTest(unittest.TestCase):
    """ggml produces the expected bytes; we must match them exactly, not approximately."""

    def setUp(self):
        import numpy as np
        self.np = np
        self.rng = np.random.default_rng(11)

    def ggml_q8_0(self, values):
        n = values.size
        out = (ctypes.c_uint8 * (n // 32 * 34))()
        GGML.quantize_row_q8_0_ref(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out), ctypes.c_int64(n))
        return bytes(out)

    def ggml_q4_0(self, values):
        n = values.size
        out = (ctypes.c_uint8 * (n // 32 * 18))()
        GGML.ggml_quantize_chunk(
            GGML_TYPE_Q4_0, values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out), 0, 1, n, None)
        return bytes(out)

    def assertIdentical(self, mine, theirs, label):
        if mine == theirs:
            return
        a = self.np.frombuffer(mine, self.np.uint8)
        b = self.np.frombuffer(theirs, self.np.uint8)
        first = int(self.np.flatnonzero(a != b)[0])
        self.fail(f"{label}: differs from ggml in {int((a != b).sum())} of {len(a)} bytes, "
                  f"first at offset {first} ({a[first]} vs {b[first]})")

    def test_q8_0_matches_ggml_byte_for_byte(self):
        for n in (32, 256, 1024):
            with self.subTest(n=n):
                values = self.rng.normal(0, 2.5, n).astype(self.np.float32)
                self.assertIdentical(quantise(values, GGML_TYPE_Q8_0),
                                     self.ggml_q8_0(values), f"q8_0 n={n}")

    def test_q4_0_matches_ggml_byte_for_byte(self):
        for n in (32, 512):
            with self.subTest(n=n):
                values = self.rng.normal(0, 2.5, n).astype(self.np.float32)
                self.assertIdentical(quantise(values, GGML_TYPE_Q4_0),
                                     self.ggml_q4_0(values), f"q4_0 n={n}")

    def test_edge_distributions_match_too(self):
        """Zeros, a single spike, and saturating values exercise the scale branches."""
        cases = {
            "all zero": self.np.zeros(64, dtype=self.np.float32),
            "one spike": self.np.eye(1, 64, 3, dtype=self.np.float32).reshape(-1) * 12.0,
            "all negative": -self.np.abs(self.rng.normal(0, 1, 64)).astype(self.np.float32),
            "tiny": (self.rng.normal(0, 1e-6, 64)).astype(self.np.float32),
            "large": (self.rng.normal(0, 1e4, 64)).astype(self.np.float32),
        }
        for label, values in cases.items():
            with self.subTest(case=label):
                self.assertIdentical(quantise(values, GGML_TYPE_Q8_0),
                                     self.ggml_q8_0(values), f"q8_0 {label}")
                self.assertIdentical(quantise(values, GGML_TYPE_Q4_0),
                                     self.ggml_q4_0(values), f"q4_0 {label}")

    def test_values_landing_exactly_on_a_rounding_boundary_match(self):
        """Designed input, because random floats cannot reach this case.

        ggml computes MIN(15, (int8_t)(x + 8.5f)), which truncates - equivalent to
        round-half-up. A plausible substitution, rint(x + 8.0), is round-half-even, and the
        two differ only when x + 8.5 is exactly an integer. Random samples never land there,
        so the oracle missed that mutation until this case existed.

        With the block maximum at -8.0 the scale is exactly 1.0, so a value of 0.5 gives
        8.5 exactly: truncation yields 9, round-half-even yields 8.
        """
        values = self.np.zeros(32, dtype=self.np.float32)
        values[0] = -8.0                     # sets d = -8 / -8 = 1.0, exact in fp16
        values[1] = 0.5                      # lands on 8.5 after the offset
        values[2] = 1.5                      # and 10.0 - a second boundary, opposite parity
        values[3] = -0.5
        self.assertIdentical(quantise(values, GGML_TYPE_Q4_0),
                             self.ggml_q4_0(values), "q4_0 exact rounding boundary")

    def test_our_dequantiser_inverts_ggml_output(self):
        """The other direction: read what ggml wrote, not only write what ggml would."""
        values = self.rng.normal(0, 2.0, 256).astype(self.np.float32)
        theirs = self.ggml_q8_0(values)
        recovered = dequantise(theirs, GGML_TYPE_Q8_0, 256)
        scale = float(self.np.abs(values).max() / 127.0)
        self.assertLessEqual(float(self.np.abs(recovered - values).max()), scale * 1.5)
