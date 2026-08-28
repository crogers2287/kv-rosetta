"""Regressions for the fail-closed contract.

Every test here reproduces a defect that was present in commit fb3ecf7 and was found
by auditing that commit against the hardening steer. They are kept because each one
passed before the fix: a corrupt, ambiguous or under-evidenced artifact must never
reach a runtime import call.
"""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kv_rosetta import container
from kv_rosetta.dtypes import convert
from kv_rosetta.store import IdentityError, Store, fingerprint


def _manifest():
    return {
        "schema": "kvx/0.2",
        "kv": {
            "layers": 2, "heads": 2, "head_dim": 4, "dtype": "f16",
            "layout": "layer,kv,token,head,dim", "byte_order": "little", "tokens": 3,
        },
    }


class StorePathContainmentTests(unittest.TestCase):
    """A prefix manifest is written by another process; its fingerprint is untrusted."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "store"
        self.store = Store(self.root)

    def test_traversal_fingerprint_cannot_escape_the_root(self):
        with self.assertRaises(IdentityError):
            self.store.path_for("prov", "model", "../../../../tmp/pwned")

    def test_malformed_digests_are_refused(self):
        for bad in ("", "..", "/etc/passwd", "a" * 63, "a" * 65, "A" * 64, "x" * 64, "g" * 64):
            with self.subTest(value=bad), self.assertRaises(IdentityError):
                self.store.path_for("prov", "model", bad)

    def test_non_string_digest_is_refused(self):
        for bad in (None, 1234, b"a" * 64, ["a" * 64]):
            with self.subTest(value=type(bad).__name__), self.assertRaises(IdentityError):
                self.store.path_for("prov", "model", bad)

    def test_valid_digest_is_used_whole_and_stays_inside_the_root(self):
        fp = fingerprint("prov", "model", "a" * 64, "b" * 64)
        path = self.store.path_for("prov", "model", fp)
        # The identity is not truncated for tidiness: a short form invites collisions.
        self.assertEqual(path.stem, fp)
        self.assertTrue(str(path).startswith(str(self.root.resolve())))


class HeaderIntegrityTests(unittest.TestCase):
    """Integrity must cover the semantic header, not only the payload bytes."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.tensor = np.zeros((2, 2, 3, 2, 4), dtype=np.float16)
        self.path = container.write_raw(self.dir / "a.kvx", _manifest(), self.tensor)

    def _mutate(self, old: bytes, new: bytes) -> Path:
        raw = bytearray(self.path.read_bytes())
        index = raw.find(old)
        self.assertGreaterEqual(index, 0, f"{old!r} not present in the header")
        raw[index:index + len(old)] = new
        out = self.dir / "mutated.kvx"
        out.write_bytes(bytes(raw))
        return out

    def test_clean_artifact_verifies(self):
        self.assertEqual(container.verify(self.path), (True, "ok"))

    def test_header_digest_is_present(self):
        self.assertEqual(len(container.read(self.path).header["integrity"]["header_sha256"]), 64)

    def test_shape_field_cannot_be_mutated_undetected(self):
        ok, reason = container.verify(self._mutate(b'"head_dim":4', b'"head_dim":8'))
        self.assertFalse(ok)
        self.assertIn("header sha256", reason)

    def test_dtype_field_cannot_be_mutated_undetected(self):
        ok, reason = container.verify(self._mutate(b'"dtype":"f16"', b'"dtype":"f32"'))
        self.assertFalse(ok)
        self.assertIn("header sha256", reason)

    def test_payload_corruption_is_still_detected(self):
        raw = bytearray(self.path.read_bytes())
        offset = container.read(self.path).blob["offset"]
        raw[offset] ^= 0xFF
        out = self.dir / "payload.kvx"
        out.write_bytes(bytes(raw))
        self.assertEqual(container.verify(out), (False, "payload sha256 mismatch"))

    def test_verifier_never_raises_on_arbitrary_bytes(self):
        rng = np.random.default_rng(0)
        cases = [
            b"", b"KVX1", b"KVX1\x01\x00\x00\x00", b"NOPE" + b"\x00" * 64,
            b"KVX1" + struct.pack("<II", 1, 10 ** 9) + b"{}",
            b"KVX1" + struct.pack("<II", 99, 2) + b"{}",
            b"KVX1" + struct.pack("<II", 1, 2) + b"@@",
            bytes(rng.integers(0, 256, 512, dtype=np.uint8)),
        ]
        for i, blob in enumerate(cases):
            path = self.dir / f"fuzz{i}.kvx"
            path.write_bytes(blob)
            with self.subTest(case=i):
                ok, reason = container.verify(path)   # must not raise
                self.assertFalse(ok)
                self.assertIsInstance(reason, str)
                self.assertTrue(reason)

    def test_missing_file_returns_a_reason(self):
        ok, reason = container.verify(self.dir / "absent.kvx")
        self.assertFalse(ok)
        self.assertIsInstance(reason, str)


class Bf16RoundingTests(unittest.TestCase):
    """bf16 conversion must round to nearest even, not truncate.

    Truncation biases every value toward zero, and across a whole KV cache the bias
    compounds rather than cancelling.
    """

    @staticmethod
    def _f32(bits: int) -> np.ndarray:
        return np.frombuffer(struct.pack("<I", bits), dtype=np.float32)

    def test_tie_breaking_matches_ieee_round_to_nearest_even(self):
        cases = (
            (0x3F80C000, 0x3F81, "above the midpoint rounds up"),
            (0x3F804000, 0x3F80, "below the midpoint rounds down"),
            (0x3F808000, 0x3F80, "exact midpoint with even lsb stays"),
            (0x3F818000, 0x3F82, "exact midpoint with odd lsb rounds up"),
        )
        for bits, expected, label in cases:
            with self.subTest(case=label):
                self.assertEqual(int(convert(self._f32(bits), "f32", "bf16")[0]), expected)

    def test_specials_survive_conversion(self):
        self.assertTrue(np.isnan(convert(convert(np.array([np.nan], np.float32), "f32", "bf16"), "bf16", "f32")[0]))
        for value in (np.inf, -np.inf, 0.0, -0.0):
            with self.subTest(value=value):
                back = convert(convert(np.array([value], np.float32), "f32", "bf16"), "bf16", "f32")[0]
                self.assertEqual(np.signbit(back), np.signbit(np.float32(value)))
                self.assertEqual(back, np.float32(value))

    def test_exactly_representable_values_are_unchanged(self):
        values = np.array([1.0, -2.5, 0.0, 256.0, -0.5], dtype=np.float32)
        back = convert(convert(values, "f32", "bf16"), "bf16", "f32")
        np.testing.assert_array_equal(back, values)


if __name__ == "__main__":
    unittest.main()
