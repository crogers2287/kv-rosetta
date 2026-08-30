"""Reading one metadata key must not decode the rest of the header.

read_metadata() materialises every array element, so asking for general.architecture built
150k Python strings from the tokenizer vocabulary. Measured at ~0.65 s per call on the 27B,
that was the entire cost of the fail-closed support check on the request path, and the whole
reason the admitted-store 2K gate lost to cold prefill.

The support decision itself is unchanged: these tests pin that classification still comes
from the file, that malformed headers fail closed, and that the lookup stops at the key.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from kv_rosetta import gguf

MAGIC = b"GGUF"
_STRING, _ARRAY, _U32 = 8, 9, 4


def gstr(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv_string(key: str, value: str) -> bytes:
    return gstr(key) + struct.pack("<I", _STRING) + gstr(value)


def kv_string_array(key: str, values: list[str]) -> bytes:
    body = gstr(key) + struct.pack("<I", _ARRAY)
    body += struct.pack("<I", _STRING) + struct.pack("<Q", len(values))
    return body + b"".join(gstr(v) for v in values)


def kv_u32(key: str, value: int) -> bytes:
    return gstr(key) + struct.pack("<I", _U32) + struct.pack("<I", value)


def write_gguf(path: Path, entries: list[bytes], *, n_kv: int | None = None) -> Path:
    head = MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 0)
    head += struct.pack("<Q", len(entries) if n_kv is None else n_kv)
    path.write_bytes(head + b"".join(entries))
    return path


class ExactKeyLookupTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    # -- the lookup stops at the key ------------------------------------------------

    def test_a_huge_later_array_is_never_read(self):
        vocabulary = [f"token{i}" for i in range(60000)]
        path = write_gguf(self.dir / "big.gguf", [
            kv_string("general.architecture", "qwen35"),
            kv_string_array("tokenizer.ggml.tokens", vocabulary),
        ])
        self.assertEqual(gguf.architecture(path), "qwen35")

    def test_an_invalid_later_value_does_not_prevent_the_lookup(self):
        # A value type the parser cannot decode, placed after the key. Reaching it would
        # raise, so a successful read proves the lookup returned before it.
        broken = gstr("later.key") + struct.pack("<I", 999) + b"\x00" * 8
        path = write_gguf(self.dir / "broken-tail.gguf", [
            kv_string("general.architecture", "llama"), broken])
        self.assertEqual(gguf.architecture(path), "llama")

    def test_the_lookup_reads_only_the_prefix_through_the_key(self):
        vocabulary = [f"token{i}" for i in range(40000)]
        entries = [kv_string("general.architecture", "qwen2"),
                   kv_string_array("tokenizer.ggml.tokens", vocabulary)]
        path = write_gguf(self.dir / "counted.gguf", entries)
        total = path.stat().st_size
        prefix = 24 + len(entries[0])
        pages = {"touched": 0}
        real_mmap = gguf.mmap.mmap

        class Counting:
            def __init__(self, *args, **kwargs):
                self._m = real_mmap(*args, **kwargs)

            def __enter__(self):
                return self._m.__enter__()

            def __exit__(self, *exc):
                return self._m.__exit__(*exc)

        # The memoryview offset reached is the direct evidence: it must stop within the
        # first entry, not run to the end of the vocabulary.
        seen = {}
        original_cursor = gguf._Cursor.string

        def tracking_string(self):
            value = original_cursor(self)
            seen["offset"] = self.offset
            return value

        gguf._Cursor.string = tracking_string
        try:
            self.assertEqual(gguf.architecture(path), "qwen2")
        finally:
            gguf._Cursor.string = original_cursor
        self.assertLessEqual(seen["offset"], prefix,
                             f"read to offset {seen['offset']} of {total}; the lookup did "
                             f"not stop at the key")
        self.assertLess(seen["offset"], total // 10)

    # -- fail-closed on malformed headers ---------------------------------------------

    def test_a_missing_architecture_key_fails_closed(self):
        path = write_gguf(self.dir / "nokey.gguf", [kv_string("general.name", "x")])
        with self.assertRaises(gguf.GGUFError) as caught:
            gguf.architecture(path)
        self.assertIn("declares no general.architecture", str(caught.exception))

    def test_a_non_string_architecture_fails_closed(self):
        path = write_gguf(self.dir / "wrongtype.gguf",
                          [kv_u32("general.architecture", 7)])
        with self.assertRaises(gguf.GGUFError) as caught:
            gguf.architecture(path)
        self.assertIn("not a string", str(caught.exception))

    def test_a_truncated_header_fails_closed(self):
        path = write_gguf(self.dir / "trunc.gguf",
                          [kv_string("general.architecture", "qwen35")])
        data = path.read_bytes()
        path.write_bytes(data[:-4])
        with self.assertRaises(gguf.GGUFError):
            gguf.architecture(path)

    def test_a_file_too_short_to_hold_a_header_fails_closed(self):
        """Found undefended by a mutation audit while adding the sliding-window reader.

        Without this check a 23-byte file is mmapped anyway, and mmap of a region shorter
        than the fixed header reads past it - so the refusal has to happen before the map,
        and it needs its own case because every other short-file test has enough bytes for
        the magic and counts.
        """
        path = self.dir / "tiny.gguf"
        path.write_bytes(MAGIC + struct.pack("<IQ", 3, 0) + b"\x00" * 7)
        self.assertEqual(path.stat().st_size, 23)
        with self.assertRaises(gguf.GGUFError) as caught:
            gguf.architecture(path)
        self.assertIn("too short", str(caught.exception))

    def test_a_header_promising_more_keys_than_it_has_fails_closed(self):
        path = write_gguf(self.dir / "short.gguf",
                          [kv_string("general.architecture", "qwen35")], n_kv=5)
        with self.assertRaises(gguf.GGUFError):
            gguf.architecture_exhaustive(path)

    def test_a_non_gguf_file_fails_closed(self):
        path = self.dir / "notgguf.bin"
        path.write_bytes(b"XXXX" + b"\x00" * 64)
        with self.assertRaises(gguf.GGUFError):
            gguf.architecture(path)

    # -- duplicates -------------------------------------------------------------------

    def test_a_conflicting_duplicate_is_refused_by_the_exhaustive_form(self):
        path = write_gguf(self.dir / "dup.gguf", [
            kv_string("general.architecture", "qwen35"),
            kv_string("general.architecture", "llama")])
        with self.assertRaises(gguf.GGUFError) as caught:
            gguf.architecture_exhaustive(path)
        self.assertIn("more than once", str(caught.exception))

    def test_an_identical_duplicate_is_accepted(self):
        path = write_gguf(self.dir / "dupsame.gguf", [
            kv_string("general.architecture", "qwen35"),
            kv_string("general.architecture", "qwen35")])
        self.assertEqual(gguf.architecture_exhaustive(path), "qwen35")

    def test_the_request_path_form_does_not_inspect_later_duplicates(self):
        # Stated rather than implied: early exit cannot see a later conflicting duplicate.
        # Admission runs the exhaustive form, so such a header never becomes an admitted
        # object; this test pins the documented limit so it cannot drift silently.
        path = write_gguf(self.dir / "dup2.gguf", [
            kv_string("general.architecture", "qwen35"),
            kv_string("general.architecture", "llama")])
        self.assertEqual(gguf.architecture(path), "qwen35")

    # -- classification is unchanged ---------------------------------------------------

    def test_classifications_are_unchanged(self):
        for arch, reusable in (("qwen2", True), ("llama", True), ("qwen35", False),
                               ("qwen35moe", False), ("mamba2", False), ("rwkv7", False),
                               ("jamba", False)):
            with self.subTest(arch=arch):
                path = write_gguf(self.dir / f"{arch}.gguf",
                                  [kv_string("general.architecture", arch)])
                self.assertEqual(gguf.architecture(path), arch)
                self.assertEqual(gguf.supports_prefix_reuse(arch)[0], reusable)

    def test_no_stale_value_is_reused_across_calls(self):
        path = write_gguf(self.dir / "mutable.gguf",
                          [kv_string("general.architecture", "qwen2")])
        self.assertEqual(gguf.architecture(path), "qwen2")
        write_gguf(path, [kv_string("general.architecture", "qwen35")])
        self.assertEqual(gguf.architecture(path), "qwen35",
                         "a cached architecture was returned after the file changed")


if __name__ == "__main__":
    unittest.main()
