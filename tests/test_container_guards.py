"""Refusals in the KVX container reader and writer.

A mutation audit found 22 of container.py's 23 guards could be deleted without any test
noticing. This is the module that decides whether a file on disk becomes tensors handed to
a runtime, so every one of those refusals is load-bearing: a truncated header, a mismatched
payload digest or an opaque-only dtype read as canonical produces garbage rather than an
error if the check is absent.
"""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kv_rosetta import container
from kv_rosetta.segments import Segment


def _manifest(dtype="f16", tokens=3):
    return {
        "schema": "kvx/0.2",
        "kv": {"layers": 2, "heads": 2, "head_dim": 4, "dtype": dtype,
               "layout": "layer,kv,token,head,dim", "byte_order": "little",
               "tokens": tokens},
    }


def _seg(name, nbytes, offset=0):
    return Segment(name=name, role="k", layer_start=0, layer_end=1, dtype="f16",
                   shape=(nbytes // 2,), layout="token", offset=offset, nbytes=nbytes)


class TmpDirCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())


class WriterGuards(TmpDirCase):
    def test_header_without_a_digest_placeholder_is_refused(self):
        # The digest covers the header containing it, so it is hashed against a
        # fixed-width placeholder. Without the field there is nothing to overwrite and
        # the payload offset would silently shift.
        with self.assertRaises(container.ContainerError) as caught:
            container._digest_placeholder(b'{"schema":"kvx/0.2"}')
        self.assertIn("header_sha256", str(caught.exception))

    def test_segmented_artifact_needs_at_least_one_segment(self):
        with self.assertRaises(container.ContainerError) as caught:
            container.write_segments(self.dir / "a.kvx", _manifest(), [])
        self.assertIn("at least one segment", str(caught.exception))

    def test_segment_declaring_the_wrong_length_is_refused(self):
        # nbytes is what readers slice by; trusting it over the actual data hands the
        # next segment's bytes to this one.
        with self.assertRaises(container.ContainerError) as caught:
            container.write_segments(self.dir / "a.kvx", _manifest(),
                                     [(_seg("k0", 8), b"\x00" * 4)])
        self.assertIn("declares 8 bytes", str(caught.exception))

    def test_segment_table_that_fails_validation_is_refused(self):
        # Two segments under one name: a lookup by name would resolve to whichever came
        # first and read the wrong region.
        with self.assertRaises(container.ContainerError) as caught:
            container.write_segments(self.dir / "a.kvx", _manifest(),
                                     [(_seg("k0", 8), b"\x00" * 8),
                                      (_seg("k0", 8), b"\x11" * 8)])
        self.assertIn("invalid segment table", str(caught.exception))

    def test_raw_tensor_must_be_five_dimensional(self):
        with self.assertRaises(container.ContainerError) as caught:
            container.write_raw(self.dir / "a.kvx", _manifest(),
                                np.zeros((2, 3, 2, 4), dtype=np.float16))
        self.assertIn("5-D", str(caught.exception))

    def test_raw_tensor_axis_one_must_be_the_kv_axis(self):
        # (layer, kv, token, head, dim) with kv != 2 means keys and values are not both
        # present, and every downstream reshape would still succeed.
        with self.assertRaises(container.ContainerError) as caught:
            container.write_raw(self.dir / "a.kvx", _manifest(),
                                np.zeros((2, 3, 3, 2, 4), dtype=np.float16))
        self.assertIn("kv axis", str(caught.exception))

    def test_opaque_format_must_be_a_non_empty_string(self):
        # opaque_format is the only record of how to decode the blob; blank means the
        # artifact is undecodable but looks well-formed.
        blob = b"\x01" * 16
        for bad in ("", "   ", None, 3):
            with self.assertRaises(container.ContainerError) as caught:
                container.write_opaque(self.dir / "o.kvx", _manifest(), blob, bad)
            self.assertIn("opaque_format", str(caught.exception))

    def test_streaming_writer_also_requires_an_opaque_format(self):
        src = self.dir / "state.bin"
        src.write_bytes(b"\x02" * 32)
        for bad in ("", "   ", None):
            with self.assertRaises(container.ContainerError) as caught:
                container.write_opaque_from_file(self.dir / "o.kvx", _manifest(), src, bad)
            self.assertIn("opaque_format", str(caught.exception))

    def test_streaming_writer_refuses_a_missing_source(self):
        with self.assertRaises(container.ContainerError) as caught:
            container.write_opaque_from_file(self.dir / "o.kvx", _manifest(),
                                             self.dir / "absent.bin", "ggsq/3")
        self.assertIn("does not exist", str(caught.exception))


class ExtractGuards(TmpDirCase):
    def setUp(self):
        super().setUp()
        self.blob = bytes(range(256)) * 8
        self.path = container.write_opaque(self.dir / "o.kvx", _manifest(), self.blob,
                                           "ggsq/3")

    def test_truncated_payload_is_refused_rather_than_short_written(self):
        # A short read must not produce a file that looks complete: the caller hands the
        # extracted path straight to llama-server.
        raw = self.path.read_bytes()
        cut = self.dir / "cut.kvx"
        cut.write_bytes(raw[:-64])
        dest = self.dir / "out.bin"
        with self.assertRaises(container.ContainerError) as caught:
            container.extract_payload(cut, dest)
        self.assertIn("truncated", str(caught.exception))
        self.assertFalse(dest.exists(), "partial extraction must not be left behind")

    def test_payload_digest_mismatch_is_refused(self):
        raw = bytearray(self.path.read_bytes())
        offset = int(container.read_header(self.path)["blob"]["offset"])
        raw[offset + 5] ^= 0xFF
        bad = self.dir / "bad.kvx"
        bad.write_bytes(bytes(raw))
        dest = self.dir / "out.bin"
        with self.assertRaises(container.ContainerError) as caught:
            container.extract_payload(bad, dest)
        self.assertIn("sha256 mismatch", str(caught.exception))
        self.assertFalse(dest.exists(), "a mismatched payload must not be left behind")


class EncodingAccessorGuards(TmpDirCase):
    def setUp(self):
        super().setUp()
        self.raw = container.write_raw(self.dir / "r.kvx", _manifest(),
                                       np.zeros((2, 2, 3, 2, 4), dtype=np.float16))
        self.opaque = container.write_opaque(self.dir / "o.kvx", _manifest(), b"\x00" * 32,
                                             "ggsq/3")

    def test_tensor_on_a_non_raw_artifact_is_refused(self):
        with self.assertRaises(container.ContainerError) as caught:
            container.read(self.opaque).tensor
        self.assertIn("not 'raw'", str(caught.exception))

    def test_opaque_on_a_non_opaque_artifact_is_refused(self):
        with self.assertRaises(container.ContainerError) as caught:
            container.read(self.raw).opaque
        self.assertIn("not 'opaque'", str(caught.exception))

    def test_segments_on_a_non_segmented_artifact_is_refused(self):
        with self.assertRaises(container.ContainerError) as caught:
            container.read(self.raw).segments
        self.assertIn("not 'segmented'", str(caught.exception))

    def test_quantised_kv_dtype_has_no_canonical_tensor(self):
        # q8_0 rows are blocks, not an array of scalars. Viewing them as one produces a
        # plausible-shaped tensor of nonsense, which is worse than a refusal.
        path = container.write_opaque(self.dir / "q.kvx", _manifest(dtype="q8_0"),
                                      b"\x00" * 32, "ggsq/3")
        art = container.read(path)
        art.header["blob"]["encoding"] = "raw"
        art.header["blob"]["shape"] = [2, 2, 3, 2, 4]
        with self.assertRaises(container.ContainerError) as caught:
            art.tensor
        self.assertIn("opaque-only", str(caught.exception))

    def test_quantised_segment_dtype_is_refused(self):
        seg = _seg("k0", 64)
        path = container.write_segments(self.dir / "s.kvx", _manifest(),
                                        [(seg, b"\x00" * 64)])
        art = container.read(path)
        art.header["blob"]["segments"][0]["dtype"] = "q8_0"
        with self.assertRaises(container.ContainerError) as caught:
            art.segment("k0")
        self.assertIn("opaque-only", str(caught.exception))

    def test_unknown_segment_name_is_refused(self):
        seg = _seg("k0", 64)
        path = container.write_segments(self.dir / "s.kvx", _manifest(),
                                        [(seg, b"\x00" * 64)])
        with self.assertRaises(container.ContainerError):
            container.read(path).segment("v0")


class HeaderParseGuards(TmpDirCase):
    """Every one of these is a file another process wrote. None may be trusted."""

    def setUp(self):
        super().setUp()
        self.good = container.write_raw(self.dir / "r.kvx", _manifest(),
                                        np.zeros((2, 2, 3, 2, 4), dtype=np.float16))

    def _write(self, name, payload):
        path = self.dir / name
        path.write_bytes(payload)
        return path

    def test_file_shorter_than_the_preamble_is_refused(self):
        path = self._write("short.kvx", b"KVX")
        with self.assertRaises(container.ContainerError) as caught:
            container.read_header(path)
        self.assertIn("preamble", str(caught.exception))

    def test_bad_magic_is_refused(self):
        path = self._write("magic.kvx", b"GGUF" + struct.pack("<II", 1, 2) + b"{}")
        with self.assertRaises(container.ContainerError) as caught:
            container.read_header(path)
        self.assertIn("magic", str(caught.exception))

    def test_unsupported_container_version_is_refused(self):
        body = b'{"blob":{}}'
        path = self._write("ver.kvx",
                           container.MAGIC + struct.pack("<II", 99, len(body)) + body)
        with self.assertRaises(container.ContainerError) as caught:
            container.read_header(path)
        self.assertIn("version", str(caught.exception))

    def test_truncated_header_is_refused(self):
        body = b'{"blob":{"offset":0,"nbytes":0}}'
        path = self._write("trunc.kvx",
                           container.MAGIC + struct.pack("<II", 1, len(body) + 40) + body)
        with self.assertRaises(container.ContainerError) as caught:
            container.read_header(path)
        self.assertIn("truncated", str(caught.exception))

    def test_header_that_is_not_a_json_object_is_refused(self):
        body = b'[1, 2, 3]'
        path = self._write("list.kvx",
                           container.MAGIC + struct.pack("<II", 1, len(body)) + body)
        with self.assertRaises(container.ContainerError) as caught:
            container.read_header(path)
        self.assertIn("JSON object", str(caught.exception))

    def test_header_without_a_blob_object_is_refused(self):
        # Everything downstream indexes header["blob"]; a missing or non-object blob
        # would surface as a KeyError or a TypeError deep in a reader instead.
        for blob in ('', ', "blob": null', ', "blob": []', ', "blob": 7'):
            body = ('{"schema":"kvx/0.2"' + blob + '}').encode()
            path = self._write("noblob.kvx",
                               container.MAGIC + struct.pack("<II", 1, len(body)) + body)
            with self.assertRaises(container.ContainerError) as caught:
                container.read_header(path)
            self.assertIn("blob", str(caught.exception))

    def test_file_shorter_than_the_declared_payload_is_refused(self):
        # np.memmap on a declared length past EOF is the failure this prevents.
        raw = self.good.read_bytes()
        path = self._write("cut.kvx", raw[:-32])
        with self.assertRaises(container.ContainerError) as caught:
            container.read(path)
        self.assertIn("shorter than the declared payload", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


class GGUFShortFileGuard(TmpDirCase):
    """A truncated GGUF must fail closed with a typed error, not an mmap ValueError.

    This runs on the request path: the model file is read to decide whether an artifact
    may be admitted for it, so an untyped exception escapes into the caller.
    """

    def test_file_too_short_to_hold_a_gguf_header_is_refused(self):
        from kv_rosetta.gguf import GGUFError, read_string_key
        for name, payload in (("empty.gguf", b""),
                              ("stub.gguf", b"GGUF" + b"\x00" * 8)):
            path = self.dir / name
            path.write_bytes(payload)
            with self.assertRaises(GGUFError) as caught:
                read_string_key(path, "general.architecture")
            self.assertIn("too short", str(caught.exception))
