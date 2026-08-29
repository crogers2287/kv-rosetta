"""llama.cpp sequence-state envelopes.

The body parser must never be handed an unknown buffer: an in-process buffer and a
GGSQ file put their state body at different offsets, and guessing silently misparses
everything downstream.
"""

import struct
import unittest
from dataclasses import replace

from kv_rosetta.adapters import ggsq_envelope as envelope
from kv_rosetta.adapters.ggsq_envelope import EnvelopeError, Source


def _file_blob(tokens=(1, 2, 3, 4), version=3, body=b"BODY-BYTES"):
    return (envelope.GGSQ_MAGIC + struct.pack("<II", version, len(tokens))
            + b"".join(struct.pack("<i", t) for t in tokens) + body)


class FileEnvelopeTests(unittest.TestCase):
    def test_body_offset_is_computed_not_guessed(self):
        tokens = (1, 2, 3, 4)
        blob = _file_blob(tokens)
        env = envelope.parse(blob)
        self.assertEqual(env.source, Source.FILE)
        self.assertEqual(env.version, 3)
        self.assertEqual(env.token_ids, tokens)
        self.assertEqual(env.body_offset, 4 + 4 + 4 + 4 * len(tokens))
        self.assertEqual(envelope.body(blob, env), b"BODY-BYTES")

    def test_zero_tokens_is_valid(self):
        env = envelope.parse(_file_blob(tokens=()))
        self.assertEqual(env.token_ids, ())
        self.assertEqual(env.body_offset, 12)

    def test_bad_magic_is_refused(self):
        with self.assertRaises(EnvelopeError):
            envelope.parse_file_envelope(b"NOPE" + struct.pack("<II", 3, 0))

    def test_unsupported_version_is_refused(self):
        with self.assertRaises(EnvelopeError) as ctx:
            envelope.parse_file_envelope(_file_blob(version=99))
        self.assertIn("99", str(ctx.exception))

    def test_truncated_inputs_are_refused(self):
        full = _file_blob()
        for cut in (0, 1, 4, 7, 11, 15):
            with self.subTest(length=cut), self.assertRaises(EnvelopeError):
                envelope.parse_file_envelope(full[:cut])

    def test_token_count_past_the_end_is_refused(self):
        blob = envelope.GGSQ_MAGIC + struct.pack("<II", 3, 10 ** 6) + b"\x00\x00\x00\x00"
        with self.assertRaises(EnvelopeError):
            envelope.parse_file_envelope(blob)


class BufferEnvelopeTests(unittest.TestCase):
    def test_buffer_without_magic(self):
        blob = struct.pack("<i", 7) + b"BODY"
        env = envelope.parse_buffer_envelope(blob)
        self.assertEqual(env.source, Source.BUFFER)
        self.assertEqual(env.token_ids, ())
        self.assertEqual(envelope.body(blob, env), b"BODY")

    def test_declared_io_magic_must_match(self):
        blob = b"IOMG" + struct.pack("<i", 0) + b"BODY"
        self.assertEqual(envelope.body(blob, envelope.parse_buffer_envelope(blob, io_magic=b"IOMG")), b"BODY")
        with self.assertRaises(EnvelopeError):
            envelope.parse_buffer_envelope(blob, io_magic=b"XXXX")


class DispatchTests(unittest.TestCase):
    def test_detect_distinguishes_the_two_sources(self):
        self.assertEqual(envelope.detect(_file_blob()), Source.FILE)
        self.assertEqual(envelope.detect(struct.pack("<i", 0) + b"BODY"), Source.BUFFER)

    def test_body_range_must_be_inside_the_blob(self):
        blob = _file_blob()
        env = envelope.parse(blob)
        with self.assertRaises(EnvelopeError):
            envelope.body(blob, replace(env, body_length=10 ** 6))


if __name__ == "__main__":
    unittest.main()
