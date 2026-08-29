"""llama.cpp sequence-state envelopes.

The body parser must never be handed an unknown buffer: an in-process buffer and a
GGSQ file put their state body at different offsets, and guessing silently misparses
everything downstream.
"""

import struct
import unittest
from pathlib import Path
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
    def test_buffer_requires_the_runtime_io_magic(self):
        """A buffer is not self-describing, so it may not be inferred from its bytes."""
        blob = b"IOMG" + struct.pack("<i", 7) + b"BODY"
        env = envelope.parse_buffer_envelope(blob, io_magic=b"IOMG")
        self.assertEqual(env.source, Source.BUFFER)
        self.assertEqual(env.token_ids, ())
        self.assertEqual(envelope.body(blob, env), b"BODY")

    def test_buffer_without_declared_magic_is_refused(self):
        with self.assertRaises(EnvelopeError):
            envelope.parse_buffer_envelope(struct.pack("<i", 7) + b"BODY", io_magic=b"")

    def test_declared_io_magic_must_match(self):
        blob = b"IOMG" + struct.pack("<i", 0) + b"BODY"
        self.assertEqual(envelope.body(blob, envelope.parse_buffer_envelope(blob, io_magic=b"IOMG")), b"BODY")
        with self.assertRaises(EnvelopeError):
            envelope.parse_buffer_envelope(blob, io_magic=b"XXXX")


class DispatchTests(unittest.TestCase):
    def test_detect_recognises_a_file_and_refuses_to_guess(self):
        """Unrecognised bytes are UNKNOWN, never BUFFER: guessing is how a real state file
        was parsed into a plausible envelope with zero tokens."""
        self.assertEqual(envelope.detect(_file_blob()), Source.FILE)
        for blob in (struct.pack("<i", 0) + b"BODY", b"", b"\x01\x02\x03\x04", b"GGSQ"):
            with self.subTest(blob=blob[:8]):
                self.assertEqual(envelope.detect(blob), Source.UNKNOWN)

    def test_parse_refuses_an_unclassifiable_blob(self):
        with self.assertRaises(EnvelopeError):
            envelope.parse(b"\x01\x02\x03\x04rest")

    def test_body_range_must_be_inside_the_blob(self):
        blob = _file_blob()
        env = envelope.parse(blob)
        with self.assertRaises(EnvelopeError):
            envelope.body(blob, replace(env, body_length=10 ** 6))


if __name__ == "__main__":
    unittest.main()


class RealFixtureTests(unittest.TestCase):
    """Parsed against a real slot file captured from llama-server, not a synthetic blob.

    Every constant here was wrong when first written from the specification alone:
    the magic is a little-endian uint32 so the bytes read "qsgg" rather than "GGSQ", the
    running binary writes version 2 while the checked-out header declares 3, and the
    "token" array is a serialized prompt structure rather than a token list.
    """

    FIXTURE = Path(__file__).parent / "fixtures" / "llamacpp_state_seq_header.bin"

    def setUp(self):
        if not self.FIXTURE.is_file():
            self.skipTest("fixture missing")
        self.raw = self.FIXTURE.read_bytes()

    def test_magic_is_little_endian_uint32(self):
        self.assertEqual(self.raw[:4], b"qsgg")
        self.assertEqual(envelope.GGSQ_MAGIC, (0x67677371).to_bytes(4, "little"))

    def test_real_file_is_detected_as_a_file(self):
        self.assertEqual(envelope.detect(self.raw), Source.FILE)

    def test_real_file_parses(self):
        env = envelope.parse(self.raw)
        self.assertEqual(env.source, Source.FILE)
        self.assertIn(env.version, envelope.SUPPORTED_VERSIONS)
        self.assertEqual(len(env.token_ids), 205)
        self.assertEqual(env.body_offset, 12 + 4 * 205)

    def test_a_real_file_is_never_parsed_as_a_buffer(self):
        """The fail-open case: a state file misread as an in-process buffer produced a
        plausible envelope with zero tokens and a body offset of 4."""
        with self.assertRaises(EnvelopeError):
            envelope.parse_buffer_envelope(self.raw)

    def test_prompt_decodes_to_real_token_ids(self):
        packed = envelope.parse(self.raw).token_ids
        tokens = envelope.decode_prompt_tokens(packed)
        self.assertEqual(len(tokens), 201, "count must match the server's n_prompt_tokens")
        self.assertTrue(all(t >= 0 for t in tokens), "a decoded token must never be the -1 marker")
        self.assertNotEqual(tokens[0], envelope.LLAMA_TOKEN_NULL)

    def test_plain_token_list_is_still_accepted(self):
        self.assertEqual(envelope.decode_prompt_tokens([5, 6, 7]), (5, 6, 7))

    def test_unsupported_prompt_state_version_is_refused(self):
        with self.assertRaises(EnvelopeError):
            envelope.decode_prompt_tokens([envelope.LLAMA_TOKEN_NULL, 99, 1, 42])

    def test_truncated_prompt_state_is_refused(self):
        with self.assertRaises(EnvelopeError):
            envelope.decode_prompt_tokens([envelope.LLAMA_TOKEN_NULL, 1, 50, 1, 2])

    def test_multimodal_prompt_yields_no_text_tokens(self):
        # marker, version, 1 token, then one media key: reusing text-only would drop media.
        packed = [envelope.LLAMA_TOKEN_NULL, 1, 1, 42, 1, 0]
        self.assertEqual(envelope.decode_prompt_tokens(packed), ())
