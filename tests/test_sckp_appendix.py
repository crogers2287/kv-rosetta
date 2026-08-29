"""Corruption cases for the SCKP checkpoint appendix.

The patched server ignores an appendix it cannot read and still reports a successful
restore, so a truncated or unknown-version appendix is indistinguishable from a good one
by looking at the restore response. These tests pin the parser's refusals: anything short
of a complete, EOF-terminated appendix must be reported as unusable, never as support.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from kv_rosetta.adapters.ggsq_envelope import (
    SCKP_MAGIC,
    SCKP_VERSION,
    CheckpointStatus,
    has_checkpoint_appendix,
    parse_checkpoint_appendix,
)

PAYLOAD = b"llama state bytes, whatever length the model and prompt produce" * 7


def buffer(nbytes: int) -> bytes:
    return struct.pack("<Q", nbytes) + bytes(nbytes)


def checkpoint(n_tokens=64, pos_min=0, pos_max=63, tgt=32, dft=0, spec=0) -> bytes:
    return (struct.pack("<qii", n_tokens, pos_min, pos_max)
            + buffer(tgt) + buffer(dft) + buffer(spec))


def appendix(count=1, version=SCKP_VERSION, body=None) -> bytes:
    body = checkpoint() * count if body is None else body
    return SCKP_MAGIC + struct.pack("<II", version, count) + body


class SCKPAppendixTest(unittest.TestCase):
    def parse(self, blob: bytes):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slot.bin"
            path.write_bytes(blob)
            return parse_checkpoint_appendix(path), has_checkpoint_appendix(path)

    def assertStatus(self, blob: bytes, status: CheckpointStatus):
        parsed, flag = self.parse(blob)
        self.assertIs(parsed.status, status)
        self.assertEqual(flag, status is CheckpointStatus.OK)
        return parsed

    # -- the appendix is present and complete -------------------------------------

    def test_complete_appendix_is_usable(self):
        parsed = self.assertStatus(PAYLOAD + appendix(), CheckpointStatus.OK)
        self.assertEqual(parsed.version, SCKP_VERSION)
        self.assertEqual(parsed.count, 1)
        self.assertEqual(parsed.offset, len(PAYLOAD))
        self.assertEqual(parsed.nbytes, len(appendix()))

    def test_several_checkpoints_are_usable(self):
        parsed = self.assertStatus(PAYLOAD + appendix(count=3), CheckpointStatus.OK)
        self.assertEqual(parsed.count, 3)

    def test_zero_checkpoints_is_a_complete_appendix(self):
        # The writer never emits this - it returns early on an empty list - but a
        # zero-count appendix is still well formed and must not read as corruption.
        self.assertStatus(PAYLOAD + appendix(count=0, body=b""), CheckpointStatus.OK)

    # -- missing ------------------------------------------------------------------

    def test_no_appendix_is_absent(self):
        self.assertStatus(PAYLOAD, CheckpointStatus.ABSENT)

    def test_empty_file_is_absent(self):
        self.assertStatus(b"", CheckpointStatus.ABSENT)

    def test_missing_file_is_absent_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "does-not-exist.bin"
            self.assertIs(parse_checkpoint_appendix(path).status, CheckpointStatus.ABSENT)
            self.assertFalse(has_checkpoint_appendix(path))

    def test_directory_is_absent_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(has_checkpoint_appendix(directory))

    # -- truncated ----------------------------------------------------------------

    def test_magic_alone_is_truncated(self):
        # The old scanner reported this as checkpoint support.
        self.assertStatus(PAYLOAD + SCKP_MAGIC, CheckpointStatus.TRUNCATED)

    def test_header_cut_mid_count_is_truncated(self):
        self.assertStatus(PAYLOAD + appendix()[:10], CheckpointStatus.TRUNCATED)

    def test_checkpoint_record_cut_short_is_truncated(self):
        self.assertStatus(PAYLOAD + appendix()[:-8], CheckpointStatus.TRUNCATED)

    def test_buffer_payload_cut_short_is_truncated(self):
        full = appendix(count=1, body=checkpoint(tgt=4096))
        self.assertStatus(PAYLOAD + full[:-2048], CheckpointStatus.TRUNCATED)

    def test_count_promises_more_checkpoints_than_present(self):
        self.assertStatus(PAYLOAD + appendix(count=4, body=checkpoint() * 2),
                          CheckpointStatus.TRUNCATED)

    # -- unknown version ----------------------------------------------------------

    def test_future_version_is_refused(self):
        parsed = self.assertStatus(PAYLOAD + appendix(version=SCKP_VERSION + 1),
                                   CheckpointStatus.UNKNOWN_VERSION)
        self.assertEqual(parsed.version, SCKP_VERSION + 1)

    def test_zero_version_is_refused(self):
        self.assertStatus(PAYLOAD + appendix(version=0), CheckpointStatus.UNKNOWN_VERSION)

    def test_absurd_version_is_refused_not_read_as_a_length(self):
        self.assertStatus(PAYLOAD + appendix(version=0xFFFFFFFF),
                          CheckpointStatus.UNKNOWN_VERSION)

    # -- malformed ----------------------------------------------------------------

    def test_count_beyond_the_writers_cap_is_malformed(self):
        # The server refuses count > 1024; so do we, without trying to allocate for it.
        self.assertStatus(PAYLOAD + appendix(count=1025, body=b""),
                          CheckpointStatus.MALFORMED)

    def test_buffer_length_beyond_the_16gib_cap_is_malformed(self):
        body = struct.pack("<qii", 64, 0, 63) + struct.pack("<Q", (1 << 34) + 1)
        self.assertStatus(PAYLOAD + appendix(count=1, body=body),
                          CheckpointStatus.MALFORMED)

    def test_malformed_appendix_does_not_allocate_its_claimed_size(self):
        # A 16 GiB length field must be rejected by arithmetic, not by reading it.
        body = struct.pack("<qii", 64, 0, 63) + struct.pack("<Q", 1 << 40)
        self.assertStatus(PAYLOAD + appendix(count=1, body=body),
                          CheckpointStatus.MALFORMED)

    # -- the magic occurring by chance --------------------------------------------

    def test_magic_inside_the_payload_is_refused(self):
        # Four bytes of KV data can spell SCKP, and nothing distinguishes that from an
        # appendix written by a future version - so the parser refuses rather than
        # claiming either. What matters is that it is never reported as support.
        parsed, flag = self.parse(SCKP_MAGIC + PAYLOAD)
        self.assertFalse(parsed.usable)
        self.assertFalse(flag)
        self.assertIsNot(parsed.status, CheckpointStatus.OK)

    def test_chance_magic_does_not_mask_a_real_appendix(self):
        blob = PAYLOAD + SCKP_MAGIC + PAYLOAD + appendix()
        parsed = self.assertStatus(blob, CheckpointStatus.OK)
        self.assertEqual(parsed.offset, len(blob) - len(appendix()))

    def test_chance_magic_does_not_mask_a_truncated_appendix(self):
        blob = SCKP_MAGIC + PAYLOAD + appendix()[:-4]
        self.assertStatus(blob, CheckpointStatus.TRUNCATED)


if __name__ == "__main__":
    unittest.main()


class CountingHandle:
    """Wraps a file object and records how many bytes were actually read."""

    def __init__(self, handle):
        self._handle = handle
        self.bytes_read = 0
        self.reads = 0

    def read(self, count=-1):
        block = self._handle.read(count)
        self.bytes_read += len(block)
        self.reads += 1
        return block

    def seek(self, *args):
        return self._handle.seek(*args)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)


class BoundedReadTest(unittest.TestCase):
    """Validation must cost the same on a 4 GiB artifact as on a small one.

    The production 256-token patched slot file is already 487,926,936 bytes and grows with
    the context. Reading it whole to check a few dozen bytes of framing cost that much
    transient memory per validation, which does not survive the 2K/8K/32K ladder.
    """

    BODY = 4 << 30      # 4 GiB, sparse - larger than any artifact measured so far

    def sparse_artifact(self, directory: Path, appendix: bytes) -> Path:
        path = Path(directory) / "slot.bin"
        with open(path, "wb") as handle:
            handle.truncate(self.BODY)      # sparse: consumes no disk blocks
            handle.seek(self.BODY)
            handle.write(appendix)
        allocated = path.stat().st_blocks * 512
        if allocated > (64 << 20):
            self.skipTest(f"filesystem did not keep the fixture sparse "
                          f"({allocated >> 20} MiB allocated); the point of the test is "
                          f"the reader, not the writer")
        return path

    def test_validation_at_a_known_offset_reads_only_the_framing(self):
        from kv_rosetta.adapters import ggsq_envelope

        with tempfile.TemporaryDirectory() as directory:
            blob = appendix(count=2)
            path = self.sparse_artifact(directory, blob)
            self.assertEqual(path.stat().st_size, self.BODY + len(blob))

            real_open = open
            counters = []

            def counting_open(*args, **kwargs):
                handle = CountingHandle(real_open(*args, **kwargs))
                counters.append(handle)
                return handle

            ggsq_envelope.open = counting_open
            try:
                result = ggsq_envelope.checkpoint_appendix_at(path, self.BODY)
            finally:
                del ggsq_envelope.open

            self.assertIs(result.status, CheckpointStatus.OK)
            self.assertEqual(result.count, 2)
            total = sum(c.bytes_read for c in counters)
            # 4 bytes of magic, a 12-byte header, then 12 + 24 per checkpoint.
            self.assertLessEqual(total, 256,
                                 f"read {total} bytes to validate {len(blob)} bytes of "
                                 f"framing on a {self.BODY >> 30} GiB file")

    def test_validation_does_not_allocate_a_declared_buffer_length(self):
        # A 16 GiB length field must be rejected by arithmetic. If the parser ever tried to
        # read or reserve it, this test would exhaust memory rather than fail.
        with tempfile.TemporaryDirectory() as directory:
            body = struct.pack("<qii", 64, 0, 63) + struct.pack("<Q", (1 << 34) + 1)
            path = self.sparse_artifact(directory, appendix(count=1, body=body))
            result = parse_checkpoint_appendix(path)
            self.assertIs(result.status, CheckpointStatus.MALFORMED)

    def test_a_length_field_pointing_past_eof_is_truncated_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            body = struct.pack("<qii", 64, 0, 63) + struct.pack("<Q", 1 << 30)
            path = self.sparse_artifact(directory, appendix(count=1, body=body))
            self.assertIs(parse_checkpoint_appendix(path).status,
                          CheckpointStatus.TRUNCATED)


class WriterLayoutTest(unittest.TestCase):
    """The record layout must match the types the patch actually writes.

    This was wrong once. The parser assumed three int32 fields, so it read 12-byte records
    while save_slot_checkpoints() writes 16: n_tokens is int64_t in
    common_prompt_checkpoint, pos_min and pos_max are llama_pos (int32). Every fixture in
    this file used the same wrong pack format, so the whole suite passed while the parser
    rejected every real appendix. It failed closed rather than open, but it proved nothing.

    The framing below was read from a real slot file written by the patched 27B server:
    n_written 331,032,532, checkpoint_bytes 156,894,416, so the appendix begins at
    174,138,116 and runs to EOF with one checkpoint of 252 tokens at positions 251..251,
    whose target buffer is 156,894,364 bytes and whose draft and speculative buffers are
    empty - a target-only launch.
    """

    OBSERVED_HEADER = bytes.fromhex("53434b50" "01000000" "01000000")
    OBSERVED_RECORD = bytes.fromhex("fc00000000000000" "fb000000" "fb000000")
    OBSERVED_BUFFER_LENGTHS = (156894364, 0, 0)

    def test_record_size_matches_the_writer(self):
        from kv_rosetta.adapters.ggsq_envelope import _SCKP_RECORD_SIZE

        self.assertEqual(_SCKP_RECORD_SIZE, 16)
        self.assertEqual(len(self.OBSERVED_RECORD), 16)

    def test_observed_framing_decodes_to_the_reported_metadata(self):
        self.assertEqual(self.OBSERVED_HEADER[:4], SCKP_MAGIC)
        version, count = struct.unpack("<II", self.OBSERVED_HEADER[4:12])
        self.assertEqual((version, count), (SCKP_VERSION, 1))
        n_tokens, pos_min, pos_max = struct.unpack("<qii", self.OBSERVED_RECORD)
        self.assertEqual((n_tokens, pos_min, pos_max), (252, 251, 251))

    def test_the_observed_framing_parses_as_a_complete_appendix(self):
        blob = self.OBSERVED_HEADER + self.OBSERVED_RECORD
        for length in self.OBSERVED_BUFFER_LENGTHS:
            # Real payload sizes with placeholder bytes, so the framing is exercised
            # without materialising 150 MB.
            blob += struct.pack("<Q", length)
        # Rebuild with payloads small enough to hold, keeping the framing shape.
        blob = self.OBSERVED_HEADER + self.OBSERVED_RECORD
        for length in (32, 0, 0):
            blob += struct.pack("<Q", length) + bytes(length)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slot.bin"
            path.write_bytes(PAYLOAD + blob)
            result = parse_checkpoint_appendix(path)
            self.assertIs(result.status, CheckpointStatus.OK)
            self.assertEqual(result.count, 1)
            self.assertEqual(result.offset, len(PAYLOAD))

    def test_a_twelve_byte_record_no_longer_parses(self):
        # The old, wrong layout must not validate, or the regression could return unnoticed.
        blob = (self.OBSERVED_HEADER + struct.pack("<iii", 252, 251, 251)
                + struct.pack("<Q", 0) * 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slot.bin"
            path.write_bytes(PAYLOAD + blob)
            self.assertIsNot(parse_checkpoint_appendix(path).status, CheckpointStatus.OK)
