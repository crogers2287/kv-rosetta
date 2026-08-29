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
    return (struct.pack("<iii", n_tokens, pos_min, pos_max)
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
        body = struct.pack("<iii", 64, 0, 63) + struct.pack("<Q", (1 << 34) + 1)
        self.assertStatus(PAYLOAD + appendix(count=1, body=body),
                          CheckpointStatus.MALFORMED)

    def test_malformed_appendix_does_not_allocate_its_claimed_size(self):
        # A 16 GiB length field must be rejected by arithmetic, not by reading it.
        body = struct.pack("<iii", 64, 0, 63) + struct.pack("<Q", 1 << 40)
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
