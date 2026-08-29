"""Staging verifies the payload in the pass that writes it, not in a separate read.

At 2K the import spent 1.986 s on container verification and 2.404 s on staging - two full
passes over the same 577 MiB payload, 66% of the total. extract_payload() already hashes
every byte as it writes and refuses before publishing, so the separate verification read was
learning the same fact twice.

Removing a check is only safe if nothing it caught now gets through. These tests assert the
refusals are unchanged and that they still happen before any restore POST.
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

from kv_rosetta import container
from kv_rosetta.adapters.base import ImportRequest
from tests.test_compound_format_agreement import genuine_artifact
from tests.test_hybrid_export_gate import ggsq_body, props
from tests.test_hybrid_support_predicate import RestoreStub, TARGET_ONLY


def corrupt_byte(source: Path, target: Path, position: int) -> Path:
    data = bytearray(source.read_bytes())
    data[position] ^= 0xFF
    target.write_bytes(bytes(data))
    return target


class OnePassStagingTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.artifact, self.save = genuine_artifact(self.directory)
        self.header = container.read_header(self.artifact)
        self.payload_offset = int(self.header["blob"]["offset"])

    def importer(self):
        slots = self.directory / f"slots{len(list(self.directory.iterdir()))}"
        slots.mkdir()
        restore = {"n_restored": 263,
                   "n_checkpoints_restored": self.save["n_checkpoints_saved"],
                   "checkpoint_bytes": self.save["checkpoint_bytes"],
                   "checkpoint_n_tokens": self.save["checkpoint_n_tokens"],
                   "checkpoint_pos_min": self.save["checkpoint_pos_min"],
                   "checkpoint_pos_max": self.save["checkpoint_pos_max"]}
        return RestoreStub(props(**TARGET_ONLY), slots, save=dict(self.save),
                           body=ggsq_body(trailer=b""), restore=restore)

    def run_import(self, artifact):
        adapter = self.importer()
        report = adapter.import_(artifact, ImportRequest(model="", slot=0))
        return adapter, report

    # -- the refusals the removed pass used to make ---------------------------------

    def test_a_corrupted_payload_byte_is_still_refused_before_restore(self):
        bad = corrupt_byte(self.artifact, self.directory / "bad-payload.kvx",
                           self.payload_offset + 16)
        self.assertFalse(container.verify(bad)[0], "fixture must actually be corrupt")
        adapter, report = self.run_import(bad)
        self.assertFalse(report.ok)
        self.assertIn("payload", report.reason.lower())
        self.assertEqual(adapter.restore_posts(), [],
                         "corrupt payload reached the runtime")

    def test_a_corrupted_header_byte_is_still_refused(self):
        bad = corrupt_byte(self.artifact, self.directory / "bad-header.kvx", 40)
        adapter, report = self.run_import(bad)
        self.assertFalse(report.ok)
        self.assertEqual(adapter.restore_posts(), [])

    def test_a_truncated_artifact_is_still_refused(self):
        data = self.artifact.read_bytes()
        bad = self.directory / "truncated.kvx"
        bad.write_bytes(data[: len(data) - 64])
        adapter, report = self.run_import(bad)
        self.assertFalse(report.ok)
        self.assertEqual(adapter.restore_posts(), [])

    def test_a_staged_copy_is_not_left_behind_after_a_payload_mismatch(self):
        bad = corrupt_byte(self.artifact, self.directory / "bad-leftover.kvx",
                           self.payload_offset + 32)
        adapter, report = self.run_import(bad)
        self.assertFalse(report.ok)
        leftover = sorted(adapter.slot_save_path.glob("*.restore.bin")) + \
            sorted(adapter.slot_save_path.glob("*.tmp"))
        self.assertEqual(leftover, [], f"left {leftover} behind")

    def test_a_payload_mismatch_returns_a_report_not_an_exception(self):
        bad = corrupt_byte(self.artifact, self.directory / "bad-report.kvx",
                           self.payload_offset + 8)
        _, report = self.run_import(bad)   # must not raise ContainerError
        self.assertFalse(report.ok)
        self.assertIn("staging", report.phases)

    def test_the_genuine_artifact_still_imports(self):
        _, report = self.run_import(self.artifact)
        self.assertTrue(report.ok, report.reason)

    # -- the second pass is actually gone -------------------------------------------

    def big_artifact(self) -> tuple[Path, dict]:
        """An artifact whose payload dominates the fixed per-import reads.

        At a few hundred bytes the header reads swamp the payload and a read-count ratio
        measures nothing.
        """
        from tests.test_hybrid_export_gate import (
            StubAdapter, ggsq_body, sckp_appendix, save_with_appendix)
        from kv_rosetta.adapters.base import ExportRequest, Representation

        appendix = sckp_appendix()
        body = ggsq_body(trailer=b"") + bytes(8 << 20) + appendix
        save = dict(save_with_appendix()[0], n_written=len(body),
                    checkpoint_bytes=len(appendix), checkpoint_n_tokens=3,
                    checkpoint_pos_max=2)
        slots = self.directory / "big-slots"
        slots.mkdir()
        adapter = StubAdapter(props(**TARGET_ONLY), slots, save=save, body=body)
        out = self.directory / "big.kvx"
        adapter.export(ExportRequest(model="", out_path=out,
                                     representation=Representation.OPAQUE))
        return out, save

    def test_the_payload_is_read_once(self):
        real_open = open
        reads = {"bytes": 0}

        class Counting:
            def __init__(self, handle):
                self._handle = handle

            def read(self, count=-1):
                block = self._handle.read(count)
                reads["bytes"] += len(block)
                return block

            def __getattr__(self, name):
                return getattr(self._handle, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._handle.__exit__(*exc)

        def counting_open(path, *args, **kwargs):
            handle = real_open(path, *args, **kwargs)
            return Counting(handle) if str(path).endswith(".kvx") else handle

        artifact, save = self.big_artifact()
        self.save = save
        nbytes = int(container.read_header(artifact)["blob"]["nbytes"])
        container.open = counting_open
        try:
            _, report = self.run_import(artifact)
        finally:
            del container.open
        self.assertTrue(report.ok, report.reason)
        self.assertLess(reads["bytes"], nbytes * 1.6,
                        f"read {reads['bytes']} bytes for a {nbytes}-byte payload: the "
                        f"second full pass is still happening")

    # -- segmented artifacts keep the full check ------------------------------------

    def test_segmented_artifacts_are_not_given_the_header_only_path(self):
        # Extraction does not check per-segment digests, so a segmented artifact must not
        # take the header-only path.
        self.assertTrue(container.verify_header(self.artifact)[0])
        from kv_rosetta.segments import Segment

        data = bytes(range(256)) * 8
        seg = Segment("k", "k", 0, 1, "f16", (1, 8, 2, 64), "layer,token,head,dim",
                      0, len(data))
        manifest = dict(self.header)
        manifest.pop("blob", None)
        target = container.write_segments(self.directory / "segmented.kvx", manifest,
                                          [(seg, data)])
        self.assertTrue(container.verify(target)[0], container.verify(target)[1])
        ok, reason = container.verify_header(target)
        self.assertFalse(ok, "a segmented artifact must not take the header-only path")
        self.assertIn("use verify()", reason)


if __name__ == "__main__":
    unittest.main()
