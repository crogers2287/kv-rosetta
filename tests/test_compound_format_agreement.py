"""A compound artifact must be recognised from the authoritative blob format.

import_() checked compatibility against header["blob"]["opaque_format"] but selected
compound behaviour from whether header["coverage"]["format"] contained a "+". Nothing
required the two to agree, and container.verify() checks integrity, not consistency. So a
correctly hashed artifact whose blob says ggsq/3+sckp/1 - matching the live runtime, so it
passes the compatibility check - but whose coverage format is missing or plain reached the
runtime with is_compound=False. That skipped the nonzero-coverage check, the restore
metadata comparison, and forced reuse verification, and let verify_reuse=False return
success on n_restored alone.
"""

import json
import tempfile
import unittest
from pathlib import Path

from kv_rosetta import container
from kv_rosetta.adapters.base import ExportRequest, ImportRequest, Representation
from tests.test_hybrid_export_gate import StubAdapter, ggsq_body, props, save_with_appendix
from tests.test_hybrid_support_predicate import RestoreStub, TARGET_ONLY

COMPOUND = "ggsq/3+sckp/1"


def genuine_artifact(directory: Path) -> tuple[Path, dict]:
    save, body = save_with_appendix(checkpoint_n_tokens=3, checkpoint_pos_max=2)
    slots = directory / "export-slots"
    slots.mkdir()
    adapter = StubAdapter(props(**TARGET_ONLY), slots, save=save, body=body)
    out = directory / "artifact.kvx"
    adapter.export(ExportRequest(model="", out_path=out,
                                 representation=Representation.OPAQUE))
    return out, save


def rewrite_header(source: Path, target: Path, mutate) -> Path:
    """Rebuild an artifact with a mutated header, correctly re-hashed.

    The point is an artifact that passes container.verify() - a corrupt one would be caught
    by integrity alone and would prove nothing about format agreement.
    """
    header = container.read_header(source)
    payload = target.parent / "payload.bin"
    container.extract_payload(source, payload)
    mutate(header)
    return container.write_opaque_from_file(
        target, header, payload, header["blob"]["opaque_format"])


class CompoundFormatAgreementTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.genuine, self.save = genuine_artifact(self.directory)
        self.assertEqual(container.read_header(self.genuine)["blob"]["opaque_format"],
                         COMPOUND)

    def importer(self, **kwargs):
        slots = self.directory / f"slots{len(list(self.directory.iterdir()))}"
        slots.mkdir()
        restore = kwargs.pop("restore", {
            "n_restored": 263,
            "n_checkpoints_restored": self.save["n_checkpoints_saved"],
            "checkpoint_bytes": self.save["checkpoint_bytes"],
            "checkpoint_n_tokens": self.save["checkpoint_n_tokens"],
            "checkpoint_pos_min": self.save["checkpoint_pos_min"],
            "checkpoint_pos_max": self.save["checkpoint_pos_max"]})
        return RestoreStub(props(**TARGET_ONLY), slots, save=dict(self.save),
                           body=ggsq_body(trailer=b""), restore=restore, **kwargs)

    def variant(self, name, mutate) -> Path:
        target = self.directory / f"{name}.kvx"
        target.parent.mkdir(parents=True, exist_ok=True)
        return rewrite_header(self.genuine, target, mutate)

    def assert_refused_before_restore(self, artifact: Path, *, verify_reuse=True):
        adapter = self.importer()
        report = adapter.import_(artifact, ImportRequest(model="", slot=0),
                                 verify_reuse=verify_reuse)
        self.assertFalse(report.ok, f"accepted with verify_reuse={verify_reuse}")
        self.assertEqual(adapter.restore_posts(), [],
                         "refused only after staging state into the runtime")
        return report

    def test_the_genuine_artifact_still_imports(self):
        adapter = self.importer()
        report = adapter.import_(self.genuine, ImportRequest(model="", slot=0))
        self.assertTrue(report.ok, report.reason)

    def test_compound_blob_with_missing_coverage_format_is_refused(self):
        def mutate(header):
            header["coverage"].pop("format", None)
        artifact = self.variant("missing-coverage-format", mutate)
        self.assertTrue(container.verify(artifact)[0], "fixture must pass integrity")
        for verify in (True, False):
            with self.subTest(verify_reuse=verify):
                self.assert_refused_before_restore(artifact, verify_reuse=verify)

    def test_compound_blob_with_plain_coverage_format_is_refused(self):
        artifact = self.variant("plain-coverage",
                                lambda h: h["coverage"].__setitem__("format", "ggsq/3"))
        for verify in (True, False):
            with self.subTest(verify_reuse=verify):
                self.assert_refused_before_restore(artifact, verify_reuse=verify)

    def test_plain_blob_with_compound_coverage_is_refused(self):
        def mutate(header):
            header["blob"]["opaque_format"] = "ggsq/3"
        artifact = self.variant("plain-blob", mutate)
        for verify in (True, False):
            with self.subTest(verify_reuse=verify):
                self.assert_refused_before_restore(artifact, verify_reuse=verify)

    def test_missing_coverage_object_is_refused(self):
        artifact = self.variant("no-coverage", lambda h: h.pop("coverage", None))
        self.assert_refused_before_restore(artifact)

    def test_malformed_numeric_coverage_is_refused_cleanly(self):
        for value in ("many", None, [3], {"n": 3}, float("nan")):
            with self.subTest(value=value):
                artifact = self.variant(
                    f"bad-{abs(hash(str(value)))}",
                    lambda h, v=value: h["coverage"].__setitem__("checkpoint_n_tokens", v))
                # A malformed value must produce a report, never an exception escaping the
                # adapter boundary.
                report = self.assert_refused_before_restore(artifact)
                self.assertIsInstance(report.reason, str)

    def test_verify_reuse_false_never_helps_a_compound_artifact(self):
        adapter = self.importer(probe={"timings": {"cache_n": 0, "prompt_n": 3}})
        report = adapter.import_(self.genuine, ImportRequest(model="", slot=0),
                                 verify_reuse=False)
        self.assertFalse(report.ok)


if __name__ == "__main__":
    unittest.main()


class PhaseTimingTest(CompoundFormatAgreementTest):
    """A total that loses to a cold prefill must be attributable to a phase."""

    def test_a_successful_import_reports_every_phase(self):
        adapter = self.importer()
        report = adapter.import_(self.genuine, ImportRequest(model="", slot=0))
        self.assertTrue(report.ok, report.reason)
        for phase in ("container_verify", "staging", "runtime_restore",
                      "reuse_probe", "pristine_restore"):
            self.assertIn(phase, report.phases)
            self.assertGreaterEqual(report.phases[phase], 0.0)
        self.assertLessEqual(sum(report.phases.values()), report.seconds + 1e-6,
                             "phases sum to more than the reported total")

    def test_a_refusal_before_staging_reports_no_staging_phase(self):
        artifact = self.variant("plain-coverage-phases",
                                lambda h: h["coverage"].__setitem__("format", "ggsq/3"))
        adapter = self.importer()
        report = adapter.import_(artifact, ImportRequest(model="", slot=0))
        self.assertFalse(report.ok)
        self.assertNotIn("staging", report.phases)
        self.assertNotIn("runtime_restore", report.phases)
