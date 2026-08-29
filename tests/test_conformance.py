"""The conformance suite must reject adapters that do not work.

A suite that a non-functional adapter passes is worse than no suite. Each adapter below
is broken in exactly one way, and each must be rejected for that reason.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kv_rosetta import container
from kv_rosetta.adapters.base import (
    Adapter, Capabilities, ExportRequest, ImportReport, ImportRequest,
    Representation, StagingMode,
)
from kv_rosetta.adapters.conformance import (
    AdapterConformanceMixin, assert_suite_rejects, read_kvx_header, run_conformance,
)
from kv_rosetta.segments import Segment

_ABI = "a" * 64
_MODEL_DIGEST = "b" * 64


def _write_artifact(path: Path, tokens: int = 4) -> Path:
    manifest = {"schema": "kvx/0.3", "kv": {
        "layers": 1, "heads": 1, "head_dim": 4, "dtype": "f16",
        "layout": "per-segment", "byte_order": "little", "tokens": tokens}}
    data = np.zeros((1, tokens, 1, 4), dtype=np.float16)
    seg = Segment("k", "k", 0, 1, "f16", data.shape, "layer,token,head,dim", 0, data.nbytes)
    return container.write_segments(path, manifest, [(seg, data.tobytes())])


class WorkingAdapter(Adapter):
    """A minimal but honest adapter: it does what it advertises."""

    name = "working"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            runtime="fake", runtime_revision="1", backend="cpu", device_arch="",
            export=frozenset({Representation.CANONICAL}),
            import_=frozenset({Representation.CANONICAL}),
            export_dtypes=frozenset({"f16"}), import_dtypes=frozenset({"f16"}),
            cache_abi_digest=_ABI, staging=frozenset({StagingMode.HOST_STAGED}),
        )

    def identity(self, model: str) -> dict:
        return {"model_digest": _MODEL_DIGEST, "cache_abi_digest": _ABI}

    def export(self, req: ExportRequest) -> Path:
        return _write_artifact(Path(req.out_path))

    def import_(self, artifact: Path, req: ImportRequest) -> ImportReport:
        ok, reason = container.verify(artifact)
        if not ok:
            return ImportReport(mode=StagingMode.HOST_STAGED, ok=False, reason=reason)
        if req.expected_cache_abi_digest and req.expected_cache_abi_digest != _ABI:
            return ImportReport(mode=StagingMode.HOST_STAGED, ok=False,
                                reason="cache ABI mismatch")
        tokens = int(read_kvx_header(artifact)["kv"]["tokens"])
        return ImportReport(mode=StagingMode.HOST_STAGED, ok=True, nbytes=artifact.stat().st_size,
                            seconds=0.0, tokens_restored=tokens)


class AlwaysFailsAdapter(WorkingAdapter):
    """Advertises import but every import fails. The original suite passed this."""

    name = "always-fails"

    def import_(self, artifact: Path, req: ImportRequest) -> ImportReport:
        return ImportReport(mode=StagingMode.HOST_STAGED, ok=False, reason="nope")


class UndemonstrableImportAdapter(WorkingAdapter):
    """Advertises importing a representation it cannot produce, so it can never be shown."""

    name = "undemonstrable"

    def capabilities(self) -> Capabilities:
        base = super().capabilities()
        return Capabilities(**{**base.as_dict(),
                               "export": frozenset({Representation.CANONICAL}),
                               "import_": frozenset({Representation.OPAQUE}),
                               "export_dtypes": base.export_dtypes,
                               "import_dtypes": base.import_dtypes,
                               "opaque_formats": frozenset(),
                               "staging": base.staging})


class PathIdentityAdapter(WorkingAdapter):
    """Derives identity from where the weights live rather than from their content."""

    name = "path-identity"

    def identity(self, model: str) -> dict:
        return {"model_digest": _MODEL_DIGEST, "cache_abi_digest": _ABI,
                "weights": "/mnt/models/qwen/model.safetensors"}


class AbiBlindAdapter(WorkingAdapter):
    """Imports regardless of the cache ABI it was asked to match."""

    name = "abi-blind"

    def import_(self, artifact: Path, req: ImportRequest) -> ImportReport:
        tokens = int(read_kvx_header(artifact)["kv"]["tokens"])
        return ImportReport(mode=StagingMode.HOST_STAGED, ok=True,
                            nbytes=artifact.stat().st_size, tokens_restored=tokens)


class CorruptionBlindAdapter(WorkingAdapter):
    """Accepts a corrupted artifact."""

    name = "corruption-blind"

    def import_(self, artifact: Path, req: ImportRequest) -> ImportReport:
        if req.expected_cache_abi_digest and req.expected_cache_abi_digest != _ABI:
            return ImportReport(mode=StagingMode.HOST_STAGED, ok=False, reason="abi")
        return ImportReport(mode=StagingMode.HOST_STAGED, ok=True, nbytes=32, tokens_restored=4)


class ConformanceIsNotVacuousTests(unittest.TestCase):
    def test_a_working_adapter_passes(self):
        result = run_conformance(WorkingAdapter(), "fake-model")
        self.assertTrue(result.wasSuccessful(),
                        f"failures={[t[0] for t in result.failures]} "
                        f"errors={[t[0] for t in result.errors]}")
        self.assertGreater(result.testsRun, 5)

    def test_an_always_failing_adapter_is_rejected(self):
        self.assertTrue(assert_suite_rejects(AlwaysFailsAdapter(), "fake-model"))

    def test_an_undemonstrable_import_claim_is_rejected(self):
        self.assertTrue(assert_suite_rejects(UndemonstrableImportAdapter(), "fake-model"))

    def test_path_derived_identity_is_rejected(self):
        self.assertTrue(assert_suite_rejects(PathIdentityAdapter(), "fake-model"))

    def test_ignoring_the_expected_cache_abi_is_rejected(self):
        self.assertTrue(assert_suite_rejects(AbiBlindAdapter(), "fake-model"))

    def test_accepting_a_corrupted_artifact_is_rejected(self):
        self.assertTrue(assert_suite_rejects(CorruptionBlindAdapter(), "fake-model"))


class WorkingAdapterConformance(AdapterConformanceMixin, unittest.TestCase):
    """The suite applied directly, so its own assertions are exercised in the main run."""

    def setUp(self):
        self.adapter = WorkingAdapter()
        self.model = "fake-model"


class HeaderReaderTests(unittest.TestCase):
    def test_reads_a_real_artifact(self):
        path = _write_artifact(Path(tempfile.mkdtemp()) / "a.kvx")
        self.assertEqual(read_kvx_header(path)["blob"]["encoding"], "segmented")

    def test_rejects_bad_magic_and_short_files(self):
        d = Path(tempfile.mkdtemp())
        (d / "bad.kvx").write_bytes(b"NOPE" + b"\x00" * 32)
        (d / "short.kvx").write_bytes(b"KV")
        for name in ("bad.kvx", "short.kvx"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                read_kvx_header(d / name)


if __name__ == "__main__":
    unittest.main()
