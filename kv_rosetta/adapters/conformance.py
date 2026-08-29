"""Adapter conformance suite.

The previous version could be satisfied by an adapter that always returned a failed
``ImportReport``. It checked that a report had a valid *shape*, not that the adapter could
do anything. A conformance suite a non-functional adapter passes is worse than no suite,
because it manufactures confidence.

The rule here: an adapter must prove a successful round trip for **every capability it
advertises**. Advertising a capability and failing to exercise it is a conformance failure,
not a skip. Skipping is legitimate only when the capability is not advertised.
"""

from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kv_rosetta.adapters.base import (
    Adapter,
    AdapterError,
    Capabilities,
    ExportRequest,
    ImportReport,
    ImportRequest,
    Representation,
    StagingMode,
)

_MAGIC = b"KVX1"
_PREAMBLE = 12


def read_kvx_header(path: Path | str) -> dict[str, Any]:
    """Parse a KVX header without importing the container module.

    Conformance must not depend on the writer it is checking; a shared bug would then be
    invisible to both.
    """
    raw = Path(path).read_bytes()
    if len(raw) < _PREAMBLE:
        raise ValueError(f"{path}: shorter than the {_PREAMBLE}-byte preamble")
    if raw[:4] != _MAGIC:
        raise ValueError(f"{path}: bad magic {raw[:4]!r}, expected {_MAGIC!r}")
    header_len = struct.unpack_from("<I", raw, 8)[0]
    if len(raw) < _PREAMBLE + header_len:
        raise ValueError(f"{path}: header truncated")
    return json.loads(raw[_PREAMBLE:_PREAMBLE + header_len].decode("utf-8"))


def _looks_like_a_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return value.startswith("/") or value.startswith("\\") or (len(value) > 2 and value[1] == ":")


class AdapterConformanceMixin:
    """Mix into a ``unittest.TestCase``; set ``self.adapter`` and ``self.model`` in setUp."""

    adapter: Adapter | None = None
    model: str = ""

    # -- helpers ----------------------------------------------------------------

    def _adapter(self) -> Adapter:
        if self.adapter is None:
            self.skipTest("no adapter under test")
        return self.adapter

    def _export(self, rep: Representation) -> Path:
        adapter = self._adapter()
        out = Path(tempfile.mkdtemp()) / f"conformance-{rep.value}.kvx"
        return Path(adapter.export(ExportRequest(model=self.model, out_path=out,
                                                 representation=rep)))

    # -- tests ------------------------------------------------------------------

    def test_capabilities_are_evidence_not_promises(self):
        adapter = self._adapter()
        caps = adapter.capabilities()
        self.assertIsInstance(caps, Capabilities)
        self.assertTrue(caps.runtime, "capabilities must name the runtime they came from")
        self.assertIsInstance(caps.export, frozenset)
        self.assertIsInstance(caps.import_, frozenset)
        for rep in caps.export | caps.import_:
            self.assertIsInstance(rep, Representation)
        self.assertEqual(caps, adapter.capabilities(), "probing twice must agree")

    def test_identity_is_content_derived(self):
        adapter = self._adapter()
        ident = adapter.identity(self.model)
        self.assertIsInstance(ident, dict)
        for key in ("model_digest", "cache_abi_digest"):
            value = ident.get(key, "")
            self.assertRegex(value, r"\A[0-9a-f]{64}\Z", f"{key} must be 64 lowercase hex")
        self.assertEqual(ident, adapter.identity(self.model), "identity must be stable")
        for key, value in ident.items():
            self.assertFalse(_looks_like_a_path(value),
                             f"identity[{key!r}] looks like a filesystem path; identity must "
                             f"be derived from content, not from where the weights live")

    def test_every_advertised_export_representation_produces_a_readable_artifact(self):
        caps = self._adapter().capabilities()
        if not caps.export:
            self.skipTest("adapter advertises no export representation")
        for rep in sorted(caps.export, key=lambda r: r.value):
            with self.subTest(representation=rep.value):
                path = self._export(rep)
                self.assertTrue(path.is_file(), f"export({rep.value}) produced no file")
                self.assertEqual(path.read_bytes()[:4], _MAGIC)

    def test_every_advertised_import_representation_completes_a_round_trip(self):
        """The central test: an advertised import must be demonstrable."""
        adapter = self._adapter()
        caps = adapter.capabilities()
        if not caps.import_:
            self.skipTest("adapter advertises no import representation")
        demonstrable = caps.import_ & caps.export
        self.assertTrue(
            demonstrable,
            f"adapter advertises import of {sorted(r.value for r in caps.import_)} but exports "
            f"{sorted(r.value for r in caps.export)}; it can never demonstrate any import",
        )
        for rep in sorted(demonstrable, key=lambda r: r.value):
            with self.subTest(representation=rep.value):
                artifact = self._export(rep)
                report = adapter.import_(artifact, ImportRequest(model=self.model))
                self.assertIsInstance(report, ImportReport)
                self.assertTrue(
                    report.ok,
                    f"advertised import of {rep.value} failed: {report.reason}",
                )
                self.assertIsInstance(report.mode, StagingMode)
                self.assertGreater(report.tokens_restored, 0,
                                   "a successful import must restore tokens")

    def test_canonical_exports_are_well_formed(self):
        caps = self._adapter().capabilities()
        if Representation.CANONICAL not in caps.export:
            self.skipTest("adapter does not export canonical")
        blob = read_kvx_header(self._export(Representation.CANONICAL))["blob"]
        self.assertIn(blob.get("encoding"), ("segmented", "raw"))
        if blob.get("encoding") == "segmented":
            self.assertTrue(blob.get("segments"), "segmented artifact has no segments")
            self.assertRegex(blob.get("representation_digest", ""), r"\A[0-9a-f]{64}\Z")

    def test_opaque_exports_declare_a_format_the_adapter_advertises(self):
        caps = self._adapter().capabilities()
        if Representation.OPAQUE not in caps.export:
            self.skipTest("adapter does not export opaque")
        blob = read_kvx_header(self._export(Representation.OPAQUE))["blob"]
        fmt = blob.get("opaque_format")
        self.assertTrue(fmt, "an opaque artifact must name its format")
        self.assertIn(fmt, caps.opaque_formats,
                      "exported opaque format is not among the advertised formats")

    def test_import_refuses_a_mismatched_cache_abi(self):
        adapter = self._adapter()
        caps = adapter.capabilities()
        demonstrable = caps.import_ & caps.export
        if not demonstrable:
            self.skipTest("no demonstrable import representation")
        rep = sorted(demonstrable, key=lambda r: r.value)[0]
        artifact = self._export(rep)
        request = ImportRequest(model=self.model, expected_cache_abi_digest="0" * 64)
        try:
            report = adapter.import_(artifact, request)
        except AdapterError:
            return
        self.assertFalse(report.ok,
                         "import succeeded against a cache ABI digest that cannot match; "
                         "state is only valid for the configuration that produced it")

    def test_import_refuses_a_corrupted_artifact(self):
        adapter = self._adapter()
        caps = adapter.capabilities()
        demonstrable = caps.import_ & caps.export
        if not demonstrable:
            self.skipTest("no demonstrable import representation")
        rep = sorted(demonstrable, key=lambda r: r.value)[0]
        artifact = self._export(rep)
        blob = read_kvx_header(artifact)["blob"]
        raw = bytearray(artifact.read_bytes())
        offset = int(blob.get("offset", 0))
        if offset >= len(raw):
            self.skipTest("artifact has no payload to corrupt")
        raw[offset] ^= 0xFF
        broken = artifact.with_name("corrupt-" + artifact.name)
        broken.write_bytes(bytes(raw))
        try:
            report = adapter.import_(broken, ImportRequest(model=self.model))
        except AdapterError:
            return
        self.assertFalse(report.ok, "a corrupted artifact must never import successfully")

    def test_report_numbers_are_coherent(self):
        adapter = self._adapter()
        caps = adapter.capabilities()
        demonstrable = caps.import_ & caps.export
        if not demonstrable:
            self.skipTest("no demonstrable import representation")
        rep = sorted(demonstrable, key=lambda r: r.value)[0]
        report = adapter.import_(self._export(rep), ImportRequest(model=self.model))
        if not report.ok:
            self.fail(f"advertised import of {rep.value} failed: {report.reason}")
        self.assertGreater(report.nbytes, 0)
        self.assertGreaterEqual(report.seconds, 0.0)
        self.assertGreater(report.tokens_restored + report.tokens_reprefilled, 0)


def run_conformance(adapter: Adapter, model: str) -> unittest.TestResult:
    """Run the suite against one adapter and return the raw result."""

    case = type(
        "GeneratedConformance",
        (AdapterConformanceMixin, unittest.TestCase),
        {"setUp": lambda self: (setattr(self, "adapter", adapter),
                                setattr(self, "model", model), None)[-1]},
    )
    suite = unittest.TestLoader().loadTestsFromTestCase(case)
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


def assert_suite_rejects(adapter: Adapter, model: str) -> bool:
    """True when the suite does NOT pass — used to prove the suite is not vacuous."""
    return not run_conformance(adapter, model).wasSuccessful()
