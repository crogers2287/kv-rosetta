from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import (
    IMPORT_MODES,
    Adapter,
    AdapterError,
    Capabilities,
    ExportRequest,
    ImportRequest,
    ImportReport,
    Tier,
)


def parse_kvx_header(path: Path) -> dict[str, Any]:
    """Parse the KVX1 header of an artifact.

    Layout: magic b"KVX1" at bytes 0..4, uint32 little-endian version at 4..8,
    uint32 little-endian header_len at 8..12, then header_len bytes of UTF-8
    JSON at offset 12.
    """
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise AdapterError(f"KVX artifact {path} is too short ({len(data)} bytes)")
    if data[0:4] != b"KVX1":
        raise AdapterError(f"KVX artifact {path} has bad magic {data[0:4]!r}")
    int.from_bytes(data[4:8], "little")  # version, retained for completeness
    header_len = int.from_bytes(data[8:12], "little")
    header_bytes = data[12:12 + header_len]
    if len(header_bytes) != header_len:
        raise AdapterError(f"KVX header truncated in {path}")
    try:
        return json.loads(header_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise AdapterError(f"KVX header in {path} is not valid UTF-8 JSON: {exc}") from exc


def write_kvx_artifact(path: Path, header: dict[str, Any], body: bytes = b"") -> Path:
    """Write a KVX1 artifact carrying the given header dict and optional body."""
    header_bytes = json.dumps(header).encode("utf-8")
    out = bytearray()
    out += b"KVX1"
    out += (1).to_bytes(4, "little")
    out += len(header_bytes).to_bytes(4, "little")
    out += header_bytes
    out += body
    Path(path).write_bytes(bytes(out))
    return Path(path)


class AdapterConformanceMixin:
    """Conformance checks for Adapter implementations.

    Subclass together with a unittest.TestCase and set self.adapter in setUp.
    Every test method begins with self._skip_unless_adapter() so the mixin is
    safe to subclass before an adapter exists.
    """

    adapter: Adapter | None = None

    def _skip_unless_adapter(self) -> None:
        if self.adapter is None:
            self.skipTest("no adapter configured (self.adapter is None)")

    def _read_header(self, artifact: Path) -> dict[str, Any]:
        return parse_kvx_header(artifact)

    def _write_kvx_artifact(self, path: Path, header: dict[str, Any], body: bytes = b"") -> Path:
        return write_kvx_artifact(path, header, body)

    def _unlink(self, path: Path) -> None:
        try:
            Path(path).unlink()
        except OSError:
            pass

    def test_capabilities_is_frozen_and_typed(self) -> None:
        self._skip_unless_adapter()
        caps = self.adapter.capabilities()
        self.assertIsInstance(caps, Capabilities)
        self.assertIsInstance(caps.tier, Tier)
        self.assertIsInstance(caps.dtypes, frozenset)
        self.assertEqual(caps, self.adapter.capabilities())
        with self.assertRaises(AttributeError):
            caps.tier = Tier.PROMPT

    def test_identity_shape(self) -> None:
        self._skip_unless_adapter()
        ident = self.adapter.identity("test-model")
        self.assertIsInstance(ident, dict)
        sha = ident["l0_sha256"]
        self.assertEqual(len(sha), 64)
        self.assertEqual(sha, sha.lower())
        int(sha, 16)

    def test_export_produces_readable_artifact(self) -> None:
        self._skip_unless_adapter()
        if not self.adapter.capabilities().can_export:
            self.skipTest("adapter cannot export")
        out_path = Path("kv_rosetta_conformance_export.kvx")
        artifact = self.adapter.export(ExportRequest(model="test-model", out_path=out_path))
        self.assertTrue(artifact.exists())
        self.assertEqual(artifact.read_bytes()[:4], b"KVX1")
        self._unlink(artifact)

    def test_canonical_declares_raw_encoding(self) -> None:
        self._skip_unless_adapter()
        if self.adapter.capabilities().tier < Tier.CANONICAL:
            self.skipTest("adapter tier below CANONICAL")
        out_path = Path("kv_rosetta_conformance_canonical.kvx")
        artifact = self.adapter.export(ExportRequest(model="test-model", out_path=out_path))
        try:
            header = self._read_header(artifact)
            blob = header["blob"]
            self.assertEqual(blob["encoding"], "raw")
            shape = blob["shape"]
            self.assertIsInstance(shape, list)
            self.assertEqual(len(shape), 5)
            self.assertEqual(shape[1], 2)
        finally:
            self._unlink(artifact)

    def test_import_report_contract(self) -> None:
        self._skip_unless_adapter()
        if not self.adapter.capabilities().can_import:
            self.skipTest("adapter cannot import")
        out_path = Path("kv_rosetta_conformance_import.kvx")
        caps = self.adapter.capabilities()
        try:
            if caps.can_export:
                artifact = self.adapter.export(ExportRequest(model="test-model", out_path=out_path))
            else:
                artifact = self._write_kvx_artifact(
                    out_path,
                    {
                        "blob": {"encoding": "raw", "shape": [1, 2, 3, 4, 5]},
                        "identity": {"l0_sha256": "0" * 64},
                    },
                )
            report = self.adapter.import_(artifact, ImportRequest(model="test-model"))
        finally:
            self._unlink(artifact)
        self.assertIsInstance(report, ImportReport)
        self.assertIn(report.mode, IMPORT_MODES)

    def test_identity_mismatch_is_refused(self) -> None:
        self._skip_unless_adapter()
        if not self.adapter.capabilities().can_import:
            self.skipTest("adapter cannot import")
        bad_path = Path("kv_rosetta_conformance_bad_identity.kvx")
        header = {
            "blob": {"encoding": "raw", "shape": [1, 2, 3, 4, 5]},
            "identity": {"l0_sha256": "0" * 64},
        }
        self._write_kvx_artifact(bad_path, header)
        try:
            try:
                report = self.adapter.import_(bad_path, ImportRequest(model="test-model"))
                ok = bool(getattr(report, "ok", True))
                self.assertFalse(
                    ok,
                    "import of mismatched identity must not silently succeed",
                )
            except AdapterError:
                pass
        finally:
            self._unlink(bad_path)
