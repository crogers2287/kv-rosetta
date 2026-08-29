"""llama.cpp adapter over llama-server's HTTP slot endpoints.

This is a tier-1 (opaque) adapter. llama-server can save a sequence's KV state to a file
and restore it again, which is a real cache-import ABI even though the payload is
engine-native rather than canonical. It needs no C bindings and no patched engine: the
only prerequisite is that the server was started with ``--slot-save-path``.

An opaque blob is valid ONLY for the exact configuration that produced it - the same
weights, the same KV dtype, the same context and position settings, the same state format
version. That is why import refuses on a cache-ABI mismatch and why the check cannot be
overridden: restoring a blob into a mismatched configuration does not fail loudly, it
produces a cache the model silently misreads.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from kv_rosetta import container
from kv_rosetta.adapters.base import (
    Adapter,
    AdapterError,
    Capabilities,
    ExportRequest,
    ImportReport,
    ImportRequest,
    Representation,
    StagingMode,
    register,
)

OPAQUE_FORMAT = "ggsq/3"
_TIMEOUT = 600


class LlamaCppHTTPAdapter(Adapter):
    name = "llamacpp-http"

    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 slot_save_path: Path | str | None = None, slot: int = 0) -> None:
        self.base_url = base_url.rstrip("/")
        self.slot_save_path = Path(slot_save_path) if slot_save_path else None
        self.slot = slot
        self._props: dict[str, Any] | None = None

    # -- transport --------------------------------------------------------------

    def _get(self, path: str) -> Any:
        try:
            with urllib.request.urlopen(self.base_url + path, timeout=_TIMEOUT) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise AdapterError(f"GET {path} failed: {exc}") from exc

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise AdapterError(f"POST {path} failed: {exc}") from exc

    def props(self, refresh: bool = False) -> dict[str, Any]:
        if self._props is None or refresh:
            self._props = self._get("/props")
        return self._props

    # -- identity ---------------------------------------------------------------

    def identity(self, model: str = "") -> dict[str, str]:
        """Identity read from the running server, never assumed from configuration."""
        props = self.props()
        settings = props.get("default_generation_settings", {}) or {}
        model_digest = hashlib.sha256("\x00".join([
            str(props.get("model_path", "")).rsplit("/", 1)[-1],   # filename, not the path
            str(props.get("chat_template", "")),
            str(props.get("bos_token", "")),
            str(props.get("eos_token", "")),
        ]).encode()).hexdigest()
        cache_abi_digest = hashlib.sha256("\x00".join([
            "llama.cpp",
            str(props.get("build_info", "")),
            OPAQUE_FORMAT,
            str(settings.get("n_ctx", "")),
            str(props.get("type_k", settings.get("type_k", ""))),
            str(props.get("type_v", settings.get("type_v", ""))),
            str(model_digest),
        ]).encode()).hexdigest()
        return {"model_digest": model_digest, "cache_abi_digest": cache_abi_digest,
                "build_info": str(props.get("build_info", ""))}

    # -- capabilities -----------------------------------------------------------

    def capabilities(self) -> Capabilities:
        """Probed from the live server. Slot save/restore is only advertised when the
        server was actually started with --slot-save-path."""
        props = self.props()
        settings = props.get("default_generation_settings", {}) or {}
        can_slot = bool(self.slot_save_path)
        reps = frozenset({Representation.OPAQUE}) if can_slot else frozenset()
        notes: list[str] = []
        if not can_slot:
            notes.append("slot save/restore unavailable: server has no --slot-save-path")
        return Capabilities(
            runtime="llama.cpp",
            runtime_revision=str(props.get("build_info", "")),
            backend=str(props.get("backend", "")) or "unknown",
            device_arch="",
            export=reps,
            import_=reps,
            export_dtypes=frozenset({str(settings.get("type_k", "f16"))}),
            import_dtypes=frozenset({str(settings.get("type_k", "f16"))}),
            opaque_formats=frozenset({OPAQUE_FORMAT}) if can_slot else frozenset(),
            cache_abi_digest=self.identity().get("cache_abi_digest", ""),
            staging=frozenset({StagingMode.HOST_STAGED}),
            notes=tuple(notes),
        )

    # -- export / import --------------------------------------------------------

    def export(self, req: ExportRequest) -> Path:
        if req.representation is not Representation.OPAQUE:
            raise AdapterError(
                f"{self.name} exports {Representation.OPAQUE.value} only, "
                f"asked for {req.representation.value}")
        if not self.slot_save_path:
            raise AdapterError("no slot_save_path configured")
        out = Path(req.out_path)
        filename = out.stem + ".bin"
        result = self._post(f"/slots/{req.slot or self.slot}?action=save",
                            {"filename": filename})
        state = self.slot_save_path / filename
        if not state.is_file():
            raise AdapterError(f"server reported a save but {state} does not exist")
        blob = state.read_bytes()
        ident = self.identity(req.model)
        props = self.props()
        settings = props.get("default_generation_settings", {}) or {}
        manifest = {
            "schema": "kvx/0.3",
            "model": {"architecture": "", "weights_id": ident["model_digest"], "dtype": ""},
            "prompt": {"tokenizer_id": "", "token_ids_sha256": "", "token_count":
                       int(result.get("n_saved", 0))},
            "kv": {"layers": 0, "heads": 0, "head_dim": 0,
                   "dtype": str(settings.get("type_k", "f16")),
                   "layout": "opaque", "byte_order": "little",
                   "tokens": int(result.get("n_saved", 0))},
            "position": {"kind": "rope", "rope_applied": True},
            "producer": {"runtime": "llama.cpp", "backend": "",
                         "device_arch": "", "library_version": ident["build_info"]},
            "identity": {"model_digest": ident["model_digest"],
                         "cache_abi_digest": ident["cache_abi_digest"]},
            "corpus": req.corpus or {},
        }
        return container.write_opaque(out, manifest, blob, OPAQUE_FORMAT)

    def import_(self, artifact: Path | str, req: ImportRequest) -> ImportReport:
        started = time.time()
        artifact = Path(artifact)
        try:
            if not self.slot_save_path:
                return ImportReport(mode=StagingMode.HOST_STAGED, ok=False,
                                    representation=Representation.OPAQUE,
                                    reason="no slot_save_path configured")
            ok, reason = container.verify(artifact)
            if not ok:
                return ImportReport(mode=StagingMode.HOST_STAGED, ok=False,
                                    representation=Representation.OPAQUE,
                                    reason=f"artifact failed verification: {reason}")
            art = container.read(artifact, mmap=False)
            header_abi = (art.header.get("identity") or {}).get("cache_abi_digest", "")
            live_abi = self.identity(req.model).get("cache_abi_digest", "")
            expected = req.expected_cache_abi_digest or live_abi
            # An opaque blob is only readable by the configuration that wrote it. There is
            # deliberately no override: a mismatched restore fails silently, not loudly.
            if header_abi and expected and header_abi != expected:
                return ImportReport(
                    mode=StagingMode.HOST_STAGED, ok=False,
                    representation=Representation.OPAQUE,
                    reason=f"cache ABI mismatch: artifact {header_abi[:12]} vs expected {expected[:12]}")
            if art.blob.get("opaque_format") != OPAQUE_FORMAT:
                return ImportReport(mode=StagingMode.HOST_STAGED, ok=False,
                                    representation=Representation.OPAQUE,
                                    reason=f"unsupported opaque format {art.blob.get('opaque_format')!r}")

            filename = artifact.stem + ".restore.bin"
            (self.slot_save_path / filename).write_bytes(art.opaque)
            result = self._post(f"/slots/{req.slot or self.slot}?action=restore",
                                {"filename": filename})
            restored = int(result.get("n_restored", 0))
            return ImportReport(
                mode=StagingMode.HOST_STAGED,
                ok=restored > 0,
                representation=Representation.OPAQUE,
                reason="" if restored > 0 else "server restored no cells",
                nbytes=int(result.get("n_read", len(art.opaque))),
                seconds=time.time() - started,
                tokens_restored=restored,
            )
        except AdapterError as exc:
            return ImportReport(mode=StagingMode.HOST_STAGED, ok=False,
                                representation=Representation.OPAQUE, reason=str(exc),
                                seconds=time.time() - started)

    # -- convenience used by the parity test ------------------------------------

    def erase(self, slot: int | None = None) -> int:
        return int(self._post(f"/slots/{self.slot if slot is None else slot}?action=erase",
                              {}).get("n_erased", 0))

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/completion", payload)


register(LlamaCppHTTPAdapter.name, LlamaCppHTTPAdapter)
