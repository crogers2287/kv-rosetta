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
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from kv_rosetta import container, weights
from kv_rosetta.identity import CacheABIIdentity, ModelIdentity, PromptIdentity
from kv_rosetta.adapters import ggsq_envelope
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

OPAQUE_FORMAT_FAMILY = "ggsq"
#: Never hardcode the sequence-state version. The installed binary here writes 2 while the
#: checked-out header declares 3, and state_seq_load_file requires an exact match - so a
#: hardcoded label would mark an artifact as loadable by builds that would refuse it.
_TIMEOUT = 600


class LlamaCppHTTPAdapter(Adapter):
    name = "llamacpp-http"

    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 slot_save_path: Path | str | None = None, slot: int = 0) -> None:
        self.base_url = base_url.rstrip("/")
        self.slot_save_path = Path(slot_save_path) if slot_save_path else None
        self.slot = slot
        self._props: dict[str, Any] | None = None
        self._state_version: int | None = None
        self._staged: list[Path] = []
        self._digest_cache = weights.DigestCache()

    def _slot(self, req_slot: int | None = None) -> int:
        """Resolve the slot every call must name explicitly.

        A multi-slot server can schedule a completion onto a different slot than the one
        restored, so a probe that does not name its slot proves nothing about the restore.
        """
        return self.slot if req_slot is None else int(req_slot)

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

    def state_version(self) -> int:
        """The sequence-state version this server actually emits.

        Probed once by saving the slot and reading the envelope, because no endpoint
        reports it. Capability is evidence from the live runtime, not a constant.
        """
        if self._state_version is not None:
            return self._state_version
        if not self.slot_save_path:
            raise AdapterError("cannot probe the state version without a slot_save_path")
        probe = "kvx-version-probe.bin"
        self._post(f"/slots/{self.slot}?action=save", {"filename": probe})
        path = self.slot_save_path / probe
        try:
            with open(path, "rb") as handle:
                head = handle.read(8)
            envelope = ggsq_envelope.parse_file_envelope(head + b"\x00" * 8)
        except ggsq_envelope.EnvelopeError as exc:
            if "truncated" not in str(exc):
                raise AdapterError(f"unrecognised slot state file: {exc}") from exc
            import struct
            if head[:4] != ggsq_envelope.GGSQ_MAGIC:
                raise AdapterError(f"slot file magic {head[:4]!r} is not a sequence state file")
            self._state_version = struct.unpack_from("<I", head, 4)[0]
            path.unlink(missing_ok=True)
            return self._state_version
        finally:
            path.unlink(missing_ok=True)
        self._state_version = envelope.version
        return self._state_version

    def opaque_format(self) -> str:
        return f"{OPAQUE_FORMAT_FAMILY}/{self.state_version()}"

    # -- identity ---------------------------------------------------------------

    def model_identity(self, model: str = "") -> ModelIdentity:
        """Content-derived model identity.

        The weights digest comes from the bytes, not the path: renaming or relocating a
        model must not change its identity, and different weights written behind the same
        name must. The digest is cached against strong file metadata so a large model is
        read once rather than on every request.

        When the weights file is not reachable from this process the digest is left empty
        rather than substituted with a path hash - an artifact that cannot prove which
        weights produced it should be unusable, not plausibly labelled.
        """
        props = self.props()
        weights_path = str(props.get("model_path", ""))
        weights_sha256 = ""
        if weights_path and Path(weights_path).is_file():
            weights_sha256 = weights.model_content_digest(weights_path, self._digest_cache)
        tokenizer_sha256 = hashlib.sha256("\x00".join([
            str(props.get("bos_token", "")),
            str(props.get("eos_token", "")),
            str(props.get("vocab_type", "")),
            str(props.get("n_vocab", "")),
        ]).encode()).hexdigest()
        return ModelIdentity(
            architecture=str(props.get("model_arch", props.get("architecture", ""))),
            weights_sha256=weights_sha256,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=hashlib.sha256(
                str(props.get("chat_template", "")).encode()).hexdigest(),
        )

    def cache_abi_identity(self, model: str = "") -> CacheABIIdentity:
        props = self.props()
        settings = props.get("default_generation_settings", {}) or {}
        return CacheABIIdentity(
            runtime="llama.cpp",
            runtime_revision=str(props.get("build_info", "")),
            # The emitted state version is part of cache identity: an artifact written at
            # version 2 is refused outright by a build expecting version 3.
            state_format=self.opaque_format() if self.slot_save_path else "",
            k_dtype=str(props.get("type_k", settings.get("type_k", ""))),
            v_dtype=str(props.get("type_v", settings.get("type_v", ""))),
            context_kind=str(settings.get("n_ctx", "")),
            byte_order="little",
        )

    def identity(self, model: str = "") -> dict[str, str]:
        """Identity read from the running server, never assumed from configuration."""
        model_ident = self.model_identity(model)
        return {
            "model_digest": model_ident.digest(),
            "cache_abi_digest": self.cache_abi_identity(model).digest(),
            "build_info": str(self.props().get("build_info", "")),
            "weights_sha256": model_ident.weights_sha256,
        }

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
            opaque_formats=frozenset({self.opaque_format()}) if can_slot else frozenset(),
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
        slot = self._slot(req.slot)
        result = self._post(f"/slots/{slot}?action=save", {"filename": filename})
        state = self.slot_save_path / filename
        if not state.is_file():
            raise AdapterError(f"server reported a save but {state} does not exist")
        # Read only the envelope header; the body can be gigabytes and never needs to be
        # in the Python heap.
        with open(state, "rb") as handle:
            head = handle.read(4096)
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
        # Label with the version actually present in the bytes, not an assumed one.
        version = ggsq_envelope.peek_version(head)
        fmt = f"{OPAQUE_FORMAT_FAMILY}/{version}"
        self._state_version = version
        manifest["identity"]["state_version"] = version
        try:
            return container.write_opaque_from_file(out, manifest, state, fmt)
        finally:
            state.unlink(missing_ok=True)   # do not leave the server's copy behind

    def import_(self, artifact: Path | str, req: ImportRequest,
                verify_reuse: bool = True) -> ImportReport:
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
            header = container.read_header(artifact)
            blob = header.get("blob", {})
            header_abi = (header.get("identity") or {}).get("cache_abi_digest", "")
            live_abi = self.identity(req.model).get("cache_abi_digest", "")
            expected = req.expected_cache_abi_digest or live_abi
            # An opaque blob is only readable by the configuration that wrote it. There is
            # deliberately no override: a mismatched restore fails silently, not loudly.
            if header_abi and expected and header_abi != expected:
                return ImportReport(
                    mode=StagingMode.HOST_STAGED, ok=False,
                    representation=Representation.OPAQUE,
                    reason=f"cache ABI mismatch: artifact {header_abi[:12]} vs expected {expected[:12]}")
            # Refuse a version this runtime cannot load, before touching the restore
            # endpoint. Relabelling a version-2 artifact as version 3 would be a lie the
            # loader discovers only after the state is already in flight.
            live_format = self.opaque_format()
            artifact_format = blob.get("opaque_format")
            if artifact_format != live_format:
                return ImportReport(
                    mode=StagingMode.HOST_STAGED, ok=False,
                    representation=Representation.OPAQUE,
                    reason=(f"opaque format mismatch: artifact is {artifact_format!r}, "
                            f"this runtime emits and loads {live_format!r}"),
                    seconds=time.time() - started)

            slot = self._slot(req.slot)
            artifact_name = f"{artifact.stem}.{os.getpid()}.restore.bin"
            staged = self.slot_save_path / artifact_name
            self._staged.append(staged)
            container.extract_payload(artifact, staged)
            result = self._post(f"/slots/{slot}?action=restore", {"filename": artifact_name})
            restored = int(result.get("n_restored", 0))
            if restored <= 0:
                return ImportReport(mode=StagingMode.HOST_STAGED, ok=False,
                                    representation=Representation.OPAQUE,
                                    reason="server restored no cells",
                                    seconds=time.time() - started)

            # A restore count is NOT evidence of a usable cache. Observed on a 27B MTP
            # model: the server reported n_restored=201 and set n_prompt_tokens, yet the
            # very next completion re-prefilled every token (cache_n=0). Reporting ok on
            # the server's own count would hand the caller a cache that does not exist.
            reuse_note = ""
            if verify_reuse:
                token_ids = self._artifact_token_ids(artifact)
                if not token_ids:
                    return ImportReport(
                        mode=StagingMode.HOST_STAGED, ok=False,
                        representation=Representation.OPAQUE,
                        reason="cannot verify reuse: artifact carries no token IDs",
                        nbytes=int(result.get("n_read", blob.get("nbytes", 0))),
                        seconds=time.time() - started, tokens_restored=restored)
                probe = self._post("/completion", {
                    "prompt": list(token_ids), "n_predict": 1, "temperature": 0.0,
                    "top_k": 1, "cache_prompt": True, "id_slot": slot})
                timings = probe.get("timings", {})
                cache_n = int(timings.get("cache_n", 0))
                prompt_n = int(timings.get("prompt_n", 0))
                # llama.cpp always reprocesses the final token, so full reuse of an L-token
                # prefix is exactly cache_n == L-1 and prompt_n == 1. Accepting cache_n > 0
                # would accept PARTIAL reuse: probing a 200-token cache with a 100-token
                # prefix returns cache_n=99, which is a different prompt sharing a prefix.
                expected_cache_n = len(token_ids) - 1
                if cache_n != expected_cache_n or prompt_n != 1:
                    self._restore_pristine(artifact_name, slot)
                    return ImportReport(
                        mode=StagingMode.HOST_STAGED, ok=False,
                        representation=Representation.OPAQUE,
                        reason=(f"slot {slot} did not reuse the full prefix: "
                                f"cache_n={cache_n} prompt_n={prompt_n}, expected "
                                f"cache_n={expected_cache_n} prompt_n=1 for "
                                f"{len(token_ids)} token(s)"),
                        nbytes=int(result.get("n_read", blob.get("nbytes", 0))),
                        seconds=time.time() - started, tokens_restored=restored)
                # The probe generated a token into the slot. Put the slot back to the exact
                # imported prefix so a caller never inherits a mutated cache.
                self._restore_pristine(artifact_name, slot)
                reuse_note = (f"verified full-prefix reuse on slot {slot}: "
                              f"cache_n={cache_n}/{expected_cache_n} prompt_n={prompt_n}")

            return ImportReport(
                mode=StagingMode.HOST_STAGED,
                ok=True,
                representation=Representation.OPAQUE,
                reason=reuse_note,
                nbytes=int(result.get("n_read", blob.get("nbytes", 0))),
                seconds=time.time() - started,
                tokens_restored=restored,
            )
        except AdapterError as exc:
            return ImportReport(mode=StagingMode.HOST_STAGED, ok=False,
                                representation=Representation.OPAQUE, reason=str(exc),
                                seconds=time.time() - started)
        finally:
            # Staged copies are the size of the cache; leaving them behind fills the disk.
            for path in self._staged:
                path.unlink(missing_ok=True)
            self._staged.clear()

    def _restore_pristine(self, filename: str, slot: int) -> None:
        """Re-restore so the slot holds exactly the imported prefix, nothing appended."""
        try:
            self._post(f"/slots/{slot}?action=erase", {})
            self._post(f"/slots/{slot}?action=restore", {"filename": filename})
        except AdapterError:
            pass

    def _artifact_token_ids(self, artifact: Path) -> tuple[int, ...]:
        """Recover the prompt token IDs from the engine-native blob.

        llama-server writes a sequence-state file, whose envelope carries the exact token
        IDs. Reusing them is what makes reuse verifiable rather than assumed.
        """
        try:
            head = container.read_payload_prefix(artifact, 65536)
            packed = ggsq_envelope.parse_file_envelope(head).token_ids
            return ggsq_envelope.decode_prompt_tokens(packed)
        except (ggsq_envelope.EnvelopeError, container.ContainerError):
            return ()

    # -- convenience used by the parity test ------------------------------------

    def erase(self, slot: int | None = None) -> int:
        return int(self._post(f"/slots/{self._slot(slot)}?action=erase", {}).get("n_erased", 0))

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/completion", payload)


register(LlamaCppHTTPAdapter.name, LlamaCppHTTPAdapter)
