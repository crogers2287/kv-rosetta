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

from kv_rosetta import container, gguf, weights
from kv_rosetta.store import ArtifactStore
from kv_rosetta.identity import ArtifactKey, CacheABIIdentity, ModelIdentity, PromptIdentity
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


def _position(value: object) -> int:
    """A reported checkpoint position, preserving 0 and rejecting non-integers."""
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


class LlamaCppHTTPAdapter(Adapter):
    name = "llamacpp-http"

    #: Largest tail of the prefix a runtime may reprocess and still count as reuse.
    #: Measured on this host: 1 token for ordinary attention, 4 for a hybrid model with
    #: persisted checkpoints. The bound exists so that "reuse" cannot quietly degrade into
    #: reprocessing an arbitrary fraction of the prompt.
    max_uncovered_tail: int = 8

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

    def checkpoint_protocol(self) -> dict[str, Any]:
        """The runtime's advertised checkpoint-persistence protocol, or {} when absent.

        A machine-readable statement of behaviour. Support is never inferred from an
        architecture name, a build id, a filename or an artifact size.
        """
        props = self.props()
        if not props.get("slot_checkpoint_persistence"):
            return {}
        return {
            "format": str(props.get("slot_checkpoint_format", "")),
            "sequence_state_version": props.get("sequence_state_version"),
            # PROVEN behaviour, not what the format happens to serialize.
            "target": bool(props.get("supports_target_checkpoint_state")),
            "draft": bool(props.get("supports_draft_checkpoint_state")),
            "speculative": bool(props.get("supports_speculative_checkpoint_state")),
            # What the on-disk format carries, reported separately because a struct member
            # is source evidence and never a runtime capability.
            "serializes": {
                "target": bool(props.get("sckp_serializes_target_state")),
                "draft": bool(props.get("sckp_serializes_draft_state")),
                "speculative": bool(props.get("sckp_serializes_speculative_state")),
            },
        }

    #: Sequence-state versions this adapter has been exercised against.
    supported_sequence_versions: frozenset[int] = frozenset({2, 3})
    #: Checkpoint formats this adapter understands.
    supported_checkpoint_formats: frozenset[str] = frozenset({"sckp/1"})

    #: (sequence-state version, checkpoint format) pairs a retained test has actually
    #: exercised end to end. The checkpoint-persistence proof is for sequence version 3;
    #: plain ggsq/2 support says nothing about whether an appendix on a version-2 state
    #: restores, so that tuple stays out until a test proves it.
    supported_compound_tuples: frozenset[tuple[int, str]] = frozenset({(3, "sckp/1")})

    #: Checkpoint state classes whose restoration this project has proven behaviourally.
    proven_state_classes: frozenset[str] = frozenset({"target"})

    def _protocol_is_complete(self, protocol: dict[str, Any]) -> tuple[bool, str]:
        """Whether the advertised protocol is complete AND exact enough to act on.

        Absent, partial, malformed or unrecognised protocols all fail closed. Support is
        never enabled by an architecture name, a filename, a strings match, an artifact
        size, or the mere presence of the SCKP magic.
        """
        if not protocol:
            return False, "runtime advertises no checkpoint-persistence protocol"
        fmt = protocol.get("format") or ""
        if fmt not in self.supported_checkpoint_formats:
            return False, (f"unrecognised checkpoint format {fmt!r}; this adapter "
                           f"understands {sorted(self.supported_checkpoint_formats)}")
        version = protocol.get("sequence_state_version")
        if not isinstance(version, int) or version not in self.supported_sequence_versions:
            return False, (f"unsupported sequence-state version {version!r}; this adapter "
                           f"has been exercised against "
                           f"{sorted(self.supported_sequence_versions)}")
        if not protocol.get("target"):
            return False, ("runtime does not advertise PROVEN target checkpoint state; "
                           "serialization alone is not a capability")
        return True, ""

    def opaque_format(self) -> str:
        """The artifact's representation label.

        When the runtime persists checkpoints the file is a GGSQ sequence state with an
        SCKP appendix, and calling that plain ggsq/N would misdescribe it: an importer
        would believe a sequence-only restore is sufficient. One file, one compound label.
        """
        base = f"{OPAQUE_FORMAT_FAMILY}/{self.state_version()}"
        protocol = self.checkpoint_protocol()
        return f"{base}+{protocol['format']}" if protocol.get("format") else base

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
            # The checkpoint contract is part of cache identity: an artifact written by a
            # build that persists draft state is not interchangeable with one that does
            # not, even at the same sequence-state version.
            flags=tuple(sorted(self._checkpoint_flags(props))),
        )

    def _checkpoint_flags(self, props: dict[str, Any]) -> list[str]:
        protocol = self.checkpoint_protocol()
        if not protocol:
            return ["sckp:none"]
        flags = [f"sckp:{protocol['format']}", f"seqver:{protocol['sequence_state_version']}"]
        for blob in ("target", "draft", "speculative"):
            if protocol.get(blob):
                flags.append(f"sckp-blob:{blob}")
        # Which classes THIS launch requires is part of cache identity: a cache written by
        # a launch that also stores draft state is not interchangeable with one that does
        # not, even at the same sequence version and format.
        active = self._active_state_classes()
        flags.extend(f"active:{name}" for name in sorted(active)) if active is not None \
            else flags.append("active:unreported")
        settings = props.get("default_generation_settings", {}) or {}
        for key in ("n_ctx_checkpoints", "checkpoint_min_step"):
            if settings.get(key) is not None:
                flags.append(f"policy:{key}={settings[key]}")
        return flags

    def identity(self, model: str = "") -> dict[str, str]:
        """Identity read from the running server, never assumed from configuration."""
        model_ident = self.model_identity(model)
        return {
            "model_digest": model_ident.digest(),
            "cache_abi_digest": self.cache_abi_identity(model).digest(),
            "build_info": str(self.props().get("build_info", "")),
            "weights_sha256": model_ident.weights_sha256,
        }

    def prefix_reuse_support(self) -> tuple[bool, str]:
        """Whether this model's restored cache can be reused for a prompt prefix."""
        path = str(self.props().get("model_path", ""))
        if not path or not Path(path).is_file():
            return False, "model file unreachable; cannot establish architecture"
        try:
            return gguf.supports_prefix_reuse(gguf.architecture(path))
        except gguf.GGUFError as exc:
            return False, f"cannot read architecture: {exc}"

    # -- capabilities -----------------------------------------------------------

    def capabilities(self) -> Capabilities:
        """Probed from the live server. Slot save/restore is only advertised when the
        server was actually started with --slot-save-path."""
        props = self.props()
        settings = props.get("default_generation_settings", {}) or {}
        can_slot = bool(self.slot_save_path)
        notes: list[str] = []
        if not can_slot:
            notes.append("slot save/restore unavailable: server has no --slot-save-path")

        # A hybrid or recurrent architecture restores its state successfully and then
        # reuses none of it, because a recurrent state is a function of the whole sequence
        # and has no prompt-prefix semantics. Advertising a capability the runtime accepts
        # but cannot honour is exactly the fail-open shape this project exists to avoid.
        if can_slot:
            # The same decision export() and import_() make. A note is not a gate: a
            # configuration whose restoration is unproven must not be advertised at all.
            supported, reason, _protocol = self.hybrid_support()
            notes.append(reason if supported else f"opaque transfer withheld: {reason}")
            can_slot = supported
        reps = frozenset({Representation.OPAQUE}) if can_slot else frozenset()
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

    def publish(self, artifact: Path | str, store: "ArtifactStore") -> object:
        """Store an artifact under its own composite key.

        Lookup keyed on the proxy prefix fingerprint alone cannot distinguish a CUDA
        opaque state from a HIP one, a canonical representation, or a translation - they
        share a prompt and nothing else. The key recorded in the artifact is recomputed
        here so a mislabelled file cannot be filed under an identity it does not have.
        """
        artifact = Path(artifact)
        ok, reason = container.verify(artifact)
        if not ok:
            raise AdapterError(f"refusing to publish an unverifiable artifact: {reason}")
        header = container.read_header(artifact)
        recorded = header.get("artifact_key") or {}
        if not recorded.get("digest"):
            raise AdapterError("artifact carries no composite key; refusing to publish")
        prompt = header.get("prompt", {})
        identity = header.get("identity", {})
        key = ArtifactKey(
            prompt=PromptIdentity(
                tokenizer_id=prompt.get("tokenizer_id", ""),
                token_ids_sha256=prompt.get("token_ids_sha256", ""),
                token_count=int(prompt.get("token_count", 0)),
                corpus_fingerprint=str((header.get("corpus") or {}).get("fingerprint", "")),
            ),
            model=self.model_identity(),
            cache_abi=self.cache_abi_identity(),
            encoding="opaque",
            format_version=str(header.get("blob", {}).get("opaque_format", "")),
            representation_digest=str(recorded.get("representation_digest", "")),
            mapper_id=str(recorded.get("mapper_id", "")),
        )
        if key.digest() != recorded["digest"]:
            raise AdapterError(
                f"artifact key does not reproduce: recorded {recorded['digest'][:12]}, "
                f"recomputed {key.digest()[:12]}")
        return store.put(key, artifact)

    def artifact_key(self, artifact: Path | str) -> ArtifactKey | None:
        """Recompute the composite key recorded in an artifact, or None when absent."""
        recorded = (container.read_header(artifact).get("artifact_key") or {})
        return recorded or None

    def hybrid_support(self) -> tuple[bool, str, dict[str, Any]]:
        """The single support decision for a hybrid runtime, used by all three surfaces.

        capabilities(), export() and import_() answered this independently and disagreed:
        capabilities advertised OPAQUE for any complete protocol, while export additionally
        required a tested compound tuple and a provably target-only launch. A caller who
        trusted capabilities and then exported got a refusal. There is one answer now.

        Returns (supported, reason, protocol). An empty protocol with supported=True means
        the architecture reuses prefixes and needs no checkpoints.
        """
        reusable, why = self.prefix_reuse_support()
        if reusable:
            return True, "architecture reuses restored prefixes", {}
        protocol = self.checkpoint_protocol()
        complete, reason = self._protocol_is_complete(protocol)
        if not complete:
            return False, (f"{why}, and {reason}. A sequence-only artifact from a hybrid "
                           f"model restores and then reuses nothing."), protocol
        version = protocol.get("sequence_state_version")
        fmt = protocol.get("format")
        if (version, fmt) not in self.supported_compound_tuples:
            return False, (f"compound tuple ggsq/{version}+{fmt} is not in the tested "
                           f"allowlist {sorted(self.supported_compound_tuples)}; refusing "
                           f"rather than assuming an appendix on this sequence version "
                           f"restores"), protocol
        active = self._active_state_classes()
        if active is None:
            return False, ("runtime does not report which checkpoint state classes its "
                           "current launch requires, so a target-only configuration cannot "
                           "be proven; refusing rather than assuming draft/speculative "
                           "state is absent"), protocol
        unproven = sorted(set(active) - self.proven_state_classes)
        if unproven:
            return False, (f"this launch requires {unproven} checkpoint state, whose "
                           f"restoration is not behaviourally proven; refusing"), protocol
        return True, f"target-only {fmt} on sequence version {version}", protocol

    def _require_exportable(self, model: str) -> dict[str, Any]:
        """Refuse before asking the server to save anything it cannot make usable.

        Capability discovery is advisory - a caller can skip it and request OPAQUE
        directly. The check belongs here, at the boundary that produces artifacts.
        """
        supported, reason, protocol = self.hybrid_support()
        if not supported:
            raise AdapterError(f"refusing to export from this runtime: {reason}")
        return protocol

    def _verify_checkpoint_appendix(self, state: Path, result: dict[str, Any],
                                    coverage: dict[str, int],
                                    protocol: dict[str, Any]) -> bool:
        """Whether this artifact may be labelled compound, from the bytes and the metadata.

        The runtime reports what it wrote, so the appendix boundary is known rather than
        searched for: offset = n_written - checkpoint_bytes. Requiring the magic exactly
        there, the appendix to run to EOF, and n_written to equal the file size is format
        evidence. A magic found somewhere in the file is not - opaque KV data contains
        those four bytes by chance, and labelling on that would tell an importer a hybrid
        restore is available when none is.
        """
        if not protocol:
            return False              # a prefix-reusing architecture needs no appendix
        size = state.stat().st_size
        n_written = int(result.get("n_written", 0) or 0)
        checkpoint_bytes = coverage["checkpoint_bytes"]
        missing = [name for name in ("n_checkpoints", "checkpoint_n_tokens")
                   if coverage[name] <= 0]
        if missing or checkpoint_bytes <= 0:
            raise AdapterError(
                f"runtime advertises {protocol['format']} but this save reported no usable "
                f"checkpoint coverage ({dict(coverage)}); a sequence-only artifact from a "
                f"hybrid model restores and then reuses nothing")
        if coverage["checkpoint_pos_min"] < 0 or coverage["checkpoint_pos_max"] < 0:
            raise AdapterError(f"checkpoint positions are not valid: {dict(coverage)}")
        if n_written != size:
            raise AdapterError(
                f"runtime reported writing {n_written} bytes but {state} is {size}; "
                f"refusing to label an artifact whose declared bounds do not match it")
        appendix = ggsq_envelope.checkpoint_appendix_at(state, n_written - checkpoint_bytes)
        if not appendix.usable:
            raise AdapterError(
                f"no usable checkpoint appendix at the declared offset "
                f"{n_written - checkpoint_bytes} ({appendix.status.value}); refusing to "
                f"label this artifact {protocol['format']}")
        return True

    def _active_state_classes(self) -> list[str] | None:
        """Which checkpoint state classes this launch actually requires, or None."""
        value = self.props().get("active_checkpoint_state_classes")
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            return None
        return value

    def export(self, req: ExportRequest) -> Path:
        if req.representation is not Representation.OPAQUE:
            raise AdapterError(
                f"{self.name} exports {Representation.OPAQUE.value} only, "
                f"asked for {req.representation.value}")
        if not self.slot_save_path:
            raise AdapterError("no slot_save_path configured")
        protocol = self._require_exportable(req.model)
        out = Path(req.out_path)
        filename = out.stem + ".bin"
        slot = self._slot(req.slot)
        result = self._post(f"/slots/{slot}?action=save", {"filename": filename})
        # What the runtime says it persisted. Recorded so import can compare declared
        # coverage against observed reuse instead of trusting either alone.
        coverage = {
            "n_checkpoints": int(result.get("n_checkpoints_saved", 0) or 0),
            "checkpoint_bytes": int(result.get("checkpoint_bytes", 0) or 0),
            "checkpoint_n_tokens": int(result.get("checkpoint_n_tokens", 0) or 0),
            # `x or -1` turns a legitimate position 0 into "absent". Position 0 is the
            # first token of the prompt and is exactly what a full-prefix checkpoint reports.
            "checkpoint_pos_min": _position(result.get("checkpoint_pos_min")),
            "checkpoint_pos_max": _position(result.get("checkpoint_pos_max")),
        }
        state = self.slot_save_path / filename
        if not state.is_file():
            raise AdapterError(f"server reported a save but {state} does not exist")
        # Read only the envelope header; the body can be gigabytes and never needs to be
        # in the Python heap. Its size depends on the prompt length, so size it from the
        # declared count rather than peeking a fixed number of bytes.
        with open(state, "rb") as handle:
            head = handle.read(12)
            head += handle.read(ggsq_envelope.header_size(head) - len(head))
        ident = self.identity(req.model)
        model_ident = self.model_identity(req.model)
        # Label with the version actually present in the bytes, not an assumed one.
        version = ggsq_envelope.peek_version(head)
        # Label from the BYTES, not from what the runtime says it can do. A file carrying
        # an SCKP appendix is not plain ggsq/N, and calling it that would let an importer
        # believe a sequence-only restore is sufficient.
        has_ckpt = self._verify_checkpoint_appendix(state, result, coverage, protocol)
        fmt_version = version
        # The prompt identity must carry the EXACT tokens this cache was built from.
        # Empty stand-ins would let an artifact be matched to a prompt it never saw.
        packed = ggsq_envelope.parse_file_envelope(head).token_ids
        prompt_tokens = ggsq_envelope.decode_prompt_tokens(packed)
        prompt_ident = PromptIdentity(
            tokenizer_id=model_ident.tokenizer_sha256,
            token_ids_sha256=hashlib.sha256(
                json.dumps(list(prompt_tokens), separators=(",", ":")).encode()).hexdigest(),
            token_count=len(prompt_tokens),
            corpus_fingerprint=str((req.corpus or {}).get("fingerprint", "")),
        )

        abi_ident = self.cache_abi_identity(req.model)
        props = self.props()
        settings = props.get("default_generation_settings", {}) or {}
        manifest = {
            "schema": "kvx/0.3",
            "model": {"architecture": "", "weights_id": ident["model_digest"], "dtype": ""},
            "prompt": {"tokenizer_id": prompt_ident.tokenizer_id,
                       "token_ids_sha256": prompt_ident.token_ids_sha256,
                       "token_count": prompt_ident.token_count},
            "kv": {"layers": 0, "heads": 0, "head_dim": 0,
                   "dtype": str(settings.get("type_k", "f16")),
                   "layout": "opaque", "byte_order": "little",
                   "tokens": int(result.get("n_saved", 0))},
            "position": {"kind": "rope", "rope_applied": True},
            "producer": {"runtime": "llama.cpp", "backend": "",
                         "device_arch": "", "library_version": ident["build_info"]},
            "identity": {"model_digest": ident["model_digest"],
                         "cache_abi_digest": ident["cache_abi_digest"],
                         "weights_sha256": model_ident.weights_sha256,
                         "prompt_digest": prompt_ident.digest()},
            "corpus": req.corpus or {},
        }
        fmt = f"{OPAQUE_FORMAT_FAMILY}/{version}"
        if has_ckpt:
            fmt = f"{fmt}+{ggsq_envelope.SCKP_FORMAT}"
        self._state_version = version
        manifest["identity"]["state_version"] = version
        # The composite key is what the store is keyed on, so it is embedded in the
        # artifact and recomputed from the file on import. An artifact that cannot
        # reproduce its own key is not the artifact the store thinks it is.
        key = ArtifactKey(
            prompt=prompt_ident, model=model_ident, cache_abi=abi_ident,
            encoding="opaque", format_version=fmt,
            representation_digest=hashlib.sha256(head[:12]).hexdigest(),
        )
        manifest["coverage"] = dict(coverage, format=fmt)
        manifest["artifact_key"] = key.as_dict()
        manifest["artifact_key"]["digest"] = key.digest()
        try:
            return container.write_opaque_from_file(out, manifest, state, fmt)
        finally:
            state.unlink(missing_ok=True)   # do not leave the server's copy behind

    def import_(self, artifact: Path | str, req: ImportRequest,
                verify_reuse: bool = True) -> ImportReport:
        started = time.time()
        artifact = Path(artifact)
        phases: dict[str, float] = {}
        try:
            if not self.slot_save_path:
                return ImportReport(mode=StagingMode.HOST_STAGED, ok=False,
                                    representation=Representation.OPAQUE,
                                    reason="no slot_save_path configured")
            phase_started = time.time()
            ok, reason = container.verify(artifact)
            phases["container_verify"] = time.time() - phase_started
            if not ok:
                return ImportReport(mode=StagingMode.HOST_STAGED, ok=False,
                                    representation=Representation.OPAQUE,
                                    reason=f"artifact failed verification: {reason}")
            header = container.read_header(artifact)
            blob = header.get("blob", {})
            recorded = header.get("identity") or {}
            header_abi = recorded.get("cache_abi_digest", "")
            live = self.identity(req.model)
            live_abi = live.get("cache_abi_digest", "")
            # Weights identity is checked separately from the cache ABI: the same runtime
            # configuration can be serving entirely different weights, and a cache restored
            # against the wrong weights is silently wrong rather than loudly broken.
            header_model = recorded.get("model_digest", "")
            if header_model and header_model != live.get("model_digest", ""):
                return ImportReport(
                    mode=StagingMode.HOST_STAGED, ok=False,
                    representation=Representation.OPAQUE,
                    reason=(f"model identity mismatch: artifact {header_model[:12]} vs "
                            f"live {live.get('model_digest', '')[:12]}"),
                    seconds=time.time() - started)
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

            # The same decision capabilities() and export() make, before the state is put
            # anywhere near the runtime.
            supported, support_reason, _protocol = self.hybrid_support()
            if not supported:
                return ImportReport(
                    mode=StagingMode.HOST_STAGED, ok=False,
                    representation=Representation.OPAQUE,
                    reason=f"refusing to import into this runtime: {support_reason}",
                    seconds=time.time() - started)

            # A compound artifact declares checkpoint coverage. Zero or missing coverage
            # means the appendix carries nothing reusable, so there is nothing to import
            # that a plain prefill would not do better.
            # The blob format is authoritative: it is what the compatibility check above
            # matched against the live runtime. Coverage is a description of that blob, so
            # it must agree exactly. Selecting compound behaviour from the coverage text
            # instead let an artifact whose blob said compound - and so passed the
            # compatibility check - be handled as plain, skipping every compound check.
            raw_coverage = header.get("coverage")
            is_compound = self._is_compound_format(artifact_format)
            if is_compound:
                if not isinstance(raw_coverage, dict):
                    return ImportReport(
                        mode=StagingMode.HOST_STAGED, ok=False,
                        representation=Representation.OPAQUE,
                        reason=(f"artifact is {artifact_format} but carries no coverage "
                                f"object; refusing before restore"),
                        seconds=time.time() - started)
                if raw_coverage.get("format") != artifact_format:
                    return ImportReport(
                        mode=StagingMode.HOST_STAGED, ok=False,
                        representation=Representation.OPAQUE,
                        reason=(f"format disagreement: blob says {artifact_format!r}, "
                                f"coverage says {raw_coverage.get('format')!r}; refusing "
                                f"before restore"),
                        seconds=time.time() - started)
            elif isinstance(raw_coverage, dict) and \
                    self._is_compound_format(raw_coverage.get("format")):
                return ImportReport(
                    mode=StagingMode.HOST_STAGED, ok=False,
                    representation=Representation.OPAQUE,
                    reason=(f"format disagreement: coverage claims "
                            f"{raw_coverage.get('format')!r} but the blob is "
                            f"{artifact_format!r}; refusing before restore"),
                    seconds=time.time() - started)
            coverage = dict(raw_coverage) if isinstance(raw_coverage, dict) else {}
            if is_compound:
                numeric, bad = self._coverage_numbers(coverage)
                if bad:
                    return ImportReport(
                        mode=StagingMode.HOST_STAGED, ok=False,
                        representation=Representation.OPAQUE,
                        reason=(f"coverage fields are not usable integers: {bad}; "
                                f"refusing before restore"),
                        seconds=time.time() - started)
                coverage.update(numeric)
                if numeric["checkpoint_n_tokens"] <= 0 or numeric["n_checkpoints"] <= 0:
                    return ImportReport(
                        mode=StagingMode.HOST_STAGED, ok=False,
                        representation=Representation.OPAQUE,
                        reason=(f"compound artifact declares no checkpoint coverage "
                                f"({coverage}); refusing before restore"),
                        seconds=time.time() - started)
                # Reuse verification is not optional for a compound artifact: the whole
                # claim is that a checkpoint survived, and only a probe shows that.
                verify_reuse = True

            slot = self._slot(req.slot)
            artifact_name = f"{artifact.stem}.{os.getpid()}.restore.bin"
            staged = self.slot_save_path / artifact_name
            self._staged.append(staged)
            phase_started = time.time()
            container.extract_payload(artifact, staged)
            phases["staging"] = time.time() - phase_started
            phase_started = time.time()
            result = self._post(f"/slots/{slot}?action=restore", {"filename": artifact_name})
            phases["runtime_restore"] = time.time() - phase_started
            restored = int(result.get("n_restored", 0))
            if is_compound:
                pairs = (("n_checkpoints", "n_checkpoints_restored"),
                         ("checkpoint_bytes", "checkpoint_bytes"),
                         ("checkpoint_n_tokens", "checkpoint_n_tokens"),
                         ("checkpoint_pos_min", "checkpoint_pos_min"),
                         ("checkpoint_pos_max", "checkpoint_pos_max"))
                differing = [
                    f"{manifest_key}: manifest {coverage.get(manifest_key)!r} vs restore "
                    f"{result.get(response_key)!r}"
                    for manifest_key, response_key in pairs
                    if coverage.get(manifest_key) != result.get(response_key)
                ]
                if differing:
                    self._erase(slot)
                    return ImportReport(
                        mode=StagingMode.HOST_STAGED, ok=False,
                        representation=Representation.OPAQUE,
                        reason=("restore metadata does not match the manifest: "
                                + "; ".join(differing)),
                        seconds=time.time() - started, tokens_restored=restored)
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
                phase_started = time.time()
                probe = self._post("/completion", {
                    "prompt": list(token_ids), "n_predict": 1, "temperature": 0.0,
                    "top_k": 1, "cache_prompt": True, "id_slot": slot})
                phases["reuse_probe"] = time.time() - phase_started
                timings = probe.get("timings", {})
                cache_n = int(timings.get("cache_n", 0))
                prompt_n = int(timings.get("prompt_n", 0))
                # A runtime always reprocesses a short tail of the prefix, and how short
                # depends on the architecture: measured here, ordinary attention reprocesses
                # exactly 1 token, while a hybrid model with persisted checkpoints
                # reprocesses exactly 4 - constant across 256/1024/4096-token prompts and
                # across --checkpoint-min-step and --ctx-checkpoints settings, so it is
                # structural rather than a granularity artifact.
                #
                # The invariant that holds for both: everything except a small tail is
                # reused, and the tokens reprocessed are EXACTLY that tail. Requiring
                # cache_n > 0 alone would accept partial reuse of a merely-shared prefix;
                # requiring cache_n == L-1 would reject a correct hybrid restore.
                uncovered = len(token_ids) - cache_n
                declared = int((header.get("coverage") or {}).get("checkpoint_n_tokens", 0) or 0)
                # When the artifact declares checkpoint coverage, the runtime must reuse
                # exactly that much. More would mean credit for cache the artifact never
                # carried; less means the checkpoint did not survive.
                if declared and cache_n != declared:
                    self._restore_pristine(artifact_name, slot)
                    return ImportReport(
                        mode=StagingMode.HOST_STAGED, ok=False,
                        representation=Representation.OPAQUE,
                        reason=(f"declared checkpoint coverage {declared} does not match "
                                f"observed reuse cache_n={cache_n}; refusing rather than "
                                f"trusting either number alone"),
                        nbytes=int(result.get("n_read", blob.get("nbytes", 0))),
                        seconds=time.time() - started, tokens_restored=restored)
                if uncovered < 1 or uncovered > self.max_uncovered_tail or prompt_n != uncovered:
                    self._restore_pristine(artifact_name, slot)
                    return ImportReport(
                        mode=StagingMode.HOST_STAGED, ok=False,
                        representation=Representation.OPAQUE,
                        reason=(f"slot {slot} did not reuse the prefix as declared: "
                                f"cache_n={cache_n} prompt_n={prompt_n} for "
                                f"{len(token_ids)} token(s) leaves {uncovered} uncovered; "
                                f"require 1..{self.max_uncovered_tail} uncovered and "
                                f"prompt_n == uncovered"),
                        nbytes=int(result.get("n_read", blob.get("nbytes", 0))),
                        seconds=time.time() - started, tokens_restored=restored)
                # The probe generated a token into the slot. Put the slot back to the exact
                # imported prefix so a caller never inherits a mutated cache. If that fails
                # the slot holds the prefix plus the probe's token, which is not the cache
                # the caller would be told it has.
                phase_started = time.time()
                pristine_ok = self._restore_pristine(artifact_name, slot)
                phases["pristine_restore"] = time.time() - phase_started
                if not pristine_ok:
                    return ImportReport(
                        mode=StagingMode.HOST_STAGED, ok=False,
                        representation=Representation.OPAQUE,
                        reason=("reuse verified, but restoring the pristine prefix "
                                "afterwards failed; the slot was erased rather than left "
                                "holding the verification probe's token"),
                        nbytes=int(result.get("n_read", blob.get("nbytes", 0))),
                        seconds=time.time() - started, tokens_restored=restored)
                reuse_note = (f"verified reuse on slot {slot}: cache_n={cache_n} of "
                              f"{len(token_ids)} token(s), {uncovered} reprocessed"
                              + (f", matching declared coverage {declared}" if declared else ""))

            return ImportReport(
                mode=StagingMode.HOST_STAGED,
                ok=True,
                representation=Representation.OPAQUE,
                reason=reuse_note,
                nbytes=int(result.get("n_read", blob.get("nbytes", 0))),
                seconds=time.time() - started,
                tokens_restored=restored,
                phases=dict(phases),
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

    def _is_compound_format(self, label: object) -> bool:
        """Whether a format label names a tested compound tuple, by exact comparison.

        Not by looking for a plus sign: an unknown tuple must not be treated as compound
        support, and a compound-looking string is not evidence of one.
        """
        if not isinstance(label, str):
            return False
        return label in {f"{OPAQUE_FORMAT_FAMILY}/{version}+{fmt}"
                         for version, fmt in self.supported_compound_tuples}

    @staticmethod
    def _coverage_numbers(coverage: dict) -> tuple[dict[str, int], list[str]]:
        """Coerce the declared coverage counts, reporting what could not be read.

        A manifest is untrusted input. A malformed value must produce a refusal, not an
        exception escaping the adapter boundary as int() would raise on 'many' or NaN.
        """
        names = ("n_checkpoints", "checkpoint_bytes", "checkpoint_n_tokens",
                 "checkpoint_pos_min", "checkpoint_pos_max")
        numbers: dict[str, int] = {}
        bad: list[str] = []
        for name in names:
            value = coverage.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                bad.append(f"{name}={value!r}")
            else:
                numbers[name] = value
        return numbers, bad

    def _erase(self, slot: int) -> None:
        """Best-effort slot clear. Used when returning ok=false after a restore."""
        try:
            self._post(f"/slots/{slot}?action=erase", {})
        except AdapterError:
            pass

    def _restore_pristine(self, filename: str, slot: int) -> bool:
        """Re-restore so the slot holds exactly the imported prefix, nothing appended.

        The verification probe appends its own token to the slot. If putting the pristine
        prefix back fails, the slot holds the prefix plus that token - so the cache the
        caller is about to be told it has is not the cache in the slot. Swallowing the
        failure handed back ok=true over exactly that state.
        """
        try:
            self._post(f"/slots/{slot}?action=erase", {})
            self._post(f"/slots/{slot}?action=restore", {"filename": filename})
            return True
        except AdapterError:
            self._erase(slot)
            return False

    def _artifact_token_ids(self, artifact: Path) -> tuple[int, ...]:
        """Recover the prompt token IDs from the engine-native blob.

        llama-server writes a sequence-state file, whose envelope carries the exact token
        IDs. Reusing them is what makes reuse verifiable rather than assumed.
        """
        try:
            prefix = container.read_payload_prefix(artifact, 12)
            head = container.read_payload_prefix(
                artifact, ggsq_envelope.header_size(prefix))
            packed = ggsq_envelope.parse_file_envelope(head).token_ids
            return ggsq_envelope.decode_prompt_tokens(packed)
        except (ggsq_envelope.EnvelopeError, container.ContainerError):
            return ()

    # -- convenience used by the parity test ------------------------------------

    def tokenize(self, text: str) -> list[int]:
        """Tokenize with the LIVE model: token IDs are model-specific, so a prefix must be
        tokenized by the runtime that will serve it, never by an assumed tokenizer."""
        return list(self._post("/tokenize", {"content": text}).get("tokens", []))

    def erase(self, slot: int | None = None) -> int:
        return int(self._post(f"/slots/{self._slot(slot)}?action=erase", {}).get("n_erased", 0))

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/completion", payload)


register(LlamaCppHTTPAdapter.name, LlamaCppHTTPAdapter)
