"""EXPERIMENTAL: restore a pre-admitted raw state in place, with no payload copy.

This is a separate path on purpose. The portable KVX container and its verification
semantics are unchanged; nothing here weakens them, and nothing here is a production API.

It reuses the adapter's own support predicate and CacheABIIdentity, so a runtime that cannot
export or import through KVX cannot admit or restore through this path either. The two
phases are the store's: admission validates everything off the request path, and the request
path resolves an already-admitted object and restores it by name.

The request path deliberately performs no payload copy and no full payload read. That claim
is only worth making if it is measured, so `RequestPathReads` counts every byte this module
reads while a restore is in flight and the live experiment asserts the count stays bounded.
Reads performed by llama-server itself are outside this process and are attributed
separately - they are not, and must not be described as, zero.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kv_rosetta import requirements
from kv_rosetta import gguf, weights
from kv_rosetta.admitted_store import AdmissionError, AdmittedObject, AdmittedStore
from kv_rosetta.adapters import ggsq_envelope
from kv_rosetta.adapters.base import AdapterError

#: The measured tail is 4; 8 is the working ceiling carried from the earlier steer.
MAX_UNCOVERED_TAIL = 8


@dataclass
class RequestPathReads:
    """Bytes this module reads. Used to substantiate the no-copy claim, not to assume it."""

    payload_bytes: int = 0
    metadata_bytes: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class AdmittedRestoreReport:
    ok: bool
    reason: str = ""
    digest: str = ""
    cache_n: int = 0
    prompt_n: int = 0
    seconds: float = 0.0
    phases: dict[str, float] = field(default_factory=dict)
    reads: RequestPathReads = field(default_factory=RequestPathReads)
    calls: list[str] = field(default_factory=list)


class AdmittedPath:
    """Admission and direct restore for one adapter and one store."""

    def __init__(self, adapter, store: AdmittedStore) -> None:
        self.adapter = adapter
        self.store = store

    # -- admission (off the request path) -------------------------------------------

    def admit(self, raw: Path | str, *, model: str, token_ids: list[int],
              save_response: dict[str, Any],
              prefix_fingerprint: str = "") -> AdmittedObject:
        """Validate a raw state completely, then publish it. Refuses on any doubt.

        Everything the KVX import would check is checked here, because this is the only
        place it will be checked: the request path never reads the payload again.
        """
        raw = Path(raw)
        supported, reason, protocol = self.adapter.hybrid_support()
        if not supported:
            raise AdmissionError(f"runtime cannot support this state: {reason}")
        # Admission is off the request path, so it can afford the exhaustive header scan
        # that refuses an ambiguous architecture. The request path uses the early-exit
        # form; an ambiguous header is rejected here and never becomes an admitted object.
        model_path = str(self.adapter.props().get("model_path", "")) or str(model)
        if model_path and Path(model_path).is_file():
            try:
                gguf.architecture_exhaustive(model_path)
            except gguf.GGUFError as exc:
                raise AdmissionError(f"model header is not unambiguous: {exc}") from exc

        with open(raw, "rb") as handle:
            head = handle.read(12)
            head += handle.read(ggsq_envelope.header_size(head) - len(head))
        version = ggsq_envelope.peek_version(head)
        tuple_label = f"{ggsq_envelope.__dict__.get('OPAQUE_FAMILY', 'ggsq')}/{version}"
        if (version, protocol.get("format")) not in self.adapter.supported_compound_tuples:
            raise AdmissionError(
                f"untested compound tuple {tuple_label}+{protocol.get('format')}")

        envelope = ggsq_envelope.parse_file_envelope(head)
        carried = list(ggsq_envelope.decode_prompt_tokens(envelope.token_ids))
        if carried != list(token_ids):
            # The comparison is on the ids; reporting only the counts printed
            # "state carries 6169 tokens, not the 6169 under test" when the lengths
            # matched and the contents did not, which reads as a bug in the checker
            # rather than a mismatch in the caller's tokens.
            if len(carried) != len(token_ids):
                raise AdmissionError(
                    f"state carries {len(carried)} tokens, not the {len(token_ids)} "
                    f"under test")
            first = next(i for i, (a, b) in enumerate(zip(carried, token_ids)) if a != b)
            raise AdmissionError(
                f"state carries {len(carried)} tokens but they are not the ones under "
                f"test: first difference at position {first}, state has {carried[first]} "
                f"and the caller passed {list(token_ids)[first]}")

        n_written = int(save_response.get("n_written", 0) or 0)
        checkpoint_bytes = int(save_response.get("checkpoint_bytes", 0) or 0)
        if n_written != raw.stat().st_size:
            raise AdmissionError(
                f"runtime wrote {n_written} bytes but the file is {raw.stat().st_size}")
        if checkpoint_bytes <= 0 or int(save_response.get("checkpoint_n_tokens", 0) or 0) <= 0:
            raise AdmissionError(f"no usable checkpoint coverage: {save_response}")
        appendix = ggsq_envelope.checkpoint_appendix_at(raw, n_written - checkpoint_bytes)
        if not appendix.usable:
            raise AdmissionError(
                f"checkpoint appendix at the declared offset is {appendix.status.value}")

        k_dtype, v_dtype = self.adapter.cache_dtypes()
        abi = self.adapter.cache_abi_identity(model)
        model_ident = self.adapter.model_identity(model)
        manifest = {
            "experimental": True,
            "raw_size": raw.stat().st_size,
            "sequence_version": version,
            "compound_tuple": f"{tuple_label}+{protocol['format']}",
            "checkpoint": {
                "offset": n_written - checkpoint_bytes,
                "bytes": checkpoint_bytes,
                "count": int(save_response.get("n_checkpoints_saved", 0) or 0),
                "n_tokens": int(save_response.get("checkpoint_n_tokens", 0) or 0),
                "pos_min": save_response.get("checkpoint_pos_min"),
                "pos_max": save_response.get("checkpoint_pos_max"),
            },
            "model_weights_sha256": model_ident.weights_sha256,
            # model_path, not model. `model` is whatever the caller names the runtime by,
            # and every caller that reaches here through the sidecar names it by its
            # llama-swap alias -- "tiel-kvx-w6800" is not a file, so digesting it raised
            # WeightsError and no artifact could be admitted at all. The resolved path a
            # few lines above is the same value this line always meant.
            "model_content_digest": (weights.model_content_digest(model_path)
                                     if model_path and Path(model_path).is_file() else ""),
            "prompt_token_digest": hashlib.sha256(
                json.dumps(list(token_ids), separators=(",", ":")).encode()).hexdigest(),
            "prompt_token_count": len(token_ids),
            # Kept so a later restore can run the mandatory reuse probe. Without them the
            # only evidence of success would be the runtime's own restore response, which
            # is the claim this project refuses to accept.
            "prompt_token_ids": list(token_ids),
            "cache_dtype_k": k_dtype,
            "cache_dtype_v": v_dtype,
            "active_state_classes": self.adapter._active_state_classes(),
            "cache_abi_digest": abi.digest(),
            "runtime_build_info": self.adapter.props().get("build_info", ""),
            # The corpus-level identity a caller asks by. Recorded here so a lookup does
            # not have to re-tokenise a prefix to find the artifact that holds it.
            "prefix_fingerprint": prefix_fingerprint,
            "runtime_model": model,
        }
        # What a runtime must provide to restore this. Recorded at admission because it is
        # a fact about the bytes just written, and checking it at restore turns a silent
        # uselessness - a hybrid state accepted by an unpatched build that then reuses
        # nothing - into an explicit refusal.
        checkpoints = manifest["checkpoint"]["count"]
        # `hybrid` is set from what the artifact demonstrably carries, not from the model's
        # architecture, which this adapter has no way to read - `active_state_classes`
        # describes checkpoint kinds, not attention structure. The consequence is that
        # for_artifact's "hybrid saved without checkpoints" note never fires from here. That
        # is the honest outcome: a note claiming the restore will be useless is worth more
        # than nothing only if it is derived from something, and here it would be a guess.
        manifest["requirements"] = requirements.for_artifact(
            hybrid=bool(checkpoints),
            checkpoints=checkpoints,
            sequence_state_version=version,
            kv_type_k=k_dtype, kv_type_v=v_dtype,
            model_identity=model_ident.weights_sha256,
            checkpoint_format=protocol.get("format", "sckp/1") or "sckp/1",
        ).as_dict()
        return self.store.admit(raw, manifest)

    # -- restore (on the request path) ------------------------------------------------

    def restore(self, digest: str, *, model: str, token_ids: list[int],
                slot: int = 0) -> AdmittedRestoreReport:
        """Resolve an admitted object and restore it in place. No payload copy or read."""
        started = time.time()
        phases: dict[str, float] = {}
        reads = RequestPathReads()
        calls: list[str] = []

        def refuse(reason: str) -> AdmittedRestoreReport:
            return AdmittedRestoreReport(ok=False, reason=reason, digest=digest,
                                         seconds=time.time() - started, phases=dict(phases),
                                         reads=reads, calls=calls)

        # The same predicate export and import use. Checked before the store is touched,
        # so an unsupported runtime never opens or links state.
        mark = time.time()
        supported, reason, _protocol = self.adapter.hybrid_support()
        phases["resolve_support"] = time.time() - mark
        if not supported:
            return refuse(f"refusing to restore into this runtime: {reason}")
        mark = time.time()
        try:
            obj = self.store.resolve(digest)
        except AdmissionError as exc:
            phases["resolve_store"] = time.time() - mark
            return refuse(str(exc))
        phases["resolve_store"] = time.time() - mark
        mark = time.time()
        manifest = obj.manifest
        reads.metadata_bytes += len(json.dumps(manifest))
        reads.notes.append("manifest read; payload not opened by kv-rosetta")

        abi = self.adapter.cache_abi_identity(model)
        phases["resolve_abi"] = time.time() - mark
        mark = time.time()
        if manifest.get("cache_abi_digest") != abi.digest():
            phases["resolve_identity"] = time.time() - mark
            return refuse(
                f"cache ABI mismatch: admitted {str(manifest.get('cache_abi_digest'))[:12]} "
                f"vs live {abi.digest()[:12]}")
        prompt_digest = hashlib.sha256(
            json.dumps(list(token_ids), separators=(",", ":")).encode()).hexdigest()
        if manifest.get("prompt_token_digest") != prompt_digest:
            phases["resolve_identity"] = time.time() - mark
            return refuse("prompt identity mismatch between the admitted state and the "
                          "tokens being restored")
        # As in admit(): `model` is how the caller names the runtime, which through the
        # sidecar is an llama-swap alias rather than a path. Digesting it raised
        # WeightsError and turned every aliased restore into a 400. The weights path comes
        # from the runtime itself.
        live_path = str(self.adapter.props().get("model_path", "")) or str(model)
        live_digest = (weights.model_content_digest(live_path)
                       if live_path and Path(live_path).is_file() else "")
        if model and manifest.get("model_content_digest") not in ("", live_digest):
            phases["resolve_identity"] = time.time() - mark
            # Reached when the digests differ AND when the live one could not be
            # resolved at all: an artifact that recorded an identity we can no longer
            # check is refused rather than restored on the strength of the record alone.
            return refuse("model identity mismatch")
        phases["resolve_identity"] = time.time() - mark

        mark = time.time()
        calls.append(f"/slots/{slot}?action=restore")
        result = self.adapter._post(f"/slots/{slot}?action=restore",
                                    {"filename": obj.basename})
        phases["runtime_restore"] = time.time() - mark

        declared = manifest["checkpoint"]
        pairs = (("count", "n_checkpoints_restored"), ("bytes", "checkpoint_bytes"),
                 ("n_tokens", "checkpoint_n_tokens"), ("pos_min", "checkpoint_pos_min"),
                 ("pos_max", "checkpoint_pos_max"))
        differing = [f"{a}: admitted {declared.get(a)!r} vs restore {result.get(b)!r}"
                     for a, b in pairs if declared.get(a) != result.get(b)]
        if differing:
            self._erase(slot, calls)
            return refuse("restore metadata does not match the admitted state: "
                          + "; ".join(differing))

        mark = time.time()
        calls.append("/completion")
        probe = self.adapter._post("/completion", {
            "prompt": list(token_ids), "n_predict": 1, "temperature": 0.0, "top_k": 1,
            "cache_prompt": True, "id_slot": slot})
        phases["reuse_probe"] = time.time() - mark
        timings = probe.get("timings", {})
        cache_n = int(timings.get("cache_n", 0))
        prompt_n = int(timings.get("prompt_n", 0))
        uncovered = len(token_ids) - cache_n
        if cache_n != int(declared["n_tokens"]):
            self._erase(slot, calls)
            return refuse(f"cache_n {cache_n} does not equal declared coverage "
                          f"{declared['n_tokens']}")
        if not 1 <= uncovered <= MAX_UNCOVERED_TAIL or prompt_n != uncovered:
            self._erase(slot, calls)
            return refuse(f"tail contract violated: cache_n={cache_n} prompt_n={prompt_n} "
                          f"uncovered={uncovered}")

        mark = time.time()
        calls.append(f"/slots/{slot}?action=erase")
        calls.append(f"/slots/{slot}?action=restore")
        try:
            self.adapter._post(f"/slots/{slot}?action=erase", {})
            self.adapter._post(f"/slots/{slot}?action=restore", {"filename": obj.basename})
        except AdapterError as exc:
            phases["pristine_restore"] = time.time() - mark
            self._erase(slot, calls)
            return refuse(f"reuse verified, but restoring the pristine prefix failed: {exc}")
        phases["pristine_restore"] = time.time() - mark

        return AdmittedRestoreReport(
            ok=True, digest=digest, cache_n=cache_n, prompt_n=prompt_n,
            reason=f"verified reuse: cache_n={cache_n} of {len(token_ids)}, "
                   f"{uncovered} reprocessed",
            seconds=time.time() - started, phases=phases, reads=reads, calls=calls)

    def _erase(self, slot: int, calls: list[str]) -> None:
        calls.append(f"/slots/{slot}?action=erase")
        try:
            self.adapter._post(f"/slots/{slot}?action=erase", {})
        except AdapterError:
            pass
