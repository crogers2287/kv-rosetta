"""Resolve a proxy prefix manifest to a stored artifact, by composite identity.

The proxy's prefix fingerprint identifies a reusable prompt prefix. It is provenance, not
an artifact key: one prefix legitimately has many artifacts - a CUDA opaque state, a HIP
one, a canonical representation, a translation, several KV dtypes, several runtime
revisions - and they must not collide.

So resolution goes through the runtime rather than around it. The prefix text is tokenized
by the live model, which is the only way to obtain the exact token IDs the identity is built
on, and the composite key is assembled from that plus the model and cache-ABI identities the
adapter probes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from kv_rosetta.identity import ArtifactKey, PromptIdentity


class ResolveError(ValueError):
    pass


class _Adapter(Protocol):
    def model_identity(self, model: str = ...) -> Any: ...
    def cache_abi_identity(self, model: str = ...) -> Any: ...
    def opaque_format(self) -> str: ...
    def tokenize(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class Resolution:
    """What a prefix resolved to, and why."""

    key: ArtifactKey | None
    record: Any | None
    reason: str

    @property
    def found(self) -> bool:
        return self.record is not None


def prefix_text(manifest: dict[str, Any]) -> str:
    """The static head the proxy recorded: system prompt followed by canonical tools.

    Concatenated in the order the proxy stores them; the tools field is already a
    canonical JSON string, so it is used verbatim rather than re-encoded.
    """
    system = str(manifest.get("system", ""))
    tools = manifest.get("tools", "")
    if isinstance(tools, (dict, list)):
        tools = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    return system + str(tools or "")


def token_digest(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(list(token_ids), separators=(",", ":")).encode()).hexdigest()


def prompt_identity(manifest: dict[str, Any], token_ids: list[int],
                    tokenizer_id: str) -> PromptIdentity:
    return PromptIdentity(
        tokenizer_id=tokenizer_id,
        token_ids_sha256=token_digest(token_ids),
        token_count=len(token_ids),
        # Carried for traceability back to the proxy, never used as the key.
        corpus_fingerprint=str(manifest.get("fingerprint", "")),
    )


def artifact_key(manifest: dict[str, Any], token_ids: list[int], adapter: _Adapter,
                 encoding: str = "opaque", mapper_id: str = "") -> ArtifactKey:
    model = adapter.model_identity()
    return ArtifactKey(
        prompt=prompt_identity(manifest, token_ids, model.tokenizer_sha256),
        model=model,
        cache_abi=adapter.cache_abi_identity(),
        encoding=encoding,
        format_version=adapter.opaque_format() if encoding == "opaque" else "",
        mapper_id=mapper_id,
    )


def resolve(manifest: dict[str, Any], adapter: _Adapter, store: Any,
            encoding: str = "opaque") -> Resolution:
    """Look up the artifact for a prefix manifest. Never raises; reports why on a miss."""
    try:
        if not isinstance(manifest, dict) or manifest.get("schema") != 1:
            return Resolution(None, None, "not a schema-1 prefix manifest")
        text = prefix_text(manifest)
        if not text:
            return Resolution(None, None, "prefix manifest carries no prompt text")
        token_ids = list(adapter.tokenize(text))
        if not token_ids:
            return Resolution(None, None, "prefix tokenized to nothing")
        key = artifact_key(manifest, token_ids, adapter, encoding=encoding)
        record = store.get(key.digest())
        if record is None:
            return Resolution(key, None, "no artifact stored under this composite identity")
        return Resolution(key, record, "found")
    except Exception as exc:   # a resolution miss must never break the caller
        return Resolution(None, None, f"resolution failed: {type(exc).__name__}: {exc}")
