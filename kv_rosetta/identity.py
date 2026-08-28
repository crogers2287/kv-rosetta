"""KV Rosetta identity model.

An artifact must carry four independent identities, and conflating them is what lets a
cache get imported into a runtime it does not belong to. This module separates them:

  * PromptIdentity      — canonical messages, tool schemas, exact token IDs.
  * ModelIdentity       — weights, tokenizer, adapters, architecture. Content-derived: a
                          filesystem path is NOT identity, so the same weights moved to a
                          new path are the same model and different weights behind the
                          same path are not.
  * CacheABIIdentity    — everything that changes the bytes of a cache: runtime revision,
                          state format, KV config, position config. An opaque import
                          requires an EXACT match of this digest; nothing may override it.
  * ArtifactKey         — the composite key. One prompt+model legitimately has many
                          artifacts (CUDA opaque, HIP opaque, canonical raw, a target
                          native translation, several KV dtypes, several runtime
                          revisions, several mapper versions) that must coexist rather
                          than overwrite one another.

Every digest() is a deterministic sha256 over a canonical JSON encoding, so it is stable
across processes and changes when any contributing field changes. Two instances with equal
fields always produce equal digests. No digest is ever truncated; a short label is only a
display aid and must never be used as a key.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "IdentityError",
    "require_digest",
    "digest_of",
    "PromptIdentity",
    "ModelIdentity",
    "CacheABIIdentity",
    "ArtifactKey",
    "coexist",
]

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


class IdentityError(ValueError):
    """Raised when an externally supplied digest or identity field is malformed."""


def require_digest(value: str, name: str = "digest") -> str:
    """Validate ``value`` as exactly 64 lowercase hex characters before it is placed in a
    path or database key. Returns ``value`` unchanged on success."""
    if not _HEX64.match(value):
        raise IdentityError(
            f"{name} must be exactly 64 lowercase hex characters, got {value!r}"
        )
    return value


def digest_of(*parts: object) -> str:
    """Deterministic sha256 hex over the canonical JSON encoding of ``parts``.

    ``None``, ``str``, ``int``, ``float``, ``bool``, ``list``, ``dict`` and nested
    combinations all work. ``sort_keys=True`` makes dict ordering irrelevant, so the result
    is identical across processes and Python versions."""
    payload = json.dumps(
        list(parts),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptIdentity:
    tokenizer_id: str = ""
    token_ids_sha256: str = ""
    token_count: int = 0
    corpus_fingerprint: str = ""

    def digest(self) -> str:
        return digest_of(
            "PromptIdentity",
            self.tokenizer_id,
            self.token_ids_sha256,
            self.token_count,
            self.corpus_fingerprint,
        )


@dataclass(frozen=True)
class ModelIdentity:
    """Content-derived. A filesystem path is NOT identity: the same weights moved to a new
    path are the same model, and different weights behind the same path are not."""

    architecture: str = ""
    weights_sha256: str = ""
    tokenizer_sha256: str = ""
    chat_template_sha256: str = ""
    adapters: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def digest(self) -> str:
        # adapters and notes are content digests; normalise their order so that identity
        # does not depend on the order they were supplied in.
        return digest_of(
            "ModelIdentity",
            self.architecture,
            self.weights_sha256,
            self.tokenizer_sha256,
            self.chat_template_sha256,
            tuple(sorted(self.adapters)),
            tuple(sorted(self.notes)),
        )


@dataclass(frozen=True)
class CacheABIIdentity:
    """Everything that changes the bytes of a cache. An opaque import requires an EXACT
    match of this digest; nothing may override that."""

    runtime: str = ""
    runtime_revision: str = ""
    state_format: str = ""
    k_dtype: str = ""
    v_dtype: str = ""
    context_kind: str = ""
    rope_kind: str = ""
    rope_base: float = 0.0
    rope_scaling: tuple[tuple[str, object], ...] = ()
    partial_rotary_dim: int = 0
    swa_window: int = 0
    hybrid_cache: str = ""
    unified_kv: bool = False
    byte_order: str = "little"
    flags: tuple[str, ...] = ()

    def digest(self) -> str:
        # rope_scaling is documented as already-sorted key/value pairs; its values are
        # arbitrary objects, so sorting here could raise TypeError. Trust the documented
        # invariant rather than re-sorting incomparable values.
        return digest_of(
            "CacheABIIdentity",
            self.runtime,
            self.runtime_revision,
            self.state_format,
            self.k_dtype,
            self.v_dtype,
            self.context_kind,
            self.rope_kind,
            self.rope_base,
            self.rope_scaling,
            self.partial_rotary_dim,
            self.swa_window,
            self.hybrid_cache,
            self.unified_kv,
            self.byte_order,
            tuple(sorted(self.flags)),
        )


@dataclass(frozen=True)
class ArtifactKey:
    """The composite key. The prompt fingerprint alone is NOT a unique artifact key: one
    prompt legitimately has many artifacts (CUDA opaque, HIP opaque, canonical raw, a
    target-native translation, several KV dtypes, several runtime revisions, several mapper
    versions) and they MUST be able to coexist."""

    prompt: PromptIdentity
    model: ModelIdentity
    cache_abi: CacheABIIdentity
    encoding: str = "raw"
    format_version: str = ""
    representation_digest: str = ""
    mapper_id: str = ""

    def digest(self) -> str:
        # Never truncate: this is the real key. digest_of the component digests plus the
        # scalar fields, in a fixed order.
        return digest_of(
            "ArtifactKey",
            self.prompt.digest(),
            self.model.digest(),
            self.cache_abi.digest(),
            self.encoding,
            self.format_version,
            self.representation_digest,
            self.mapper_id,
        )

    def label(self) -> str:
        """First 12 chars of digest(), for logs only. Never used as a key."""
        return self.digest()[:12]

    def as_dict(self) -> dict[str, Any]:
        """Every component digest plus the scalar fields, JSON-safe."""
        return {
            "prompt": self.prompt.digest(),
            "model": self.model.digest(),
            "cache_abi": self.cache_abi.digest(),
            "encoding": self.encoding,
            "format_version": self.format_version,
            "representation_digest": self.representation_digest,
            "mapper_id": self.mapper_id,
        }


def coexist(a: ArtifactKey, b: ArtifactKey) -> bool:
    """True when ``a`` and ``b`` describe the same prompt+model but differ in any of
    encoding, cache_abi, format_version, representation_digest or mapper_id — i.e. they are
    different artifacts that must be storable side by side rather than overwriting."""
    if a.prompt != b.prompt or a.model != b.model:
        return False
    return (
        a.encoding != b.encoding
        or a.cache_abi != b.cache_abi
        or a.format_version != b.format_version
        or a.representation_digest != b.representation_digest
        or a.mapper_id != b.mapper_id
    )
