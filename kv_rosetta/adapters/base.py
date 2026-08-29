"""Adapter contract for the KV Rosetta cache-portability layer.

The whole point of this module is that a cache adapter can never be described by a
single "tier" or by a pair of booleans. A real engine may export canonical KVX segments
while importing only its own opaque blob, support f16 but refuse bf16, or accept an
export at one runtime revision and reject another. A coarse tier leaves the caller
blind: it cannot know, before attempting a transfer, whether that specific transfer is
even legal.

So instead of a maximum tier plus ``can_export``/``can_import`` flags, this module
defines a *directional capability report* -- :class:`Capabilities` -- that is treated
as evidence gathered from the live runtime, never as a hard-coded promise. A caller
inspects the report to decide whether a given (representation, dtype, opaque_format)
transfer is permitted, and :meth:`Adapter.supports_transfer` answers the common
"can I round-trip this?" question in a single call.

The contract (:class:`Adapter`) is deliberately backend-neutral: CUDA, HIP/ROCm, CPU,
and future backends are producers and consumers of the same canonical tensor layout and
are not part of cache identity. Identity is captured by 64-hex digests
(:meth:`Adapter.identity`) rather than by any backend-specific blob.
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Type


class AdapterError(ValueError):
    """Raised for any contract violation in the adapter layer."""


class Representation(str, Enum):
    PROMPT = "prompt"        # can only replay the prompt to re-prefill
    OPAQUE = "opaque"        # engine-native blob
    CANONICAL = "canonical"  # KVX canonical segments


class StagingMode(str, Enum):
    ZERO_COPY = "zero-copy"
    DEVICE_CONVERT = "device-convert"
    HOST_STAGED = "host-staged"
    REPLAY = "replay"


@dataclass(frozen=True)
class Capabilities:
    runtime: str = ""
    runtime_revision: str = ""
    backend: str = ""
    device_arch: str = ""
    export: frozenset[Representation] = frozenset()
    import_: frozenset[Representation] = frozenset()
    export_dtypes: frozenset[str] = frozenset()
    import_dtypes: frozenset[str] = frozenset()
    opaque_formats: frozenset[str] = frozenset()
    cache_abi_digest: str = ""
    staging: frozenset[StagingMode] = frozenset()
    notes: tuple[str, ...] = ()

    def can_export(self, rep: Representation, dtype: str = "") -> bool:
        if rep not in self.export:
            return False
        if dtype != "" and dtype not in self.export_dtypes:
            return False
        return True

    def can_import(self, rep: Representation, dtype: str = "", opaque_format: str = "") -> bool:
        if rep not in self.import_:
            return False
        if dtype != "" and dtype not in self.import_dtypes:
            return False
        if rep == Representation.OPAQUE and opaque_format != "" and opaque_format not in self.opaque_formats:
            return False
        return True

    def as_dict(self) -> Dict[str, object]:
        def _sorted(values) -> List[str]:
            return sorted(str(v) for v in values)

        return {
            "runtime": self.runtime,
            "runtime_revision": self.runtime_revision,
            "backend": self.backend,
            "device_arch": self.device_arch,
            "export": _sorted(self.export),
            "import_": _sorted(self.import_),
            "export_dtypes": _sorted(self.export_dtypes),
            "import_dtypes": _sorted(self.import_dtypes),
            "opaque_formats": _sorted(self.opaque_formats),
            "cache_abi_digest": self.cache_abi_digest,
            "staging": _sorted(self.staging),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ExportRequest:
    model: str
    out_path: Path
    representation: Representation = Representation.CANONICAL
    seq_id: int = 0
    slot: int = 0
    corpus: Optional[dict] = None


@dataclass(frozen=True)
class ImportRequest:
    model: str
    seq_id: int = 0
    slot: int = 0
    expected_cache_abi_digest: str = ""


@dataclass
class ImportReport:
    mode: StagingMode
    ok: bool
    reason: str = ""
    representation: Representation = Representation.CANONICAL
    nbytes: int = 0
    seconds: float = 0.0
    tokens_restored: int = 0
    tokens_reprefilled: int = 0
    lossy_conversions: tuple[str, ...] = ()
    #: Seconds attributed to each phase of an import: outer container verification,
    #: payload staging, runtime restore, the reuse probe, and the pristine re-restore.
    #: Reported so a total that loses to a cold prefill can be attributed rather than
    #: merely observed.
    phases: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, StagingMode):
            try:
                self.mode = StagingMode(self.mode)
            except ValueError:
                raise AdapterError(f"unknown StagingMode: {self.mode!r}") from None
        if not isinstance(self.representation, Representation):
            try:
                self.representation = Representation(self.representation)
            except ValueError:
                raise AdapterError(f"unknown Representation: {self.representation!r}") from None
        if (
            self.nbytes < 0
            or self.seconds < 0
            or self.tokens_restored < 0
            or self.tokens_reprefilled < 0
        ):
            raise AdapterError("ImportReport numeric fields must be non-negative")


class Adapter(abc.ABC):
    name: str = ""

    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abc.abstractmethod
    def identity(self, model: str) -> Dict[str, str]:
        """Return cache identity for ``model``.

        The result MUST contain ``"model_digest"`` and ``"cache_abi_digest"``, each a
        64-character lowercase hex string.
        """

    @abc.abstractmethod
    def export(self, req: ExportRequest) -> Path: ...

    @abc.abstractmethod
    def import_(self, artifact: Path, req: ImportRequest) -> ImportReport: ...

    def supports_transfer(
        self, rep: Representation, dtype: str = "", opaque_format: str = ""
    ) -> bool:
        caps = self.capabilities()
        return caps.can_export(rep, dtype) and caps.can_import(rep, dtype, opaque_format)

    @staticmethod
    def l0_sha256(
        model_path: str,
        model_ftype: str,
        chat_template: str,
        bos_token: str,
        eos_token: str,
    ) -> str:
        payload = "\x00".join(
            [model_path, model_ftype, chat_template, bos_token, eos_token]
        )
        return hashlib.sha256(payload.encode()).hexdigest()


_REGISTRY: Dict[str, Type[Adapter]] = {}


def register(name: str, cls: Type[Adapter]) -> None:
    if not isinstance(cls, type) or not issubclass(cls, Adapter):
        raise AdapterError(f"{cls!r} is not an Adapter subclass")
    if name in _REGISTRY:
        raise AdapterError(f"adapter name already registered: {name!r}")
    _REGISTRY[name] = cls


def get_adapter(name: str) -> Type[Adapter]:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = sorted(_REGISTRY)
        raise AdapterError(f"unknown adapter {name!r}; known adapters: {known}") from None


def available() -> List[str]:
    return sorted(_REGISTRY)
