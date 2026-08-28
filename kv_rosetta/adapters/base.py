from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any


class AdapterError(ValueError):
    pass


class Tier(IntEnum):
    PROMPT = 0
    OPAQUE = 1
    CANONICAL = 2
    MAPPED = 3


@dataclass(frozen=True)
class Capabilities:
    tier: Tier
    can_export: bool
    can_import: bool
    dtypes: frozenset[str]
    backend: str
    device_arch: str
    runtime: str
    library_version: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExportRequest:
    model: str
    out_path: Path
    seq_id: int = 0
    slot: int = 0
    corpus: dict[str, Any] | None = None


@dataclass(frozen=True)
class ImportRequest:
    model: str
    seq_id: int = 0
    slot: int = 0
    allow_identity_mismatch: bool = False


IMPORT_MODES = ("zero-copy", "device-convert", "host-staged", "replay")


@dataclass
class ImportReport:
    mode: str
    ok: bool
    reason: str = ""
    nbytes: int = 0
    seconds: float = 0.0
    tokens_restored: int = 0
    tokens_reprefilled: int = 0

    def __post_init__(self) -> None:
        if self.mode not in IMPORT_MODES:
            raise AdapterError(
                f"import mode {self.mode!r} is not one of {IMPORT_MODES}"
            )
        for name, value in (
            ("nbytes", self.nbytes),
            ("seconds", self.seconds),
            ("tokens_restored", self.tokens_restored),
            ("tokens_reprefilled", self.tokens_reprefilled),
        ):
            if value < 0:
                raise AdapterError(f"ImportReport.{name} must be non-negative")


class Adapter(ABC):
    name: str = ""

    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def identity(self, model: str) -> dict[str, Any]: ...

    @abstractmethod
    def export(self, req: ExportRequest) -> Path: ...

    @abstractmethod
    def import_(self, artifact: Path, req: ImportRequest) -> ImportReport: ...

    def supports(self, tier: Tier) -> bool:
        return self.capabilities().tier >= tier

    @staticmethod
    def l0_sha256(
        model_path: str,
        model_ftype: str,
        chat_template: str,
        bos_token: str,
        eos_token: str,
    ) -> str:
        payload = "\x00".join(
            (model_path, model_ftype, chat_template, bos_token, eos_token)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
