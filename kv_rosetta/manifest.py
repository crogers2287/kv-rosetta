"""KVX manifest validation and cross-model compatibility planning.

The KVX file format is deliberately device-neutral. CUDA, HIP/ROCm, CPU, and
future backends are producers/consumers of the same canonical tensor layout;
they are not part of cache identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "kvx/0.1"
SCHEMA_LATEST = "kvx/0.2"
ACCEPTED_SCHEMAS = frozenset({"kvx/0.1", "kvx/0.2"})
LAYOUT = "layer,kv,token,head,dim"
DTYPES = {"f32", "f16", "bf16", "fp8_e4m3", "q8_0", "q4_0"}


class ManifestError(ValueError):
    pass


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{name} must be an object")
    return value


def _text(obj: dict[str, Any], key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{where}.{key} must be a non-empty string")
    return value


def _bool(obj: dict[str, Any], key: str, where: str, default: bool) -> bool:
    if key not in obj:
        return default
    value = obj[key]
    if not isinstance(value, bool):
        raise ManifestError(f"{where}.{key} must be a boolean")
    return value


def _non_negative_int(obj: dict[str, Any], key: str, where: str, default: int) -> int:
    if key not in obj:
        return default
    value = obj[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ManifestError(f"{where}.{key} must be a non-negative integer")
    return value


def _positive_int(obj: dict[str, Any], key: str, where: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManifestError(f"{where}.{key} must be a positive integer")
    return value


@dataclass(frozen=True)
class ModelABI:
    architecture: str
    weights_id: str
    tokenizer_id: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype: str
    position_kind: str
    position_params: tuple[tuple[str, Any], ...]
    rope_applied: bool = True
    kv_tokens: int = 0

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> "ModelABI":
        if data.get("schema") not in ACCEPTED_SCHEMAS:
            raise ManifestError(f"schema must be one of {sorted(ACCEPTED_SCHEMAS)}")
        model = _object(data.get("model"), "model")
        kv = _object(data.get("kv"), "kv")
        prompt = _object(data.get("prompt"), "prompt")
        position = _object(data.get("position"), "position")

        dtype = _text(kv, "dtype", "kv")
        if dtype not in DTYPES:
            raise ManifestError(f"kv.dtype must be one of {sorted(DTYPES)}")
        if kv.get("layout") != LAYOUT:
            raise ManifestError(f"kv.layout must be {LAYOUT!r}")
        if kv.get("byte_order") != "little":
            raise ManifestError("kv.byte_order must be 'little'")
        _positive_int(prompt, "token_count", "prompt")
        _text(prompt, "token_ids_sha256", "prompt")

        ignored = {"kind", "rope_applied"}
        position_params = tuple(
            sorted((key, value) for key, value in position.items() if key not in ignored)
        )
        return cls(
            architecture=_text(model, "architecture", "model"),
            weights_id=_text(model, "weights_id", "model"),
            tokenizer_id=_text(prompt, "tokenizer_id", "prompt"),
            layers=_positive_int(kv, "layers", "kv"),
            kv_heads=_positive_int(kv, "heads", "kv"),
            head_dim=_positive_int(kv, "head_dim", "kv"),
            dtype=dtype,
            position_kind=_text(position, "kind", "position"),
            position_params=position_params,
            rope_applied=_bool(position, "rope_applied", "position", True),
            kv_tokens=_non_negative_int(kv, "tokens", "kv", 0),
        )

    def fingerprint(self) -> str:
        """Identity digest.

        Deliberately covers an explicit key subset rather than ``self.__dict__``:
        fields added after kvx/0.1 (``rope_applied``, ``kv_tokens``) MUST NOT change
        the digest of a manifest that already exists in the wild.
        """
        payload = json.dumps(
            {
                "architecture": self.architecture,
                "weights_id": self.weights_id,
                "tokenizer_id": self.tokenizer_id,
                "layers": self.layers,
                "kv_heads": self.kv_heads,
                "head_dim": self.head_dim,
                "dtype": self.dtype,
                "position_kind": self.position_kind,
                "position_params": self.position_params,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def load(path: str | Path) -> tuple[dict[str, Any], ModelABI]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be an object")
    abi = ModelABI.from_manifest(data)
    producer = data.get("producer")
    if producer is not None:
        producer = _object(producer, "producer")
        _text(producer, "runtime", "producer")
        _text(producer, "backend", "producer")
        _text(producer, "device_arch", "producer")
    return data, abi


def compatibility(source: ModelABI, target: ModelABI) -> dict[str, Any]:
    if source == target:
        return {
            "mode": "exact",
            "eligible": True,
            "lossless": True,
            "requirements": ["backend import adapter", "matching prompt token IDs"],
        }

    requirements = ["paired calibration traces", "quality gate", "reprefill fallback"]
    matched_kv = source.kv_heads == target.kv_heads and source.head_dim == target.head_dim
    same_tokens = source.tokenizer_id == target.tokenizer_id
    rope_pair = source.position_kind == target.position_kind == "rope"
    if matched_kv and same_tokens and rope_pair:
        return {
            "mode": "linear-candidate",
            "eligible": True,
            "lossless": False,
            "requirements": requirements + [
                "strip source RoPE",
                "per-head top-k layer mapper",
                "apply target RoPE",
            ],
        }

    return {
        "mode": "learned-candidate",
        "eligible": True,
        "lossless": False,
        "requirements": requirements
        + (["byte-span token alignment"] if not same_tokens else [])
        + (["head/dimension projector"] if not matched_kv else [])
        + (["position encoder adapter"] if not rope_pair else []),
    }
