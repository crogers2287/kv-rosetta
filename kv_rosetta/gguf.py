"""Minimal GGUF metadata reader, and which architectures support prefix reuse.

Only the key/value header is parsed - never the tensor data - so reading a 20 GB model's
architecture costs a few kilobytes.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

MAGIC = b"GGUF"

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING = 8
_ARRAY = 9

#: Architectures llama.cpp classifies as hybrid attention + recurrent, mirroring
#: llm_arch_is_hybrid() in src/llama-arch.cpp. A recurrent state is a function of the whole
#: processed sequence, so it has no common-prefix semantics: a restored cache cannot be
#: matched against a prompt prefix the way an attention KV cache can.
HYBRID_ARCHITECTURES = frozenset({
    "jamba", "falcon-h1", "plamo2", "granitehybrid", "lfm2", "lfm2moe",
    "nemotron_h", "nemotron_h_moe", "qwen3next", "kimi-linear", "bailingmoe3",
    "kimi-k3", "qwen35", "qwen35moe", "qwen4exp", "deepseek4", "minimax-01",
})

#: Purely recurrent architectures, from llm_arch_is_recurrent().
RECURRENT_ARCHITECTURES = frozenset({
    "mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7",
})


class GGUFError(ValueError):
    pass


def read_metadata(path: Path | str, keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Read the GGUF key/value header. `keys` filters by substring; empty reads all."""
    path = Path(path)
    try:
        handle = open(path, "rb")
    except OSError as exc:
        raise GGUFError(f"cannot open {path}: {exc}") from exc
    with handle:
        if handle.read(4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file")
        struct.unpack("<I", handle.read(4))          # version
        struct.unpack("<Q", handle.read(8))          # tensor count
        n_kv = struct.unpack("<Q", handle.read(8))[0]

        def read_string() -> str:
            n = struct.unpack("<Q", handle.read(8))[0]
            return handle.read(n).decode("utf-8", "replace")

        def read_value(kind: int) -> Any:
            if kind in _SCALARS:
                fmt = _SCALARS[kind]
                return struct.unpack(fmt, handle.read(struct.calcsize(fmt)))[0]
            if kind == _STRING:
                return read_string()
            if kind == _ARRAY:
                element = struct.unpack("<I", handle.read(4))[0]
                length = struct.unpack("<Q", handle.read(8))[0]
                # Every element must be consumed even when discarded, or the reader
                # desynchronises and every later key is garbage.
                values = [read_value(element) for _ in range(length)]
                return values if length <= 8 else f"[{length} items]"
            raise GGUFError(f"unknown GGUF value type {kind}")

        out: dict[str, Any] = {}
        for _ in range(n_kv):
            key = read_string()
            kind = struct.unpack("<I", handle.read(4))[0]
            value = read_value(kind)
            if not keys or any(k in key for k in keys):
                out[key] = value
        return out


def architecture(path: Path | str) -> str:
    return str(read_metadata(path, ("general.architecture",)).get("general.architecture", ""))


def supports_prefix_reuse(arch: str) -> tuple[bool, str]:
    """Whether a restored cache of this architecture can be reused for a prompt prefix.

    Measured, then confirmed against llama.cpp's own classification: two qwen35 models (one
    MTP, one not) restored successfully - n_restored equal to the saved cell count - and
    then reported cache_n=0 on the very next completion, re-prefilling every token. The
    same test on a qwen2 model reuses the full prefix.
    """
    arch = (arch or "").strip().lower()
    if not arch:
        return False, "unknown architecture"
    if arch in RECURRENT_ARCHITECTURES:
        return False, (f"{arch} is recurrent: its state depends on the whole sequence, "
                       f"so a restored cache has no reusable prompt prefix")
    if arch in HYBRID_ARCHITECTURES:
        return False, (f"{arch} is a hybrid attention+recurrent architecture: the "
                       f"recurrent state is a function of the entire processed sequence, "
                       f"so a restored cache cannot be matched against a prompt prefix")
    return True, "ok"
