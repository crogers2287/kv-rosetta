"""Minimal GGUF metadata reader, and which architectures support prefix reuse.

Only the key/value header is parsed - never the tensor data - so reading a 20 GB model's
architecture costs a few kilobytes.
"""

from __future__ import annotations

import mmap
import os
import struct
from pathlib import Path
from typing import Any

MAGIC = b"GGUF"

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING = 8
_ARRAY = 9

#: Architectures llama.cpp classifies as hybrid attention + recurrent, mirroring
#: llm_arch_is_hybrid() in src/llama-arch.cpp. These models resume from a context
#: CHECKPOINT rather than from KV cells alone, and llama.cpp's slot save does not persist
#: checkpoints (upstream issue #25913), so a restored cache reports every cell back and
#: reuses none of it. The state itself is restorable after an exact token sequence; the
#: persistence seam is what is missing.
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


class TruncatedArray:
    """An array read_metadata declined to materialise. It refuses to be compared.

    The summary used to be the plain string ``"[48 items]"``, and that string is the same
    for every 48-element array in every model. gemma4 declares its KV head count as one
    value per layer, so two different gemma4 models - different head counts, different
    layer roles - compared **equal** on that field, and scripts/cross_model_gate.py's
    same-geometry check passed a pair it exists to refuse. A summary that compares equal to
    a different model's summary is worse than no summary.

    So equality raises. There is no true answer available from a value that was discarded,
    and returning False would be as wrong as returning True: it would report two identical
    models as differing. The caller is told which key to re-read in full instead.
    """

    __slots__ = ("length", "element_type")

    def __init__(self, length: int, element_type: int) -> None:
        self.length, self.element_type = length, element_type

    def __repr__(self) -> str:
        return f"[{self.length} items]"

    __str__ = __repr__

    def __len__(self) -> int:
        return self.length

    def __eq__(self, other: object) -> bool:
        raise GGUFError(
            f"this array of {self.length} elements was summarised rather than read, so it "
            f"cannot be compared; read the key with read_metadata(..., "
            f"full_arrays=(<key>,)) if its elements are the answer")

    __hash__ = None                      # type: ignore[assignment]


def read_metadata(path: Path | str, keys: tuple[str, ...] = (), *,
                  full_arrays: tuple[str, ...] = ()) -> dict[str, Any]:
    """Read the GGUF key/value header. `keys` filters by substring; empty reads all.

    An array of more than eight elements is summarised as a TruncatedArray, because a
    tokenizer vocabulary is 150k Python strings and almost no caller wants it. `full_arrays`
    names - by the same substring match as `keys` - the arrays whose **elements are the
    answer**: a per-layer head count or sliding-window pattern is unusable as a summary, and
    was silently unusable before this existed.
    """
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

        def read_value(kind: int, *, keep: bool = False) -> Any:
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
                if keep or length <= 8:
                    return values
                return TruncatedArray(length, element)
            raise GGUFError(f"unknown GGUF value type {kind}")

        out: dict[str, Any] = {}
        for _ in range(n_kv):
            key = read_string()
            kind = struct.unpack("<I", handle.read(4))[0]
            value = read_value(kind, keep=any(k in key for k in full_arrays))
            if not keys or any(k in key for k in keys):
                out[key] = value
        return out


#: Byte width of each scalar GGUF type, so a value can be stepped over without decoding it.
_SCALAR_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class _Cursor:
    """Walks a GGUF key/value section over an mmap, decoding nothing it is not asked for.

    read_metadata() materialises every array element - a tokenizer vocabulary becomes
    150k Python strings - even when the caller wanted one key. Measured at about 0.65 s
    per call on the 27B, which is the entire cost of the fail-closed support check on the
    request path.

    Stepping over a value is arithmetic here: scalars and scalar arrays advance by a
    computed width, and string arrays walk lengths through the mapping without allocating.
    No element is turned into a Python object unless it is the value being returned.
    """

    def __init__(self, view: memoryview, size: int) -> None:
        self.view, self.size, self.offset = view, size, 0

    def take(self, count: int) -> memoryview:
        end = self.offset + count
        if count < 0 or end > self.size:
            raise GGUFError(f"metadata truncated at offset {self.offset}")
        chunk = self.view[self.offset:end]
        self.offset = end
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return bytes(self.take(self.u64())).decode("utf-8", "replace")

    def skip_string(self) -> None:
        self.take(self.u64())

    def skip_value(self, kind: int) -> None:
        if kind in _SCALAR_WIDTH:
            self.take(_SCALAR_WIDTH[kind])
        elif kind == _STRING:
            self.skip_string()
        elif kind == _ARRAY:
            element, length = self.u32(), self.u64()
            if element in _SCALAR_WIDTH:
                self.take(_SCALAR_WIDTH[element] * length)     # one step, any length
            elif element == _STRING:
                for _ in range(length):
                    self.skip_string()
            elif element == _ARRAY:
                for _ in range(length):
                    self.skip_value(_ARRAY)
            else:
                raise GGUFError(f"unknown GGUF value type {element} inside an array")
        else:
            raise GGUFError(f"unknown GGUF value type {kind}")


def read_string_key(path: Path | str, key: str, *, exhaustive: bool = False) -> str:
    """The value of one string-typed metadata key, without decoding anything else.

    Fails closed: a missing key, a non-string value, or a truncated header all raise rather
    than returning a guess.

    Two modes, because duplicate detection and request-path cost are in tension. Scanning
    every key to find a later duplicate means stepping over the tokenizer vocabulary, which
    measures about 0.4 s on the 27B even without decoding it. Returning at the first match
    measures under a millisecond, because general.architecture is written near the front.

    So `exhaustive=True` is used at admission, off the request path, where an ambiguous
    header is rejected once and never becomes an admitted object. The default early-exit
    mode is used on the request path and does NOT inspect keys after the match - that limit
    is stated here rather than left implied.
    """
    path = Path(path)
    try:
        handle = open(path, "rb")
    except OSError as exc:
        raise GGUFError(f"cannot open {path}: {exc}") from exc
    with handle:
        size = os.fstat(handle.fileno()).st_size
        if size < 24:
            raise GGUFError(f"{path} is too short to be a GGUF file")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            view = memoryview(mapped)
            try:
                cursor = _Cursor(view, size)
                if bytes(cursor.take(4)) != MAGIC:
                    raise GGUFError(f"{path} is not a GGUF file")
                cursor.u32()                                    # version
                cursor.u64()                                    # tensor count
                n_kv = cursor.u64()
                found: str | None = None
                for _ in range(n_kv):
                    name = cursor.string()
                    kind = cursor.u32()
                    if name != key:
                        cursor.skip_value(kind)
                        continue
                    if kind != _STRING:
                        raise GGUFError(
                            f"{key} is GGUF type {kind}, not a string; refusing to "
                            f"interpret it")
                    value = cursor.string()
                    if found is not None and found != value:
                        raise GGUFError(
                            f"{key} appears more than once with different values "
                            f"({found!r} then {value!r}); refusing an ambiguous header")
                    found = value
                    if not exhaustive:
                        return found
                if found is None:
                    raise GGUFError(f"{path} declares no {key}")
                return found
            finally:
                view.release()


def architecture(path: Path | str) -> str:
    """The model's declared architecture, read without decoding the rest of the header.

    Request-path form: returns at the key and does not inspect later keys. Admission uses
    architecture_exhaustive() instead, which refuses an ambiguous header.
    """
    return read_string_key(path, "general.architecture")


def architecture_exhaustive(path: Path | str) -> str:
    """As architecture(), but scans the whole header and refuses a conflicting duplicate.

    Costs a full step over the metadata section, so it belongs at admission rather than on
    the request path.
    """
    return read_string_key(path, "general.architecture", exhaustive=True)


def uses_sliding_window(path: Path | str) -> bool:
    """Does this model keep sliding-window attention state?

    Read rather than inferred from the architecture name. A sliding-window model is not
    "hybrid" by any label in this module, and it still cannot reuse a restored prefix
    unless the runtime persists checkpoints: measured on gemma4-12b, a stock build restored
    586 cells and then reused 0 of 583 tokens, while the same model on a
    checkpoint-persisting build reused 578.
    """
    md = read_metadata(path)
    arch = md.get("general.architecture")
    if not arch:
        return False
    window = md.get(f"{arch}.attention.sliding_window")
    return bool(window)


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
        return False, (f"{arch} is a hybrid attention+recurrent architecture: it resumes "
                       f"from a context checkpoint, and this runtime's slot save does not "
                       f"persist checkpoints, so a restored cache reuses nothing")
    return True, "ok"
