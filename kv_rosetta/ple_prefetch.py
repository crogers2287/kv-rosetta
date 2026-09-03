"""Warm the PLE n-gram table pages a known prefix will read, before the request arrives.

Qwen3.8-Flash-Next (`qwen4exp`) carries a 28.8 GB hash-addressed embedding table
(`per_layer_token_embd.weight`) that this fleet keeps on the CPU as a memory-mapped file
and reads for every token: 16 rows of 90 bytes, one row per hash head. Addressing is
deterministic -- the rows a token reads depend only on it and its two predecessors -- so
for a prefix whose token ids are already stored (every admitted attachment carries them)
the exact pages the prefill will fault in are known before any compute runs.

Measured on this host (REQ-089): cold prefill of that table from the NAS ran at ~40 tok/s
against ~720 warm, and decode 8 against 64. This module closes that gap for KNOWN prefixes
by asking the kernel for precisely those pages, nothing more.

What it is not: it does not skip prefill, and it does not patch the table. The table is a
per-token residual at one layer; the KV cache that prefill produces is a different object.
(The "hot-swappable knowledge" work this was prompted by writes rows; this only reads.)

The hash is a port of llama.cpp's `llm_graph_input_ple::set_input` (src/models/qwen4exp.cpp
in the build that serves this fleet), cross-checked against the MIT-licensed reference in
ortegaalfredo/ngram-knowledge-injector, whose Python hash is tested against C++ golden
vectors. Any drift between those two and this file is a bug here.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from kv_rosetta import gguf

PAGE = 4096
M64 = (1 << 64) - 1

#: (block size, bytes per block) for the ggml types a PLE table plausibly ships in.
#: Kept deliberately short: an unknown type is a refusal, not a guess.
_QUANT_SIZES = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    20: (32, 18),    # IQ4_NL
    23: (256, 136),  # IQ4_XS
    30: (1, 2),      # BF16
}
_TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
               20: "IQ4_NL", 23: "IQ4_XS", 30: "BF16"}

TABLE_TENSOR = "per_layer_token_embd.weight"


class PlePrefetchError(ValueError):
    pass


# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PleConstants:
    ngram_size: int
    heads_per_ngram: int
    multipliers: tuple[int, ...]
    head_offsets: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]
    eos_token_id: int

    @property
    def n_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram


def load_constants(first_shard: Path | str) -> PleConstants:
    """Read the `<arch>.ple.*` keys. Raises PlePrefetchError for a model with no PLE table."""
    meta = gguf.read_metadata(first_shard, keys=(".ple.", "general.architecture"),
                              full_arrays=(".ple.",))
    arch = str(meta.get("general.architecture", ""))
    prefix = f"{arch}.ple."
    try:
        ngram = int(meta[prefix + "ngram_size"])
        heads = int(meta[prefix + "heads_per_ngram"])
        mult = tuple(int(x) for x in meta[prefix + "layer_multipliers"])
        offs = tuple(int(x) for x in meta[prefix + "head_offsets"])
        vocab = tuple(int(x) for x in meta[prefix + "head_vocab_sizes"])
        eos = int(meta[prefix + "eos_token_id"])
    except KeyError as exc:
        raise PlePrefetchError(f"{first_shard} ({arch or 'unknown arch'}) has no PLE table: "
                               f"missing {exc}") from exc
    n_heads = (ngram - 1) * heads
    if len(offs) != n_heads or len(vocab) != n_heads:
        raise PlePrefetchError(f"head tables have {len(offs)}/{len(vocab)} entries for "
                               f"{n_heads} heads")
    if len(mult) < ngram:
        raise PlePrefetchError(f"{len(mult)} multipliers for ngram_size {ngram}")
    return PleConstants(ngram, heads, mult, offs, vocab, eos)


# --------------------------------------------------------------------------------------
# the hash
# --------------------------------------------------------------------------------------

def rows_for_token(tok: int, prev: Sequence[int], c: PleConstants) -> list[int]:
    """The n_heads rows one token reads, given its predecessors oldest-first.

    Mirrors set_input exactly: an EOS or missing predecessor cuts everything older than
    itself to EOS; the token's own EOS does not cut its own context.
    """
    ctx = [int(tok)]
    cut = False
    for s in range(1, c.ngram_size):
        t = c.eos_token_id if (cut or len(prev) < s) else int(prev[len(prev) - s])
        cut = cut or t < 0 or t == c.eos_token_id
        ctx.append(c.eos_token_id if cut else t)
    rows: list[int] = []
    for n in range(2, c.ngram_size + 1):
        mixed = (ctx[0] * c.multipliers[0]) & M64
        for j in range(1, n):
            mixed ^= (ctx[j] * c.multipliers[j]) & M64
        base = (n - 2) * c.heads_per_ngram
        for g in range(c.heads_per_ngram):
            h = base + g
            rows.append(mixed % c.head_vocab_sizes[h] + c.head_offsets[h])
    return rows


def rows_for_sequence(tokens: Sequence[int], c: PleConstants) -> list[int]:
    """Every row the prefill of `tokens` will read, in position order (duplicates kept)."""
    n_prev = c.ngram_size - 1
    out: list[int] = []
    for i, tok in enumerate(tokens):
        prev = tokens[max(0, i - n_prev):i]
        out.extend(rows_for_token(int(tok), prev, c))
    return out


# --------------------------------------------------------------------------------------
# where the table lives on disk
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TableLocation:
    path: Path             # the shard holding the tensor
    abs_offset: int        # absolute byte offset of row 0 in that file
    n_rows: int
    row_dim: int
    ggml_type: int
    bytes_per_row: int

    @property
    def n_bytes(self) -> int:
        return self.n_rows * self.bytes_per_row


def _tensor_infos(path: Path) -> tuple[dict[str, tuple[int, int, tuple[int, ...]]], int]:
    """(name -> (type, relative offset, dims), absolute data-section start).

    The repo's gguf reader stops at the key/value section; this walks the tensor infos
    that follow it, using the same cursor so no array is decoded.
    """
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        view = memoryview(mm)
        cur = gguf._Cursor(view, size)
        try:
            if bytes(cur.take(4)) != gguf.MAGIC:
                raise PlePrefetchError(f"{path} is not a GGUF file")
            cur.u32()                          # version
            n_tensors = cur.u64()
            n_kv = cur.u64()
            alignment = 32
            for _ in range(n_kv):
                key = cur.string()
                kind = cur.u32()
                if key == "general.alignment" and kind == 4:
                    alignment = cur.u32()
                else:
                    cur.skip_value(kind)
            infos: dict[str, tuple[int, int, tuple[int, ...]]] = {}
            for _ in range(n_tensors):
                name = cur.string()
                n_dims = cur.u32()
                dims = tuple(cur.u64() for _ in range(n_dims))
                ttype = cur.u32()
                offset = cur.u64()
                infos[name] = (ttype, offset, dims)
            data_start = (cur.offset + alignment - 1) // alignment * alignment
        finally:
            # Every slice the cursor handed out is a view into `mm`; the mapping refuses
            # to close while any survive, so drop the cursor's reference and release
            # the parent view before closing.
            cur.view = None
            view.release()
            mm.close()
    return infos, data_start


def _sibling_shards(first_shard: Path) -> list[Path]:
    name = first_shard.name
    marker = "-00001-of-"
    if marker not in name:
        return [first_shard]
    stem, rest = name.split(marker, 1)
    total = int(rest.split(".")[0])
    return [first_shard.with_name(f"{stem}-{i:05d}-of-{total:05d}.gguf")
            for i in range(1, total + 1)]


def locate_table(first_shard: Path | str) -> TableLocation:
    """Find `per_layer_token_embd.weight` across the model's shards."""
    first_shard = Path(first_shard)
    for shard in _sibling_shards(first_shard):
        if not shard.is_file():
            raise PlePrefetchError(f"shard missing: {shard}")
        infos, data_start = _tensor_infos(shard)
        if TABLE_TENSOR not in infos:
            continue
        ttype, rel, dims = infos[TABLE_TENSOR]
        if ttype not in _QUANT_SIZES:
            raise PlePrefetchError(f"table type {ttype} is not one this module sizes")
        if len(dims) != 2:
            raise PlePrefetchError(f"table has {len(dims)} dims, expected 2")
        row_dim, n_rows = int(dims[0]), int(dims[1])
        blck, per_block = _QUANT_SIZES[ttype]
        if row_dim % blck:
            raise PlePrefetchError(f"row_dim {row_dim} not divisible by block {blck}")
        bpr = row_dim // blck * per_block
        loc = TableLocation(shard, data_start + rel, n_rows, row_dim, ttype, bpr)
        if loc.abs_offset + loc.n_bytes > shard.stat().st_size:
            raise PlePrefetchError(f"table runs past the end of {shard}")
        return loc
    raise PlePrefetchError(f"{TABLE_TENSOR} not found in any shard of {first_shard}")


# --------------------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------------------

def pages_for_rows(rows: Iterable[int], loc: TableLocation) -> list[int]:
    """Sorted, unique 4 KB page offsets (absolute, in loc.path) covering every row."""
    pages: set[int] = set()
    n_rows = loc.n_rows
    for r in rows:
        if not 0 <= r < n_rows:
            raise PlePrefetchError(f"row {r} outside table of {n_rows} rows")
        start = loc.abs_offset + r * loc.bytes_per_row
        end = start + loc.bytes_per_row - 1
        pages.add(start // PAGE * PAGE)
        pages.add(end // PAGE * PAGE)         # a row may straddle a page boundary
    return sorted(pages)


def coalesce(pages: Sequence[int]) -> list[tuple[int, int]]:
    """Merge adjacent pages into (offset, length) runs, so one fadvise covers a run."""
    runs: list[tuple[int, int]] = []
    for p in pages:
        if runs and runs[-1][0] + runs[-1][1] == p:
            runs[-1] = (runs[-1][0], runs[-1][1] + PAGE)
        else:
            runs.append((p, PAGE))
    return runs


# --------------------------------------------------------------------------------------
# kernel calls
# --------------------------------------------------------------------------------------

def advise(path: Path, runs: Sequence[tuple[int, int]]) -> float:
    """posix_fadvise(WILLNEED) each run. Advisory, unprivileged, no resident cost here."""
    started = time.time()
    fd = os.open(str(path), os.O_RDONLY)
    try:
        for offset, length in runs:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_WILLNEED)
    finally:
        os.close(fd)
    return time.time() - started


def residency(path: Path, pages: Sequence[int]) -> tuple[int, int]:
    """(resident, total) for the given page offsets, via mincore on a private mapping.

    Measurement only. It maps the file read-only and asks the kernel which of those pages
    are in the page cache; nothing is faulted in.
    """
    if not pages:
        return 0, 0
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    size = os.path.getsize(path)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_PRIVATE, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise PlePrefetchError(f"mmap failed: errno {ctypes.get_errno()}")
        try:
            n_pages = (size + PAGE - 1) // PAGE
            vec = ctypes.create_string_buffer(n_pages)
            if libc.mincore(addr, size, vec) != 0:
                raise PlePrefetchError(f"mincore failed: errno {ctypes.get_errno()}")
            raw = vec.raw
            resident = sum(1 for p in pages if raw[p // PAGE] & 1)
            return resident, len(pages)
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------------------
# the operation
# --------------------------------------------------------------------------------------

@dataclass
class PrefetchReport:
    tokens: int
    rows: int
    pages: int
    runs: int
    bytes: int
    advise_seconds: float
    resident_before: int | None = None
    resident_after: int | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def warm_for_tokens(first_shard: Path | str, token_ids: Sequence[int], *,
                    measure: bool = False) -> PrefetchReport:
    """Prefetch every table page the prefill of `token_ids` will read."""
    if not token_ids:
        raise PlePrefetchError("no token ids to prefetch for")
    c = load_constants(first_shard)
    loc = locate_table(first_shard)
    rows = rows_for_sequence(token_ids, c)
    pages = pages_for_rows(rows, loc)
    runs = coalesce(pages)
    before = residency(loc.path, pages)[0] if measure else None
    took = advise(loc.path, runs)
    after = residency(loc.path, pages)[0] if measure else None
    return PrefetchReport(tokens=len(token_ids), rows=len(rows), pages=len(pages),
                          runs=len(runs), bytes=len(pages) * PAGE, advise_seconds=took,
                          resident_before=before, resident_after=after)


def has_ple_table(first_shard: Path | str) -> bool:
    """Cheap gate for the restore path: only qwen4exp-style models have anything to warm."""
    try:
        load_constants(first_shard)
        return True
    except (PlePrefetchError, gguf.GGUFError, OSError):
        return False
