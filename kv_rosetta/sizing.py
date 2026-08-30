"""How large a saved sequence state will be, derived from the format rather than fitted.

The sidecar has to answer "will this fit" before it generates anything, and a projection that
over-predicts by 2.4x either wastes space or refuses work that would have succeeded. RA-003 is
open on exactly that question.

Every term here is read off the writer in `llama-kv-cache.cpp` and confirmed against real
artifacts byte for byte. Nothing is fitted. The distinction matters: a line drawn through two
points reproduces those two points by construction, and only a prediction far outside the
range it came from is evidence of anything.

One term is easy to miss and was: llama.cpp stores the prompt's **token ids** in the file
header, four bytes each, so each additional cached token costs four bytes beyond its cache
entry. Accounting for the cells alone lands 4 bytes per token short - invisible at 128 tokens,
128 KB adrift at 32,000.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kv_rosetta import gguf
from kv_rosetta.adapters.llamacpp_ggsq import GGML_TYPES

#: magic, version, n_token_count.
FILE_HEADER = 12
#: Each cached token's id, stored in the header ahead of the body.
BYTES_PER_HEADER_TOKEN = 4
#: n_stream, then cell_count for the single stream.
SECTION_HEADER = 8
#: Per cell: pos (int32), n_seq_id (uint32), one seq_id (int32). `cell_ext` is written only
#: when has_cell_ext() is true - M-RoPE or PLE - and is not covered here.
BYTES_PER_CELL_META = 12
#: v_trans, n_layer.
DATA_HEADER = 8
#: Per tensor: type id (int32) then row size (uint64). Written once per layer for keys and
#: again for values.
BYTES_PER_TENSOR_HEADER = 12


class SizingError(ValueError):
    """Raised when a size cannot be derived. Never an approximation."""


@dataclass(frozen=True)
class KVGeometry:
    """What decides a state file's size, and nothing else."""

    n_layer: int
    n_kv_head: int
    head_dim: int
    architecture: str = ""

    @property
    def n_embd_kv(self) -> int:
        """Elements in one layer's key row for one token."""
        return self.n_kv_head * self.head_dim

    def validate(self) -> list[str]:
        return [f"{name} must be positive" for name in ("n_layer", "n_kv_head", "head_dim")
                if getattr(self, name) <= 0]


def row_size(type_name: str, n_elements: int) -> int:
    """ggml_row_size: whole blocks only, so a row must be a multiple of the block size."""
    for name, block, per_block in GGML_TYPES.values():
        if name != type_name:
            continue
        if n_elements % block:
            raise SizingError(f"{n_elements} elements is not a whole number of {type_name} "
                              f"blocks of {block}")
        return n_elements // block * per_block
    raise SizingError(f"unknown ggml type {type_name!r}")


def geometry_of(model: Path | str) -> KVGeometry:
    """Read the geometry from the GGUF. Nothing here is defaulted or guessed."""
    arch = gguf.architecture(model)
    wanted = (f"{arch}.block_count", f"{arch}.attention.head_count",
              f"{arch}.attention.head_count_kv", f"{arch}.embedding_length")
    found = gguf.read_metadata(model, wanted)
    missing = [key for key in wanted if key not in found]
    if missing:
        raise SizingError(f"{model} does not declare {', '.join(missing)}")
    heads = found[f"{arch}.attention.head_count"]
    kv_heads = found[f"{arch}.attention.head_count_kv"]
    if isinstance(kv_heads, (list, tuple)):
        raise SizingError(f"{arch} declares per-layer head_count_kv {list(kv_heads)}; a "
                          f"single row size does not describe this model")
    if not heads:
        raise SizingError(f"{arch} declares head_count {heads}")
    return KVGeometry(n_layer=int(found[f"{arch}.block_count"]), n_kv_head=int(kv_heads),
                      head_dim=int(found[f"{arch}.embedding_length"]) // int(heads),
                      architecture=arch)


def state_bytes(geometry: KVGeometry, cells: int, *, kv_type: str = "f16",
                header_tokens: int | None = None) -> int:
    """Exact size of a non-hybrid, single-stream sequence state file.

    `header_tokens` defaults to `cells`. A slot saved after a completion can carry a few more
    token ids than cache cells, which is worth four bytes each and is why the parameter is
    exposed rather than assumed equal.

    Hybrid models are refused rather than approximated: their body continues into recurrent
    state sized per layer, which none of these terms describe.
    """
    problems = geometry.validate()
    if problems:
        raise SizingError("; ".join(problems))
    if cells < 0:
        raise SizingError(f"cell count {cells} is negative")
    tokens = cells if header_tokens is None else header_tokens
    if tokens < 0:
        raise SizingError(f"header token count {tokens} is negative")
    per_row = row_size(kv_type, geometry.n_embd_kv)
    body = SECTION_HEADER + BYTES_PER_CELL_META * cells + DATA_HEADER
    # Keys for every layer, then values, each preceded by its type and row size.
    body += 2 * geometry.n_layer * (BYTES_PER_TENSOR_HEADER + cells * per_row)
    return FILE_HEADER + BYTES_PER_HEADER_TOKEN * tokens + body


def bytes_per_token(geometry: KVGeometry, *, kv_type: str = "f16") -> int:
    """The marginal cost of one more cached token, header id included."""
    return (BYTES_PER_HEADER_TOKEN + BYTES_PER_CELL_META
            + 2 * geometry.n_layer * row_size(kv_type, geometry.n_embd_kv))
