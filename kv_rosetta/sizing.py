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
from kv_rosetta.gguf import HYBRID_ARCHITECTURES, RECURRENT_ARCHITECTURES

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
    """What decides a state file's size, and nothing else.

    Keys and values are sized independently because the format sizes them independently -
    `n_embd_k_gqa` and `n_embd_v_gqa` are separate quantities with their own row sizes, and a
    model may declare different key and value lengths.
    """

    n_layer: int
    n_kv_head: int
    head_dim: int
    architecture: str = ""
    value_head_dim: int | None = None

    @property
    def value_dim(self) -> int:
        return self.head_dim if self.value_head_dim is None else self.value_head_dim

    @property
    def n_embd_k_gqa(self) -> int:
        """Elements in one layer's key row for one token."""
        return self.n_kv_head * self.head_dim

    @property
    def n_embd_v_gqa(self) -> int:
        return self.n_kv_head * self.value_dim

    def validate(self) -> list[str]:
        problems = [f"{name} must be positive"
                    for name in ("n_layer", "n_kv_head", "head_dim")
                    if getattr(self, name) <= 0]
        if self.value_dim <= 0:
            problems.append("value_head_dim must be positive")
        return problems


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
    extra = (f"{arch}.attention.key_length", f"{arch}.attention.value_length")
    read = gguf.read_metadata(model, wanted + extra)
    found = {key: value for key, value in read.items() if key in wanted}
    optional = {key: value for key, value in read.items() if key in extra}
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

    # key_length/value_length are authoritative when present. Deriving instead cost a
    # confident wrong answer: qwen35 declares key_length 256 while embedding_length divided
    # by head_count is 5120/24 = 213.33, which floored to 213 and produced a plausible file
    # size that was wrong by a fifth.
    explicit = optional.get(f"{arch}.attention.key_length")
    if explicit:
        key_dim = int(explicit)
    else:
        embd = int(found[f"{arch}.embedding_length"])
        if embd % int(heads):
            raise SizingError(
                f"{arch} declares neither attention.key_length nor an embedding_length "
                f"({embd}) divisible by head_count ({heads}); there is no head dimension "
                f"to read and flooring the division would invent one")
        key_dim = embd // int(heads)
    value_dim = int(optional.get(f"{arch}.attention.value_length") or key_dim)
    return KVGeometry(n_layer=int(found[f"{arch}.block_count"]), n_kv_head=int(kv_heads),
                      head_dim=key_dim, value_head_dim=value_dim, architecture=arch)


def state_bytes(geometry: KVGeometry, cells: int, *, kv_type: str = "f16",
                header_tokens: int | None = None) -> int:
    """Exact size of a non-hybrid, single-stream sequence state file.

    `header_tokens` defaults to `cells`. A slot saved after a completion can carry a few more
    token ids than cache cells, which is worth four bytes each and is why the parameter is
    exposed rather than assumed equal.

    Hybrid and recurrent models are refused. Their body continues past the attention section
    into recurrent state sized per layer, and not every layer carries attention KV at all, so
    every term here is wrong for them in two directions at once. An earlier version said this
    in prose and did not check it, which is worse than not saying it: a caller reading the
    docstring would have taken an attention-only number for a whole file.
    """
    if geometry.architecture in HYBRID_ARCHITECTURES:
        raise SizingError(
            f"{geometry.architecture} is hybrid: its state continues past the attention "
            f"section into recurrent state, and not every layer carries attention KV. These "
            f"terms describe neither, and the result would be wrong in both directions")
    if geometry.architecture in RECURRENT_ARCHITECTURES:
        raise SizingError(f"{geometry.architecture} is recurrent and has no attention KV "
                          f"cells for these terms to size")
    problems = geometry.validate()
    if problems:
        raise SizingError("; ".join(problems))
    if cells < 0:
        raise SizingError(f"cell count {cells} is negative")
    tokens = cells if header_tokens is None else header_tokens
    if tokens < 0:
        raise SizingError(f"header token count {tokens} is negative")
    k_row = row_size(kv_type, geometry.n_embd_k_gqa)
    v_row = row_size(kv_type, geometry.n_embd_v_gqa)
    body = SECTION_HEADER + BYTES_PER_CELL_META * cells + DATA_HEADER
    # Keys for every layer, then values, each preceded by its own type and row size.
    body += geometry.n_layer * (2 * BYTES_PER_TENSOR_HEADER + cells * (k_row + v_row))
    return FILE_HEADER + BYTES_PER_HEADER_TOKEN * tokens + body


def bytes_per_token(geometry: KVGeometry, *, kv_type: str = "f16") -> int:
    """The marginal cost of one more cached token, header id included.

    Derived as the difference between two adjacent totals so it cannot drift from
    state_bytes, and so it inherits the same refusals.
    """
    return (state_bytes(geometry, 1, kv_type=kv_type)
            - state_bytes(geometry, 0, kv_type=kv_type))
