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
#: llama_kv_cell_ext: two llama_pos for the 2D position and one llama_token. Written per cell
#: only when has_cell_ext() is true.
BYTES_PER_CELL_EXT = 12
#: The recurrent section writes cell_count and nothing before it - no n_stream, unlike the
#: attention section.
RECURRENT_SECTION_HEADER = 4
#: Per recurrent cell: pos and n_seq_id. A per-sequence save writes no sequence ids.
BYTES_PER_RECURRENT_CELL_META = 8
#: s_trans, n_layer.
RECURRENT_DATA_HEADER = 8

#: has_cell_ext() is true when n_pos_per_embd() > 1, and n_pos_per_embd is decided by a
#: hardcoded switch on the architecture in llama_model_rope_type - not by any GGUF key. These
#: return M-RoPE or interleaved M-RoPE unconditionally.
CELL_EXT_ARCHITECTURES = frozenset({
    "qwen2vl", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "qwen3tts", "qwen4exp",
    "paddleocr",
})

#: These decide their rope type from model state rather than from the architecture alone, so
#: whether they write cell_ext cannot be answered from the architecture string. They are
#: refused rather than assumed either way.
CONDITIONAL_CELL_EXT_ARCHITECTURES = frozenset({"glm4", "glm4moe", "hunyuan_vl", "dflash"})


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


def writes_cell_ext(architecture: str) -> bool:
    """Whether each cell carries a 12-byte llama_kv_cell_ext.

    Refused, not guessed, for architectures whose rope type depends on model state: an
    architecture string alone cannot answer it, and 12 bytes per cell is 3 MB adrift at 256K
    cells while looking entirely reasonable at 128.
    """
    if architecture in CONDITIONAL_CELL_EXT_ARCHITECTURES:
        raise SizingError(f"{architecture} chooses its rope type from model state, so "
                          f"whether it writes cell_ext cannot be read off the architecture")
    return architecture in CELL_EXT_ARCHITECTURES


@dataclass(frozen=True)
class HybridGeometry:
    """A hybrid model's two halves. Neither describes the other."""

    attention_layers: int
    recurrent_layers: int
    n_kv_head: int
    head_dim: int
    conv_row_bytes: int
    ssm_row_bytes: int
    architecture: str = ""
    value_head_dim: int | None = None

    @property
    def attention(self) -> KVGeometry:
        return KVGeometry(n_layer=self.attention_layers, n_kv_head=self.n_kv_head,
                          head_dim=self.head_dim, architecture=self.architecture,
                          value_head_dim=self.value_head_dim)


def hybrid_geometry_of(model: Path | str) -> HybridGeometry:
    """Split a hybrid model into its attention and recurrent halves, from the GGUF.

    The layer split follows llama.cpp: layer i is recurrent unless (i+1) is a multiple of
    full_attention_interval, and the NextN/MTP layers appended past n_layer are not recurrent.
    They are also **not in the KV cache**: a decoded Qwen3.8-27B artifact reports 16 attention
    layers where the rule counted 17, and the extra one is the MTP block.
    """
    arch = gguf.architecture(model)
    if arch not in HYBRID_ARCHITECTURES:
        raise SizingError(f"{arch} is not hybrid; use geometry_of")
    keys = (f"{arch}.block_count", f"{arch}.attention.head_count_kv",
            f"{arch}.attention.key_length", f"{arch}.ssm.inner_size",
            f"{arch}.ssm.state_size", f"{arch}.ssm.conv_kernel", f"{arch}.ssm.group_count")
    optional = (f"{arch}.full_attention_interval", f"{arch}.nextn_predict_layers",
                f"{arch}.attention.value_length", f"{arch}.attention.recurrent_layers")
    found = gguf.read_metadata(model, keys + optional)
    missing = [key for key in keys if key not in found]
    if missing:
        raise SizingError(f"{model} does not declare {', '.join(missing)}")
    if found.get(f"{arch}.attention.recurrent_layers") is not None:
        raise SizingError(f"{arch} declares an explicit recurrent_layers array; this derives "
                          f"the split from full_attention_interval and would ignore it")
    # `or 4` here would turn a declared 0 into llama.cpp's default and hide it. Absent means
    # take the default; present means use what it says, or refuse it.
    declared = found.get(f"{arch}.full_attention_interval")
    interval = 4 if declared is None else int(declared)
    if interval < 1:
        raise SizingError(f"full_attention_interval {interval} is not positive")
    n_all = int(found[f"{arch}.block_count"])
    n_main = n_all - int(found.get(f"{arch}.nextn_predict_layers") or 0)
    if n_main <= 0:
        raise SizingError(f"{n_all} blocks less the NextN layers leaves {n_main}")
    recurrent = sum(1 for i in range(n_main) if (i + 1) % interval)
    attention = n_main - recurrent

    d_inner = int(found[f"{arch}.ssm.inner_size"])
    d_state = int(found[f"{arch}.ssm.state_size"])
    d_conv = int(found[f"{arch}.ssm.conv_kernel"])
    n_group = int(found[f"{arch}.ssm.group_count"])
    if min(d_inner, d_state, d_conv, n_group) <= 0:
        raise SizingError(f"{arch} declares a non-positive SSM dimension")
    key_dim = int(found[f"{arch}.attention.key_length"])
    return HybridGeometry(
        attention_layers=attention, recurrent_layers=recurrent,
        n_kv_head=int(found[f"{arch}.attention.head_count_kv"]), head_dim=key_dim,
        # f32 throughout: recurrent state is not quantised.
        conv_row_bytes=(d_conv - 1) * (d_inner + 2 * n_group * d_state) * 4,
        ssm_row_bytes=d_inner * d_state * 4, architecture=arch,
        value_head_dim=int(found.get(f"{arch}.attention.value_length") or key_dim))


def hybrid_state_bytes(geometry: HybridGeometry, cells: int, *, checkpoints: int,
                       kv_type: str = "f16", header_tokens: int | None = None,
                       recurrent_cells: int = 1) -> int:
    """Exact size of a hybrid sequence state file: header, attention section, recurrent tail.

    The recurrent tail does not grow with tokens. Measured on Qwen3.8-27B it is identical to
    the byte between a 256- and a 257-token save, and it is 90% of the file at 256 tokens -
    which is why a per-token rate taken from a hybrid artifact over-predicts so badly.

    `checkpoints` has no default and must be stated. A build carrying the context-checkpoint
    patch appends an SCKP record per checkpoint, each holding a recurrent-only state, and
    they are not small: the 2,048-token artifact behind RA-003 is 604,958,676 bytes where
    these terms give 291,169,840, and the difference is 2.0000 times the recurrent section
    plus 124 bytes. Defaulting to zero would have made this function under-predict that file
    by half - a space guard failing open, which is the direction that runs a disk out
    mid-admission.

    Only `checkpoints=0` is accepted for now. The 2.0000 multiple above is an inference from
    one file's arithmetic, not a decoded appendix, and the 124 bytes of framing are not
    accounted for at all. Predicting from it would dress a guess as a derivation, which is
    what every other term here exists to avoid.
    """
    if checkpoints < 0:
        raise SizingError(f"checkpoint count {checkpoints} is negative")
    if checkpoints:
        raise SizingError(
            f"{checkpoints} context checkpoints requested; their appendix has not been "
            f"decoded, only inferred from one file's arithmetic, and predicting from that "
            f"would dress a guess as a derivation. Decode an artifact from a patched build "
            f"first")
    attention = geometry.attention
    problems = attention.validate()
    if geometry.recurrent_layers < 0:
        problems.append("recurrent_layers must not be negative")
    if min(geometry.conv_row_bytes, geometry.ssm_row_bytes) <= 0:
        problems.append("recurrent row sizes must be positive")
    if recurrent_cells < 1:
        problems.append("recurrent_cells must be at least one")
    if problems:
        raise SizingError("; ".join(problems))
    if cells < 0:
        raise SizingError(f"cell count {cells} is negative")
    tokens = cells if header_tokens is None else header_tokens
    if tokens < 0:
        raise SizingError(f"header token count {tokens} is negative")

    per_cell = BYTES_PER_CELL_META
    if writes_cell_ext(geometry.architecture):
        per_cell += BYTES_PER_CELL_EXT
    k_row = row_size(kv_type, attention.n_embd_k_gqa)
    v_row = row_size(kv_type, attention.n_embd_v_gqa)
    attn_bytes = (SECTION_HEADER + per_cell * cells + DATA_HEADER
                  + geometry.attention_layers
                  * (2 * BYTES_PER_TENSOR_HEADER + cells * (k_row + v_row)))
    rec_bytes = (RECURRENT_SECTION_HEADER
                 + BYTES_PER_RECURRENT_CELL_META * recurrent_cells + RECURRENT_DATA_HEADER
                 + geometry.recurrent_layers
                 * (2 * BYTES_PER_TENSOR_HEADER
                    + recurrent_cells * (geometry.conv_row_bytes + geometry.ssm_row_bytes)))
    return FILE_HEADER + BYTES_PER_HEADER_TOKEN * tokens + attn_bytes + rec_bytes
