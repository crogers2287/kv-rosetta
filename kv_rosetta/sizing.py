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

#: SCKP appendix: magic, version, count.
CHECKPOINT_APPENDIX_HEADER = 12
#: Per checkpoint: int64 n_tokens, int32 pos_min, int32 pos_max.
BYTES_PER_CHECKPOINT_RECORD = 16
#: Each of data_tgt, data_dft, data_spec is written length-prefixed and interleaved with its
#: payload, so an empty buffer still costs its uint64 length.
BYTES_PER_CHECKPOINT_BUFFER_LENGTH = 8
#: data_tgt opens with eight bytes ahead of its recurrent section whose meaning has not been
#: identified. Their SIZE is confirmed on two independently produced artifacts - a 256-token
#: one-checkpoint file written here and the 2,048-token two-checkpoint file behind RA-003 -
#: and both totals come out exact. What they hold is still unknown, and this constant says so
#: rather than naming them something plausible.
CHECKPOINT_PAYLOAD_PREAMBLE = 8

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

    def layered(self) -> "LayeredGeometry":
        """The same model as a per-layer geometry: every layer identical, none sliding.

        Exists so the uniform and per-layer size laws are one law rather than two that
        happen to agree. A second implementation of the same arithmetic is a place for the
        two to drift apart, and only the artifacts of one of them would notice.
        """
        return LayeredGeometry(
            layers=tuple(LayerKV(index=index, n_kv_head=self.n_kv_head,
                                 head_dim=self.head_dim, value_head_dim=self.value_dim)
                         for index in range(self.n_layer)),
            architecture=self.architecture)


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
    # A model with more than eight layers arrives here as a TruncatedArray, not a list: the
    # reader summarises long arrays. The isinstance check below caught the four-layer case
    # and missed gemma4's forty-eight-layer one, which then reached int() and raised a bare
    # ValueError reading "invalid literal for int() with base 10: '[48 items]'" - a refusal
    # by accident, from the wrong exception type, saying nothing about geometry.
    if isinstance(kv_heads, (list, tuple, gguf.TruncatedArray)):
        raise SizingError(f"{arch} declares per-layer head_count_kv {kv_heads}; a single "
                          f"row size does not describe this model - use "
                          f"layered_geometry_of")
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
    appendix = 0
    if checkpoints:
        # Each checkpoint carries a recurrent-only state, so the fixed tail is paid again
        # per checkpoint - which is why one of them roughly doubles a short artifact.
        payload = CHECKPOINT_PAYLOAD_PREAMBLE + rec_bytes
        appendix = CHECKPOINT_APPENDIX_HEADER + checkpoints * (
            BYTES_PER_CHECKPOINT_RECORD + 3 * BYTES_PER_CHECKPOINT_BUFFER_LENGTH + payload)
    return (FILE_HEADER + BYTES_PER_HEADER_TOKEN * tokens + attn_bytes + rec_bytes
            + appendix)


#: Architectures whose loader in the pinned tree sets `swa_type = LLAMA_SWA_TYPE_STANDARD`,
#: which is the only window shape whose retained-cell count these terms can derive. The
#: window type is NOT a GGUF key - it is assigned in C++ - so it cannot be read off a model
#: file, and an architecture not listed here is refused rather than assumed standard. A
#: chunked window keeps whole chunks and a symmetric one keeps half a window either side;
#: taking one for the other silently misestimates the sliding section, which is 74% of the
#: gemma4 artifact below.
STANDARD_SWA_ARCHITECTURES = frozenset({"gemma4", "gemma4-assistant"})


@dataclass(frozen=True)
class LayerKV:
    """One layer's KV row geometry, and whether it lives in the sliding-window cache.

    `index` is the model's own layer number, not a position in a list. A sliding-window
    model's state file holds two sections, each covering a filtered subset of layers, so a
    span's position within its section is not the layer it came from - and attributing a
    tensor to the wrong layer is the failure this field exists to prevent.
    """

    index: int
    n_kv_head: int
    head_dim: int
    value_head_dim: int
    sliding: bool = False

    @property
    def n_embd_k_gqa(self) -> int:
        return self.n_kv_head * self.head_dim

    @property
    def n_embd_v_gqa(self) -> int:
        return self.n_kv_head * self.value_head_dim

    def validate(self) -> list[str]:
        return [f"layer {self.index}: {name} must be positive"
                for name in ("n_kv_head", "head_dim", "value_head_dim")
                if getattr(self, name) <= 0]


@dataclass(frozen=True)
class LayeredGeometry:
    """A model whose layers need not agree, which is the only kind gemma4 has.

    KVGeometry carries one head count and one head dimension for the whole model. gemma4-12b
    declares forty-eight of each: its sliding layers hold 8 KV heads of 256, its
    full-attention layers 1 head of 512. No scalar describes that, and any scalar chosen
    from it is wrong for forty of the forty-eight layers or for the other eight.
    """

    layers: tuple[LayerKV, ...]
    architecture: str = ""
    sliding_window: int = 0

    @property
    def n_layer(self) -> int:
        return len(self.layers)

    @property
    def base_layers(self) -> tuple[LayerKV, ...]:
        """Layers in the non-SWA cache, in the order the writer walks them."""
        return tuple(layer for layer in self.layers if not layer.sliding)

    @property
    def sliding_layers(self) -> tuple[LayerKV, ...]:
        return tuple(layer for layer in self.layers if layer.sliding)

    @property
    def is_uniform(self) -> bool:
        """Whether one row size would have described every layer after all."""
        shapes = {(layer.n_kv_head, layer.head_dim, layer.value_head_dim, layer.sliding)
                  for layer in self.layers}
        return len(shapes) <= 1

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.layers:
            problems.append("a geometry with no layers describes no cache")
        seen = [layer.index for layer in self.layers]
        if len(set(seen)) != len(seen):
            problems.append(f"layer indices {seen} repeat")
        for layer in self.layers:
            problems.extend(layer.validate())
        if self.sliding_layers and self.sliding_window <= 0:
            problems.append("sliding layers need a positive sliding_window")
        if self.sliding_window and not self.sliding_layers:
            problems.append(f"sliding_window {self.sliding_window} is declared but no "
                            f"layer is a sliding-window layer")
        return problems

    def head_layout(self) -> dict[int, tuple[int, int, int]]:
        """layer index -> (kv heads, key head dim, value head dim), for a tensor decoder."""
        return {layer.index: (layer.n_kv_head, layer.head_dim, layer.value_head_dim)
                for layer in self.layers}


def _per_layer(value, n_layer: int, key: str, arch: str) -> tuple[int, ...]:
    """One declared value broadcast, or one per layer. Never a truncated summary."""
    if isinstance(value, gguf.TruncatedArray):
        raise SizingError(
            f"{arch} declares {key} as an array of {len(value)} elements that was "
            f"summarised rather than read; re-read it with full_arrays")
    if isinstance(value, (list, tuple)):
        if len(value) != n_layer:
            raise SizingError(f"{arch} declares {len(value)} values for {key} against "
                              f"{n_layer} layers; the two cannot be matched up")
        return tuple(int(item) for item in value)
    return tuple(int(value) for _ in range(n_layer))


def layered_geometry_of(model: Path | str) -> LayeredGeometry:
    """Per-layer KV geometry from the GGUF, including the sliding-window split.

    Reads what llama.cpp reads: `attention.head_count_kv` (scalar or one per layer),
    `attention.sliding_window_pattern` (true where the layer is a sliding-window layer),
    and the two pairs of head dimensions - `key_length`/`value_length` for full-attention
    layers, `key_length_swa`/`value_length_swa` for sliding ones. Mirrors
    `llama_hparams::n_embd_k_gqa(il)`, which is `n_embd_head_k(il) * n_head_kv(il)` with
    both terms selected by `is_swa(il)`.
    """
    arch = gguf.architecture(model)
    layer_key = f"{arch}.block_count"
    kv_key = f"{arch}.attention.head_count_kv"
    pattern_key = f"{arch}.attention.sliding_window_pattern"
    window_key = f"{arch}.attention.sliding_window"
    dims = (f"{arch}.attention.key_length", f"{arch}.attention.value_length",
            f"{arch}.attention.key_length_swa", f"{arch}.attention.value_length_swa")
    read = gguf.read_metadata(model, (layer_key, kv_key, pattern_key, window_key) + dims,
                              full_arrays=(kv_key, pattern_key))
    for required in (layer_key, kv_key):
        if required not in read:
            raise SizingError(f"{model} does not declare {required}")
    n_layer = int(read[layer_key])
    if n_layer <= 0:
        raise SizingError(f"{arch} declares {n_layer} blocks")

    pattern = read.get(pattern_key)
    if pattern is None:
        sliding = (False,) * n_layer
    elif isinstance(pattern, (list, tuple)):
        if len(pattern) != n_layer:
            raise SizingError(f"{arch} declares {len(pattern)} sliding_window_pattern "
                              f"entries against {n_layer} layers")
        sliding = tuple(bool(item) for item in pattern)
    else:
        # gemma3 wrote this key as a period ("every n-th layer is full attention") and
        # gemma4 writes one flag per layer. llama.cpp's get_key_or_arr broadcasts a scalar
        # to every layer, which under the period reading would make every layer sliding.
        # The two meanings disagree about forty of gemma4-12b's forty-eight layers, so a
        # scalar is refused rather than read under either.
        raise SizingError(
            f"{arch} declares a scalar {pattern_key} ({pattern}); that key has meant both "
            f"a per-layer flag and a repeat period, and this cannot tell which")

    key_dim = read.get(dims[0])
    if key_dim is None:
        raise SizingError(f"{model} does not declare {dims[0]}; a per-layer geometry has "
                          f"no single embedding_length to divide")
    value_dim = int(read.get(dims[1]) or key_dim)
    if any(sliding):
        if read.get(dims[2]) is None:
            raise SizingError(
                f"{arch} marks {sum(sliding)} layers as sliding-window but declares no "
                f"{dims[2]}; those layers use a different head dimension and assuming the "
                f"full-attention one would misstate every one of them")
        swa_key_dim = int(read[dims[2]])
        swa_value_dim = int(read.get(dims[3]) or swa_key_dim)
    else:
        swa_key_dim = swa_value_dim = 0

    heads = _per_layer(read[kv_key], n_layer, kv_key, arch)
    layers = tuple(
        LayerKV(index=index, n_kv_head=heads[index],
                head_dim=swa_key_dim if sliding[index] else int(key_dim),
                value_head_dim=swa_value_dim if sliding[index] else value_dim,
                sliding=sliding[index])
        for index in range(n_layer))
    window = int(read.get(window_key) or 0)
    geometry = LayeredGeometry(layers=layers, architecture=arch, sliding_window=window)
    problems = geometry.validate()
    if problems:
        raise SizingError("; ".join(problems))
    return geometry


def sliding_cells(geometry: LayeredGeometry, cells: int) -> int:
    """How many cells the sliding-window section holds for a `cells`-cell sequence.

    From the writer, not from a fit: `llama_kv_cache::state_write` skips a cell when
    `is_masked_swa(n_swa, swa_type, pos, pos_max)` is true, and for LLAMA_SWA_TYPE_STANDARD
    that is `pos_max - pos >= n_swa`. Over contiguous positions exactly the last `n_swa`
    survive, so the section holds `min(cells, n_swa)` - a constant beyond the window, which
    is why a per-token rate taken from a long gemma4 artifact over-predicts a longer one.
    """
    if not geometry.sliding_layers:
        raise SizingError(f"{geometry.architecture} has no sliding-window layers, so it "
                          f"has no sliding section to count cells for")
    if geometry.architecture not in STANDARD_SWA_ARCHITECTURES:
        raise SizingError(
            f"the window shape of {geometry.architecture} is set in llama.cpp rather than "
            f"declared in the GGUF, and it is not one of the architectures verified to use "
            f"the standard window; state the sliding cell count rather than deriving it")
    if cells < 0:
        raise SizingError(f"cell count {cells} is negative")
    return min(cells, geometry.sliding_window)


def _attention_section_bytes(layers: tuple[LayerKV, ...], cells: int, *, kv_type: str,
                             per_cell: int) -> int:
    """One `llama_kv_cache::state_write` section, single stream.

    A zero-cell section is the writer's early `continue`: it emits n_stream and cell_count
    and nothing else - no meta, no v_trans/n_layer, no per-layer headers.
    """
    total = SECTION_HEADER
    if cells == 0:
        return total
    total += per_cell * cells + DATA_HEADER
    for layer in layers:
        total += 2 * BYTES_PER_TENSOR_HEADER + cells * (
            row_size(kv_type, layer.n_embd_k_gqa) + row_size(kv_type, layer.n_embd_v_gqa))
    return total


def layered_state_bytes(geometry: LayeredGeometry, cells: int, *,
                        checkpoint_cells: tuple[int, ...], sliding_cell_count: int | None = None,
                        kv_type: str = "f16", header_tokens: int | None = None) -> int:
    """Exact size of a sequence state file whose layers need not share a geometry.

    A sliding-window model does not write one attention section. `llama_kv_cache_iswa`
    holds two caches - one for the full-attention layers, one for the sliding ones - and
    `state_write` emits **the base section then the SWA section, back to back**, each
    declaring only its own filtered layer count. Treating the file as a single section with
    forty-eight layers is wrong twice over: wrong layer counts and wrong row sizes.

    `checkpoint_cells` has no default and must be stated, one entry per SCKP record giving
    that checkpoint's sliding cell count. Checkpoints are written with PARTIAL_ONLY, which
    `llama_kv_cache_iswa::state_write` honours by skipping the base cache, so a checkpoint
    payload is a sliding section and nothing else. Defaulting it to none would under-predict
    the checkpoint-persisting build's gemma4 artifact by 60% - a space guard failing open.
    """
    if geometry.architecture in HYBRID_ARCHITECTURES:
        raise SizingError(f"{geometry.architecture} is hybrid: its state continues past "
                          f"the attention sections into recurrent state; use "
                          f"hybrid_state_bytes")
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

    base, sliding = geometry.base_layers, geometry.sliding_layers
    if sliding and sliding_cell_count is None:
        raise SizingError(
            f"{geometry.architecture} has {len(sliding)} sliding-window layers whose "
            f"section holds fewer cells than the sequence; state that count, or take it "
            f"from sliding_cells()")
    if sliding_cell_count is not None and not sliding:
        raise SizingError(f"{geometry.architecture} has no sliding-window layers, so a "
                          f"sliding cell count of {sliding_cell_count} describes nothing")
    swa_cells = 0 if sliding_cell_count is None else sliding_cell_count
    if swa_cells < 0:
        raise SizingError(f"sliding cell count {swa_cells} is negative")
    if swa_cells > cells:
        raise SizingError(f"the sliding section cannot hold {swa_cells} cells out of a "
                          f"{cells}-cell sequence")
    if checkpoint_cells and not sliding:
        raise SizingError(
            f"a checkpoint of {geometry.architecture} is not a sliding section - with no "
            f"SWA cache, PARTIAL_ONLY writes the whole attention section instead, and that "
            f"payload has never been checked against a real artifact")
    if any(count < 0 or count > swa_cells for count in checkpoint_cells):
        raise SizingError(f"checkpoint cell counts {list(checkpoint_cells)} do not all lie "
                          f"between 0 and the sliding section's {swa_cells}")

    per_cell = BYTES_PER_CELL_META
    if writes_cell_ext(geometry.architecture):
        per_cell += BYTES_PER_CELL_EXT
    body = _attention_section_bytes(base, cells, kv_type=kv_type, per_cell=per_cell)
    if sliding:
        body += _attention_section_bytes(sliding, swa_cells, kv_type=kv_type,
                                         per_cell=per_cell)
    appendix = 0
    if checkpoint_cells:
        appendix = CHECKPOINT_APPENDIX_HEADER + sum(
            BYTES_PER_CHECKPOINT_RECORD + 3 * BYTES_PER_CHECKPOINT_BUFFER_LENGTH
            + CHECKPOINT_PAYLOAD_PREAMBLE
            + _attention_section_bytes(sliding, count, kv_type=kv_type, per_cell=per_cell)
            for count in checkpoint_cells)
    return FILE_HEADER + BYTES_PER_HEADER_TOKEN * tokens + body + appendix
