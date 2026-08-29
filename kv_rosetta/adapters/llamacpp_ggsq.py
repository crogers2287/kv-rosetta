"""Fail-closed reader for the ggsq/3 body. First slice: exact section bounds.

`docs/ggsq-layout.md` derived this layout from the writer in the pinned tree. Nothing here
scans for a magic or infers an offset from a pattern: the SCKP record was assumed to be 12
bytes when the writer emits 16, and every test passed against that assumption for a day. So
each field is read at a computed offset, every advance is bounds-checked against the declared
section, and a body with bytes left over is refused rather than accepted as "close enough".

This slice locates and validates the sections. It does not yet materialise tensors.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO

#: ggml type id -> (name, block size, bytes per block). Only types this decoder can account
#: for are listed; an unlisted id is refused rather than guessed at, because a wrong element
#: width silently desynchronises every later field.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("f32", 1, 4),
    1: ("f16", 1, 2),
    2: ("q4_0", 32, 18),
    3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22),
    7: ("q5_1", 32, 24),
    8: ("q8_0", 32, 34),
    9: ("q8_1", 32, 40),
    30: ("bf16", 1, 2),
}


class GGSQError(ValueError):
    """Raised when the body cannot be read exactly. Never a warning, never a guess."""


def row_size(type_id: int, n_elements: int) -> int:
    """ggml_row_size: elements are stored in blocks, so this is not n * itemsize."""
    if type_id not in GGML_TYPES:
        raise GGSQError(f"unsupported ggml type id {type_id}; refusing to guess its width")
    _, block, per_block = GGML_TYPES[type_id]
    if n_elements % block:
        raise GGSQError(f"{n_elements} elements is not a multiple of the {block}-element "
                        f"block for type {GGML_TYPES[type_id][0]}")
    return n_elements // block * per_block


def elements_from_row_size(type_id: int, nbytes: int) -> int:
    """Invert row_size. The file records the stride, not the width, so this recovers it."""
    if type_id not in GGML_TYPES:
        raise GGSQError(f"unsupported ggml type id {type_id}")
    _, block, per_block = GGML_TYPES[type_id]
    if nbytes % per_block:
        raise GGSQError(f"row size {nbytes} is not a multiple of the {per_block}-byte block "
                        f"for type {GGML_TYPES[type_id][0]}")
    return nbytes // per_block * block


@dataclass
class Reader:
    """Bounded cursor. Every read is checked against the end of the region."""

    handle: BinaryIO
    start: int
    end: int
    offset: int = 0

    def __post_init__(self) -> None:
        self.offset = self.start

    def take(self, count: int) -> bytes:
        if count < 0:
            raise GGSQError(f"negative read of {count} bytes")
        if self.offset + count > self.end:
            raise GGSQError(f"read of {count} bytes at {self.offset} passes the end of the "
                            f"section at {self.end}")
        self.handle.seek(self.offset)
        block = self.handle.read(count)
        if len(block) != count:
            raise GGSQError(f"file ended {count - len(block)} bytes short at {self.offset}")
        self.offset += count
        return block

    def skip(self, count: int) -> int:
        """Advance without reading. Payloads are located, not materialised."""
        if count < 0:
            raise GGSQError(f"negative skip of {count} bytes")
        if self.offset + count > self.end:
            raise GGSQError(f"skip of {count} bytes at {self.offset} passes the end of the "
                            f"section at {self.end}")
        position = self.offset
        self.offset += count
        return position

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]


@dataclass(frozen=True)
class TensorSpan:
    """Where one layer's payload lives, and how to interpret it. Bytes stay on disk."""

    layer_index: int
    kind: str                 # "k", "v", "r", "s"
    type_id: int
    type_name: str
    row_size: int
    n_elements: int
    offset: int
    nbytes: int
    transposed: bool = False


@dataclass
class AttentionSection:
    n_stream: int
    cell_count: int
    v_trans: bool
    n_layer: int
    positions: tuple[int, ...] = field(default_factory=tuple)
    spans: tuple[TensorSpan, ...] = field(default_factory=tuple)
    end_offset: int = 0


def read_attention_section(handle: BinaryIO, start: int, end: int, *,
                           has_cell_ext: bool, cell_ext_size: int = 0) -> AttentionSection:
    """Parse one attention KV section and locate every tensor payload.

    `has_cell_ext` is not discoverable from the file - it is an architecture property, and
    guessing it desynchronises the whole parse - so the caller must supply it from the
    resolved source geometry.
    """
    reader = Reader(handle, start, end)
    n_stream = reader.u32()
    if not 1 <= n_stream <= 64:
        raise GGSQError(f"implausible n_stream {n_stream}")
    if n_stream != 1:
        raise GGSQError(f"n_stream {n_stream} is unsupported by this decoder; only a single "
                        f"stream has been exercised against a real artifact")

    cell_count = reader.u32()
    if cell_count == 0:
        return AttentionSection(n_stream, 0, False, 0, end_offset=reader.offset)

    positions = []
    for index in range(cell_count):
        pos = reader.i32()
        n_seq_id = reader.u32()
        if n_seq_id > 64:
            raise GGSQError(f"cell {index} claims {n_seq_id} sequence ids")
        if has_cell_ext:
            reader.skip(cell_ext_size)
        reader.skip(n_seq_id * 4)
        positions.append(pos)

    v_trans = bool(reader.u32())
    n_layer = reader.u32()
    if not 1 <= n_layer <= 512:
        raise GGSQError(f"implausible n_layer {n_layer}")

    spans: list[TensorSpan] = []
    for layer in range(n_layer):
        type_id = reader.i32()
        stride = reader.u64()
        nbytes = cell_count * stride
        offset = reader.skip(nbytes)
        spans.append(TensorSpan(
            layer_index=layer, kind="k", type_id=type_id,
            type_name=GGML_TYPES.get(type_id, ("?", 0, 0))[0], row_size=stride,
            n_elements=elements_from_row_size(type_id, stride),
            offset=offset, nbytes=nbytes))

    for layer in range(n_layer):
        type_id = reader.i32()
        if not v_trans:
            stride = reader.u64()
            nbytes = cell_count * stride
            offset = reader.skip(nbytes)
            spans.append(TensorSpan(
                layer_index=layer, kind="v", type_id=type_id,
                type_name=GGML_TYPES.get(type_id, ("?", 0, 0))[0], row_size=stride,
                n_elements=elements_from_row_size(type_id, stride),
                offset=offset, nbytes=nbytes))
        else:
            size_el = reader.u32()
            n_embd_v_gqa = reader.u32()
            if type_id not in GGML_TYPES:
                raise GGSQError(f"unsupported ggml type id {type_id}")
            if size_el != GGML_TYPES[type_id][2] // GGML_TYPES[type_id][1]:
                raise GGSQError(
                    f"declared element size {size_el} disagrees with type "
                    f"{GGML_TYPES[type_id][0]}; refusing a header that contradicts itself")
            nbytes = n_embd_v_gqa * cell_count * size_el
            offset = reader.skip(nbytes)
            spans.append(TensorSpan(
                layer_index=layer, kind="v", type_id=type_id,
                type_name=GGML_TYPES[type_id][0], row_size=size_el * cell_count,
                n_elements=n_embd_v_gqa, offset=offset, nbytes=nbytes, transposed=True))

    return AttentionSection(n_stream=n_stream, cell_count=cell_count, v_trans=v_trans,
                            n_layer=n_layer, positions=tuple(positions),
                            spans=tuple(spans), end_offset=reader.offset)


@dataclass
class RecurrentSection:
    """R and S state for the layers that carry it."""

    n_stream: int
    cell_count: int
    s_trans: bool
    n_layer: int
    positions: tuple[int, ...] = field(default_factory=tuple)
    spans: tuple[TensorSpan, ...] = field(default_factory=tuple)
    end_offset: int = 0


def read_recurrent_section(handle: BinaryIO, start: int, end: int, *,
                           recurrent_layers: tuple[int, ...],
                           single_sequence: bool = True) -> RecurrentSection:
    """Parse the recurrent state section.

    `recurrent_layers` is required and cannot be derived from the file. The writer skips
    null layers with a bare `continue`, recording only the total layer count, so the number
    of R and S blocks present is unrecoverable from the bytes. Walking until the section ends
    would appear to work - the recurrent section is last - but it would silently accept a
    wrong layer attribution, which is the failure mode this decoder exists to prevent.

    `single_sequence` reflects the writer's `n_seq_id = seq_id == -1 ? size : 0`: a
    per-sequence save writes zero sequence ids, unlike the attention meta.
    """
    reader = Reader(handle, start, end)
    n_stream = reader.u32()
    if n_stream != 1:
        raise GGSQError(f"recurrent n_stream {n_stream} is unsupported")
    cell_count = reader.u32()
    if cell_count == 0:
        return RecurrentSection(n_stream, 0, False, 0, end_offset=reader.offset)

    positions = []
    for index in range(cell_count):
        pos = reader.i32()
        n_seq_id = reader.u32()
        if single_sequence and n_seq_id != 0:
            raise GGSQError(
                f"recurrent cell {index} declares {n_seq_id} sequence ids; a per-sequence "
                f"save writes none, so this body was not written for one sequence")
        if n_seq_id > 64:
            raise GGSQError(f"recurrent cell {index} claims {n_seq_id} sequence ids")
        reader.skip(n_seq_id * 4)
        positions.append(pos)

    s_trans = bool(reader.u32())
    if s_trans:
        raise GGSQError("transposed recurrent S state is unsupported by this decoder; the "
                        "writer emits s_trans=0 and the transposed path has never been "
                        "exercised against a real artifact")
    n_layer = reader.u32()
    if not 1 <= n_layer <= 512:
        raise GGSQError(f"implausible recurrent n_layer {n_layer}")
    if any(not 0 <= i < n_layer for i in recurrent_layers):
        raise GGSQError(f"recurrent layer map {recurrent_layers} references a layer outside "
                        f"0..{n_layer - 1}")
    if len(set(recurrent_layers)) != len(recurrent_layers):
        raise GGSQError("recurrent layer map repeats a layer")

    spans: list[TensorSpan] = []
    for kind in ("r", "s"):
        for layer in recurrent_layers:
            type_id = reader.i32()
            stride = reader.u64()
            nbytes = cell_count * stride
            offset = reader.skip(nbytes)
            spans.append(TensorSpan(
                layer_index=layer, kind=kind, type_id=type_id,
                type_name=GGML_TYPES.get(type_id, ("?", 0, 0))[0], row_size=stride,
                n_elements=elements_from_row_size(type_id, stride),
                offset=offset, nbytes=nbytes))

    return RecurrentSection(n_stream=n_stream, cell_count=cell_count, s_trans=s_trans,
                            n_layer=n_layer, positions=tuple(positions),
                            spans=tuple(spans), end_offset=reader.offset)


@dataclass
class CheckpointBlob:
    """One SCKP record. Its target payload is itself a recurrent state serialization."""

    n_tokens: int
    pos_min: int
    pos_max: int
    target: RecurrentSection
    target_offset: int
    target_nbytes: int


def read_checkpoint_appendix(handle: BinaryIO, start: int, end: int, *,
                             recurrent_layers: tuple[int, ...]) -> list[CheckpointBlob]:
    """Parse the SCKP appendix, decoding each target payload rather than forwarding it.

    `data_tgt` is another native recurrent-state serialization, written with PARTIAL_ONLY.
    Carrying it forward as an opaque blob would put an unparsed native payload inside a
    canonical artifact - a portability claim the format cannot honour - so it is parsed
    recursively here. Nonempty draft and speculative payloads are refused outright: their
    restoration has never been behaviourally verified.
    """
    reader = Reader(handle, start, end)
    if bytes(reader.take(4)) != b"SCKP":
        raise GGSQError(f"no SCKP magic at offset {start}")
    version = reader.u32()
    if version != 1:
        raise GGSQError(f"checkpoint appendix version {version} is not 1")
    count = reader.u32()
    if count > 1024:
        raise GGSQError(f"checkpoint count {count} exceeds the writer's own cap of 1024")

    blobs: list[CheckpointBlob] = []
    for index in range(count):
        record = reader.take(16)                 # int64 n_tokens, int32 pos_min, pos_max
        n_tokens, pos_min, pos_max = struct.unpack("<qii", record)
        if n_tokens < 0 or pos_min < 0 or pos_max < pos_min:
            raise GGSQError(f"checkpoint {index} has impossible extent "
                            f"n_tokens={n_tokens} pos={pos_min}..{pos_max}")
        payloads = []
        for name in ("data_tgt", "data_dft", "data_spec"):
            length = reader.u64()
            if length > (1 << 34):
                raise GGSQError(f"checkpoint {index} {name} claims {length} bytes")
            offset = reader.skip(length)
            payloads.append((name, offset, length))
        for name, _offset, length in payloads[1:]:
            if length:
                raise GGSQError(
                    f"checkpoint {index} carries {length} bytes of {name}; draft and "
                    f"speculative state restoration is not behaviourally proven, so it is "
                    f"refused rather than forwarded")
        _, tgt_offset, tgt_length = payloads[0]
        if tgt_length == 0:
            raise GGSQError(f"checkpoint {index} has no target state")
        target = read_recurrent_section(handle, tgt_offset, tgt_offset + tgt_length,
                                        recurrent_layers=recurrent_layers)
        if target.end_offset != tgt_offset + tgt_length:
            raise GGSQError(
                f"checkpoint {index} target state ends at {target.end_offset}, not at its "
                f"declared end {tgt_offset + tgt_length}")
        blobs.append(CheckpointBlob(n_tokens=n_tokens, pos_min=pos_min, pos_max=pos_max,
                                    target=target, target_offset=tgt_offset,
                                    target_nbytes=tgt_length))
    if reader.offset != end:
        raise GGSQError(f"checkpoint appendix ends at {reader.offset}, not at {end}")
    return blobs
