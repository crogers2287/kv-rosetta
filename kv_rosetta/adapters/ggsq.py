from __future__ import annotations

import dataclasses
import struct


class GGSQError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class LayerEntry:
    index: int
    kind: str
    ggml_type: int
    row_size: int
    n_embd_v_gqa: int
    offset: int
    nbytes: int


@dataclasses.dataclass(frozen=True)
class StreamInfo:
    index: int
    cell_count: int
    positions: tuple[int, ...]
    seq_ids: tuple[tuple[int, ...], ...]
    v_trans: bool
    n_layer: int
    layers: tuple[LayerEntry, ...]


@dataclasses.dataclass(frozen=True)
class GGSQBlob:
    n_stream: int
    streams: tuple[StreamInfo, ...]
    total_bytes: int

    @property
    def token_count(self) -> int:
        return sum(s.cell_count for s in self.streams)


GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
    40: "NVFP4",
    41: "Q1_0",
    42: "Q2_0",
    43: "COUNT",
}

DEQUANTIZABLE: frozenset[int] = frozenset({0, 1})


def _require(blob: bytes, offset: int, size: int, field: str) -> int:
    if offset < 0 or offset + size > len(blob):
        raise GGSQError(
            f"unexpected end of blob: cannot read {size}-byte field '{field}' "
            f"at offset {offset} (blob is {len(blob)} bytes)"
        )
    return offset + size


def _read_u32(blob: bytes, offset: int, field: str) -> tuple[int, int]:
    end = _require(blob, offset, 4, field)
    return struct.unpack_from("<I", blob, offset)[0], end


def _read_i32(blob: bytes, offset: int, field: str) -> tuple[int, int]:
    end = _require(blob, offset, 4, field)
    return struct.unpack_from("<i", blob, offset)[0], end


def _read_u64(blob: bytes, offset: int, field: str) -> tuple[int, int]:
    end = _require(blob, offset, 8, field)
    return struct.unpack_from("<Q", blob, offset)[0], end


def _check_data(blob: bytes, offset: int, nbytes: int, context: str) -> int:
    if offset + nbytes > len(blob):
        raise GGSQError(
            f"{context} data extends past end of blob: needs {nbytes} bytes "
            f"at offset {offset} (blob is {len(blob)} bytes)"
        )
    return offset + nbytes


def parse(blob: bytes, has_cell_ext: bool = False, cell_ext_size: int = 0) -> GGSQBlob:
    n_stream, off = _read_u32(blob, 0, "n_stream")
    streams: list[StreamInfo] = []

    for s in range(n_stream):
        cell_count, off = _read_u32(blob, off, f"stream[{s}].cell_count")

        if cell_count == 0:
            continue

        positions: list[int] = []
        seq_ids: list[tuple[int, ...]] = []
        for c in range(cell_count):
            pos, off = _read_i32(blob, off, f"stream[{s}].meta[{c}].pos")
            n_seq_id, off = _read_u32(blob, off, f"stream[{s}].meta[{c}].n_seq_id")

            if has_cell_ext:
                off = _require(
                    blob, off, cell_ext_size, f"stream[{s}].meta[{c}].cell_ext"
                )

            ids: list[int] = []
            for q in range(n_seq_id):
                sid, off = _read_i32(blob, off, f"stream[{s}].meta[{c}].seq_id[{q}]")
                ids.append(sid)
            positions.append(pos)
            seq_ids.append(tuple(ids))

        v_trans, off = _read_u32(blob, off, f"stream[{s}].data.v_trans")
        n_layer, off = _read_u32(blob, off, f"stream[{s}].data.n_layer")

        layer_entries: list[LayerEntry] = []

        for li in range(n_layer):
            k_type, off = _read_i32(
                blob, off, f"stream[{s}].data.layer[{li}].k_type"
            )
            k_size_row, off = _read_u64(
                blob, off, f"stream[{s}].data.layer[{li}].k_size_row"
            )
            nbytes = cell_count * k_size_row
            data_off = off
            off = _check_data(blob, off, nbytes, f"stream[{s}] layer[{li}] key")
            layer_entries.append(
                LayerEntry(
                    index=li,
                    kind="k",
                    ggml_type=k_type,
                    row_size=k_size_row,
                    n_embd_v_gqa=0,
                    offset=data_off,
                    nbytes=nbytes,
                )
            )

        if v_trans == 0:
            for li in range(n_layer):
                v_type, off = _read_i32(
                    blob, off, f"stream[{s}].data.layer[{li}].v_type"
                )
                v_size_row, off = _read_u64(
                    blob, off, f"stream[{s}].data.layer[{li}].v_size_row"
                )
                nbytes = cell_count * v_size_row
                data_off = off
                off = _check_data(
                    blob, off, nbytes, f"stream[{s}] layer[{li}] value"
                )
                layer_entries.append(
                    LayerEntry(
                        index=li,
                        kind="v",
                        ggml_type=v_type,
                        row_size=v_size_row,
                        n_embd_v_gqa=0,
                        offset=data_off,
                        nbytes=nbytes,
                    )
                )
        else:
            for li in range(n_layer):
                v_type, off = _read_i32(
                    blob, off, f"stream[{s}].data.layer[{li}].v_type"
                )
                v_size_el, off = _read_u32(
                    blob, off, f"stream[{s}].data.layer[{li}].v_size_el"
                )
                n_embd_v_gqa, off = _read_u32(
                    blob,
                    off,
                    f"stream[{s}].data.layer[{li}].n_embd_v_gqa",
                )
                nbytes = n_embd_v_gqa * cell_count * v_size_el
                data_off = off
                off = _check_data(
                    blob,
                    off,
                    nbytes,
                    f"stream[{s}] layer[{li}] transposed value",
                )
                layer_entries.append(
                    LayerEntry(
                        index=li,
                        kind="v",
                        ggml_type=v_type,
                        row_size=v_size_el,
                        n_embd_v_gqa=n_embd_v_gqa,
                        offset=data_off,
                        nbytes=nbytes,
                    )
                )

        streams.append(
            StreamInfo(
                index=s,
                cell_count=cell_count,
                positions=tuple(positions),
                seq_ids=tuple(seq_ids),
                v_trans=v_trans != 0,
                n_layer=n_layer,
                layers=tuple(layer_entries),
            )
        )

    return GGSQBlob(
        n_stream=n_stream,
        streams=tuple(streams),
        total_bytes=len(blob),
    )


def can_decode(blob: GGSQBlob) -> tuple[bool, str]:
    for stream in blob.streams:
        for entry in stream.layers:
            if entry.ggml_type not in DEQUANTIZABLE:
                name = GGML_TYPE_NAMES.get(
                    entry.ggml_type, f"GGML_TYPE_{entry.ggml_type}"
                )
                return (False, f"{name} is not losslessly decodable")
    return (True, "ok")
