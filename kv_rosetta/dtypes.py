from __future__ import annotations

import numpy as np

NUMPY_DTYPE: dict[str, str] = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "uint16",
}

OPAQUE_ONLY: frozenset[str] = frozenset({"fp8_e4m3", "q8_0", "q4_0"})


class DtypeError(ValueError):
    pass


def is_canonical(name: str) -> bool:
    return name in NUMPY_DTYPE


def itemsize(name: str) -> int:
    if name not in NUMPY_DTYPE:
        raise DtypeError(f"unknown dtype: {name!r}")
    return {"f32": 4, "f16": 2, "bf16": 2}[name]


def to_numpy(buf: bytes, name: str, shape: tuple[int, ...]) -> "np.ndarray":
    if name not in NUMPY_DTYPE:
        raise DtypeError(f"unknown dtype: {name!r}")
    return np.frombuffer(buf, dtype=NUMPY_DTYPE[name]).reshape(shape)


def convert(arr: "np.ndarray", src: str, dst: str) -> "np.ndarray":
    if src not in NUMPY_DTYPE or dst not in NUMPY_DTYPE:
        raise DtypeError(f"unknown dtype: src={src!r} dst={dst!r}")
    if src == dst:
        return arr

    def to_f32(x: "np.ndarray") -> "np.ndarray":
        if src == "f32":
            return x.astype(np.float32)
        if src == "f16":
            return x.astype(np.float32)
        # bf16: widen uint16 to uint32, left-shift 16, view as float32
        u = np.asarray(x).astype(np.uint32)
        u = u << 16
        return u.view(np.float32)

    def from_f32(f32: "np.ndarray") -> "np.ndarray":
        if dst == "f32":
            return f32
        if dst == "f16":
            return f32.astype(np.float16)
        # bf16: keep the top 16 bits of the float32, rounding to nearest even rather
        # than truncating. Truncation biases every converted value toward zero, and the
        # bias compounds across a whole KV cache; RNE is the defined IEEE behaviour.
        u = np.asarray(f32, dtype=np.float32).view(np.uint32).astype(np.uint64)
        lsb = (u >> np.uint64(16)) & np.uint64(1)
        rounded = u + np.uint64(0x7FFF) + lsb
        out = (rounded >> np.uint64(16)).astype(np.uint16)
        # NaN must stay NaN: rounding can carry a NaN payload into an infinity pattern.
        nan_mask = np.isnan(np.asarray(f32, dtype=np.float32))
        if nan_mask.any():
            out = np.where(nan_mask, np.uint16(0x7FC0), out).astype(np.uint16)
        return out

    return from_f32(to_f32(arr))
