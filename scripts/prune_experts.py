"""Build a small MoE "prefill warmer" by slicing experts out of a GGUF.

The point is a model that produces a KV cache the *full* model can read. That
only holds if every field that determines cache geometry is preserved byte for
byte, so the fields below are asserted unchanged after the write rather than
merely left alone. Expert weights are sliced as raw bytes -- the expert index is
the outermost dimension, so no dequantisation is involved and no value changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402

# Changing any of these changes the shape or count of the KV cache, which is the
# one thing a warmer may never do.
CACHE_FIELDS = (
    "block_count", "embedding_length",
    "attention.head_count", "attention.head_count_kv",
    "attention.key_length", "attention.value_length",
    "full_attention_interval",
    "ssm.conv_kernel", "ssm.state_size", "ssm.group_count",
    "ssm.time_step_rank", "ssm.inner_size",
)
EXPERT_TENSOR_SUFFIXES = ("ffn_down_exps.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight")
ROUTER_SUFFIX = "ffn_gate_inp.weight"
SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")


def cache_geometry(reader: GGUFReader, arch: str) -> dict:
    out = {}
    for suffix in CACHE_FIELDS:
        field = reader.fields.get(f"{arch}.{suffix}")
        if field is not None:
            out[suffix] = field.contents()
    if not out:
        raise ValueError(f"no cache-geometry fields found for architecture {arch!r}")
    return out


def architecture(reader: GGUFReader) -> str:
    field = reader.fields.get("general.architecture")
    if field is None:
        raise ValueError("file has no general.architecture")
    return str(field.contents())


def expert_count(reader: GGUFReader, arch: str) -> int:
    field = reader.fields.get(f"{arch}.expert_count")
    if field is None:
        raise ValueError(f"{arch} declares no expert_count; this is not an MoE file")
    return int(field.contents())


def plan_slices(tensors, n_experts: int, current: int) -> dict[str, int]:
    """Map tensor name -> axis-0 length to keep, refusing anything unexpected."""
    plan = {}
    for tensor in tensors:
        if tensor.name.endswith(EXPERT_TENSOR_SUFFIXES) or tensor.name.endswith(ROUTER_SUFFIX):
            axis0 = int(tensor.data.shape[0])
            if axis0 != current:
                raise ValueError(
                    f"{tensor.name}: axis 0 is {axis0}, expected the expert count {current}. "
                    "Refusing to slice a tensor whose expert dimension is not outermost."
                )
            plan[tensor.name] = n_experts
    if not plan:
        raise ValueError("found no expert or router tensors to slice")
    return plan


def metadata_shard(readers: list[GGUFReader]) -> GGUFReader:
    """The split shard carrying general.architecture; it may hold no tensors."""
    for reader in readers:
        if reader.fields.get("general.architecture") is not None:
            return reader
    raise ValueError("no shard carries general.architecture")


def prune(src_shards: list[Path], dst: Path, n_experts: int) -> dict:
    readers = [GGUFReader(str(p)) for p in src_shards]
    meta = metadata_shard(readers)
    arch = architecture(meta)
    current = expert_count(meta, arch)
    if n_experts < 1:
        raise ValueError(f"n_experts must be >= 1, got {n_experts}")
    if n_experts > current:
        raise ValueError(f"cannot keep {n_experts} experts; the file only has {current}")
    before = cache_geometry(meta, arch)

    tensors = [t for r in readers for t in r.tensors]
    if not tensors:
        raise ValueError("no tensors found across the given shards")
    plan = plan_slices(tensors, n_experts, current)

    used_key = f"{arch}.expert_used_count"
    used_field = meta.fields.get(used_key)
    new_used = min(int(used_field.contents()), n_experts) if used_field else n_experts

    writer = GGUFWriter(str(dst), arch)
    overrides = {f"{arch}.expert_count": n_experts, used_key: new_used}
    for field in meta.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name in SPLIT_KEYS:
            continue
        if field.name in overrides:
            writer.add_key_value(field.name, overrides.pop(field.name), field.types[0])
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    for key, val in overrides.items():
        writer.add_key_value(key, val, GGUFValueType.UINT32)

    sliced = []
    for tensor in tensors:
        data = tensor.data[: plan[tensor.name]] if tensor.name in plan else tensor.data
        sliced.append((tensor, data))
        writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    kept = 0
    for tensor, data in sliced:
        writer.write_tensor_data(data)
        kept += data.nbytes
    writer.close()

    check = GGUFReader(str(dst))
    after = cache_geometry(check, architecture(check))
    if after != before:
        raise ValueError(f"cache geometry changed: {before} -> {after}")
    if expert_count(check, architecture(check)) != n_experts:
        raise ValueError("expert_count was not written correctly")
    return {"path": dst, "geometry": before, "experts_before": current,
            "experts_after": n_experts, "bytes": kept}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--experts", type=int, required=True)
    args = ap.parse_args(argv)
    report = prune(args.src, args.out, args.experts)
    print(f"cache geometry preserved: {report['geometry']}")
    print(f"experts {report['experts_before']} -> {report['experts_after']}; "
          f"wrote {report['bytes'] / 1e9:.2f} GB to {report['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
