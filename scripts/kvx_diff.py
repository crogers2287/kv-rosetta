#!/usr/bin/env python3
"""Where do two models' caches actually differ, for byte-identical input? RESEARCH ONLY.

End-to-end agreement says whether a foreign cache works. It does not say why one fails, and
it cannot tell a difference that a converter could undo from one that destroys information.
This decomposes the difference instead: same prompt, same token ids, one state file per
model, decoded to canonical (token, head, dim) float32 and compared per layer, per kind and
per head.

The comparison is designed around what a converter could be. Cosine measures direction and
is invariant to scale; the norm ratio measures scale alone; relative error measures both
together. A difference that is pure scale is a per-layer scalar away from being undone. A
difference that keeps direction but not magnitude is an affine. A difference that loses
direction is not a transform this project can build, and saying so early is the point.

Cosine is REPORTED and never admitted on - the project's standing rule, enforced by there
being no verdict in this file at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production_matrix import free_port, sha256_file  # noqa: E402
from reader_determinism import Reader  # noqa: E402

from kv_rosetta.adapters import ggsq_envelope, llamacpp_ggsq  # noqa: E402
from kv_rosetta import sizing  # noqa: E402


class DiffError(RuntimeError):
    """A refusal. Two caches that are not over identical input are not comparable."""


def decode(path: Path, *, n_head_kv: int, head_dim: int,
           has_cell_ext: bool = False) -> dict:
    """Every layer's K and V as canonical float32, plus the token ids they cover."""
    raw = path.read_bytes()
    envelope = ggsq_envelope.parse_file_envelope(raw)
    with open(path, "rb") as handle:
        section = llamacpp_ggsq.read_attention_section(
            handle, envelope.body_offset, len(raw), has_cell_ext=has_cell_ext,
            cell_ext_size=sizing.BYTES_PER_CELL_EXT if has_cell_ext else 0)
        tensors = {}
        for span in section.spans:
            tensors[(span.layer_index, span.kind)] = llamacpp_ggsq.materialise(
                handle, span, n_head=n_head_kv, head_dim=head_dim,
                cell_count=section.cell_count)
    return {"token_ids": tuple(envelope.token_ids), "cell_count": section.cell_count,
            "n_layer": section.n_layer, "tensors": tensors}


def require_same_input(a: dict, b: dict, label_a: str, label_b: str) -> None:
    """Refuse caches built over different tokens.

    Two models given different token ids produce different caches for a reason that has
    nothing to do with their weights, and every per-layer number below would be measuring
    that instead.
    """
    if a["token_ids"] != b["token_ids"]:
        shared = sum(1 for x, y in zip(a["token_ids"], b["token_ids"]) if x == y)
        raise DiffError(
            f"{label_a} and {label_b} were built over different token ids "
            f"({len(a['token_ids'])} vs {len(b['token_ids'])}, {shared} shared prefix); "
            f"their caches are not comparable")
    if a["cell_count"] != b["cell_count"]:
        raise DiffError(f"{label_a} holds {a['cell_count']} cells against {label_b}'s "
                        f"{b['cell_count']}")


def compare_tensor(x: np.ndarray, y: np.ndarray) -> dict:
    """Direction, scale and combined error, per head and pooled.

    Reported separately on purpose. A pure scale difference is undone by one number per
    layer; a direction difference is not undone by anything this project can build. A
    single "error" figure hides which of those is happening.
    """
    flat_x = x.reshape(x.shape[0], -1).astype(np.float64)
    flat_y = y.reshape(y.shape[0], -1).astype(np.float64)
    num = float((flat_x * flat_y).sum())
    den = float(np.linalg.norm(flat_x) * np.linalg.norm(flat_y))
    cosine = num / den if den else float("nan")
    norm_x, norm_y = float(np.linalg.norm(flat_x)), float(np.linalg.norm(flat_y))
    rel = float(np.linalg.norm(flat_x - flat_y) / norm_x) if norm_x else float("nan")

    per_head = []
    for head in range(x.shape[1]):
        hx = x[:, head, :].reshape(-1).astype(np.float64)
        hy = y[:, head, :].reshape(-1).astype(np.float64)
        d = float(np.linalg.norm(hx) * np.linalg.norm(hy))
        per_head.append({
            "cosine": float((hx * hy).sum() / d) if d else float("nan"),
            "norm_ratio": float(np.linalg.norm(hy) / np.linalg.norm(hx))
            if np.linalg.norm(hx) else float("nan"),
        })
    return {
        "cosine": cosine,
        "norm_ratio": norm_y / norm_x if norm_x else float("nan"),
        "relative_error": rel,
        "per_head": per_head,
    }


def parse_models(entries: list[str]) -> list[tuple[str, str]]:
    """NAME=PATH pairs; the first is the reference every other is compared against."""
    pairs = []
    for entry in entries:
        if "=" not in entry:
            raise DiffError(f"--model expects NAME=PATH, got {entry!r}")
        name, path = entry.split("=", 1)
        if not name or not path:
            raise DiffError(f"--model entry {entry!r} has an empty name or path")
        pairs.append((name, path))
    if len(pairs) < 2:
        raise DiffError("at least two models are needed to compare anything")
    names = [n for n, _ in pairs]
    if len(set(names)) != len(names):
        raise DiffError(f"duplicate model names {names}; each names a state file and the "
                        f"second would overwrite the first")
    return pairs


def capture(binary: str, model: str, prompt: str, slots: Path, name: str,
            extra: list[str], n_ctx: int, log: Path) -> dict:
    """Prefill one prompt on one model and save the slot, recording the token ids."""
    reader = Reader(binary, model, free_port(), log, extra, n_ctx, str(slots))
    reader.start()
    try:
        token_ids = reader.post("/tokenize", {"content": prompt})["tokens"]
        reader.post("/slots/0?action=erase", {})
        reader.post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0.0,
                                    "cache_prompt": True, "id_slot": 0})
        saved = reader.post("/slots/0?action=save", {"filename": f"{name}.state"})
    finally:
        reader.stop()
    return {"name": name, "model": model, "model_sha256": sha256_file(Path(model)),
            "tokenized": len(token_ids), "saved": saved}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=PATH", help="repeatable; the first is the reference")
    ap.add_argument("--n-head-kv", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--prompt-repeat", type=int, default=60)
    ap.add_argument("--out", required=True)
    # Some models do not fit without placement flags -- Flash-Next needs its
    # per-layer embedding table forced to CPU. Both models get the same extras,
    # which is what keeps the two captures comparable.
    ap.add_argument("--arch", default="",
                    help="model architecture; decides whether cells carry a 12-byte cell_ext")
    ap.add_argument("--extra", action="append", default=[],
                    help="repeatable; passed verbatim to every llama-server launch. Values starting with a dash need the equals form: --extra=--flag")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    pairs = parse_models(args.model)

    prompt = ("You are a meticulous systems engineer. " * args.prompt_repeat) + \
             "\nSummarise the invariant in one sentence."
    slots = Path(args.slots)
    slots.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)

    captures = [capture(args.binary, path, prompt, slots, name, args.extra, args.n_ctx,
                        out.with_suffix(f".{name}.log")) for name, path in pairs]

    # 12 bytes per cell that the parser must skip; misreading it turns the very
    # first cell's sequence-id count into garbage rather than failing quietly.
    has_cell_ext = sizing.writes_cell_ext(args.arch) if args.arch else False
    decoded = {name: decode(slots / f"{name}.state", n_head_kv=args.n_head_kv,
                            head_dim=args.head_dim, has_cell_ext=has_cell_ext)
               for name, _ in pairs}
    reference = pairs[0][0]
    for name, _ in pairs[1:]:
        require_same_input(decoded[reference], decoded[name], reference, name)

    comparisons = {}
    for name, _ in pairs[1:]:
        layers = []
        for layer in range(decoded[reference]["n_layer"]):
            entry = {"layer": layer}
            for kind in ("k", "v"):
                key = (layer, kind)
                if key not in decoded[reference]["tensors"]:
                    continue
                entry[kind] = compare_tensor(decoded[reference]["tensors"][key],
                                             decoded[name]["tensors"][key])
            layers.append(entry)
        comparisons[f"{reference}_vs_{name}"] = layers

    record = {
        "prompt_tokens": decoded[reference]["cell_count"],
        "geometry": {"n_head_kv": args.n_head_kv, "head_dim": args.head_dim,
                     "n_layer": decoded[reference]["n_layer"]},
        "captures": captures,
        "comparisons": comparisons,
    }
    out.write_text(json.dumps(record, indent=1))

    for pair_name, layers in comparisons.items():
        print(f"\n=== {pair_name} ===")
        for kind in ("k", "v"):
            cos = [l[kind]["cosine"] for l in layers if kind in l]
            rel = [l[kind]["relative_error"] for l in layers if kind in l]
            ratio = [l[kind]["norm_ratio"] for l in layers if kind in l]
            if not cos:
                continue
            print(f"  {kind}: cosine min={min(cos):.4f} median={sorted(cos)[len(cos)//2]:.4f} "
                  f"max={max(cos):.4f} | rel_err median={sorted(rel)[len(rel)//2]:.4f} "
                  f"| norm_ratio median={sorted(ratio)[len(ratio)//2]:.4f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
