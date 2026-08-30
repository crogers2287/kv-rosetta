#!/usr/bin/env python3
"""Can a per-layer linear map convert one model's cache into another's? RESEARCH ONLY.

The difference between two same-geometry models is not scale - `kvx_diff` measured norm
ratios of 1.000 - and it grows with depth, which is the signature of accumulation through
the residual stream rather than a fixed per-layer transform. This asks the next question
directly: fit a square per-head map on one set of prompts and score it on prompts it never
saw.

Two things this refuses to do, both learned the expensive way in research-findings §20:

  Split by prompt, never by token. Adjacent tokens share context, so a token-level split
  leaks. §20's honest median R^2 was 0.55; the leaked split reported 0.98.

  Report a fitted score without the do-nothing baseline beside it. If copying the source
  across unchanged already scores 0.95, a fit that reaches 0.96 has bought nothing, and a
  bare "R^2 = 0.96" reads as success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvx_diff import DiffError, capture, decode, parse_models, require_same_input  # noqa: E402

from kv_rosetta.mappers.linear import fit_ridge  # noqa: E402

PROMPTS = [
    "You are a meticulous systems engineer reviewing a cache format. ",
    "The harbour master kept a ledger of every vessel that entered the bay. ",
    "In functional programming a fold reduces a structure to a single value. ",
    "Photosynthesis converts light energy into chemical energy in chloroplasts. ",
    "The treaty was signed after eleven months of intermittent negotiation. ",
    "A red-black tree maintains balance through colour invariants on insertion. ",
]


def plan_split(pairs: list[tuple[str, str]], holdout: int, n_prompts: int) -> int:
    """Validate the direction and the prompt split, returning how many prompts to fit on.

    The split is by whole prompt. §20 measured what a token-level split costs: median
    held-out R^2 went from an honest 0.55 to a leaked 0.98, because adjacent tokens share
    context. There is no option here to split any other way.
    """
    if len(pairs) != 2:
        raise DiffError(f"expected exactly two models, source then target, got {len(pairs)}")
    if holdout < 1:
        raise DiffError("at least one prompt must be held out; scoring a fit on the "
                        "prompts it was fitted to measures memorisation")
    if holdout >= n_prompts:
        raise DiffError(f"holding out {holdout} of {n_prompts} prompts leaves nothing "
                        f"to fit on")
    return n_prompts - holdout


def r_squared(target: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of the target's variance the prediction accounts for.

    Against the target's own mean, so a prediction no better than "always guess the
    average vector" scores 0 rather than something flattering.
    """
    residual = float(((target - predicted) ** 2).sum())
    spread = float(((target - target.mean(axis=0)) ** 2).sum())
    if spread == 0.0:
        return float("nan")
    return 1.0 - residual / spread


def stack(decoded: list[dict], layer: int, kind: str, head: int) -> np.ndarray:
    """All tokens from the given captures for one (layer, kind, head), as (tokens, dim)."""
    parts = [d["tensors"][(layer, kind)][:, head, :] for d in decoded]
    return np.concatenate(parts, axis=0).astype(np.float64)


def evaluate_layer(source_fit: list[dict], target_fit: list[dict],
                   source_held: list[dict], target_held: list[dict],
                   layer: int, kind: str, n_head: int, ridge: float) -> dict:
    """Fit per head on the fit prompts, score on the held-out ones, with the baseline."""
    fitted, baseline = [], []
    for head in range(n_head):
        x_fit = stack(source_fit, layer, kind, head)
        y_fit = stack(target_fit, layer, kind, head)
        x_held = stack(source_held, layer, kind, head)
        y_held = stack(target_held, layer, kind, head)
        weights, bias = fit_ridge(x_fit, y_fit, ridge)
        fitted.append(r_squared(y_held, x_held @ weights + bias))
        # The do-nothing converter: hand the source across untouched.
        baseline.append(r_squared(y_held, x_held))
    return {
        "fitted_r2_mean": float(np.mean(fitted)),
        "baseline_r2_mean": float(np.mean(baseline)),
        "gain": float(np.mean(fitted) - np.mean(baseline)),
        "per_head_fitted": [float(v) for v in fitted],
        "per_head_baseline": [float(v) for v in baseline],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH",
                    help="exactly two: source then target")
    ap.add_argument("--n-head-kv", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--repeat", type=int, default=40, help="how long each prompt is")
    ap.add_argument("--holdout", type=int, default=2, help="prompts held out entirely")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs = parse_models(args.model)
    split = plan_split(pairs, args.holdout, len(PROMPTS))

    slots = Path(args.slots)
    slots.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    texts = [(seed * args.repeat) + "\nSummarise it." for seed in PROMPTS]

    captures, decoded = [], {}
    for name, path in pairs:
        per_prompt = []
        for index, text in enumerate(texts):
            tag = f"{name}-p{index}"
            captures.append(capture(args.binary, path, text, slots, tag, [], args.n_ctx,
                                    out.with_suffix(f".{tag}.log")))
            per_prompt.append(decode(slots / f"{tag}.state", n_head_kv=args.n_head_kv,
                                     head_dim=args.head_dim))
        decoded[name] = per_prompt

    source_name, target_name = pairs[0][0], pairs[1][0]
    for index in range(len(texts)):
        require_same_input(decoded[source_name][index], decoded[target_name][index],
                           f"{source_name}-p{index}", f"{target_name}-p{index}")

    n_layer = decoded[source_name][0]["n_layer"]
    results = []
    for layer in range(n_layer):
        entry = {"layer": layer}
        for kind in ("k", "v"):
            if (layer, kind) not in decoded[source_name][0]["tensors"]:
                continue
            entry[kind] = evaluate_layer(
                decoded[source_name][:split], decoded[target_name][:split],
                decoded[source_name][split:], decoded[target_name][split:],
                layer, kind, args.n_head_kv, args.ridge)
        results.append(entry)

    record = {
        "source": source_name, "target": target_name,
        "prompts_fit": split, "prompts_held_out": args.holdout,
        "split": "by whole prompt; a token-level split leaks context between adjacent rows",
        "ridge": args.ridge,
        "geometry": {"n_head_kv": args.n_head_kv, "head_dim": args.head_dim,
                     "n_layer": n_layer},
        "captures": captures,
        "layers": results,
    }
    out.write_text(json.dumps(record, indent=1))

    for kind in ("k", "v"):
        fitted = [l[kind]["fitted_r2_mean"] for l in results if kind in l]
        base = [l[kind]["baseline_r2_mean"] for l in results if kind in l]
        if not fitted:
            continue
        gains = [f - b for f, b in zip(fitted, base)]
        print(f"{kind}: fitted R2 median={sorted(fitted)[len(fitted)//2]:.4f} "
              f"min={min(fitted):.4f} | do-nothing baseline median="
              f"{sorted(base)[len(base)//2]:.4f} | gain median={sorted(gains)[len(gains)//2]:+.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
