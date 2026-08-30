#!/usr/bin/env python3
"""How many layers can come from a foreign model before the output goes wrong? RESEARCH ONLY.

§29 measured that two same-geometry models agree almost exactly in their early layers and
drift apart with depth, and that no linear map recovers the drift. That suggests the damage
is concentrated in the deep layers rather than spread evenly, and if so a cache does not
have to be used whole.

This tests it directly by building hybrid state files: layers below a cutoff taken from the
foreign model, layers at or above it from the target's own cache, spliced byte-for-byte.
Sweeping the cutoff from 0 to n_layer walks from "entirely the target's own cache" to
"entirely foreign", and both endpoints are already-known quantities, so the curve is
anchored at both ends rather than floating.

This is a diagnostic, not a shipping feature. llama.cpp restores a state file whole; there
is no partial restore, so a production version would need the runtime to reuse some layers
and recompute the rest. What this establishes is whether that would be worth building.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_model_gate import (  # noqa: E402
    GateError, compare_forced, teacher_forced, tokenize,
)
from kvx_diff import capture  # noqa: E402
from production_matrix import free_port, toks  # noqa: E402
from reader_determinism import Reader  # noqa: E402

from kv_rosetta.adapters import ggsq_envelope, llamacpp_ggsq  # noqa: E402


def read_spans(path: Path) -> tuple[bytes, list]:
    raw = path.read_bytes()
    envelope = ggsq_envelope.parse_file_envelope(raw)
    with open(path, "rb") as handle:
        section = llamacpp_ggsq.read_attention_section(
            handle, envelope.body_offset, len(raw), has_cell_ext=False)
    return raw, list(section.spans)


def require_spliceable(spans_a: list, spans_b: list, size_a: int, size_b: int) -> None:
    """Refuse two files whose payloads are not laid out identically.

    A splice copies bytes by offset. If the two files disagree on any span's position,
    length, quantisation type or row size, the copy lands in the wrong place and produces a
    file that is structurally valid, restorable, and silently wrong - the worst outcome
    available. Checked field by field rather than trusting that equal geometry implies
    equal layout.
    """
    if size_a != size_b:
        raise GateError(f"state files differ in size ({size_a} vs {size_b}); they do not "
                        f"describe the same cache shape")
    if len(spans_a) != len(spans_b):
        raise GateError(f"{len(spans_a)} spans against {len(spans_b)}")
    for index, (x, y) in enumerate(zip(spans_a, spans_b)):
        for field in ("layer_index", "kind", "type_id", "row_size", "offset", "nbytes",
                      "transposed"):
            if getattr(x, field) != getattr(y, field):
                raise GateError(
                    f"span {index} differs in {field}: {getattr(x, field)!r} against "
                    f"{getattr(y, field)!r}; a byte splice would land in the wrong place")


def foreign_layers(count: int, n_layer: int, direction: str) -> set[int]:
    """Which layer indices come from the foreign cache.

    Both directions are needed because the two effects run opposite ways. An early layer
    differs by very little but its difference propagates through every layer above it; a
    late layer differs by much more but influences only what is left. Sweeping only one end
    would measure their sum and conclude the cost is simply linear in layer count.
    """
    if count < 0:
        raise GateError(f"count {count} is negative")
    if count > n_layer:
        raise GateError(f"count {count} exceeds {n_layer} layers")
    if direction == "early":
        return set(range(count))
    if direction == "late":
        return set(range(n_layer - count, n_layer))
    raise GateError(f"direction must be 'early' or 'late', got {direction!r}")


def splice(raw_foreign: bytes, raw_own: bytes, spans: list, layers: set[int]) -> bytes:
    """The named layers from the foreign cache, every other layer from the target's own.

    An empty set yields the target's own file unchanged and the full set yields the foreign
    one, so both endpoints of any sweep are quantities already measured elsewhere.
    """
    out = bytearray(raw_own)
    replaced = 0
    for span in spans:
        if span.layer_index in layers:
            out[span.offset:span.offset + span.nbytes] = \
                raw_foreign[span.offset:span.offset + span.nbytes]
            replaced += 1
    return bytes(out), replaced


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--foreign-model", required=True)
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--prompt-repeat", type=int, default=220)
    ap.add_argument("--forced-positions", type=int, default=64)
    ap.add_argument("--step", type=int, default=4, help="cutoff sweep step")
    ap.add_argument("--direction", choices=("early", "late"), default="early",
                    help="take the foreign layers from the bottom or the top of the stack")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = ("You are a meticulous systems engineer working on a portable KV cache "
              "format. " * args.prompt_repeat) + "\nState the single most important invariant."
    slots = Path(args.slots)
    slots.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)

    captures = [
        capture(args.binary, args.foreign_model, prompt, slots, "foreign", [], args.n_ctx,
                out.with_suffix(".foreign.log")),
        capture(args.binary, args.target_model, prompt, slots, "own", [], args.n_ctx,
                out.with_suffix(".own.log")),
    ]
    raw_foreign, spans_foreign = read_spans(slots / "foreign.state")
    raw_own, spans_own = read_spans(slots / "own.state")
    require_spliceable(spans_foreign, spans_own, len(raw_foreign), len(raw_own))

    n_layer = max(s.layer_index for s in spans_own) + 1
    cutoffs = sorted(set(list(range(0, n_layer + 1, args.step)) + [n_layer]))
    for cutoff in cutoffs:
        layers = foreign_layers(cutoff, n_layer, args.direction)
        blob, replaced = splice(raw_foreign, raw_own, spans_own, layers)
        (slots / f"splice-{cutoff:02d}.state").write_bytes(blob)

    reader = Reader(args.binary, args.target_model, free_port(), out.with_suffix(".gate.log"),
                    [], args.n_ctx, str(slots))
    reader.start()
    results = []
    try:
        prompt_ids = tokenize(reader, prompt)
        reader.post("/slots/0?action=erase", {})
        native = reader.post("/completion", {
            "prompt": prompt, "n_predict": args.forced_positions, "temperature": 0.0,
            "seed": 1, "cache_prompt": True, "id_slot": 0, "n_probs": 5})
        continuation = toks(native)[:args.forced_positions]

        scores = {}
        for cutoff in cutoffs:
            reader.post("/slots/0?action=erase", {})
            restored = reader.post("/slots/0?action=restore",
                                   {"filename": f"splice-{cutoff:02d}.state"})
            scores[cutoff] = teacher_forced(reader, prompt_ids, continuation, 0)
            results.append({"cutoff": cutoff,
                            "layers_from_foreign": cutoff,
                            "n_restored": restored.get("n_restored")})
        reference = scores[0]          # cutoff 0 is the target's own cache, untouched
        for entry in results:
            entry["forced_vs_own"] = compare_forced(scores[entry["cutoff"]], reference)
    finally:
        reader.stop()

    record = {"foreign_model": args.foreign_model, "target_model": args.target_model,
              "direction": args.direction,
              "n_layer": n_layer, "forced_positions": len(continuation),
              "captures": captures, "results": results}
    out.write_text(json.dumps(record, indent=1))

    print(f"\nforeign layers taken from the {args.direction} end")
    print(f"count | top-1 | mean delta | max delta")
    for entry in results:
        c = entry["forced_vs_own"]
        mean = c.get("mean_abs_logprob_delta")
        bar = "#" * int(round((mean or 0) * 60))
        print(f"  {entry['cutoff']:2d}/{n_layer} | {c['top1_agreement']:.4f} | "
              f"{mean:.4f} | {c['max_abs_logprob_delta']:6.3f} {bar}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
