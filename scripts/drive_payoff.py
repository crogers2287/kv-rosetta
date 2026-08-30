#!/usr/bin/env python3
"""Does a model's drive attachment actually pay, on this exact binary? RESEARCH ONLY.

The drive predicts reuse from the architecture and the runtime's advertised checkpoint
persistence. That prediction has been wrong twice: once by ignoring the runtime entirely
(a hybrid attachment called worthless was a 3.7x win on a patched build), and the reason
it can be wrong again is that architectures differ in ways the flag does not name --
sliding-window attention keeps state that a slot save may or may not carry, and a model
using it is not "hybrid" by any label this code checks.

So this measures instead of predicting, and prints the prediction beside the measurement
so a disagreement is visible rather than absorbed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production_matrix import free_port, sha256_file  # noqa: E402
from reader_determinism import Reader  # noqa: E402

from kv_rosetta.gguf import architecture, uses_sliding_window  # noqa: E402
from kv_rosetta.identity import CacheABIIdentity, ModelIdentity  # noqa: E402
from kv_rosetta.shared import Content, Entry, SharedDrive, expected_reuse  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--args", default="")
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--repeat", type=int, default=25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    slots = Path(args.out or "/tmp") .parent / f"payoff-{args.label}"
    slots.mkdir(parents=True, exist_ok=True)
    for stale in slots.glob("*.state"):
        stale.unlink()
    drive = SharedDrive(slots / "drive")

    arch = architecture(args.model)
    swa = uses_sliding_window(args.model)
    reader = Reader(args.binary, args.model, free_port(), slots / "server.log",
                    args.args.split(), args.n_ctx, str(slots))
    reader.start()
    try:
        props = reader.get("/props")
        persistence = props.get("slot_checkpoint_persistence")
        entries = []
        for name, role, text in (
                ("system", "system",
                 "You are a careful assistant that answers precisely. " * args.repeat),
                ("tools", "tools", '{"name":"search"} ' * args.repeat),
                ("recall", "memory",
                 "Remembered: the user prefers metric units. " * args.repeat)):
            ids = reader.post("/tokenize", {"content": text})["tokens"]
            entries.append(Entry(name, role, text, tuple(ids)))
        content = Content(tokenizer_id=f"tok-{arch}", entries=tuple(entries))
        digest = drive.publish(content)
        model = ModelIdentity(architecture=arch, weights_sha256=sha256_file(Path(args.model)),
                              tokenizer_sha256="b" * 64, chat_template_sha256="c" * 64)
        abi = CacheABIIdentity(runtime="llama.cpp", runtime_revision=str(props.get("build_info")),
                               state_format="ggsq/3", k_dtype="f16", v_dtype="f16",
                               rope_kind="normal", rope_base=1e6)

        def once(label, staged):
            reader.post("/slots/0?action=erase", {})
            restored = None
            if staged:
                restored = reader.post("/slots/0?action=restore",
                                       {"filename": staged}).get("n_restored")
            r = reader.post("/completion", {
                "prompt": content.text + "\nAnswer in one sentence.", "n_predict": 4,
                "temperature": 0.0, "seed": 1, "cache_prompt": True, "id_slot": 0})
            t = r["timings"]
            print("  %-30s n_restored=%-7s cache_n=%-7s prefilled=%-6s ms=%.0f"
                  % (label, restored, t["cache_n"], t["prompt_n"], t["prompt_ms"]))
            return t

        print(f"\n{args.label}: arch={arch} sliding_window={swa} "
              f"tokens={len(content.token_ids)} checkpoint_persistence={persistence}")
        cold = once("cold, no attachment", None)
        saved = reader.post("/slots/0?action=save", {"filename": f"{args.label}.state"})
        raw = (slots / f"{args.label}.state").read_bytes()
        drive.attach(digest, model, abi, slots / f"{args.label}.state", architecture=arch,
                     sliding_window=swa,
                     checkpoint_persistence=(persistence if isinstance(persistence, bool)
                                             else None))
        warm = once("via the drive attachment", drive.stage(digest, model, abi, slots))
    finally:
        reader.stop()

    predicted, reason = expected_reuse(
        arch, sliding_window=swa,
        checkpoint_persistence=persistence if isinstance(persistence, bool) else None)
    reused = warm["cache_n"] or 0
    measured = reused > 0
    record = {
        "label": args.label, "architecture": arch, "binary": args.binary,
        "sliding_window": swa,
        "checkpoint_persistence": persistence,
        "sckp_appendix_present": b"SCKP" in raw,
        "saved_cells": saved.get("n_saved"), "saved_bytes": saved.get("n_written"),
        "tokens": cold["prompt_n"],
        "cold_ms": cold["prompt_ms"], "warm_ms": warm["prompt_ms"],
        "cache_n": reused, "prefilled_after_restore": warm["prompt_n"],
        "speedup": (cold["prompt_ms"] / warm["prompt_ms"]) if warm["prompt_ms"] else None,
        "predicted_pays": predicted, "prediction_reason": reason, "measured_pays": measured,
        # An unknown prediction is not a wrong one. A stock build does not advertise
        # slot_checkpoint_persistence at all, so the honest prediction there is None, and
        # scoring None against a measured False as a disagreement would flag every
        # unpatched run as a contradiction and train the reader to ignore the warning.
        "prediction_contradicted": (predicted is not None and predicted is not measured),
        "prediction_was_made": predicted is not None,
    }
    print("  SCKP appendix: %s | predicted pays=%s | measured pays=%s | speedup=%.2fx"
          % (record["sckp_appendix_present"], predicted, measured, record["speedup"] or 0))
    if record["prediction_contradicted"]:
        print("  *** PREDICTION CONTRADICTED BY MEASUREMENT ***")
    elif predicted is None:
        print("  (no prediction available: this runtime does not advertise checkpoint "
              "persistence)")
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1))
    return 3 if record["prediction_contradicted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
