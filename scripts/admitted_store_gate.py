#!/usr/bin/env python3
"""Three process-owned 2K repetitions through the admitted-state path. RESEARCH ONLY.

One object is admitted once, then every repetition starts a fresh llama-server, measures a
genuinely cold request in that process, and restores the same admitted object. Pairing cold
and restore inside one process removes load-to-load variation from the comparison.

Reports median and range across repetitions, never the best run, and retains the admission
cost with the break-even restore count it implies: a request-path win is not a lifecycle win.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production_matrix import (  # noqa: E402
    N_PROBS,
    Server,
    binary_digests,
    build_flags,
    free_port,
    git_state,
    probs,
    prompt_text,
    require_clean_worktree,
    sha256_file,
    toks,
    vectors_agree,
)

from kv_rosetta import gguf  # noqa: E402
from kv_rosetta.admitted_store import AdmittedStore, _file_facts  # noqa: E402
from kv_rosetta.adapters.admitted_path import AdmittedPath  # noqa: E402
from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter  # noqa: E402

PHASE_TOLERANCE_S = 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("--unpatched", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--storage-note", default="")
    ap.add_argument("--out", default="bench/admitted-store-2k.json")
    args = ap.parse_args()
    repo_commit = require_clean_worktree()

    slots = Path(args.slots)
    slots.mkdir(parents=True, exist_ok=True)
    log_dir = slots.parent / "admitted-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # The store IS the slot directory: llama-server restores by basename relative to it, so
    # an object anywhere else would have to be copied or linked in, which is the cost being
    # removed. AdmittedStore enforces 0700 on it.
    store = AdmittedStore(slots)

    # ---- admission, once, off the request path ---------------------------------------
    first = Server(args.patched, args.model, str(slots), free_port(), log_dir / "admit.log")
    first_pid = first.start()
    try:
        ids = first.post("/tokenize", {"content": prompt_text(args.prompt_tokens)}
                         )["tokens"][:args.prompt_tokens]
        if len(ids) != args.prompt_tokens:
            raise RuntimeError(f"tokenizer produced {len(ids)} tokens")
        request = {"prompt": ids, "n_predict": 8, "temperature": 0.0, "top_k": 1,
                   "n_probs": N_PROBS, "cache_prompt": True, "id_slot": 0}
        first.post("/slots/0?action=erase", {})
        first.post("/completion", request)
        native = first.post("/completion", request)      # in-memory reuse, the parity target
        native_tokens, native_content = toks(native), native["content"]
        native_vectors = probs(native)

        first.post("/slots/0?action=erase", {})
        first.post("/completion", dict(request, n_predict=0))   # prefix only
        saved = first.post("/slots/0?action=save", {"filename": "to-admit.bin"})
        raw = slots / "to-admit.bin"

        adapter = LlamaCppHTTPAdapter(first.url, str(slots))
        path = AdmittedPath(adapter, store)
        admit_started = time.time()
        obj = path.admit(raw, model=args.model, token_ids=ids, save_response=saved)
        admission_seconds = time.time() - admit_started
        raw.unlink(missing_ok=True)
        admitted_facts = dict(obj.facts)
        props_a = first.props()
    finally:
        first.stop()

    # ---- repetitions, each in a fresh process ----------------------------------------
    repetitions = []
    for index in range(1, args.repeats + 1):
        server = Server(args.patched, args.model, str(slots), free_port(),
                        log_dir / f"rep{index}.log")
        pid = server.start()
        try:
            if pid == first_pid:
                raise RuntimeError("no new process was started")
            ids2 = server.post("/tokenize", {"content": prompt_text(args.prompt_tokens)}
                               )["tokens"][:args.prompt_tokens]
            if ids2 != ids:
                raise RuntimeError("tokenization differs across processes")
            request2 = dict(request, prompt=ids2)

            t0 = time.time()
            cold = server.post("/completion", request2)
            cold_wall = time.time() - t0
            if cold["timings"]["cache_n"] != 0:
                raise RuntimeError(f"rep {index}: fresh process reported reuse before "
                                   f"any restore")
            server.post("/slots/0?action=erase", {})

            rep_adapter = LlamaCppHTTPAdapter(server.url, str(slots))
            rep_path = AdmittedPath(rep_adapter, store)
            report = rep_path.restore(obj.digest, model=args.model, token_ids=ids2)
            if not report.ok:
                raise RuntimeError(f"rep {index}: admitted restore failed: {report.reason}")
            if report.cache_n != 2044 or report.prompt_n != 4:
                raise RuntimeError(f"rep {index}: cache_n={report.cache_n} "
                                   f"prompt_n={report.prompt_n}, expected 2044/4")
            if report.reads.payload_bytes:
                raise RuntimeError(f"rep {index}: request path read "
                                   f"{report.reads.payload_bytes} payload bytes")

            t0 = time.time()
            after = server.post("/completion", request2)
            tail_wall = time.time() - t0
            if toks(after) != native_tokens or after["content"] != native_content:
                raise RuntimeError(f"rep {index}: output differs from native reuse")
            after_vectors = probs(after)
            if not all(after_vectors) or not vectors_agree(after_vectors, native_vectors):
                raise RuntimeError(f"rep {index}: probability vectors differ from native")

            phase_sum = sum(report.phases.values())
            if abs(report.seconds - phase_sum) > PHASE_TOLERANCE_S:
                raise RuntimeError(
                    f"rep {index}: phases account for {phase_sum:.3f}s of "
                    f"{report.seconds:.3f}s")
            now_facts = _file_facts(store.root / f"{obj.digest}.state")
            if now_facts != admitted_facts:
                raise RuntimeError(f"rep {index}: admitted object changed: "
                                   f"{admitted_facts} -> {now_facts}")
            total = report.seconds + tail_wall
            repetitions.append({
                "repetition": index, "pid": pid,
                "cold_wall_s": cold_wall,
                "restore_import_s": report.seconds,
                "tail_completion_s": tail_wall,
                "total_s": total,
                "phases": dict(report.phases),
                "phase_sum_s": phase_sum,
                "unclassified_s": report.seconds - phase_sum,
                "cache_n": report.cache_n, "prompt_n": report.prompt_n,
                "request_path_payload_bytes": report.reads.payload_bytes,
                "request_path_metadata_bytes": report.reads.metadata_bytes,
                "endpoint_calls": report.calls,
                "cheaper_than_cold": total < cold_wall,
                "ratio_to_cold": total / cold_wall,
            })
            print(f"  rep {index}: restore {report.seconds:.3f}s + tail {tail_wall:.3f}s "
                  f"= {total:.3f}s vs cold {cold_wall:.3f}s -> "
                  f"{'CHEAPER' if total < cold_wall else 'not cheaper'}", flush=True)
        finally:
            server.stop()

    # ---- unpatched control ------------------------------------------------------------
    control = Server(args.unpatched, args.model, str(slots), free_port(),
                     log_dir / "control.log")
    control.start()
    try:
        control_adapter = LlamaCppHTTPAdapter(control.url, str(slots))
        control_path = AdmittedPath(control_adapter, store)
        control_report = control_path.restore(obj.digest, model=args.model, token_ids=ids)
        if control_report.ok:
            raise RuntimeError("the unpatched runtime accepted an admitted object")
        if control_report.calls:
            raise RuntimeError(f"the unpatched runtime was contacted before refusing: "
                               f"{control_report.calls}")
    finally:
        control.stop()

    totals = [r["total_s"] for r in repetitions]
    colds = [r["cold_wall_s"] for r in repetitions]
    wins = sum(1 for r in repetitions if r["cheaper_than_cold"])
    median_total, median_cold = statistics.median(totals), statistics.median(colds)
    saving = median_cold - median_total
    record = {
        "kind": "admitted-store-2k-gate",
        "warning": "RESEARCH ONLY. Experimental local path, not a production API.",
        "repo_commit": repo_commit,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "storage_note": args.storage_note,
        "slots_path": str(slots),
        "filesystem": subprocess.run(["findmnt", "-no", "FSTYPE", "--target", str(slots)],
                                     capture_output=True, text=True).stdout.strip(),
        "prompt_tokens": args.prompt_tokens,
        "model_path": args.model,
        "architecture": gguf.architecture(args.model),
        "model_content_digest": obj.manifest.get("model_content_digest"),
        "kv_dtype_k": props_a.get("target_cache_type_k"),
        "kv_dtype_v": props_a.get("target_cache_type_v"),
        "build_info": props_a.get("build_info"),
        "binary_digests": binary_digests(Path(args.patched)),
        "build_flags": build_flags(Path(args.patched)),
        "source_trees": {"patched": git_state(str(Path(args.patched).parent.parent.parent))},
        "admission": {"seconds": admission_seconds, "digest": obj.digest,
                      "manifest": obj.manifest, "file_facts": admitted_facts},
        "repetitions": repetitions,
        "unpatched_control": {"ok": control_report.ok, "reason": control_report.reason,
                              "endpoint_calls": control_report.calls},
        "summary": {
            "median_total_s": median_total, "total_range_s": [min(totals), max(totals)],
            "median_cold_s": median_cold, "cold_range_s": [min(colds), max(colds)],
            "paired_wins": wins, "repetitions": len(repetitions),
            "median_saving_s": saving,
            "median_ratio_to_cold": median_total / median_cold,
            "break_even_restores": (int(-(-admission_seconds // saving))
                                    if saving > 0 else None),
            "passes": (median_total < median_cold and wins >= 2),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    s = record["summary"]
    print(f"  median {s['median_total_s']:.3f}s (range {s['total_range_s'][0]:.3f}-"
          f"{s['total_range_s'][1]:.3f}) vs cold {s['median_cold_s']:.3f}s "
          f"(range {s['cold_range_s'][0]:.3f}-{s['cold_range_s'][1]:.3f})")
    print(f"  paired wins {s['paired_wins']}/{s['repetitions']}  "
          f"break-even after {s['break_even_restores']} restores  "
          f"PASSES={s['passes']}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
