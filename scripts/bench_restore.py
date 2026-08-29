#!/usr/bin/env python3
"""Measure total user-visible restore latency against native prefill.

Emits one machine-readable record per rung so a claim can be traced back to the exact
commit, build, model identity, cache ABI, storage medium and token digest that produced it.

Accounting, as the steer specifies - server-only restore time is not the user-visible cost:

    total restore latency = artifact read + integrity and identity verification
                          + runtime restore + reuse verification

Usage:
    python3 scripts/bench_restore.py --url http://127.0.0.1:8781 \
        --slots /dev/shm/kvx-slots --medium tmpfs --rungs 256,2048,8192 --repeats 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kv_rosetta import container, gguf  # noqa: E402
from kv_rosetta.adapters.base import ExportRequest, ImportRequest, Representation  # noqa: E402
from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter  # noqa: E402


def vram_peak_mb() -> float:
    """Peak VRAM in use across visible devices, or 0 when there is no GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.split()
        return max((float(x) for x in out), default=0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def break_even_tokens(rungs: list[dict]) -> float | None:
    """Context length where total restore latency equals native prefill.

    Both grow roughly linearly in tokens, so the crossing is interpolated from the two
    rungs that straddle it. Returns None when every rung falls on the same side - which is
    itself the answer: the cache path either always wins or always loses in this range.
    """
    ordered = sorted(rungs, key=lambda r: r["tokens"])
    for lo, hi in zip(ordered, ordered[1:]):
        lo_delta = lo["total_restore_s"] * 1000 - lo["native_prefill_ms"]
        hi_delta = hi["total_restore_s"] * 1000 - hi["native_prefill_ms"]
        if lo_delta == 0:
            return float(lo["tokens"])
        if (lo_delta > 0) != (hi_delta > 0):
            span = hi["tokens"] - lo["tokens"]
            return lo["tokens"] + span * abs(lo_delta) / (abs(lo_delta) + abs(hi_delta))
    return None


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    except OSError:
        return ""


def rung(adapter: LlamaCppHTTPAdapter, ids: list[int], slot: int, out_dir: Path) -> dict:
    request = {"prompt": ids, "n_predict": 8, "temperature": 0.0, "top_k": 1,
               "n_probs": 5, "cache_prompt": True, "seed": 1, "id_slot": slot}

    adapter.erase(slot)
    cold = adapter.complete(request)
    assert cold["timings"]["cache_n"] == 0, "cold control reused a cache"
    reference = [c["id"] for c in cold.get("completion_probabilities", [])]

    # CONTROL: does the RUNTIME's own prompt-cache reuse reproduce the cold output, with no
    # artifact involved at all? With quantized KV it does not, because a fresh prefill
    # quantizes computed values while a reused cache reads back already-quantized ones.
    # Without this control, that runtime property would be misattributed to the restore.
    native = adapter.complete(request)
    native_cache_parity = (
        native["content"] == cold["content"]
        and [c["id"] for c in native.get("completion_probabilities", [])] == reference)

    # Determinism control: two cold prefills with no cache reuse must agree, otherwise
    # neither parity number above means anything.
    adapter.erase(slot)
    cold_again = adapter.complete(request)
    deterministic = cold_again["content"] == cold["content"]

    started = time.time()
    artifact = Path(adapter.export(ExportRequest(
        model="", out_path=out_dir / f"bench-{len(ids)}.kvx",
        representation=Representation.OPAQUE, slot=slot)))
    export_s = time.time() - started

    # Erase, then prove the cache is actually gone before claiming a restore worked.
    adapter.erase(slot)
    control = adapter.complete(request)
    assert control["timings"]["cache_n"] == 0, "erase did not drop the cache"

    started = time.time()
    ok, _ = container.verify(artifact)
    verify_s = time.time() - started
    assert ok, "artifact failed verification"

    adapter.erase(slot)
    started = time.time()
    report = adapter.import_(artifact, ImportRequest(model="", slot=slot))
    import_s = time.time() - started          # restore + reuse verification + re-restore
    assert report.ok, report.reason

    warm = adapter.complete(request)
    parity = (warm["content"] == cold["content"]
              and [c["id"] for c in warm.get("completion_probabilities", [])] == reference)

    nbytes = artifact.stat().st_size
    total_restore_s = verify_s + import_s
    header = container.read_header(artifact)
    return {
        "tokens": len(ids),
        "artifact_bytes": nbytes,
        "bytes_per_token": nbytes / len(ids),
        "artifact_digest": header["blob"]["sha256"],
        "native_prefill_ms": cold["timings"]["prompt_ms"],
        "warm_prefill_ms": warm["timings"]["prompt_ms"],
        "export_s": export_s,
        "verify_s": verify_s,
        "import_s": import_s,
        "total_restore_s": total_restore_s,
        "restore_cheaper_than_prefill": total_restore_s * 1000 < cold["timings"]["prompt_ms"],
        "cache_n": warm["timings"]["cache_n"],
        "prompt_n": warm["timings"]["prompt_n"],
        "parity": parity,
        "native_cache_parity": native_cache_parity,
        "model_is_deterministic": deterministic,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_vram_mb": vram_peak_mb(),
        "ms_saved": cold["timings"]["prompt_ms"] - total_restore_s * 1000,
        "speedup_ratio": (cold["timings"]["prompt_ms"] / (total_restore_s * 1000)
                          if total_restore_s > 0 else 0.0),
        "read_throughput_mb_s": (nbytes / 2**20) / verify_s if verify_s > 0 else 0.0,
        "restore_throughput_mb_s": (nbytes / 2**20) / import_s if import_s > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--medium", required=True, help="tmpfs or nvme; reported, never mixed")
    ap.add_argument("--rungs", default="256,2048,8192")
    ap.add_argument("--repeats", type=int, default=3,
                    help="three or more; medians and ranges are reported, never a single run")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--kv-type", default="",
                    help="KV cache type this server was launched with; /props does not "
                         "report type_k/type_v, so it cannot be probed and must be stated")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.repeats < 3:
        print(f"refusing --repeats {args.repeats}: a median of fewer than three runs is a "
              f"single noisy measurement wearing a statistic", file=sys.stderr)
        return 2

    adapter = LlamaCppHTTPAdapter(args.url, args.slots)
    props = adapter.props()
    settings = props.get("default_generation_settings", {}) or {}
    model_path = str(props.get("model_path", ""))

    reusable, why = adapter.prefix_reuse_support()
    if not reusable:
        print(json.dumps({"error": "model cannot reuse a restored prefix", "reason": why,
                          "architecture": gguf.architecture(model_path) if model_path else ""},
                         indent=2))
        return 2

    text = "In the year 1892 the naturalist recorded observations. " * 4000
    pool = adapter._post("/tokenize", {"content": text})["tokens"]
    out_dir = Path(tempfile.mkdtemp())

    results = []
    for size in [int(x) for x in args.rungs.split(",")]:
        if size > len(pool):
            continue
        ids = pool[:size]
        runs = [rung(adapter, ids, args.slot, out_dir) for _ in range(args.repeats)]
        merged = dict(runs[0])
        for field in ("native_prefill_ms", "export_s", "verify_s", "import_s",
                      "total_restore_s", "warm_prefill_ms", "ms_saved", "speedup_ratio",
                      "read_throughput_mb_s", "restore_throughput_mb_s", "peak_vram_mb"):
            values = [r[field] for r in runs]
            merged[field] = statistics.median(values)
            merged[f"{field}_range"] = [min(values), max(values)]
        merged["repeats"] = len(runs)
        merged["parity"] = all(r["parity"] for r in runs)
        merged["native_cache_parity"] = all(r["native_cache_parity"] for r in runs)
        merged["model_is_deterministic"] = all(r["model_is_deterministic"] for r in runs)
        merged["restore_cheaper_than_prefill"] = (
            merged["total_restore_s"] * 1000 < merged["native_prefill_ms"])
        merged["token_ids_sha256"] = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        results.append(merged)
        print(f"  {size:>6} tok | prefill {merged['native_prefill_ms']:8.0f} ms | "
              f"restore {merged['total_restore_s']*1000:8.0f} ms | "
              f"{merged['bytes_per_token']/1024:6.1f} KB/tok | "
              f"cheaper={merged['restore_cheaper_than_prefill']} "
              f"x{merged['speedup_ratio']:.1f} parity={merged['parity']}/"
              f"{merged['native_cache_parity']}",
              flush=True)

    record = {
        "repo_commit": git_commit(),
        "llama_build": str(props.get("build_info", "")),
        "model_path": model_path,
        "architecture": gguf.architecture(model_path) if model_path else "",
        "identity": adapter.identity(),
        "cache_abi": adapter.cache_abi_identity().as_dict()
        if hasattr(adapter.cache_abi_identity(), "as_dict") else {},
        "kv_type_k": str(props.get("type_k", settings.get("type_k", ""))),
        "kv_type_v": str(props.get("type_v", settings.get("type_v", ""))),
        "kv_type_declared": args.kv_type,
        "caveats": [
            "kv_type_k/kv_type_v are empty because llama-server /props does not report "
            "them; kv_type_declared is the launch flag, corroborated by bytes_per_token",
            "peak_vram_mb is host-wide from nvidia-smi and is NOT attributed to this "
            "process; it is meaningless when the server runs with -ngl 0",
        ],
        "n_ctx": settings.get("n_ctx"),
        "slot": args.slot,
        "storage_medium": args.medium,
        "slot_save_path": args.slots,
        "opaque_format": adapter.opaque_format(),
        "rungs": results,
        "break_even_tokens": break_even_tokens(results),
        "repeats": args.repeats,
    }
    out = Path(args.out) if args.out else Path("bench") / f"restore-{args.medium}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
