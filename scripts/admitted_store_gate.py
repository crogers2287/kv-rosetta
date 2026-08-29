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
import os
import shutil
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

#: Filesystems that do not survive a process restart in any meaningful sense. A record
#: produced on one of these is not persistent-storage evidence, whatever the path looks like.
MEMORY_BACKED = {"tmpfs", "ramfs"}


def storage_evidence(path: Path, model: Path) -> dict:
    """Identify the actual mounted source behind a path, not what the path looks like.

    A name like /mnt/storage says nothing about the device; on this host it is a
    FUSE-mounted SATA volume while the NVMe is mounted at /. The record has to carry the
    mount source, filesystem type, options, and the backing device's rotational flag so a
    reader can tell what was actually measured.
    """
    def findmnt(field: str) -> str:
        result = subprocess.run(["findmnt", "-no", field, "--target", str(path)],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    source = findmnt("SOURCE")
    evidence = {
        "resolved_path": str(path.resolve()),
        "stat_device_id": path.stat().st_dev,
        "mount_source": source,
        "mount_target": findmnt("TARGET"),
        "filesystem": findmnt("FSTYPE"),
        "mount_options": findmnt("OPTIONS"),
        "same_mount_as_model": findmnt("TARGET") == subprocess.run(
            ["findmnt", "-no", "TARGET", "--target", str(model)],
            capture_output=True, text=True).stdout.strip(),
        "available_bytes": shutil.disk_usage(path).free,
    }
    # Walk the device-mapper/LVM chain to the physical block device, so an LVM name does
    # not hide whether the bytes land on rotational media.
    backing, rotational = None, None
    try:
        # -s walks toward the parents, so an LVM or partition resolves to its disk.
        result = subprocess.run(["lsblk", "-npso", "NAME,TYPE,ROTA", source],
                                capture_output=True, text=True)
        rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
        for name, kind, rota in rows:
            if kind == "disk":
                backing, rotational = Path(name).name, rota == "1"
                break
    except (OSError, ValueError):
        pass
    if backing is None:
        pvs = subprocess.run(["pvs", "--noheadings", "-o", "pv_name,vg_name"],
                             capture_output=True, text=True).stdout
        for line in pvs.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] and parts[1] in source:
                device = Path(parts[0]).name.rstrip("0123456789").rstrip("p")
                flag = Path(f"/sys/block/{device}/queue/rotational")
                backing = device
                rotational = flag.read_text().strip() == "1" if flag.is_file() else None
                break
    evidence["backing_device"] = backing
    evidence["rotational"] = rotational
    return evidence


def resident_pages(path: Path) -> tuple[int, int] | None:
    """(resident, total) pages of a file, via mincore. None when it cannot be measured.

    Turns "the cache was dropped" from an assertion into a measurement. Without it an
    eviction that silently failed would be published as a cold-cache result. The mapping is
    made through libc rather than the mmap module because mincore needs the address, and a
    read-only Python mmap will not surrender one.
    """
    import ctypes
    import ctypes.util

    PROT_READ, MAP_SHARED = 0x1, 0x01
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_ubyte)]
        size = path.stat().st_size
        if size == 0:
            return (0, 0)
        page = os.sysconf("SC_PAGESIZE")
        pages = (size + page - 1) // page
        fd = os.open(path, os.O_RDONLY)
        try:
            address = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
            if address in (None, ctypes.c_void_p(-1).value, -1):
                return None
            try:
                buffer = (ctypes.c_ubyte * pages)()
                if libc.mincore(ctypes.c_void_p(address), size, buffer) != 0:
                    return None
                return (sum(1 for b in buffer if b & 1), pages)
            finally:
                libc.munmap(ctypes.c_void_p(address), size)
        finally:
            os.close(fd)
    except (OSError, ValueError, AttributeError):
        return None


def evict_file(path: Path) -> dict:
    """Drop one file from the page cache. File-scoped on purpose.

    The system-wide drop_caches control would also evict the model weights, changing the
    cold-prefill baseline and making the comparison uninterpretable. POSIX_FADV_DONTNEED
    touches only this file.
    """
    before = resident_pages(path)
    status, error = "ok", None
    try:
        # Dirty pages cannot be dropped, so flush first. Without this the eviction is
        # silently partial and a warm file would be reported as cold.
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError as exc:
        status, error = "failed", str(exc)
    after = resident_pages(path)
    return {
        "mechanism": "posix_fadvise(POSIX_FADV_DONTNEED), file-scoped",
        "status": status, "error": error,
        "resident_pages_before": before[0] if before else None,
        "resident_pages_after": after[0] if after else None,
        "total_pages": before[1] if before else (after[1] if after else None),
        "residency_measured": before is not None and after is not None,
        "resident_fraction_after": (after[0] / after[1] if after and after[1] else None),
    }


def predict_space(prompt_tokens: int, bytes_per_token: float, free_bytes: int,
                  margin: float = 0.20) -> dict:
    """Predict the object size and peak transient use, and say whether it fits.

    Admission copies the raw state into the store before the source is removed, so peak
    usage is roughly twice the object. Running out of space mid-admission would leave a
    partial object and a useless record, so this is checked before anything is generated
    rather than discovered during the run.
    """
    predicted = int(prompt_tokens * bytes_per_token)
    peak = predicted * 2
    required = int(peak * (1 + margin))
    return {
        "bytes_per_token": bytes_per_token,
        "predicted_object_bytes": predicted,
        "predicted_peak_transient_bytes": peak,
        "required_with_margin_bytes": required,
        "safety_margin": margin,
        "free_bytes": free_bytes,
        "fits": free_bytes >= required,
        "headroom_bytes": free_bytes - required,
    }


def require_persistent(evidence: dict) -> None:
    """Refuse to produce a persistent-storage record on memory-backed storage."""
    fs = evidence.get("filesystem", "")
    if not fs or not evidence.get("mount_source"):
        raise SystemExit(f"cannot identify the mount behind {evidence['resolved_path']}; "
                         f"refusing to call an unresolved target persistent")
    if fs in MEMORY_BACKED:
        raise SystemExit(f"{evidence['resolved_path']} is on {fs}, which is memory-backed; "
                         f"refusing to record it as persistent storage")
    if fs == "overlay":
        raise SystemExit(f"{evidence['resolved_path']} is on an overlay mount whose lower "
                         f"layers are not established here; refusing")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("--unpatched", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--storage-note", default="")
    ap.add_argument("--bytes-per-token", type=float, default=295390.0,
                    help="predicted artifact bytes per prompt token; the default is "
                         "measured from the retained 2K record (604,958,676 / 2048)")
    ap.add_argument("--require-persistent", action="store_true",
                    help="refuse to run when the store is on memory-backed storage")
    ap.add_argument("--evict-state-before-restore", action="store_true",
                    help="drop the admitted object from page cache before each timed "
                         "restore, file-scoped; never uses drop_caches")
    ap.add_argument("--page-cache-policy", default="natural",
                    help="explicit page-cache state for the record, e.g. "
                         "'natural after admission and process restarts'")
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
    evidence = storage_evidence(slots, Path(args.model))
    if args.require_persistent:
        require_persistent(evidence)
    space = predict_space(args.prompt_tokens, args.bytes_per_token,
                          evidence["available_bytes"])
    print(f"  predicted object {space['predicted_object_bytes']/2**30:.2f} GiB, peak "
          f"{space['predicted_peak_transient_bytes']/2**30:.2f} GiB, need "
          f"{space['required_with_margin_bytes']/2**30:.2f} GiB with margin, free "
          f"{space['free_bytes']/2**30:.2f} GiB -> fits={space['fits']}", flush=True)
    if not space["fits"]:
        raise SystemExit(
            f"refusing to generate: predicted peak transient "
            f"{space['predicted_peak_transient_bytes']/2**30:.2f} GiB plus a "
            f"{space['safety_margin']:.0%} margin needs "
            f"{space['required_with_margin_bytes']/2**30:.2f} GiB, but only "
            f"{space['free_bytes']/2**30:.2f} GiB is free on "
            f"{evidence['mount_target']}")
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

            eviction = None
            if args.evict_state_before_restore:
                eviction = evict_file(store.root / f"{obj.digest}.state")
                if eviction["status"] != "ok":
                    raise RuntimeError(f"rep {index}: eviction failed: {eviction['error']}")
                if not eviction["residency_measured"]:
                    raise RuntimeError(
                        f"rep {index}: residency could not be measured, so a cold-cache "
                        f"claim cannot be published")
                if eviction["resident_fraction_after"] > 0.01:
                    raise RuntimeError(
                        f"rep {index}: {eviction['resident_pages_after']} of "
                        f"{eviction['total_pages']} pages "
                        f"({eviction['resident_fraction_after']*100:.1f}%) still resident "
                        f"after eviction; refusing to publish a cold-cache claim")

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
                "eviction": eviction,
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
        "storage": evidence,
        "space_prediction": space,
        "page_cache_policy": args.page_cache_policy,
        "filesystem": evidence["filesystem"],
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
