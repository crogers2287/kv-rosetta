#!/usr/bin/env python3
"""Lower bound for a trusted, pre-admitted local cache. RESEARCH ONLY, NOT PRODUCTION.

The public adapter spends most of its import copying and hashing the payload. This measures
what remains if that copy could be eliminated entirely: the raw ggsq/3+sckp/1 state is fully
validated OUTSIDE the timed window, stored under its content digest inside the slot
directory, and then restored in place with no extraction and no byte copy.

This is a lower bound, not a design. It is not permission to skip integrity on arbitrary
files: every admitted byte is verified before admission, and the file is proven unchanged
after the timed restore. A filename or a prior hash alone is not proof that the bytes
restored are the bytes admitted, which is why the digest is recomputed at the end.

    python3 scripts/direct_raw_lowerbound.py --model ... --patched ... --unpatched ...
        --slots /dev/shm/kvx-direct --prompt-tokens 2048 --out bench/direct-raw-2k.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production_matrix import (  # noqa: E402
    LOGPROB_TOLERANCE,
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

from kv_rosetta import gguf, weights  # noqa: E402
from kv_rosetta.adapters import ggsq_envelope  # noqa: E402
from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter  # noqa: E402

MAX_UNCOVERED_TAIL = 8


def file_facts(path: Path) -> dict:
    """Identity of the bytes on disk, beyond the name."""
    st = path.stat()
    return {"device": st.st_dev, "inode": st.st_ino, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
            "sha256": sha256_file(path)}


def admit(raw: Path, adapter: LlamaCppHTTPAdapter, saved: dict, model: str,
          token_ids: list[int]) -> dict:
    """Validate everything about a raw state file, before any timed work.

    Admission is the whole safety argument for restoring in place: if any of this is
    skipped, the timed path below is restoring bytes nobody checked.
    """
    started = time.time()
    problems: list[str] = []

    supported, reason, protocol = adapter.hybrid_support()
    if not supported:
        problems.append(f"runtime unsupported: {reason}")
    version = ggsq_envelope.peek_version(raw.read_bytes()[:12])
    if (version, protocol.get("format")) not in adapter.supported_compound_tuples:
        problems.append(f"untested compound tuple ggsq/{version}+{protocol.get('format')}")
    active = adapter._active_state_classes()
    if active is None or set(active) - adapter.proven_state_classes:
        problems.append(f"active state classes not proven: {active}")
    k_dtype, v_dtype = adapter.cache_dtypes()
    if not k_dtype or not v_dtype:
        problems.append("runtime does not advertise its K/V cache types")

    # Sequence framing and the prompt the state actually carries.
    with open(raw, "rb") as handle:
        head = handle.read(12)
        head += handle.read(ggsq_envelope.header_size(head) - len(head))
    envelope = ggsq_envelope.parse_file_envelope(head)
    packed = ggsq_envelope.decode_prompt_tokens(envelope.token_ids)
    if list(packed) != list(token_ids):
        problems.append(f"state carries {len(packed)} tokens, not the {len(token_ids)} "
                        f"prompt tokens under test")

    # The SCKP appendix at the offset the runtime declared, not one found by scanning.
    n_written = int(saved.get("n_written", 0) or 0)
    checkpoint_bytes = int(saved.get("checkpoint_bytes", 0) or 0)
    if n_written != raw.stat().st_size:
        problems.append(f"runtime wrote {n_written} bytes but the file is "
                        f"{raw.stat().st_size}")
    appendix = ggsq_envelope.checkpoint_appendix_at(raw, n_written - checkpoint_bytes)
    if not appendix.usable:
        problems.append(f"checkpoint appendix at the declared offset is "
                        f"{appendix.status.value}")
    if int(saved.get("checkpoint_n_tokens", 0) or 0) <= 0:
        problems.append("no checkpoint coverage declared")

    facts = file_facts(raw)      # full payload digest, over every byte
    model_ident = adapter.model_identity(model)
    prompt_digest = hashlib.sha256(
        json.dumps(list(token_ids), separators=(",", ":")).encode()).hexdigest()

    if problems:
        raise RuntimeError("refusing to admit the raw state: " + "; ".join(problems))
    return {
        "admission_seconds": time.time() - started,
        "file": facts,
        "sequence_version": version,
        "compound_tuple": f"ggsq/{version}+{protocol.get('format')}",
        "active_state_classes": active,
        "cache_dtype_k": k_dtype,
        "cache_dtype_v": v_dtype,
        "appendix": {"status": appendix.status.value, "offset": appendix.offset,
                     "count": appendix.count, "nbytes": appendix.nbytes},
        "checkpoint_coverage": {k: saved.get(k) for k in (
            "n_checkpoints_saved", "checkpoint_bytes", "checkpoint_n_tokens",
            "checkpoint_pos_min", "checkpoint_pos_max")},
        "model_weights_sha256": model_ident.weights_sha256,
        "model_content_digest": weights.model_content_digest(model),
        "prompt_digest": prompt_digest,
        "token_count": len(token_ids),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("--unpatched", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--storage-note", default="")
    ap.add_argument("--out", default="bench/direct-raw-2k.json")
    args = ap.parse_args()
    repo_commit = require_clean_worktree()

    slots = Path(args.slots)
    slots.mkdir(parents=True, exist_ok=True)
    log_dir = slots.parent / "direct-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- untimed: produce and admit the raw state ------------------------------------
    first = Server(args.patched, args.model, str(slots), free_port(), log_dir / "a.log")
    first_pid = first.start()
    try:
        text = prompt_text(args.prompt_tokens)
        ids = first.post("/tokenize", {"content": text})["tokens"][:args.prompt_tokens]
        if len(ids) != args.prompt_tokens:
            raise RuntimeError(f"tokenizer produced {len(ids)} tokens")
        request = {"prompt": ids, "n_predict": 8, "temperature": 0.0, "top_k": 1,
                   "n_probs": N_PROBS, "cache_prompt": True, "id_slot": 0}
        first.post("/slots/0?action=erase", {})
        t0 = time.time()
        cold = first.post("/completion", request)
        cold_wall = time.time() - t0
        if cold["timings"]["cache_n"] != 0:
            raise RuntimeError("cold run reused a cache")
        native = first.post("/completion", request)
        native_vectors = probs(native)

        # Prefix only: save before any generation, so the state's boundary is the prompt.
        first.post("/slots/0?action=erase", {})
        first.post("/completion", dict(request, n_predict=0))
        saved = first.post("/slots/0?action=save", {"filename": "direct-staging.bin"})
        produced = slots / "direct-staging.bin"

        adapter = LlamaCppHTTPAdapter(first.url, str(slots))
        admission = admit(produced, adapter, saved, args.model, ids)
        # Content-addressed, inside the slot directory, so restore needs no copy at all.
        admitted_name = f"admitted-{admission['file']['sha256'][:32]}.bin"
        produced.rename(slots / admitted_name)
        admitted = slots / admitted_name
        admission["file"] = file_facts(admitted)
        admission["admitted_name"] = admitted_name
        props_a = first.props()
    finally:
        first.stop()

    # ---- timed: restore in place across a real restart -------------------------------
    second = Server(args.patched, args.model, str(slots), free_port(), log_dir / "b.log")
    second_pid = second.start()
    try:
        if second_pid == first_pid:
            raise RuntimeError("no new process was started")
        ids2 = second.post("/tokenize", {"content": prompt_text(args.prompt_tokens)}
                           )["tokens"][:args.prompt_tokens]
        if ids2 != ids:
            raise RuntimeError("tokenization differs across processes")
        request2 = dict(request, prompt=ids2)
        control = second.post("/completion", request2)
        if control["timings"]["cache_n"] != 0:
            raise RuntimeError("fresh process reported reuse before any restore")
        second.post("/slots/0?action=erase", {})

        phases: dict[str, float] = {}
        timed_start = time.time()
        mark = time.time()
        restored = second.post("/slots/0?action=restore", {"filename": admitted_name})
        phases["runtime_restore"] = time.time() - mark

        declared = admission["checkpoint_coverage"]
        pairs = (("n_checkpoints_saved", "n_checkpoints_restored"),
                 ("checkpoint_bytes", "checkpoint_bytes"),
                 ("checkpoint_n_tokens", "checkpoint_n_tokens"),
                 ("checkpoint_pos_min", "checkpoint_pos_min"),
                 ("checkpoint_pos_max", "checkpoint_pos_max"))
        differing = [f"{a}: admitted {declared.get(a)!r} vs restore {restored.get(b)!r}"
                     for a, b in pairs if declared.get(a) != restored.get(b)]
        if differing:
            raise RuntimeError("restore metadata differs from admitted: "
                               + "; ".join(differing))

        mark = time.time()
        probe = second.post("/completion", {"prompt": ids2, "n_predict": 1,
                                            "temperature": 0.0, "top_k": 1,
                                            "cache_prompt": True, "id_slot": 0})
        phases["reuse_probe"] = time.time() - mark
        cache_n = int(probe["timings"]["cache_n"])
        prompt_n = int(probe["timings"]["prompt_n"])
        uncovered = len(ids2) - cache_n
        if cache_n != int(declared["checkpoint_n_tokens"]):
            raise RuntimeError(f"cache_n {cache_n} != declared coverage "
                               f"{declared['checkpoint_n_tokens']}")
        if not 1 <= uncovered <= MAX_UNCOVERED_TAIL or prompt_n != uncovered:
            raise RuntimeError(f"tail contract violated: cache_n={cache_n} "
                               f"prompt_n={prompt_n} uncovered={uncovered}")

        mark = time.time()
        second.post("/slots/0?action=erase", {})
        second.post("/slots/0?action=restore", {"filename": admitted_name})
        phases["pristine_restore"] = time.time() - mark
        timed_import = time.time() - timed_start

        mark = time.time()
        after = second.post("/completion", request2)
        tail_wall = time.time() - mark

        if toks(after) != toks(native) or after["content"] != native["content"]:
            raise RuntimeError("output after direct restore differs from native reuse")
        after_vectors = probs(after)
        if not vectors_agree(after_vectors, native_vectors):
            raise RuntimeError("probability vectors differ from native reuse")
    finally:
        second.stop()

    after_facts = file_facts(admitted)
    if after_facts != admission["file"]:
        raise RuntimeError(f"the admitted file changed during the timed window: "
                           f"{admission['file']} -> {after_facts}")
    leftovers = sorted(p.name for p in slots.iterdir() if p.name != admitted_name)
    if leftovers:
        raise RuntimeError(f"temporary files left in the slot directory: {leftovers}")

    # ---- unpatched control -----------------------------------------------------------
    third = Server(args.unpatched, args.model, str(slots), free_port(), log_dir / "c.log")
    third.start()
    try:
        control_adapter = LlamaCppHTTPAdapter(third.url, str(slots))
        supported, reason, _ = control_adapter.hybrid_support()
        if supported:
            raise RuntimeError("the unpatched runtime claimed hybrid support")
        control_refusal = reason
    finally:
        third.stop()

    total = timed_import + tail_wall
    record = {
        "kind": "direct-raw-prefix-restore-lower-bound",
        "warning": "RESEARCH ONLY. Not a production path and not an admitted design.",
        "repo_commit": repo_commit,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "storage_note": args.storage_note,
        "slots_path": str(slots),
        "prompt_tokens": args.prompt_tokens,
        "model_path": args.model,
        "architecture": gguf.architecture(args.model),
        "source_trees": {"patched": git_state(str(Path(args.patched).parent.parent.parent))},
        "binary_digests": binary_digests(Path(args.patched)),
        "build_flags": build_flags(Path(args.patched)),
        "build_info": props_a.get("build_info"),
        "kv_dtype_k": props_a.get("target_cache_type_k"),
        "kv_dtype_v": props_a.get("target_cache_type_v"),
        "admission": admission,
        "pids": {"producer": first_pid, "restorer": second_pid},
        "native_cold_wall_s": cold_wall,
        "native_reuse": {"cache_n": native["timings"]["cache_n"],
                         "prompt_n": native["timings"]["prompt_n"]},
        "control_before_restore": {"cache_n": control["timings"]["cache_n"],
                                   "prompt_n": control["timings"]["prompt_n"]},
        "timed": {"phases": phases, "import_s": timed_import,
                  "tail_completion_s": tail_wall, "total_s": total,
                  "cache_n": cache_n, "prompt_n": prompt_n, "uncovered": uncovered},
        "file_unchanged_after_restore": True,
        "unpatched_control_refusal": control_refusal,
        "verdict": {
            "direct_total_s": total,
            "native_cold_prefill_s": cold_wall,
            "restore_is_cheaper": total < cold_wall,
            "ratio_to_cold": total / cold_wall,
        },
    }
    phase_sum = sum(phases.values())
    record["timed"]["phase_sum_s"] = phase_sum
    record["timed"]["unclassified_s"] = timed_import - phase_sum
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"  admission {admission['admission_seconds']:.3f}s (untimed)")
    print(f"  timed direct import {timed_import:.3f}s + tail {tail_wall:.3f}s "
          f"= {total:.3f}s vs cold {cold_wall:.3f}s -> "
          f"{'CHEAPER' if total < cold_wall else 'not cheaper'}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
