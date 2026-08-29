#!/usr/bin/env python3
"""Paired patched/unpatched restart matrix on the production 27B.

Both legs run the same lifecycle - start, prime, save, stop, start fresh, restore, measure -
against the same model and prompt, differing only in the binary. One JSON record is written
only when BOTH legs complete: a skipped leg is not a passing matrix, and the unpatched leg is
the half that proves the patched result means anything.

The unpatched leg FAILS if it is handed a binary that advertises checkpoint persistence,
rather than skipping. Silently dropping the negative half would leave a record that looks
complete and proves nothing.

    python3 scripts/production_matrix.py \
        --model /mnt/ai-models/qwen38-27b/Qwen3.8-27B-UD-Q5_K_XL.gguf \
        --patched /mnt/storage/llama-kvx-patched/build/bin/llama-server \
        --unpatched ~/llama.cpp/build/bin/llama-server \
        --slots /path/to/slots --out bench/production-27b-matrix.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kv_rosetta import gguf, weights  # noqa: E402

PROMPT_TOKENS = 256

#: The measured uncovered tail is 4; 8 is the working ceiling from the prior steer.
MAX_UNCOVERED_TAIL = 8
# The production 27B lives on the rclone VFS mount, which reads well under 100 MB/s
# cold. The first load of a leg warms it; later loads are local-cache speed.
BOOT_TIMEOUT = 1800

#: llama-swap's unload endpoint - the sanctioned way to reclaim the fleet's GPUs.
FLEET_UNLOAD = "http://127.0.0.1:9069/unload"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def binary_digests(binary: Path) -> dict:
    """Digest the launcher AND the implementation it loads.

    llama-server is a ~18 KB launcher; the server code lives in
    libllama-server-impl.so beside it. Hashing only the launcher would give the
    patched and unpatched legs identical digests and prove nothing about which
    build actually ran.
    """
    out = {"launcher_sha256": sha256_file(binary)}
    for name in ("libllama-server-impl.so", "libllama.so", "libggml-base.so"):
        sibling = binary.parent / name
        if sibling.exists():
            out[name + "_sha256"] = sha256_file(sibling)
    impl = binary.parent / "libllama-server-impl.so"
    if impl.exists():
        # Source-side evidence, independent of what the server advertises at runtime.
        # The SCKP magic is a constexpr uint32, not a string, so it never appears in the
        # binary - these are the strings the patch actually emits.
        blob = impl.read_bytes()
        out["patch_markers"] = {
            marker: blob.count(marker.encode())
            for marker in ("sckp/1", "slot_checkpoint_persistence",
                           "context checkpoint appendix")
        }
    if len(out) == 1:
        raise RuntimeError(f"no implementation library found beside {binary}; "
                           f"a launcher digest alone cannot identify the build")
    return out


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class Server:
    def __init__(self, binary: str, model: str, slots: str, port: int, log: Path):
        self.binary, self.model, self.slots, self.port = binary, model, slots, port
        self.log, self.proc = log, None
        self.url = f"http://127.0.0.1:{port}"

    def healthy(self, timeout=3) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=timeout) as r:
                return b'"ok"' in r.read()
        except (urllib.error.URLError, OSError):
            return False

    def start(self, attempts: int = 3) -> int:
        """Start the server, yielding to the fleet if it has reclaimed the GPUs.

        llama-swap loads a model whenever real traffic arrives, so a load started while
        the fleet is idle can still lose the GPUs partway through. That is the fleet
        behaving correctly, not a failure of the matrix, so we unload and retry rather
        than recording a leg that never ran.
        """
        for attempt in range(1, attempts + 1):
            try:
                return self._start_once()
            except RuntimeError as exc:
                if "out of memory" not in str(exc) and "unable to allocate" not in str(exc):
                    raise
                if attempt == attempts:
                    raise
                print(f"    fleet reclaimed the GPUs; unloading and retrying "
                      f"({attempt}/{attempts - 1})", flush=True)
                try:
                    urllib.request.urlopen(FLEET_UNLOAD, timeout=120).read()
                except (urllib.error.URLError, OSError) as unload_error:
                    raise RuntimeError(f"could not unload the fleet: {unload_error}") from exc
                time.sleep(10)
        raise AssertionError("unreachable")

    def _start_once(self) -> int:
        if self.healthy():
            raise RuntimeError(f"{self.url} already answers; refusing to attribute its "
                               f"behaviour to this run")
        with open(self.log, "wb") as out:
            self.proc = subprocess.Popen(
                [self.binary, "--model", self.model, "--host", "127.0.0.1",
                 "--port", str(self.port), "-ngl", "99", "-c", "8192", "--parallel", "1",
                 "-fa", "on", "--split-mode", "layer", "--tensor-split", "1,1",
                 "--slot-save-path", self.slots.rstrip("/") + "/", "--no-warmup"],
                stdout=out, stderr=subprocess.STDOUT)
        started = time.time()
        deadline = started + BOOT_TIMEOUT
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = "\n".join(Path(self.log).read_text(errors="replace").splitlines()[-6:])
                raise RuntimeError(f"server exited rc={self.proc.returncode}\n{tail}")
            if self.healthy():
                return self.proc.pid
            waited = int(time.time() - started)
            if waited and waited % 120 < 2:
                print(f"    ...still loading ({waited}s)", flush=True)
            time.sleep(2)
        self.stop()
        raise RuntimeError(f"server not healthy within {BOOT_TIMEOUT}s")

    def stop(self) -> None:
        if self.proc is None:
            return
        pid = self.proc.pid
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        deadline = time.time() + 90
        while time.time() < deadline:
            if not Path(f"/proc/{pid}").exists() and not self.healthy(timeout=2):
                self.proc = None
                return
            time.sleep(1)
        raise RuntimeError(f"pid {pid} still present after stop")

    def post(self, path: str, payload: dict, timeout=900):
        request = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return json.loads(r.read())

    def props(self) -> dict:
        with urllib.request.urlopen(self.url + "/props", timeout=60) as r:
            return json.loads(r.read())


def protocol_of(props: dict) -> dict:
    if not props.get("slot_checkpoint_persistence"):
        return {}
    return {
        "format": props.get("slot_checkpoint_format"),
        "sequence_state_version": props.get("sequence_state_version"),
        "supports_target": props.get("supports_target_checkpoint_state"),
        "supports_draft": props.get("supports_draft_checkpoint_state"),
        "supports_speculative": props.get("supports_speculative_checkpoint_state"),
    }


def run_leg(name: str, binary: str, model: str, slots: str, expect_patched: bool,
            log_dir: Path) -> dict:
    """One full save / stop / start / restore lifecycle."""
    slot_file = "matrix-" + name + ".bin"
    port = free_port()
    first = Server(binary, model, slots, port, log_dir / f"{name}-1.log")
    first_pid = first.start()
    try:
        props = first.props()
        protocol = protocol_of(props)
        # The negative half must not be quietly dropped: a patched binary here would make
        # the record look complete while proving nothing.
        if expect_patched and not protocol:
            raise RuntimeError(f"{name}: expected a patched binary, but it advertises no "
                               f"checkpoint protocol")
        if not expect_patched and protocol:
            raise RuntimeError(f"{name}: expected an UNPATCHED binary, but it advertises "
                               f"{protocol}. Refusing to skip the negative leg.")

        text = "In the year 1892 the naturalist recorded. " * 40
        ids = first.post("/tokenize", {"content": text})["tokens"][:PROMPT_TOKENS]
        request = {"prompt": ids, "n_predict": 8, "temperature": 0.0, "top_k": 1,
                   "n_probs": 5, "cache_prompt": True, "id_slot": 0}

        first.post("/slots/0?action=erase", {})
        t0 = time.time()
        cold = first.post("/completion", request)
        cold_wall = time.time() - t0
        if cold["timings"]["cache_n"] != 0:
            raise RuntimeError(f"{name}: cold run reused a cache")

        # Native in-memory reuse, the behaviour a restore must reproduce.
        native = first.post("/completion", request)

        saved = first.post("/slots/0?action=save", {"filename": slot_file})
        artifact = Path(slots) / slot_file
        artifact_digest = sha256_file(artifact)
        artifact_bytes = artifact.stat().st_size
    finally:
        first.stop()

    if Path(f"/proc/{first_pid}").exists():
        raise RuntimeError(f"{name}: first process {first_pid} survived stop")
    if first.healthy(timeout=2):
        raise RuntimeError(f"{name}: port {port} still answers after the process exited; "
                           f"a later restore could be measured against the old process")

    second = Server(binary, model, slots, free_port(), log_dir / f"{name}-2.log")
    second_pid = second.start()
    try:
        if second_pid == first_pid:
            raise RuntimeError(f"{name}: no new process was started")
        text = "In the year 1892 the naturalist recorded. " * 40
        ids2 = second.post("/tokenize", {"content": text})["tokens"][:PROMPT_TOKENS]
        if ids2 != ids:
            raise RuntimeError(f"{name}: tokenization differs across processes")
        request2 = dict(request, prompt=ids2)

        control = second.post("/completion", request2)
        if control["timings"]["cache_n"] != 0:
            raise RuntimeError(f"{name}: fresh process reported reuse before any restore")

        if sha256_file(artifact) != artifact_digest:
            raise RuntimeError(f"{name}: artifact changed between processes")

        second.post("/slots/0?action=erase", {})
        t0 = time.time()
        restored = second.post("/slots/0?action=restore", {"filename": slot_file})
        restore_wall = time.time() - t0
        warm = second.post("/completion", request2)
    finally:
        second.stop()

    def toks(response):
        return [c["id"] for c in response.get("completion_probabilities", [])]

    def probs(response):
        out = []
        for entry in response.get("completion_probabilities", []):
            top = entry.get("top_probs", entry.get("probs", []))
            out.append({str(t["id"]): t.get("prob", t.get("logprob")) for t in top})
        return out

    # The acceptance criteria, enforced rather than merely recorded. A record that reports
    # numbers without checking them is a log, not a gate.
    warm_cache_n = warm["timings"]["cache_n"]
    warm_prompt_n = warm["timings"]["prompt_n"]
    problems = []
    if toks(cold) != toks(warm) or cold["content"] != warm["content"]:
        problems.append("restored output differs from the cold run")
    if toks(native) != toks(warm) or native["content"] != warm["content"]:
        problems.append("restored output differs from native in-memory reuse")
    if probs(native) != probs(warm):
        problems.append("restored probability vectors differ from native in-memory reuse")
    if warm_cache_n + warm_prompt_n != len(ids):
        problems.append(f"cache_n + prompt_n = {warm_cache_n + warm_prompt_n}, not {len(ids)}")
    if expect_patched:
        declared = int(saved.get("checkpoint_n_tokens", 0) or 0)
        if warm_cache_n != declared:
            problems.append(f"cache_n {warm_cache_n} != declared checkpoint_n_tokens {declared}")
        if not 1 <= warm_prompt_n <= MAX_UNCOVERED_TAIL:
            problems.append(f"uncovered tail {warm_prompt_n} outside 1..{MAX_UNCOVERED_TAIL}")
        for field in ("checkpoint_bytes", "checkpoint_n_tokens",
                      "checkpoint_pos_min", "checkpoint_pos_max"):
            if saved.get(field) != restored.get(field):
                problems.append(f"{field} differs between save and restore: "
                                f"{saved.get(field)} vs {restored.get(field)}")
        if saved.get("n_checkpoints_saved") != restored.get("n_checkpoints_restored"):
            problems.append("checkpoint count differs between save and restore")
    else:
        if warm_cache_n != 0 or warm_prompt_n != len(ids):
            problems.append(f"unpatched leg reused a prefix: cache_n={warm_cache_n}")
    if problems:
        raise RuntimeError(f"{name}: " + "; ".join(problems))

    settings = props.get("default_generation_settings", {}) or {}
    return {
        "active_checkpoint_state_classes": props.get("active_checkpoint_state_classes"),
        "acceptance_checked": True,
        "leg": name,
        "expect_patched": expect_patched,
        "binary": binary,
        "binary_digests": binary_digests(Path(binary)),
        "build_info": props.get("build_info"),
        "protocol": protocol,
        "model_path": model,
        "model_content_digest": weights.model_content_digest(model),
        "architecture": gguf.architecture(model),
        "token_count": len(ids),
        "token_ids_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "n_ctx": settings.get("n_ctx"),
        "first_pid": first_pid,
        "second_pid": second_pid,
        "artifact_sha256": artifact_digest,
        "artifact_bytes": artifact_bytes,
        "save_response": saved,
        "restore_response": restored,
        "cold": {"cache_n": cold["timings"]["cache_n"],
                 "prompt_n": cold["timings"]["prompt_n"],
                 "prompt_ms": cold["timings"]["prompt_ms"],
                 "wall_s": cold_wall,
                 "tokens": toks(cold), "content": cold["content"]},
        "native_reuse": {"cache_n": native["timings"]["cache_n"],
                         "prompt_n": native["timings"]["prompt_n"],
                         "tokens": toks(native), "content": native["content"]},
        "control_after_restart": {"cache_n": control["timings"]["cache_n"],
                                  "prompt_n": control["timings"]["prompt_n"]},
        "warm_after_restore": {"cache_n": warm["timings"]["cache_n"],
                               "prompt_n": warm["timings"]["prompt_n"],
                               "prompt_ms": warm["timings"]["prompt_ms"],
                               "tokens": toks(warm), "content": warm["content"],
                               "top_probs": probs(warm)},
        "restore_wall_s": restore_wall,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("--unpatched", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--out", default="bench/production-27b-matrix.json")
    ap.add_argument("--patches-dir", default="patches/llama.cpp")
    args = ap.parse_args()

    log_dir = Path(args.slots).parent / "matrix-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.slots).mkdir(parents=True, exist_ok=True)

    legs = {}
    for name, binary, expect in (("patched", args.patched, True),
                                 ("unpatched", args.unpatched, False)):
        print(f"=== {name} leg ===", flush=True)
        legs[name] = run_leg(name, binary, args.model, args.slots, expect, log_dir)
        w = legs[name]["warm_after_restore"]
        print(f"  pids {legs[name]['first_pid']} -> {legs[name]['second_pid']} | "
              f"after restore cache_n={w['cache_n']} prompt_n={w['prompt_n']}", flush=True)

    patches = {}
    for patch in sorted(Path(args.patches_dir).glob("*.patch")):
        patches[patch.name] = sha256_file(patch)

    record = {
        "kind": "production-27b-paired-restart-matrix",
        "repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip(),
        "patches": patches,
        "prompt_tokens": PROMPT_TOKENS,
        "legs": legs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
