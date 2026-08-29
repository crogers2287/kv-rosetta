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
from kv_rosetta.adapters.base import (  # noqa: E402
    AdapterError,
    ExportRequest,
    ImportRequest,
    Representation,
)
from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter  # noqa: E402

PROMPT_TOKENS = 256

#: The measured uncovered tail is 4; 8 is the working ceiling from the prior steer.
MAX_UNCOVERED_TAIL = 8

#: Alternatives requested per generated token.
N_PROBS = 5

#: Restoring a checkpoint should reproduce the same forward pass, so the vectors are
#: expected to be identical; the tolerance covers only nondeterministic reduction order.
LOGPROB_TOLERANCE = 1e-6
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


def git_state(tree: str) -> dict:
    """HEAD and working-tree state of a source tree, so a build is identifiable.

    A patched tree carries its patches as uncommitted changes, so HEAD alone does not
    describe it. The diff digest does.
    """
    def run(*args):
        result = subprocess.run(["git", "-C", tree, *args], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else None

    head = (run("rev-parse", "HEAD") or "").strip()
    diff = run("diff", "HEAD")
    return {
        "tree": tree,
        "head": head or None,
        "worktree_diff_sha256": (hashlib.sha256(diff.encode()).hexdigest()
                                 if diff else None),
        "worktree_clean": diff == "" if diff is not None else None,
    }


def build_flags(binary: Path) -> dict:
    """The cmake settings a build was configured with, read from its CMakeCache.txt."""
    cache = binary.parent.parent / "CMakeCache.txt"
    if not cache.is_file():
        return {}
    wanted = ("CMAKE_BUILD_TYPE", "GGML_CUDA", "GGML_CCACHE", "LLAMA_CURL",
              "CMAKE_CUDA_ARCHITECTURES", "GGML_NATIVE")
    out = {}
    for line in cache.read_text(errors="replace").splitlines():
        name, _, value = line.partition(":")
        if name in wanted:
            out[name] = value.partition("=")[2]
    return out


def require_clean_worktree() -> str:
    """Refuse to produce evidence from an uncommitted implementation.

    The previous record named a commit that did not contain the acceptance logic, because
    the modified runner was executed before it was committed. A record whose runner cannot
    be recovered from the named commit is not reproducible evidence.
    """
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(
            "refusing to run: the worktree has uncommitted changes, so the record could "
            "not name the code that produced it. Commit the runner first.\n" + dirty)
    return subprocess.run(["git", "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


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
        self.argv = [self.binary, "--model", self.model, "--host", "127.0.0.1",
                     "--port", str(self.port), "-ngl", "99", "-c", "8192",
                     "--parallel", "1", "-fa", "on", "--split-mode", "layer",
                     "--tensor-split", "1,1",
                     "--slot-save-path", self.slots.rstrip("/") + "/", "--no-warmup"]
        with open(self.log, "wb") as out:
            self.proc = subprocess.Popen(self.argv, stdout=out, stderr=subprocess.STDOUT)
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
                   "n_probs": N_PROBS, "cache_prompt": True, "id_slot": 0}

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

        # The adapter's own view of this runtime, and a real KVX export through it. This
        # is the contract a caller actually uses; the raw endpoint measurements above say
        # nothing about whether the adapter would let them near it.
        adapter = LlamaCppHTTPAdapter(first.url, slots)
        supported, support_reason, _ = adapter.hybrid_support()
        caps = adapter.capabilities()
        adapter_view = {
            "hybrid_support": supported,
            "hybrid_support_reason": support_reason,
            "capabilities_export": sorted(r.value for r in caps.export),
            "capabilities_import": sorted(r.value for r in caps.import_),
            "active_checkpoint_state_classes": first.props().get(
                "active_checkpoint_state_classes"),
        }
        kvx_path = Path(slots) / f"matrix-{name}.kvx"
        export_started = time.time()
        try:
            adapter.export(ExportRequest(model=model, out_path=kvx_path,
                                         representation=Representation.OPAQUE, slot=0))
            adapter_view["export_refused"] = None
            adapter_view["export_seconds"] = time.time() - export_started
            adapter_view["kvx_bytes"] = kvx_path.stat().st_size
        except AdapterError as exc:
            adapter_view["export_refused"] = str(exc)
            adapter_view["export_seconds"] = time.time() - export_started
        # Capability and export must not disagree on a live runtime either.
        advertised = Representation.OPAQUE.value in adapter_view["capabilities_export"]
        if advertised != (adapter_view["export_refused"] is None):
            raise RuntimeError(
                f"{name}: capabilities advertised export={advertised} but export "
                f"{'refused' if adapter_view['export_refused'] else 'succeeded'}")
        if expect_patched and adapter_view["export_refused"]:
            raise RuntimeError(f"{name}: adapter refused to export from the patched "
                               f"runtime: {adapter_view['export_refused']}")
        if not expect_patched and not adapter_view["export_refused"]:
            raise RuntimeError(f"{name}: adapter exported from an unpatched runtime")
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
        t0 = time.time()
        warm = second.post("/completion", request2)
        tail_wall = time.time() - t0

        # End-to-end through the adapter: outer KVX verification, staging, runtime
        # restore, and the mandatory reuse probe. Save time is deliberately excluded -
        # it is not paid on the request path - but everything a caller waits for is in.
        adapter2 = LlamaCppHTTPAdapter(second.url, slots)
        if expect_patched:
            second.post("/slots/0?action=erase", {})
            started_import = time.time()
            report = adapter2.import_(kvx_path, ImportRequest(model=model, slot=0))
            adapter_view["import_seconds_end_to_end"] = time.time() - started_import
            adapter_view["import_ok"] = report.ok
            adapter_view["import_reason"] = report.reason
            adapter_view["import_reported_seconds"] = report.seconds
            adapter_view["import_tokens_restored"] = report.tokens_restored
            if not report.ok:
                raise RuntimeError(f"{name}: adapter import failed: {report.reason}")
        else:
            adapter_view["import_ok"] = None
            adapter_view["import_reason"] = "no artifact: export was refused, as required"
    finally:
        second.stop()

    def toks(response):
        return [c["id"] for c in response.get("completion_probabilities", [])]

    def probs(response):
        """The per-token alternative vectors, from the field llama.cpp actually uses.

        With the default response contract the alternatives are under `top_logprobs` and
        carry `logprob`; `top_probs`/`prob` appear only when post_sampling_probs=true is
        requested. Reading the wrong key yields a list of empty dicts, and comparing two
        such lists passes while comparing nothing at all.
        """
        out = []
        for entry in response.get("completion_probabilities", []):
            top = entry.get("top_logprobs")
            if top is None:
                raise RuntimeError(
                    f"completion entry has no top_logprobs; keys were {sorted(entry)}. "
                    f"Refusing to compare vectors that were never returned.")
            out.append({int(t["id"]): float(t["logprob"]) for t in top})
        return out

    def check_vectors(vectors, label):
        if len(vectors) != len(toks(warm)):
            raise RuntimeError(f"{name}: {label} has {len(vectors)} vectors for "
                               f"{len(toks(warm))} tokens")
        for i, (vector, token) in enumerate(zip(vectors, toks(warm))):
            if not vector:
                raise RuntimeError(f"{name}: {label} vector {i} is empty")
            if token not in vector:
                raise RuntimeError(f"{name}: {label} vector {i} omits its own token {token}")
            if len(vector) > N_PROBS:
                raise RuntimeError(f"{name}: {label} vector {i} has {len(vector)} entries, "
                                   f"more than the {N_PROBS} requested")
        return vectors

    def vectors_agree(a, b):
        """Equal keys per position and logprobs within the declared tolerance."""
        for va, vb in zip(a, b):
            if set(va) != set(vb):
                return False
            if any(abs(va[k] - vb[k]) > LOGPROB_TOLERANCE for k in va):
                return False
        return True

    # The acceptance criteria, enforced rather than merely recorded. A record that reports
    # numbers without checking them is a log, not a gate.
    warm_cache_n = warm["timings"]["cache_n"]
    warm_prompt_n = warm["timings"]["prompt_n"]
    problems = []
    if toks(cold) != toks(warm) or cold["content"] != warm["content"]:
        problems.append("restored output differs from the cold run")
    if toks(native) != toks(warm) or native["content"] != warm["content"]:
        problems.append("restored output differs from native in-memory reuse")
    native_vectors = check_vectors(probs(native), "native")
    warm_vectors = check_vectors(probs(warm), "restored")
    if not vectors_agree(native_vectors, warm_vectors):
        problems.append("restored probability vectors differ from native in-memory reuse "
                        f"by more than {LOGPROB_TOLERANCE}")
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
        "build_flags": build_flags(Path(binary)),
        "launch_argv": first.argv,
        "second_launch_argv": second.argv,
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
                               "top_logprobs": warm_vectors},
        "native_reuse_top_logprobs": native_vectors,
        "logprob_tolerance": LOGPROB_TOLERANCE,
        "restore_wall_s": restore_wall,
        "tail_completion_wall_s": tail_wall,
        "adapter": adapter_view,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("--unpatched", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--out", default="bench/production-27b-matrix.json")
    ap.add_argument("--patches-dir", default="patches/llama.cpp")
    ap.add_argument("--deployment-note", default="",
                    help="whether this exact model file is the deployed production SKU")
    args = ap.parse_args()
    repo_commit = require_clean_worktree()

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
        # Proven by the runner digest, not merely asserted: the worktree was clean, so this
        # commit contains the exact file below.
        "repo_commit": repo_commit,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_trees": {
            "patched": git_state(str(Path(args.patched).parent.parent.parent)),
            "unpatched": git_state(str(Path(args.unpatched).parent.parent.parent)),
        },
        "deployment_identity": args.deployment_note,
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
