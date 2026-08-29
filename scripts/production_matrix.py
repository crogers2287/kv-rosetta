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
import resource
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

#: Overridden by --prompt-tokens; the ladder rungs are 256, 2048, 8192, 32768.
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


def peak_rss_bytes(pid: int) -> int | None:
    """VmHWM: the high-water mark of a process's resident set, read before it exits."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def vram_used_bytes() -> list[int]:
    """Per-device VRAM in use, so the artifact's cost in memory is recorded, not assumed."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [int(line) * 1024 * 1024 for line in result.stdout.split() if line.isdigit()]


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
    def __init__(self, binary: str, model: str, slots: str, port: int, log: Path,
                 n_ctx: int = 8192):
        self.binary, self.model, self.slots, self.port = binary, model, slots, port
        self.n_ctx = n_ctx
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


def toks(response):
    return [c["id"] for c in response.get("completion_probabilities", [])]


def probs(response):
    """The per-token alternative vectors, from the field llama.cpp actually uses.

    With the default response contract the alternatives are under `top_logprobs` and carry
    `logprob`; `top_probs`/`prob` appear only when post_sampling_probs=true is requested.
    Reading the wrong key yields a list of empty dicts, and comparing two such lists passes
    while comparing nothing at all.
    """
    out = []
    for entry in response.get("completion_probabilities", []):
        top = entry.get("top_logprobs")
        if top is None:
            raise RuntimeError(f"completion entry has no top_logprobs; keys were "
                               f"{sorted(entry)}. Refusing to compare vectors that were "
                               f"never returned.")
        out.append({int(t["id"]): float(t["logprob"]) for t in top})
    return out


def check_vectors(vectors, label, expected_tokens, leg):
    if len(vectors) != len(expected_tokens):
        raise RuntimeError(f"{leg}: {label} has {len(vectors)} vectors for "
                           f"{len(expected_tokens)} tokens")
    for i, (vector, token) in enumerate(zip(vectors, expected_tokens)):
        if not vector:
            raise RuntimeError(f"{leg}: {label} vector {i} is empty")
        if token not in vector:
            raise RuntimeError(f"{leg}: {label} vector {i} omits its own token {token}")
        if len(vector) > N_PROBS:
            raise RuntimeError(f"{leg}: {label} vector {i} has {len(vector)} entries, "
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


def prompt_text(tokens: int) -> str:
    """Deterministic source text long enough to yield `tokens` tokens.

    Roughly ten tokens per repetition; over-generate and slice, so the exact token count is
    the tokenizer's answer rather than an assumption about words per token.
    """
    return "In the year 1892 the naturalist recorded. " * max(4, tokens // 4)


#: Scheduling noise tolerated when reconciling named phases against the reported total.
PHASE_TOLERANCE_S = 0.05


def reconcile_phases(report_seconds: float, phases: dict) -> dict:
    """How much of a reported import second is attributed to a named phase.

    An economic conclusion that cannot say where its time went is not an attribution. The
    unclassified remainder is recorded rather than absorbed into a neighbouring phase.
    """
    total = sum(phases.values())
    return {
        "phase_sum_s": total,
        "reported_seconds": report_seconds,
        "unclassified_s": report_seconds - total,
        "reconciled": abs(report_seconds - total) <= PHASE_TOLERANCE_S,
    }


def compute_verdict(patched_leg: dict) -> dict | None:
    """The steer's decision rule, from the timers that belong to the adapter path.

    The tail must be the completion issued after the ADAPTER import, not the one after the
    raw-endpoint restore. Substituting a neighbouring measurement would report a number for
    a path that was never timed.
    """
    adapter = patched_leg.get("adapter", {})
    if adapter.get("import_seconds_end_to_end") is None:
        return None
    tail = adapter.get("adapter_tail_completion_wall_s")
    if tail is None:
        raise RuntimeError("adapter import succeeded but its tail completion was not timed; "
                           "refusing to substitute the raw-endpoint tail")
    total = adapter["import_seconds_end_to_end"] + tail
    cold = patched_leg["cold"]["wall_s"]
    return {
        "adapter_import_s": adapter["import_seconds_end_to_end"],
        "adapter_tail_completion_s": tail,
        "adapter_import_plus_tail_s": total,
        "native_cold_prefill_s": cold,
        "restore_is_cheaper": total < cold,
        "ratio_to_cold": total / cold,
        # Kept for diagnostic comparison only; never used in the verdict above.
        "raw_endpoint_tail_s": patched_leg.get("tail_completion_wall_s"),
        "phase_reconciliation": reconcile_phases(
            adapter.get("import_reported_seconds", 0.0),
            adapter.get("import_phases", {})),
    }


def record_calls(adapter):
    """Wrap _post so every endpoint call is recorded in order.

    Lets "refused before any save/restore POST" be checked mechanically rather than
    asserted in prose.
    """
    adapter.calls = []
    original = adapter._post

    def logging_post(path, payload, *args, **kwargs):
        adapter.calls.append(path)
        return original(path, payload, *args, **kwargs)

    adapter._post = logging_post
    return adapter


def describe_artifact(path: Path) -> dict:
    """Everything needed to identify a KVX artifact without re-reading its payload."""
    from kv_rosetta import container

    header = container.read_header(path)
    blob = header.get("blob", {})
    coverage = header.get("coverage", {})
    total = path.stat().st_size
    checkpoint_bytes = coverage.get("checkpoint_bytes")
    return {
        "container_sha256": sha256_file(path),
        "payload_sha256": blob.get("sha256"),
        "opaque_format": blob.get("opaque_format"),
        "coverage": coverage,
        "artifact_key": header.get("artifact_key"),
        "total_bytes": total,
        "payload_bytes": blob.get("nbytes"),
        "checkpoint_bytes": checkpoint_bytes,
        "sequence_bytes": (blob.get("nbytes") - checkpoint_bytes
                           if isinstance(blob.get("nbytes"), int)
                           and isinstance(checkpoint_bytes, int) else None),
        "container_overhead_bytes": (total - blob["nbytes"]
                                     if isinstance(blob.get("nbytes"), int) else None),
    }


def run_leg(name: str, binary: str, model: str, slots: str, expect_patched: bool,
            log_dir: Path, patched_artifact: Path | None = None) -> dict:
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

        text = prompt_text(PROMPT_TOKENS)
        ids = first.post("/tokenize", {"content": text})["tokens"][:PROMPT_TOKENS]
        if len(ids) != PROMPT_TOKENS:
            raise RuntimeError(f"{name}: tokenizer produced {len(ids)} tokens, wanted "
                               f"{PROMPT_TOKENS}; lengthen the source text")
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
        native_vectors = check_vectors(probs(native), "native", toks(native), name)

        saved = first.post("/slots/0?action=save", {"filename": slot_file})
        artifact = Path(slots) / slot_file
        artifact_digest = sha256_file(artifact)
        artifact_bytes = artifact.stat().st_size
        memory = {"server_peak_rss_bytes_first": peak_rss_bytes(first_pid),
                  "vram_used_bytes_after_load": vram_used_bytes()}

        # The adapter's own view of this runtime, and a real KVX export through it. This
        # is the contract a caller actually uses; the raw endpoint measurements above say
        # nothing about whether the adapter would let them near it.
        adapter = record_calls(LlamaCppHTTPAdapter(first.url, slots))
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
        # A distinct stem: export() derives its slot filename from the artifact stem and
        # unlinks that file afterwards, which would delete the raw artifact under test.
        # Export the PREFIX, not the prefix plus this run's generation. The raw legs save
        # a slot holding 256 prompt + 8 generated tokens and then replay a 256-token
        # request, so their uncovered tail is 256 - 252 = 4. An artifact carrying all 263
        # tokens has a tail of 11, because the checkpoint still covers 252: the tail is a
        # function of how far the sequence ran past the last checkpoint, not a constant.
        # A prefix cache holds the prefix, so that is what the adapter exports.
        first.post("/slots/0?action=erase", {})
        first.post("/completion", dict(request, n_predict=0))
        kvx_path = Path(slots) / f"matrix-{name}-adapter.kvx"
        # Capability discovery itself issues a save: state_version() probes the emitted
        # sequence version by saving a slot, because no endpoint reports it. That POST is
        # real and is recorded, but it is not part of the export, so the "refused before
        # any save POST" window starts here.
        adapter_view["calls_during_capability_probe"] = list(adapter.calls)
        adapter.calls.clear()
        export_started = time.time()
        try:
            adapter.export(ExportRequest(model=model, out_path=kvx_path,
                                         representation=Representation.OPAQUE, slot=0))
            adapter_view["export_refused"] = None
            adapter_view["export_seconds"] = time.time() - export_started
            adapter_view["kvx_bytes"] = kvx_path.stat().st_size
            adapter_view["artifact"] = describe_artifact(kvx_path)
            adapter_view["kvx_path"] = str(kvx_path)
        except AdapterError as exc:
            adapter_view["export_refused"] = str(exc)
            adapter_view["export_seconds"] = time.time() - export_started
        # "Before any save POST" is mechanically checkable, not asserted in prose.
        adapter_view["calls_during_export"] = list(adapter.calls)
        if adapter_view["export_refused"] and any(
                "action=save" in call for call in adapter.calls):
            raise RuntimeError(
                f"{name}: export refused only after a save POST.\n"
                f"  calls during export: {adapter.calls}\n"
                f"  hybrid_support: {supported} ({support_reason})\n"
                f"  refusal: {adapter_view['export_refused']}")

        # Capability and the support predicate must agree on a live runtime as they do
        # offline. An artifact-level refusal - incomplete checkpoint coverage on this
        # particular save - is a different thing and is reported separately.
        advertised = Representation.OPAQUE.value in adapter_view["capabilities_export"]
        if advertised != supported:
            raise RuntimeError(
                f"{name}: capabilities advertised export={advertised} but the support "
                f"predicate said {supported} ({support_reason})")
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
        ids2 = second.post("/tokenize", {"content": prompt_text(PROMPT_TOKENS)}
                           )["tokens"][:PROMPT_TOKENS]
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
        adapter2 = record_calls(LlamaCppHTTPAdapter(second.url, slots))
        if expect_patched:
            second.post("/slots/0?action=erase", {})
            adapter2.calls.clear()
            started_import = time.time()
            report = adapter2.import_(kvx_path, ImportRequest(model=model, slot=0))
            adapter_view["import_seconds_end_to_end"] = time.time() - started_import
            adapter_view["import_ok"] = report.ok
            adapter_view["import_reason"] = report.reason
            adapter_view["import_reported_seconds"] = report.seconds
            adapter_view["import_tokens_restored"] = report.tokens_restored
            adapter_view["import_phases"] = dict(report.phases)
            adapter_view["calls_during_import"] = list(adapter2.calls)
            if not report.ok:
                raise RuntimeError(f"{name}: adapter import failed: {report.reason}")
            # The staged copy is the size of the cache; leaving it behind fills the disk.
            leftover = sorted(str(x) for x in Path(slots).glob("*.restore.bin"))
            adapter_view["staged_copies_left"] = leftover
            if leftover:
                raise RuntimeError(f"{name}: staged copies not removed: {leftover}")
            # Parity of the adapter-restored cache against native in-memory reuse, not
            # just of the raw-endpoint restore measured above.
            # Timed here, not borrowed from the raw path. Those states should behave
            # alike, but an economic record must measure the path it names.
            t0 = time.time()
            after_adapter = second.post("/completion", request2)
            adapter_view["adapter_tail_completion_wall_s"] = time.time() - t0
            adapter_view["after_adapter_import"] = {
                "cache_n": after_adapter["timings"]["cache_n"],
                "prompt_n": after_adapter["timings"]["prompt_n"],
                "tokens": toks(after_adapter),
                "content": after_adapter["content"],
            }
            adapter_vectors = check_vectors(probs(after_adapter), "adapter-restored",
                                            toks(after_adapter), name)
            if toks(after_adapter) != toks(native) or \
                    after_adapter["content"] != native["content"]:
                raise RuntimeError(f"{name}: output after adapter import differs from "
                                   f"native in-memory reuse")
            if not vectors_agree(adapter_vectors, native_vectors):
                raise RuntimeError(f"{name}: probability vectors after adapter import "
                                   f"differ from native in-memory reuse")
            adapter_view["after_adapter_top_logprobs"] = adapter_vectors
        else:
            adapter_view["import_ok"] = None
            adapter_view["import_reason"] = "no artifact: export was refused, as required"
            # The unpatched runtime must also refuse an artifact the patched one produced,
            # before staging or restoring it - and verify_reuse=False must not change that.
            if patched_artifact and Path(patched_artifact).is_file():
                refusals = {}
                for verify in (True, False):
                    adapter2.calls.clear()
                    report = adapter2.import_(Path(patched_artifact),
                                              ImportRequest(model=model, slot=0),
                                              verify_reuse=verify)
                    refusals[f"verify_reuse={verify}"] = {
                        "ok": report.ok, "reason": report.reason,
                        "calls": list(adapter2.calls),
                        "phases": dict(report.phases),
                    }
                    if report.ok:
                        raise RuntimeError(f"{name}: unpatched runtime accepted a patched "
                                           f"compound artifact (verify_reuse={verify})")
                    if any("action=restore" in call for call in adapter2.calls):
                        raise RuntimeError(f"{name}: refused only after a restore POST: "
                                           f"{adapter2.calls}")
                    staged = sorted(Path(slots).glob("*.restore.bin"))
                    if staged:
                        raise RuntimeError(f"{name}: staged copies left behind: {staged}")
                adapter_view["cross_import_refusals"] = refusals
        memory["server_peak_rss_bytes_second"] = peak_rss_bytes(second_pid)
        memory["vram_used_bytes_after_restore"] = vram_used_bytes()
        memory["runner_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    finally:
        second.stop()

    # The acceptance criteria, enforced rather than merely recorded. A record that reports
    # numbers without checking them is a log, not a gate.
    warm_cache_n = warm["timings"]["cache_n"]
    warm_prompt_n = warm["timings"]["prompt_n"]
    problems = []
    if toks(cold) != toks(warm) or cold["content"] != warm["content"]:
        problems.append("restored output differs from the cold run")
    if toks(native) != toks(warm) or native["content"] != warm["content"]:
        problems.append("restored output differs from native in-memory reuse")
    warm_vectors = check_vectors(probs(warm), "restored", toks(warm), name)
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
        # The live target context's cache types, from the key the runtime actually
        # advertises. Reading a key nothing populates records an empty string beside a
        # successful import that required a non-empty one - a record contradicting itself.
        "kv_dtype_k": str(props.get("target_cache_type_k", "")),
        "kv_dtype_v": str(props.get("target_cache_type_v", "")),
        "draft_kv_dtype_k": str(props.get("draft_cache_type_k", "")),
        "draft_kv_dtype_v": str(props.get("draft_cache_type_v", "")),
        # Recorded alongside so the record itself shows they are different things.
        "model_ftype": str(props.get("model_ftype", "")),
        "prompt_tokens_requested": PROMPT_TOKENS,
        "memory": memory,
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
    global PROMPT_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("--unpatched", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--out", default="bench/production-27b-matrix.json")
    ap.add_argument("--patches-dir", default="patches/llama.cpp")
    ap.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS,
                    help="exact prompt tokens; the ladder rungs are 256, 2048, 8192, 32768")
    ap.add_argument("--repeats", type=int, default=1,
                    help="clean repetitions, each with fresh processes for both legs")
    ap.add_argument("--storage-note", default="",
                    help="where --slots lives: tmpfs isolates compute and serialization, "
                         "NVMe measures the deployable path")
    ap.add_argument("--deployment-note", default="",
                    help="whether this exact model file is the deployed production SKU")
    args = ap.parse_args()
    repo_commit = require_clean_worktree()
    PROMPT_TOKENS = args.prompt_tokens

    log_dir = Path(args.slots).parent / "matrix-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.slots).mkdir(parents=True, exist_ok=True)

    repetitions = []
    for repetition in range(1, args.repeats + 1):
        legs = {}
        patched_artifact = None
        # The patched leg runs first so its artifact can be offered to the unpatched
        # runtime, which must refuse it before staging or restoring.
        for name, binary, expect in (("patched", args.patched, True),
                                     ("unpatched", args.unpatched, False)):
            print(f"=== repetition {repetition}/{args.repeats}: {name} leg ===", flush=True)
            legs[name] = run_leg(name, binary, args.model, args.slots, expect, log_dir,
                                 patched_artifact=patched_artifact)
            if expect:
                kept = Path(args.slots) / "patched-artifact.kvx"
                produced = Path(legs[name]["adapter"].get("kvx_path", ""))
                if produced.is_file():
                    produced.replace(kept)
                    patched_artifact = kept
            w = legs[name]["warm_after_restore"]
            print(f"  after restore cache_n={w['cache_n']} prompt_n={w['prompt_n']}",
                  flush=True)
        # The steer's decision rule, evaluated rather than left to the reader.
        patched = legs["patched"]
        verdict = compute_verdict(patched)
        if verdict:
            rec = verdict["phase_reconciliation"]
            print(f"  decision: {verdict['adapter_import_plus_tail_s']:.3f}s adapter+tail "
                  f"vs {verdict['native_cold_prefill_s']:.3f}s cold -> "
                  f"{'CHEAPER' if verdict['restore_is_cheaper'] else 'not cheaper'}",
                  flush=True)
            print(f"  phases account for {rec['phase_sum_s']:.3f}s of "
                  f"{rec['reported_seconds']:.3f}s "
                  f"({rec['unclassified_s']:.3f}s unclassified)", flush=True)
            # Required, not merely reported. An economic record whose time cannot be
            # attributed is not an attribution, and a gap here means a code path was added
            # outside every named phase.
            if not rec["reconciled"]:
                raise RuntimeError(
                    f"phase attribution does not reconcile: {rec['phase_sum_s']:.3f}s of "
                    f"{rec['reported_seconds']:.3f}s named, "
                    f"{rec['unclassified_s']:.3f}s unclassified (tolerance "
                    f"{PHASE_TOLERANCE_S}s). Some work is outside every phase.")
        repetitions.append({"repetition": repetition, "legs": legs, "verdict": verdict})
        if patched_artifact and Path(patched_artifact).is_file():
            Path(patched_artifact).unlink()

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
        "slots_path": str(args.slots),
        "storage_note": args.storage_note,
        "repeats": args.repeats,
        "repetitions": repetitions,
        # The first repetition's legs, kept at the old key so existing readers still work.
        "legs": repetitions[0]["legs"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
