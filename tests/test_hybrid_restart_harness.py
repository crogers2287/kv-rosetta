"""Automated restart proof: the harness owns BOTH process lifetimes.

tests/test_hybrid_checkpoint_restart.py retains in-process properties only - its docstring
says restart orchestration is the caller's job, so a full-restart result obtained by running
a shell script by hand is measured once, not proven by a retained test. This module closes
that gap: it starts llama-server, saves, stops it and verifies the process is gone, starts a
second server and verifies it is a different process, then restores and checks reuse.

Verifying the first process actually died is the point. Every later benchmark depends on
knowing the old in-memory checkpoints are gone; without that, a "restored" prefix could
simply be one that was never dropped.

    KVX_CKPT_BIN=/mnt/storage/llama-kvx-patched/build/bin/llama-server \
    KVX_CKPT_MODEL=/mnt/storage/gguf-models/OpenMythos-Q6_K.gguf \
    KVX_CKPT_SLOTS=/path/to/slots \
      python3 -m unittest tests.test_hybrid_restart_harness -v
"""

import json
import os
import signal
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

_BIN = os.environ.get("KVX_CKPT_BIN", "")
_MODEL = os.environ.get("KVX_CKPT_MODEL", "")
_SLOTS = os.environ.get("KVX_CKPT_SLOTS", "")
def _free_port() -> int:
    """Ask the OS for an unused port rather than hoping a hardcoded one is free.

    Port 8787 on this host is held by an unrelated service, which surfaced as
    "couldn't bind HTTP server socket" and looked exactly like a leaked server of our own.
    A restart harness that fights over a fixed port produces failures that have nothing to
    do with what it is testing.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


_PORT = int(os.environ["KVX_CKPT_PORT"]) if os.environ.get("KVX_CKPT_PORT") else _free_port()
_URL = f"http://127.0.0.1:{_PORT}"
_PROMPT_TOKENS = 256
_BOOT_TIMEOUT = 300
#: The hybrid test model needs roughly this much to load with its KV cache.
_MIN_FREE_VRAM_MB = float(os.environ.get("KVX_CKPT_MIN_VRAM_MB", "12000"))


def _free_vram_mb() -> float:
    """Largest free block across visible GPUs, or inf when there is no GPU to contend for."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.split()
        return max((float(x) for x in out), default=0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return float("inf")


def _post(path, payload, timeout=600):
    request = urllib.request.Request(
        _URL + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _healthy(timeout=3):
    try:
        with urllib.request.urlopen(_URL + "/health", timeout=timeout) as response:
            return b'"ok"' in response.read()
    except (urllib.error.URLError, OSError):
        return False


class _Server:
    """One llama-server process, owned outright."""

    def __init__(self):
        self.proc = None

    def start(self):
        if _healthy():
            raise RuntimeError(f"something is already listening on {_URL}; refusing to "
                               f"attribute its behaviour to this test")
        self.proc = subprocess.Popen(
            [_BIN, "--model", _MODEL, "--host", "127.0.0.1", "--port", str(_PORT),
             "-ngl", "99", "-c", "8192", "--parallel", "1", "-fa", "on",
             "--split-mode", "layer", "--tensor-split", "1,1",
             "--slot-save-path", str(Path(_SLOTS)) + "/", "--no-warmup"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.time() + _BOOT_TIMEOUT
        while time.time() < deadline:
            if self.proc.poll() is not None:
                # A harness that hides why the server died is worse than no harness.
                err = (self.proc.stderr.read() or b"").decode(errors="replace")
                tail = "\n".join(err.strip().splitlines()[-6:])
                raise RuntimeError(
                    f"server exited during boot, rc={self.proc.returncode}\n{tail}")
            if _healthy():
                return self.proc.pid
            time.sleep(2)
        self.stop()
        raise RuntimeError(f"server did not become healthy within {_BOOT_TIMEOUT}s")

    def stop(self):
        """Stop and confirm the process is really gone, not merely asked to leave."""
        if self.proc is None:
            return
        if self.proc.poll() is not None:      # already dead, just reap it
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc = None
            return
        pid = self.proc.pid
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
        deadline = time.time() + 60
        while time.time() < deadline:
            if not Path(f"/proc/{pid}").exists() and not _healthy(timeout=2):
                if self.proc.stderr is not None:
                    self.proc.stderr.close()
                self.proc = None
                return
            time.sleep(1)
        raise RuntimeError(f"pid {pid} still present after stop")


@unittest.skipUnless(_BIN and _MODEL and _SLOTS,
                     "set KVX_CKPT_BIN, KVX_CKPT_MODEL and KVX_CKPT_SLOTS to run")
class RestartOwnedByTheHarness(unittest.TestCase):
    def setUp(self):
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            try:
                server.stop()
            except Exception:
                pass

    def _require_patched(self):
        from runtime_matrix import PATCHED, detect_runtime
        actual = detect_runtime(_URL, _SLOTS)
        if actual != PATCHED:
            self.skipTest(f"restart proof needs a patched runtime; connected one is {actual}")

    def _boot(self):
        # This host shares its GPUs with a model router that reloads on demand. Taking
        # memory it is using is not this test's call, so skip loudly rather than either
        # OOM-ing or unloading someone else's models as a side effect.
        free = _free_vram_mb()
        if free < _MIN_FREE_VRAM_MB:
            self.skipTest(
                f"only {free:.0f} MiB VRAM free, need >= {_MIN_FREE_VRAM_MB}; another "
                f"process holds the GPUs. Free them deliberately before running this.")
        server = _Server()
        # Register BEFORE starting. A server that dies partway through boot has still
        # forked a process and may hold the port; registering only on success leaks it,
        # and the next run then fails with "couldn't bind HTTP server socket".
        self.servers.append(server)
        pid = server.start()
        return server, pid

    def _tokens(self):
        text = "In the year 1892 the naturalist recorded. " * 30
        return _post("/tokenize", {"content": text})["tokens"][:_PROMPT_TOKENS]

    def test_reuse_survives_a_harness_owned_restart(self):
        request_tail = {"n_predict": 8, "temperature": 0.0, "top_k": 1, "n_probs": 5,
                        "cache_prompt": True, "id_slot": 0}

        first, first_pid = self._boot()
        self._require_patched()
        ids = self._tokens()
        request = {"prompt": ids, **request_tail}

        _post("/slots/0?action=erase", {})
        cold = _post("/completion", request)
        self.assertEqual(cold["timings"]["cache_n"], 0, "cold run reused a cache")
        reference = [c["id"] for c in cold.get("completion_probabilities", [])]
        saved = _post("/slots/0?action=save", {"filename": "harness-restart.bin"})
        self.assertGreater(saved["n_saved"], 0)

        first.stop()
        self.assertFalse(Path(f"/proc/{first_pid}").exists(),
                         "the first server is still alive; its in-memory checkpoints could "
                         "explain any reuse seen afterwards")
        self.assertFalse(_healthy(timeout=2), "something still answers on the port")

        second, second_pid = self._boot()
        self.assertNotEqual(second_pid, first_pid, "no new process was started")

        # A fresh process must have nothing before a restore. Without this the test cannot
        # tell persistence from a cache that was never dropped.
        control = _post("/completion", request)
        self.assertEqual(control["timings"]["cache_n"], 0,
                         "a freshly started server reported reuse before any restore")

        _post("/slots/0?action=erase", {})
        restored = _post("/slots/0?action=restore", {"filename": "harness-restart.bin"})
        self.assertGreater(restored["n_restored"], 0)

        warm = _post("/completion", request)
        uncovered = len(ids) - warm["timings"]["cache_n"]
        self.assertGreater(warm["timings"]["cache_n"], 0,
                           "no reuse after restoring into a fresh process")
        self.assertLessEqual(uncovered, 8, f"{uncovered} tokens uncovered; expected a short tail")
        self.assertEqual(warm["timings"]["prompt_n"], uncovered,
                         "tokens reprocessed do not match the uncovered tail")
        self.assertEqual(warm["content"], cold["content"],
                         "output changed across the restart")
        self.assertEqual([c["id"] for c in warm.get("completion_probabilities", [])],
                         reference, "token IDs changed across the restart")
        second.stop()

    def test_the_harness_refuses_to_share_a_port(self):
        """If another server is already listening, this test's conclusions would be about
        that process instead."""
        first, _ = self._boot()
        with self.assertRaises(RuntimeError):
            _Server().start()
        first.stop()


if __name__ == "__main__":
    unittest.main()
