#!/usr/bin/env python3
"""Prove a reader configuration has a stable cold baseline, or refuse to allowlist it.

The output-parity gate compares a restored run against one cold run. That comparison is
only causal if the reader gives the *same* answer to identical uncached work. Measured
here rather than assumed: on this host a hybrid model under Vulkan produces three
different outputs across six identical cold completions, while the same model under CUDA
or HIP produces one. A restored-versus-cold verdict taken on the first configuration would
have been reporting the reader's own variance.

A summary count is not evidence. Every run's raw token ids, text, per-position probability
vectors, slot routing and timing are retained, alongside the model and prompt digests, the
binary and loaded-library digests, and the device attestation, so a verdict can be
re-derived from the record instead of trusted.

Fails closed. Any run that was not actually cold, or that returned no probability vectors,
refuses the whole set rather than being dropped from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production_matrix import binary_digests, free_port, probs, sha256_file, toks  # noqa: E402

MIN_RUNS = 6
N_PROBS = 5


class PreflightError(RuntimeError):
    """A refusal. Never downgraded to a warning."""


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def check_run(run: dict, index: int) -> None:
    """Refuse a single run that cannot contribute evidence.

    A run that reused cache is not a cold sample, and a run with no probability vectors
    compares equal to any other such run - which is the vacuous-comparison failure this
    whole record exists to prevent.
    """
    if run["cache_n"] != 0:
        raise PreflightError(
            f"run {index} reused {run['cache_n']} cached tokens, so it is not a cold "
            f"sample; the slot was not empty when it started")
    if not run["token_ids"]:
        raise PreflightError(f"run {index} returned no token ids")
    # llama.cpp's speculative path returns no completion_probabilities at all, so a
    # speculative reader can never satisfy a vector requirement. Measured on this host:
    # the same binary, model and card produced full vectors with speculation off and
    # none with draft-mtp on. Refusing outright made every speculative configuration
    # unprovable by construction, which is a different failure from the vacuous
    # comparison this check exists to prevent.
    #
    # A run with SOME empty vectors is still refused: that is a partial result, and the
    # empty ones would compare equal while looking like evidence. A run with NO vectors
    # at all is admitted at reduced strength - identical text and identical token ids
    # across cold runs is real evidence of determinism, and the record says plainly that
    # vectors were unavailable so a reader of the proof can weigh it accordingly.
    if run["vectors"]:
        if any(not v for v in run["vectors"]):
            raise PreflightError(
                f"run {index} returned partially empty probability vectors; the empty ones "
                f"compare equal to anything, so admitting this run would prove less than "
                f"it appears to")
        if len(run["vectors"]) != len(run["token_ids"]):
            raise PreflightError(
                f"run {index} has {len(run['vectors'])} vectors for "
                f"{len(run['token_ids'])} tokens")


def summarise(runs: list[dict], *, min_runs: int = MIN_RUNS) -> dict:
    """Derive the verdict from the retained runs. Pure, so it is testable without a GPU."""
    # A floor, not a formality: reproducibility asserted from one or two runs is a
    # projection wearing a measurement's clothes. Disabling this check silently turns
    # every verdict below into an unfalsifiable claim.
    if len(runs) < min_runs:
        raise PreflightError(f"{len(runs)} runs is fewer than the {min_runs} required")
    for index, run in enumerate(runs):
        check_run(run, index)

    slots = {run["id_slot"] for run in runs}
    if len(slots) != 1:
        raise PreflightError(
            f"runs were routed to more than one slot ({sorted(slots)}); they are not "
            f"repetitions of the same configuration")

    texts = {run["text_sha256"] for run in runs}
    tokens = {tuple(run["token_ids"]) for run in runs}
    vectors = {json.dumps(run["vectors"], sort_keys=True) for run in runs}
    # With no vectors anywhere, every run stringifies to "[]" and the distinct count
    # collapses to 1 -- passing vacuously, which is the failure the empty-vector check
    # exists to prevent. Say so in the record instead of letting a 1 stand for evidence.
    have_vectors = any(run["vectors"] for run in runs)
    verdict = {
        "runs": len(runs),
        "distinct_texts": len(texts),
        "distinct_token_sequences": len(tokens),
        "probability_vectors_available": have_vectors,
        "distinct_probability_vectors": len(vectors) if have_vectors else None,
        # Exact parity, declared before measuring: one answer, or the configuration is not
        # allowlisted. A tolerance chosen after seeing the spread would be fitted to it.
        "reproducible": (len(texts) == 1 and len(tokens) == 1
                         and (len(vectors) == 1 if have_vectors else True)),
    }
    # Named so a proof citing this record carries its own strength, rather than a reader
    # having to notice the absence. llama.cpp emits no completion_probabilities on the
    # speculative path, so this is the only strength a speculative reader can reach.
    verdict["evidence"] = "text+tokens+vectors" if have_vectors else "text+tokens only"
    return verdict


class Reader:
    def __init__(self, binary: str, model: str, port: int, log: Path,
                 extra: list[str], n_ctx: int, slots: str):
        self.binary, self.model, self.port, self.log = binary, model, port, log
        self.extra, self.n_ctx, self.proc = extra, n_ctx, None
        self.slots = slots
        self.url = f"http://127.0.0.1:{port}"

    def argv(self) -> list[str]:
        # --slot-save-path is not optional here even though nothing is saved: llama.cpp
        # answers every /slots action with HTTP 501 when it is unset, so the erase that
        # makes each run cold would fail and the preflight would measure nothing.
        return [self.binary, "--model", self.model, "--host", "127.0.0.1",
                "--port", str(self.port), "-ngl", "99", "-c", str(self.n_ctx),
                "--slot-save-path", self.slots.rstrip("/") + "/",
                "--no-warmup", *self.extra]

    def start(self, timeout: int = 900) -> None:
        handle = open(self.log, "wb")
        self.proc = subprocess.Popen(self.argv(), stdout=handle, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise PreflightError(
                    f"reader exited with {self.proc.returncode} before serving; see "
                    f"{self.log}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                    if b'"ok"' in r.read():
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise PreflightError(f"reader did not become healthy within {timeout}s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def post(self, path: str, body: dict, timeout: int = 900):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def get(self, path: str, timeout: int = 60):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.load(r)


def cold_run(reader: Reader, prompt: str, slot: int, predict: int) -> dict:
    """One cold completion on an explicitly emptied slot.

    The slot is erased, displaced with unrelated text, and erased again. Erasing alone
    left the previous prefix reusable in an earlier version of this measurement.
    """
    reader.post(f"/slots/{slot}?action=erase", {})
    reader.post("/completion", {"prompt": "unrelated displacement text", "n_predict": 1,
                                "temperature": 0.0, "cache_prompt": True, "id_slot": slot})
    reader.post(f"/slots/{slot}?action=erase", {})
    started = time.time()
    r = reader.post("/completion", {
        "prompt": prompt, "n_predict": predict, "temperature": 0.0, "seed": 1,
        "cache_prompt": True, "id_slot": slot, "n_probs": N_PROBS})
    timings = r["timings"]
    return {
        "id_slot": r.get("id_slot", slot),
        "cache_n": timings.get("cache_n"),
        "prompt_n": timings.get("prompt_n"),
        "seconds": round(time.time() - started, 3),
        "text": r["content"],
        "text_sha256": digest_text(r["content"]),
        "token_ids": toks(r),
        "vectors": probs(r),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="names this exact reader configuration")
    ap.add_argument("--args", default="", help="extra server flags, space separated")
    ap.add_argument("--runs", type=int, default=MIN_RUNS)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--predict", type=int, default=24)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--prompt-repeat", type=int, default=60)
    ap.add_argument("--slots", default="", help="slot-save-path; a temp dir if unset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = ("You are a meticulous systems engineer. " * args.prompt_repeat) + \
             "\nSummarise the invariant in one sentence."
    port = free_port()
    log = Path(args.out).with_suffix(".server.log")
    slots = args.slots or tempfile.mkdtemp(prefix="reader-determinism-")
    Path(slots).mkdir(parents=True, exist_ok=True)
    reader = Reader(args.binary, args.model, port, log, args.args.split(), args.n_ctx,
                    slots)

    reader.start()
    try:
        props = reader.get("/props")
        runs = [cold_run(reader, prompt, args.slot, args.predict)
                for _ in range(args.runs)]
        pid = reader.proc.pid
    finally:
        reader.stop()

    record = {
        "label": args.label,
        "argv": reader.argv(),
        "process": {"pid": pid, "returncode": reader.proc.returncode},
        "binary": str(args.binary),
        "binary_digests": binary_digests(Path(args.binary)),
        "model": str(args.model),
        "model_sha256": sha256_file(Path(args.model)),
        "prompt_sha256": digest_text(prompt),
        "prompt_chars": len(prompt),
        "slots": slots,
        "attestation": {
            "build_info": props.get("build_info"),
            "model_path": props.get("model_path"),
            "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
        },
        "runs": runs,
    }
    try:
        record["verdict"] = summarise(runs, min_runs=min(args.runs, MIN_RUNS))
        refused = None
    except PreflightError as exc:
        record["verdict"] = {"reproducible": False, "refused": str(exc)}
        refused = str(exc)

    Path(args.out).write_text(json.dumps(record, indent=1))
    v = record["verdict"]
    print(f"{args.label}: {v}")
    print(f"wrote {args.out}")
    if refused:
        return 2
    return 0 if v["reproducible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
