#!/usr/bin/env python3
"""Prove a REJECTED restore leaves the target slot exactly as it was.

A refusal is not safety. llama.cpp logs that it restores a foreign checkpoint appendix
before it rejects the main state body, so the interesting question is not whether the
request returns 400 - it does - but whether anything reached the slot on the way to that
400. Nothing previously issued a completion afterwards, so nothing had established it.

The comparison is only causal on a reader already proven reproducible against itself; see
scripts/reader_determinism.py. On a reader that answers identical work differently, a
post-rejection difference would be indistinguishable from its own variance, so this runner
refuses to render a verdict unless it is handed that proof.

Retains the baseline and post-rejection runs in full, plus the refusal status and the
server log lines around the attempt.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production_matrix import binary_digests, free_port, sha256_file  # noqa: E402
from reader_determinism import (  # noqa: E402
    N_PROBS, PreflightError, Reader, check_run, cold_run, digest_text,
)


class PoisonError(RuntimeError):
    """A refusal, or a proven poisoning. Never downgraded."""


def load_reproducible(path: str, label: str) -> dict:
    """Require a passing determinism record for exactly this reader configuration."""
    record = json.loads(Path(path).read_text())
    if record.get("label") != label:
        raise PoisonError(
            f"determinism record is for {record.get('label')!r}, not {label!r}; a proof "
            f"for a different configuration proves nothing about this one")
    verdict = record.get("verdict", {})
    if not verdict.get("reproducible"):
        raise PoisonError(
            f"reader {label!r} is not reproducible against itself ({verdict}); a "
            f"post-rejection difference could not be attributed to the restore")
    return record


def compare(baseline: dict, after: dict) -> dict:
    """Was the slot left as it was? Text, tokens and vectors must all agree."""
    return {
        "text_matches": baseline["text_sha256"] == after["text_sha256"],
        "tokens_match": baseline["token_ids"] == after["token_ids"],
        "vectors_match": baseline["vectors"] == after["vectors"],
        "pristine": (baseline["text_sha256"] == after["text_sha256"]
                     and baseline["token_ids"] == after["token_ids"]
                     and baseline["vectors"] == after["vectors"]),
    }


def stage_foreign(state: str, slots: str) -> Path:
    """Place the foreign artifact where the server will look for it, by name.

    Checked rather than assumed: llama.cpp resolves the restore filename inside its own
    slot-save-path, so a missing source would surface as the server's own "file not found"
    refusal and be indistinguishable from the model-mismatch refusal being tested.
    """
    foreign = Path(state)
    if not foreign.is_file():
        raise PoisonError(f"foreign state file does not exist: {foreign}")
    staged = Path(slots) / foreign.name
    if staged.resolve() != foreign.resolve():
        staged.write_bytes(foreign.read_bytes())
    return foreign


def attempt_restore(reader: Reader, slot: int, filename: str) -> dict:
    """Try to restore a foreign artifact. A success here is itself the failure."""
    body = json.dumps({"filename": filename}).encode()
    req = urllib.request.Request(f"{reader.url}/slots/{slot}?action=restore", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return {"refused": False, "status": r.status,
                    "body": json.loads(r.read().decode())}
    except urllib.error.HTTPError as exc:
        return {"refused": True, "status": exc.code,
                "body": exc.read().decode()[:2000]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True, help="the TARGET model, which must refuse")
    ap.add_argument("--label", required=True)
    ap.add_argument("--determinism-record", required=True,
                    help="a passing reader_determinism record for this exact reader")
    ap.add_argument("--foreign-state", required=True,
                    help="a state file written by a DIFFERENT model")
    ap.add_argument("--slots", required=True)
    ap.add_argument("--args", default="")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--predict", type=int, default=24)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--prompt-repeat", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proof = load_reproducible(args.determinism_record, args.label)

    foreign = stage_foreign(args.foreign_state, args.slots)

    prompt = ("You are a meticulous systems engineer. " * args.prompt_repeat) + \
             "\nSummarise the invariant in one sentence."
    log = Path(args.out).with_suffix(".server.log")
    reader = Reader(args.binary, args.model, free_port(), log, args.args.split(),
                    args.n_ctx, args.slots)
    reader.start()
    try:
        baseline = cold_run(reader, prompt, args.slot, args.predict)
        check_run(baseline, 0)
        attempt = attempt_restore(reader, args.slot, foreign.name)
        # No erase between the attempt and the next completion: erasing would destroy the
        # evidence this runner exists to look for.
        after = reader.post("/completion", {
            "prompt": prompt, "n_predict": args.predict, "temperature": 0.0, "seed": 1,
            "cache_prompt": True, "id_slot": args.slot, "n_probs": N_PROBS})
        from reader_determinism import probs, toks
        after_run = {
            "id_slot": after.get("id_slot", args.slot),
            "cache_n": after["timings"].get("cache_n"),
            "prompt_n": after["timings"].get("prompt_n"),
            "text": after["content"], "text_sha256": digest_text(after["content"]),
            "token_ids": toks(after), "vectors": probs(after),
        }
    finally:
        reader.stop()

    tail = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else []
    record = {
        "label": args.label,
        "argv": reader.argv(),
        "binary_digests": binary_digests(Path(args.binary)),
        "target_model": args.model,
        "target_model_sha256": sha256_file(Path(args.model)),
        "foreign_state": str(foreign),
        "foreign_state_sha256": sha256_file(foreign),
        "determinism_proof": {
            "record": args.determinism_record,
            "verdict": proof["verdict"],
        },
        "restore_attempt": attempt,
        "baseline": baseline,
        "after_rejection": after_run,
        "comparison": compare(baseline, after_run),
        "server_log_tail": tail,
    }
    record["verdict"] = {
        "restore_refused": attempt["refused"],
        "slot_pristine": record["comparison"]["pristine"],
        "safe": attempt["refused"] and record["comparison"]["pristine"],
    }
    Path(args.out).write_text(json.dumps(record, indent=1))
    print(f"{args.label}: {record['verdict']}")
    print(f"wrote {args.out}")
    return 0 if record["verdict"]["safe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
