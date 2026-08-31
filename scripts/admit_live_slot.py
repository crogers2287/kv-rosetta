"""Admit whatever a live slot is holding as a restorable attachment.

prewarm builds an attachment by replaying a prefix captured by cfrproxy. Traffic that does
not pass through the proxy -- a harness pointed straight at llama-swap -- never produces a
manifest, so no attachment can be built for it and every load prefills cold no matter how
well the restore path works.

A slot that has just served such a request already holds exactly the tokens that matter.
This saves it and admits it under a fingerprint derived from its own token sequence, which
is self-describing: the artifact carries the ids it covers, so a later restore is checked
against the same bytes it was admitted from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kv_rosetta.adapters import ggsq_envelope  # noqa: E402


class LiveCaptureError(ValueError):
    pass


def prefix_fingerprint(token_ids) -> str:
    """A stable 64-hex identity for a token sequence. Shared with the capture daemon so a
    manual capture and an automatic one produce the same identity for the same tokens."""
    from kv_rosetta.daemon.capture import prefix_fingerprint as _fp
    try:
        return _fp(token_ids)
    except ValueError as exc:
        raise LiveCaptureError(str(exc)) from exc


def wait_for_idle(get, slot: int, *, timeout: float, poll: float = 5.0) -> dict:
    """Block until the slot stops processing, so a partial prefill is never admitted."""
    if timeout <= 0:
        raise LiveCaptureError(f"timeout {timeout} is not positive")
    deadline = time.time() + timeout
    while time.time() < deadline:
        slots = get("/slots")
        found = [s for s in slots if int(s.get("id", -1)) == slot]
        if not found:
            raise LiveCaptureError(f"no slot {slot} on this runtime")
        if not found[0].get("is_processing"):
            return found[0]
        time.sleep(poll)
    raise LiveCaptureError(f"slot {slot} still busy after {timeout:.0f}s")


def require_tokens(held: int, minimum: int) -> None:
    if held < minimum:
        raise LiveCaptureError(
            f"slot holds {held} tokens, below the {minimum} worth admitting; a short prefix "
            f"costs a restore and saves almost no prefill")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--swap", default="http://127.0.0.1:9069")
    ap.add_argument("--store-root", default="~/.kvrosetta/admitted")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--min-tokens", type=int, default=8000)
    ap.add_argument("--wait", type=float, default=1800.0)
    args = ap.parse_args(argv)

    base = f"{args.swap.rstrip('/')}/upstream/{args.model}"
    store_root = Path(args.store_root).expanduser()

    def get(path, timeout=60):
        return json.load(urllib.request.urlopen(base + path, timeout=timeout))

    def post(path, payload, timeout=1800):
        req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    state = wait_for_idle(get, args.slot, timeout=args.wait)
    held = int(state.get("n_prompt_tokens") or 0)
    require_tokens(held, args.min_tokens)
    print(f"slot {args.slot} idle holding {held:,} tokens")

    name = f"live-{args.model}-slot{args.slot}-{held}.state"
    saved = post(f"/slots/{args.slot}?action=save", {"filename": name})
    raw = (store_root / name).read_bytes()
    token_ids = list(ggsq_envelope.decode_prompt_tokens(
        ggsq_envelope.parse_file_envelope(raw).token_ids))
    fingerprint = prefix_fingerprint(token_ids)

    from kv_rosetta.admitted_store import AdmittedStore
    from kv_rosetta.adapters.admitted_path import AdmittedPath
    from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter
    store = AdmittedStore(store_root, create=True)
    obj = AdmittedPath(LlamaCppHTTPAdapter(base, str(store_root)), store).admit(
        store_root / name, model=args.model, token_ids=token_ids,
        save_response=saved, prefix_fingerprint=fingerprint)
    print(f"admitted {obj.digest[:16]} for {args.model}: {len(token_ids):,} tokens, "
          f"{len(raw) / 1e6:.0f} MB, prefix {fingerprint[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
