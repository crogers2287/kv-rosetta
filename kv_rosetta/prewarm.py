"""Give a model its own attachment for a prefix, before it is needed.

A cache attachment is per model, and that is a property of the tensors, not a limitation of
the drive: a model reads the shared content and keeps its own reading of it. So the reason a
model faces a cold prefill is never that some other model owns the cache - it is that nobody
has warmed this one yet. Warming it ahead of time is the whole of the fix, and it works
across architectures and geometries because nothing is being shared.

What this must not become is the warmer it replaces. kvwarm woke parked models on a timer and
paid a full prefill whether or not anyone was going to use the result. So pre-warming here is
explicit and refuses to wake anything: a model that is not already loaded is left alone unless
the caller says otherwise in as many words.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class PrewarmError(ValueError):
    """A refusal. Warming the wrong slot publishes an artifact that describes other text."""


def replay_body(manifest: dict[str, Any], model: str) -> dict[str, Any]:
    """The request that reproduces a corpus prefix exactly.

    Composed the way cfrproxy's warmer composes it - system, then a single "." user turn,
    then the tool schemas - because the token sequence has to match live traffic or the
    artifact is a cache of something nobody will ask for. The "." is not decorative; it is
    what makes the request valid without extending the prefix being established.
    """
    system = manifest.get("system")
    if not isinstance(system, str) or not system.strip():
        raise PrewarmError("manifest carries no system prefix to warm")
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": "."}],
        "max_tokens": 16, "temperature": 0, "stream": False,
    }
    tools = manifest.get("tools")
    if tools:
        parsed = json.loads(tools) if isinstance(tools, str) else tools
        if parsed:
            body["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object"})}} for t in parsed]
    return body


def require_loaded(running: list[str], model: str, *, allow_wake: bool = False) -> None:
    """Refuse to warm a model that is not already loaded.

    This is the line between a pre-warm and the recompute warmer this project removed. That
    one woke parked models on a schedule; the cost was invisible because it landed on the
    fleet rather than on the request that caused it.
    """
    if model in running or allow_wake:
        return
    raise PrewarmError(
        f"{model!r} is not loaded and allow_wake is off; refusing to wake a parked model "
        f"to warm it. Loaded: {sorted(running) or 'none'}")


def best_prior(artifacts: list[Any], model: str) -> Any | None:
    """The largest attachment this model already holds, to build the next one on top of.

    Warming from nothing recomputes text the model has already read. Restoring its own
    previous attachment first means the replay prefills only what is new: measured on a
    growing prefix, 820 of 892 tokens were reused and the prefill fell from 108ms to 19ms.

    Only this model's own artifacts are considered. A different model's attachment is not a
    head start, it is the wrong tensors - and llama.cpp would not detect that, because it
    checks the token prefix and not the weights.

    Choosing the largest is safe even when it is not a prefix of what is being warmed: the
    runtime compares tokens itself and reuses only the common head, so a wrong guess costs a
    re-prefill rather than a wrong artifact.
    """
    mine = [a for a in artifacts if (a.manifest or {}).get("runtime_model") == model]
    if not mine:
        return None
    return max(mine, key=lambda a: int((a.manifest or {}).get("prompt_token_count") or 0))


@dataclass(frozen=True)
class SlotChoice:
    slot_id: int
    held_tokens: int


def choose_slot(slots: list[dict[str, Any]], expected_tokens: int, *,
                generated: int = 0, tolerance: int = 8) -> SlotChoice:
    """Which slot holds the prefix that was just replayed.

    llama.cpp assigns slots itself, so the replay lands where it lands and the slot must be
    identified after the fact rather than requested. Slots busy with other traffic are never
    considered, and a slot whose token count does not match the replay is refused outright:
    saving the wrong one publishes an artifact whose bytes describe somebody else's prompt
    while claiming to be this prefix. That happened during the first live wiring - a stale
    6,169-token slot was saved for a 12,298-token replay - and only a count check caught it.
    """
    if expected_tokens <= 0:
        raise PrewarmError(f"expected token count {expected_tokens} is not positive")
    if generated < 0:
        raise PrewarmError(f"generated token count {generated} cannot be negative")
    free = [s for s in slots if not s.get("is_processing")]
    if not free:
        raise PrewarmError("every slot is busy with live traffic; refusing to disturb one")
    # The replay asks for tokens as well as sending them, so the slot legitimately ends
    # up holding the prompt plus whatever was generated. Subtracting the reported completion
    # keeps the check on the thing it exists to catch -- a stale slot holding somebody else's
    # prompt -- instead of failing on the model's own reply.
    want = expected_tokens + generated
    best = min(free, key=lambda s: abs(int(s.get("n_prompt_tokens") or 0) - want))
    held = int(best.get("n_prompt_tokens") or 0)
    if abs(held - want) > tolerance:
        raise PrewarmError(
            f"no free slot holds the replayed prefix: closest is slot {best.get('id')} with "
            f"{held} tokens against {expected_tokens} replayed plus {generated} generated. "
            f"Saving it would publish an artifact describing different text")
    return SlotChoice(slot_id=int(best["id"]), held_tokens=held)


def prewarm_cli(args) -> int:
    """Replay one corpus prefix on a loaded model and admit the result.

    Every refusal above applies: a parked model is not woken, a busy slot is not disturbed,
    and a slot that does not hold the replayed prefix is not saved.
    """
    import json as _json
    import urllib.request
    from pathlib import Path

    from kv_rosetta.admitted_store import AdmittedStore
    from kv_rosetta.adapters import ggsq_envelope
    from kv_rosetta.adapters.admitted_path import AdmittedPath
    from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter

    swap = args.swap.rstrip("/")
    base = f"{swap}/upstream/{args.model}"
    store_root = Path(args.store_root).expanduser()
    manifest = _json.loads(Path(args.manifest).read_text())

    def get(path, timeout=60):
        return _json.load(urllib.request.urlopen(base + path, timeout=timeout))

    def post(path, payload, timeout=1800):
        req = urllib.request.Request(base + path, data=_json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.load(resp)

    running = [m["model"] for m in
               _json.load(urllib.request.urlopen(swap + "/running", timeout=30))["running"]]
    require_loaded(running, args.model, allow_wake=args.allow_wake)

    store = AdmittedStore(store_root)
    prior = best_prior(store.list_objects(), args.model) if not args.from_scratch else None
    if prior is not None:
        held = int((prior.manifest or {}).get("prompt_token_count") or 0)
        free = [s for s in get("/slots") if not s.get("is_processing")]
        if free:
            slot_id = int(free[0]["id"])
            try:
                post(f"/slots/{slot_id}?action=restore", {"filename": prior.basename})
                print(f"restored this model's prior attachment into slot {slot_id}: "
                      f"{held} tokens; the replay will prefill only what is new")
            except Exception as exc:                 # a stale artifact is not fatal here
                print(f"prior attachment not usable ({str(exc)[:80]}); warming from scratch")
        else:
            print("all slots busy; warming without a head start")
    else:
        print("no prior attachment for this model; this first warm pays a full prefill")

    body = replay_body(manifest, args.model)
    response = post("/v1/chat/completions", body)
    usage = response.get("usage") or {}
    replayed = int(usage.get("prompt_tokens") or 0)
    cached = int(usage.get("prompt_tokens_cached") or usage.get("cached_tokens") or 0)
    generated = int(usage.get("completion_tokens") or 0)
    choice = choose_slot(get("/slots"), replayed, generated=generated)
    print(f"replayed {replayed} tokens ({cached} already cached); "
          f"slot {choice.slot_id} holds {choice.held_tokens}")

    fingerprint = str(manifest.get("fingerprint", ""))
    name = f"prewarm-{args.model}-{fingerprint[:16]}.state"
    saved = post(f"/slots/{choice.slot_id}?action=save", {"filename": name})
    raw = (store_root / name).read_bytes()
    token_ids = list(ggsq_envelope.decode_prompt_tokens(
        ggsq_envelope.parse_file_envelope(raw).token_ids))
    obj = AdmittedPath(LlamaCppHTTPAdapter(base, str(store_root)), store).admit(
        store_root / name, model=args.model, token_ids=token_ids,
        save_response=saved, prefix_fingerprint=fingerprint)
    print(f"admitted {obj.digest[:16]} for {args.model}: {len(token_ids)} tokens, "
          f"{len(raw) / 1e6:.0f} MB")
    return 0
