"""Save warm slots before they are lost, without becoming a warmer.

A model's cache is worth exactly as much as it costs to rebuild, and that cost is only paid
when it is already too late: an unload, a swap, a restart, and tens of thousands of tokens
have to be re-read. Waiting for someone to ask for a capture means the capture never happens,
because nobody knows the eviction is coming.

The distinction from the recompute warmer this project removed is precise and worth stating,
because the surface looks similar. That warmer SENT REQUESTS: it woke parked models and paid a
full prefill on a schedule whether or not the result was ever used. This sends none. It reads
slot status and calls save on state that already exists, for models that are already loaded.
It cannot wake anything, cannot cause a prefill, and cannot change what a slot holds.
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Any

#: Below this, re-reading the prompt is cheaper than the machinery of saving it. A 200-token
#: prefix is not worth a 30 MB artifact.
DEFAULT_MIN_TOKENS = 4096


class CaptureRefused(ValueError):
    """A slot that must not be captured. Never downgraded to a retry."""


@dataclass(frozen=True)
class Candidate:
    model: str
    slot_id: int
    tokens: int


def slot_is_capturable(slot: dict[str, Any], *, min_tokens: int = DEFAULT_MIN_TOKENS) -> bool:
    """Is this slot holding finished work worth preserving?

    A slot mid-request is excluded: its state describes a prompt that is still being built,
    and saving it captures a partial prefix that would later be restored as though complete.
    """
    if slot.get("is_processing"):
        return False
    return int(slot.get("n_prompt_tokens") or 0) >= min_tokens


def choose_candidates(slots_by_model: dict[str, list[dict[str, Any]]], *,
                      min_tokens: int = DEFAULT_MIN_TOKENS,
                      already: frozenset[tuple[str, int]] = frozenset()) -> list[Candidate]:
    """Which slots are worth saving now.

    `already` carries (model, token count) pairs that have been captured before, so a slot
    that has not changed is not written again every cycle. Token count is a coarse key on
    purpose: it is available from slot status without reading a 200 MB payload, and the cost
    of an occasional redundant save is one file, while the cost of reading every payload each
    cycle would be paid continuously.
    """
    out: list[Candidate] = []
    for model, slots in sorted(slots_by_model.items()):
        for slot in slots:
            if not slot_is_capturable(slot, min_tokens=min_tokens):
                continue
            tokens = int(slot.get("n_prompt_tokens") or 0)
            if (model, tokens) in already:
                continue
            out.append(Candidate(model=model, slot_id=int(slot["id"]), tokens=tokens))
    # Biggest first: the largest context is the one whose loss hurts most.
    return sorted(out, key=lambda c: c.tokens, reverse=True)


def require_loaded_only(candidates: list[Candidate], loaded: frozenset[str]) -> list[Candidate]:
    """Drop anything not already loaded.

    The one rule that keeps this from becoming the warmer it replaces. A capture must never
    be the reason a model is woken; if it is not resident, there is no warm state to save and
    loading it to find that out would cost the prefill this exists to avoid.
    """
    return [c for c in candidates if c.model in loaded]


# REQ-111: llama-server parks a slot save issued while that slot is generating (SLOT_SAVE on
# is_processing() -> queue_tasks.defer) and only releases it when the slot is released. The
# capture tick is single-threaded, so one parked save froze every model's captures for the
# length of that generation (5m39s measured, bounded only by the 900 s read timeout). A save
# that has not answered in this long is parked, not slow: 610 MB..1.4 GB captures complete in
# single-digit seconds on this host.
SLOT_SAVE_TIMEOUT_S = 60.0


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, TimeoutError) or "timed out" in str(exc).lower()


class CaptureLoop:
    """Poll loaded models and preserve warm slots as they settle.

    Runs beside the sidecar rather than being asked for. The operator's working context was
    lost twice to model unloads and each loss cost a full re-ingest of tens of thousands of
    tokens; the state existed the whole time and nothing had written it down.
    """

    def __init__(self, swap: str, store_root, *, min_tokens: int = DEFAULT_MIN_TOKENS,
                 interval: float = 20.0, log=None, restorer=None, admitter=None) -> None:
        self.swap = swap.rstrip("/")
        self.store_root = store_root
        self.min_tokens = min_tokens
        self.interval = interval
        self._log = log or (lambda msg: None)
        self._seen: set[tuple[str, int]] = set()
        # Injected rather than constructed here so the loop stays testable without a
        # sidecar, a store, or a running llama-swap. Called as restorer(model, slot).
        self.restorer = restorer
        # Saving a slot writes bytes nothing can find: artifacts are looked up by prefix
        # fingerprint and runtime model, and a bare .state carries neither. Admitting is
        # what turns a capture into something restore-on-load can actually use.
        self.admitter = admitter
        self._previous: frozenset[str] = frozenset()
        self.captured = 0
        self.refused = 0
        self.parked = 0            # REQ-111: saves parked behind a busy slot
        self.restored = 0
        self.restore_refused = 0
        self.admitted = 0
        self.admit_refused = 0
        self.skipped_small_growth = 0
        #: Tokens last captured from each (model, slot). Growth is measured against this,
        #: not against the model's biggest artifact -- see `_slot_history`.
        self._captured_on_slot: dict[tuple[str, int], int] = {}
        #: (model, slot) pairs already holding an attachment we put there. Dropped when the
        #: slot is seen non-empty, so once real work displaces the prefix we warm it again.
        self._warmed: set[tuple[str, int]] = set()
        #: (model, slot) whose restore was refused. Deliberately NOT cleared when the slot
        #: changes -- a refused restore leaves its own tokens in the slot, so clearing on
        #: slot content is what made this retry every tick. Cleared only when a new
        #: attachment is admitted for the model, i.e. when there is genuinely something new
        #: to try.
        self._restore_refused_for: set[tuple[str, int]] = set()

    # -- runtime access ----------------------------------------------------------------

    def _json(self, url: str, payload=None, timeout: float = 900):
        import json as _json_mod
        import urllib.request
        if payload is None:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return _json_mod.load(r)
        req = urllib.request.Request(url, data=_json_mod.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _json_mod.load(r)

    def loaded_models(self) -> frozenset[str]:
        running = self._json(f"{self.swap}/running", timeout=30).get("running", [])
        return frozenset(m["model"] for m in running if m.get("state") == "ready")

    def slots_for(self, model: str) -> list[dict[str, Any]]:
        try:
            return self._json(f"{self.swap}/upstream/{model}/slots", timeout=30)
        except Exception:
            # A runtime without a slot API - vLLM has none - simply has nothing to capture.
            return []

    # -- one pass ----------------------------------------------------------------------

    def restore_fresh(self, loaded: frozenset[str],
                      by_model: dict[str, list[dict[str, Any]]]) -> list[str]:
        """Put an attachment back into each model that has just appeared.

        This is the half that was missing: capture without restore writes caches nothing
        ever reads, and the request that would benefit from a restore is precisely the one
        already paying for the prefill.

        Not primed on the first poll. A daemon restart therefore treats everything resident
        as freshly appeared, which is safe because restorable() will only hand back a slot
        that is idle AND empty -- a model in use is never restored over.
        """
        if self.restorer is None:
            return []
        done = []
        # Not only models that just appeared. A model that stays resident never appears
        # again, so gating on appearance meant a long-lived model was warmed once at load
        # and never afterwards -- a session reset against it prefills from scratch even
        # though an attachment exists. Any idle EMPTY slot is a place to keep the prefix
        # resident, and llama.cpp assigns slots by prefix match, so a new conversation can
        # land on it. The marker stops us restoring into the same empty slot every tick.
        for model in sorted(loaded):
            slot = restorable(by_model.get(model, []))
            if slot is None:
                if model in newly_loaded(loaded, self._previous):
                    self._log(f"{model} appeared with no idle empty slot; leaving it to "
                              f"prefill natively")
                continue
            if (model, slot) in self._warmed:
                continue
            if (model, slot) in self._restore_refused_for:
                continue
            try:
                info = self.restorer(model, slot)
            except Exception as exc:
                self.restore_refused += 1
                self._restore_refused_for.add((model, slot))
                self._log(f"restore refused for {model} slot {slot}: {str(exc)[:160]} "
                          f"-- not retrying until a new attachment is admitted")
                continue
            if info is None:
                self._log(f"{model} appeared but no attachment matches it yet")
                continue
            self.restored += 1
            self._warmed.add((model, slot))
            # What was just restored is already an artifact. Without this the capture pass
            # in the same tick saves the restored slot straight back and admits a duplicate
            # under a fresh fingerprint, which is how the store gained a second copy of a
            # 9,146-token attachment moments after restoring it.
            covered = info.get("covers_tokens") if isinstance(info, dict) else None
            if covered:
                self._seen.add((model, int(covered)))
            self._log(f"restored {model} slot {slot}: {info}")
            done.append(model)
        return done

    def _slot_history(self, model: str, slot_id: int) -> int:
        """What we last captured from THIS slot, which is what growth is relative to.

        Growth is a property of one conversation extending on one slot, not of a model.
        Comparing against the model's largest artifact instead suppressed every prefix
        shorter than the biggest one ever seen: with a 75,523-token attachment on disk a
        31,366-token Hermes prompt needed 90,627 tokens to be "worth" capturing, so it was
        never captured, so the load restore never had it to choose. Different conversations
        are not small versions of the largest one.
        """
        return self._captured_on_slot.get((model, slot_id), 0)

    def _discard_raw(self, name: str) -> None:
        """Remove a raw capture that will never be admitted. A leftover is waste, not an error."""
        from pathlib import Path
        try:
            Path(self.store_root).expanduser().joinpath(name).unlink()
        except OSError:
            pass

    def tick(self) -> list[Candidate]:
        loaded = self.loaded_models()
        by_model = {m: self.slots_for(m) for m in sorted(loaded)}
        for model, slots in by_model.items():
            for slot in slots:
                if int(slot.get("n_prompt_tokens") or 0) > 0:
                    self._warmed.discard((model, int(slot.get("id", -1))))
        self.restore_fresh(loaded, by_model)
        self._previous = loaded
        candidates = require_loaded_only(
            choose_candidates(by_model, min_tokens=self.min_tokens,
                              already=frozenset(self._seen)), loaded)
        done = []
        for cand in candidates:
            if not worth_capturing(cand.tokens, self._slot_history(cand.model, cand.slot_id)):
                self.skipped_small_growth += 1
                self._seen.add((cand.model, cand.tokens))
                continue
            name = f"auto-{cand.model}-slot{cand.slot_id}-{cand.tokens}.state"
            try:
                saved = self._json(
                    f"{self.swap}/upstream/{cand.model}/slots/{cand.slot_id}?action=save",
                    {"filename": name}, timeout=SLOT_SAVE_TIMEOUT_S)
            except Exception as exc:
                self.refused += 1
                if _is_timeout(exc):
                    # Parked behind a request that landed on the slot after we read /slots.
                    # The server will still run it on release; we do not wait for it and do
                    # not re-park the loop on the same prefix (REQ-111).
                    self.parked += 1
                    self._seen.add((cand.model, cand.tokens))
                    self._log(f"capture parked for {cand.model} slot {cand.slot_id}: slot "
                              f"busy, no answer in {SLOT_SAVE_TIMEOUT_S:.0f}s; not retried")
                    continue
                self._log(f"capture refused for {cand.model} slot {cand.slot_id}: "
                          f"{str(exc)[:120]}")
                continue
            self._seen.add((cand.model, cand.tokens))
            self._captured_on_slot[(cand.model, cand.slot_id)] = cand.tokens
            self.captured += 1
            self._log(f"captured {cand.model} slot {cand.slot_id}: "
                      f"{saved.get('n_saved')} cells, "
                      f"{(saved.get('n_written') or 0) / 1e6:.0f} MB -> {name}")
            if self.admitter is not None:
                try:
                    info = self.admitter(cand.model, name, saved)
                except Exception as exc:          # an unusable capture is not fatal
                    self.admit_refused += 1
                    # Wide enough to keep the save response's checkpoint fields: a refusal
                    # cut at 160 chars lost 'n_checkpoints_saved', which was the whole
                    # explanation of why a 4,096-token prompt could not be admitted.
                    self._log(f"admit refused for {cand.model}: {str(exc)[:600]}")
                    # A refused capture is bytes nothing will ever look up. REQ-095 removes
                    # the raw file only after a successful admit; the refusal path left a
                    # 232 MB state behind for a prompt that can never be admitted at this
                    # checkpoint spacing, and would write it again after every restart.
                    self._discard_raw(name)
                else:
                    if info is not None:
                        self.admitted += 1
                        # Something new to try: let refused slots attempt a restore again.
                        self._restore_refused_for = {
                            key for key in self._restore_refused_for
                            if key[0] != cand.model}
                        self._log(f"admitted {cand.model}: {info}")
            done.append(cand)
        return done

    def run_forever(self) -> None:
        import time
        self._log(f"capture loop: every {self.interval:.0f}s, "
                  f"minimum {self.min_tokens} tokens, "
                  f"{'restore on load enabled' if self.restorer else 'save only'}")
        while True:
            try:
                self.tick()
            except Exception as exc:                 # a poll failure is not fatal
                self._log(f"capture tick failed: {str(exc)[:150]}")
            time.sleep(self.interval)


def restorable(slots: list[dict[str, Any]], *, min_tokens: int = DEFAULT_MIN_TOKENS) -> int | None:
    """A slot that is idle AND empty, into which an attachment may be restored.

    Empty is the load-bearing half. A slot already holding a prompt is either serving live
    work or holds a context worth more than the one being restored, and overwriting it turns
    a warm session cold - which is the exact harm this whole loop exists to prevent.
    """
    for slot in slots:
        if slot.get("is_processing"):
            continue
        if int(slot.get("n_prompt_tokens") or 0) == 0:
            return int(slot["id"])
    return None


def prefix_fingerprint(token_ids) -> str:
    """A stable 64-hex identity for a token sequence.

    Derived from the ids rather than the text: the cache is keyed to tokens, so two strings
    that tokenise identically are the same prefix for reuse purposes. This is what lets a
    slot captured from unproxied traffic be admitted and found later, without a manifest.
    """
    import hashlib
    if not token_ids:
        raise ValueError("cannot fingerprint an empty token sequence")
    digest = hashlib.sha256()
    for tok in token_ids:
        digest.update(int(tok).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


#: A growing conversation passes through many token counts, and capturing each one admits
#: a near-duplicate of the last at full size -- 11 artifacts averaging 600 MB were written
#: from a single session before this was noticed. A capture must add materially more than
#: the previous one for that model to be worth its disk.
MIN_CAPTURE_GROWTH = 0.20

#: How far a slot's token count must FALL before it reads as a different conversation
#: rather than a compaction of the same one. Half: a harness rewriting its history rarely
#: loses more than that, and a genuinely new conversation on a long-running slot starts far
#: below it.
SHRINK_IS_A_NEW_CONVERSATION = 0.5


def worth_capturing(tokens: int, previous: int, *, growth: float = MIN_CAPTURE_GROWTH) -> bool:
    """Whether a capture at this size adds enough over the last one to be worth storing."""
    if tokens <= 0:
        raise ValueError(f"token count {tokens} is not positive")
    if previous <= 0:
        return True
    # A large shrink means the slot was reset and refilled with a different conversation,
    # and capturing it is the only way a second prefix on a busy slot reaches the store.
    # A SMALL shrink means no such thing: a harness that compacts its context makes the
    # count oscillate (86,952 -> 80,013 -> 103,866 -> 87,999 was observed live), and
    # treating each dip as a new conversation fired 29 captures of 1.2-1.5 GB in one
    # session. Each one blocks the slot while llama.cpp writes it, so an eager rule here
    # stalls the very conversation it is trying to preserve.
    if tokens < previous * SHRINK_IS_A_NEW_CONVERSATION:
        return True
    return tokens >= previous * (1.0 + growth)


#: Half-life, in hours, applied to an attachment's covered-token count when ranking a
#: model-load restore. Six hours: long enough that a system prompt captured earlier today
#: still outranks a trivial capture from a minute ago, short enough that yesterday's big
#: attachment loses to today's real traffic.
RESTORE_HALF_LIFE_HOURS = 6.0


def restore_score(covered: int, age_hours: float,
                  *, half_life: float = RESTORE_HALF_LIFE_HOURS) -> float:
    """Expected usefulness of an attachment: how much it covers, decayed by how stale it is.

    Neither term works alone, and this project has now shipped both failures. Ranking by
    size restored a 75,523-token attachment on every load while the prompt traffic actually
    used was never chosen. Ranking by recency then restored a 6,468-token capture from a
    passing interaction over the 72,465-token system prompt an agent harness sends every
    time, so 84k tokens were re-prefilled cold on every load.

    Coverage is what a restore is worth; recency is the probability it still applies. The
    product is the thing to maximise.
    """
    if covered <= 0:
        return 0.0
    if age_hours <= 0:
        return float(covered)
    return covered * (0.5 ** (age_hours / half_life))


def rank_restore_candidates(candidates, *, now=None):
    """Order attachments for a model-load restore: own first, then most recently admitted.

    There is no request to be relevant to when a model loads, so the choice is a guess.
    The property that matters in a guess is that being wrong repairs it. Recency has that:
    a miss is prefilled by the server, capture admits that prefix, and it is the newest
    candidate next time. Size does not -- a 75,523-token attachment outranked every shorter
    prompt on this host indefinitely, so each load restored 75k tokens the request shared no
    prefix with and paid the full prefill anyway. A wrong restore costs the restore AND the
    prefill, so it is worse than not restoring at all.

    Each candidate is ``(own, seen_at, covered, fingerprint)``, `seen_at` a unix time.
    Ownership still dominates -- a foreign prefix cannot be restored onto this model at any
    score -- and the score orders the rest.
    """
    if now is None:
        now = time.time()
    def key(candidate):
        own, seen_at, covered, fingerprint = candidate
        age_hours = max(0.0, (now - float(seen_at)) / 3600.0)
        return (own, restore_score(covered, age_hours), fingerprint)
    return sorted(candidates, key=key, reverse=True)


def same_model(capture_label: str, runtime_model: str) -> bool:
    """Whether a prefix captured under one name belongs to this runtime model.

    A prefix is only reusable by the model whose traffic produced it: the cache is keyed to
    an exact token prefix, so an attachment captured from a different harness matches nothing
    and merely occupies the slot. Names differ because clients call an alias -- traffic
    labelled qwen38-flash-next-3090 is served by qwen38-flash-next-kvx -- so the comparison
    drops one trailing dash-segment from each and compares stems.

    A heuristic, deliberately conservative: a false negative costs a smaller attachment, a
    false positive costs a restore that matches nothing.
    """
    left = (capture_label or "").strip().lower()
    right = (runtime_model or "").strip().lower()
    if not left or not right:
        return False
    if left == right:
        return True
    return left.rsplit("-", 1)[0] == right.rsplit("-", 1)[0]


def newly_loaded(current: frozenset[str], previous: frozenset[str]) -> frozenset[str]:
    """Models that appeared since the last poll.

    Restoring is attempted once per appearance rather than continuously: a model that is up
    and being used has its own cache, and re-restoring over it would be the overwrite the
    rule above forbids.
    """
    return current - previous


#: How many attachments to keep per runtime model. An evolving conversation yields a
#: genuinely NEW prefix on every turn, so no amount of deduplication bounds the store --
#: 30 distinct ornith artifacts totalling 29.1 GB came from a single session and filled
#: the disk to 93%. The restore ranking is recency-weighted, so anything far down that
#: order is paying gigabytes for a prefix that will never be chosen.
MAX_ARTIFACTS_PER_MODEL = 4


def prune_model_artifacts(store, model: str, *, keep: int = MAX_ARTIFACTS_PER_MODEL) -> int:
    """Drop all but the `keep` most recent attachments for one model. Returns how many went.

    Called after a successful admit, so the store is bounded at the moment it grows rather
    than by a sweep that may never run.
    """
    held = []
    for obj in store.list_objects():
        manifest = obj.manifest or {}
        if manifest.get("runtime_model") != model:
            continue
        try:
            held.append((obj.path.stat().st_mtime, obj))
        except OSError:                     # vanished under us; nothing to prune
            continue
    if len(held) <= keep:
        return 0
    held.sort(key=lambda pair: pair[0], reverse=True)
    dropped = 0
    for _, obj in held[keep:]:
        try:
            obj.path.unlink()
            manifest_path = obj.path.with_name(obj.path.name.replace(".state",
                                                                    ".manifest.json"))
            if manifest_path.exists():
                manifest_path.unlink()
            dropped += 1
        except OSError:                     # a failed eviction is waste, not an error
            continue
    return dropped
