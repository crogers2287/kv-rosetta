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


class CaptureLoop:
    """Poll loaded models and preserve warm slots as they settle.

    Runs beside the sidecar rather than being asked for. The operator's working context was
    lost twice to model unloads and each loss cost a full re-ingest of tens of thousands of
    tokens; the state existed the whole time and nothing had written it down.
    """

    def __init__(self, swap: str, store_root, *, min_tokens: int = DEFAULT_MIN_TOKENS,
                 interval: float = 20.0, log=None, restorer=None) -> None:
        self.swap = swap.rstrip("/")
        self.store_root = store_root
        self.min_tokens = min_tokens
        self.interval = interval
        self._log = log or (lambda msg: None)
        self._seen: set[tuple[str, int]] = set()
        # Injected rather than constructed here so the loop stays testable without a
        # sidecar, a store, or a running llama-swap. Called as restorer(model, slot).
        self.restorer = restorer
        self._previous: frozenset[str] = frozenset()
        self.captured = 0
        self.refused = 0
        self.restored = 0
        self.restore_refused = 0

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
        for model in sorted(newly_loaded(loaded, self._previous)):
            slot = restorable(by_model.get(model, []))
            if slot is None:
                self._log(f"{model} appeared with no idle empty slot; leaving it to "
                          f"prefill natively")
                continue
            try:
                info = self.restorer(model, slot)
            except Exception as exc:
                self.restore_refused += 1
                self._log(f"restore refused for {model} slot {slot}: {str(exc)[:160]}")
                continue
            if info is None:
                self._log(f"{model} appeared but no attachment matches it yet")
                continue
            self.restored += 1
            self._log(f"restored {model} slot {slot}: {info}")
            done.append(model)
        return done

    def tick(self) -> list[Candidate]:
        loaded = self.loaded_models()
        by_model = {m: self.slots_for(m) for m in sorted(loaded)}
        self.restore_fresh(loaded, by_model)
        self._previous = loaded
        candidates = require_loaded_only(
            choose_candidates(by_model, min_tokens=self.min_tokens,
                              already=frozenset(self._seen)), loaded)
        done = []
        for cand in candidates:
            name = f"auto-{cand.model}-slot{cand.slot_id}-{cand.tokens}.state"
            try:
                saved = self._json(
                    f"{self.swap}/upstream/{cand.model}/slots/{cand.slot_id}?action=save",
                    {"filename": name})
            except Exception as exc:
                self.refused += 1
                self._log(f"capture refused for {cand.model} slot {cand.slot_id}: "
                          f"{str(exc)[:120]}")
                continue
            self._seen.add((cand.model, cand.tokens))
            self.captured += 1
            self._log(f"captured {cand.model} slot {cand.slot_id}: "
                      f"{saved.get('n_saved')} cells, "
                      f"{(saved.get('n_written') or 0) / 1e6:.0f} MB -> {name}")
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


def newly_loaded(current: frozenset[str], previous: frozenset[str]) -> frozenset[str]:
    """Models that appeared since the last poll.

    Restoring is attempted once per appearance rather than continuously: a model that is up
    and being used has its own cache, and re-restoring over it would be the overwrite the
    rule above forbids.
    """
    return current - previous
