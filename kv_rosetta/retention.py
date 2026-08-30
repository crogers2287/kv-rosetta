"""Which cached prefixes to keep when they do not all fit. The warmer's failure, in storage.

kvwarm cycled up to 51 prefixes through a server with 2 slots every 900 seconds. Thirteen of
twenty observed cycles were complete re-prefills costing 15 to 84 seconds, because each pass
evicted what the previous pass had just built. **Fifty-one things do not fit in two slots, and
no scheduling of them will make it work.** The failure was not the cadence or the warming
strategy; it was attempting to keep more than could be kept.

A durable store has the same arithmetic with a different unit. So the policy here starts from
what will not fit rather than from what to load: a caller declares a budget, pins what must
survive, and everything else is admitted only while there is room and evicted by least-recent
use when there is not.

Two refusals carry the design. Pins that do not fit the budget are rejected outright rather
than silently dropped later - a pin that can be evicted is not a pin. And a prefix that will
not fit even alone is refused rather than triggering an eviction sweep that frees space for
something that still cannot be stored.

Nothing here reduces correctness if it is wrong: an evicted prefix costs a prefill, which is
what would have happened without any cache. That asymmetry is why the policy errs toward
keeping fewer things properly rather than more things briefly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class RetentionError(ValueError):
    """Raised when a retention policy cannot be satisfied. Never a silent drop."""


@dataclass(frozen=True)
class Entry:
    """One stored prefix, as the policy sees it."""

    digest: str
    nbytes: int
    last_used: float
    pinned: bool = False

    def validate(self) -> list[str]:
        problems = []
        if len(self.digest) != 64:
            problems.append("digest must be a full sha256")
        if self.nbytes <= 0:
            problems.append(f"nbytes {self.nbytes} must be positive")
        return problems


@dataclass(frozen=True)
class Plan:
    """What to keep, what to drop, and why - never just a list of victims."""

    keep: tuple[str, ...]
    evict: tuple[str, ...]
    kept_bytes: int
    budget_bytes: int
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def headroom(self) -> int:
        return self.budget_bytes - self.kept_bytes


def plan_retention(entries, budget_bytes: int) -> Plan:
    """Decide what survives. Pinned entries always do, or the budget is refused.

    Ordering is by `last_used` ascending among unpinned entries, so the least recently used are
    dropped first. Ties break on digest so a store holding two equally stale prefixes always
    makes the same choice - an unstable eviction would make one bad prefix look intermittent.
    """
    if budget_bytes < 0:
        raise RetentionError(f"budget {budget_bytes} is negative")
    items = list(entries)
    seen = set()
    for index, entry in enumerate(items):
        problems = entry.validate()
        if problems:
            raise RetentionError(f"entry[{index}] {entry.digest[:12]}: {'; '.join(problems)}")
        if entry.digest in seen:
            raise RetentionError(f"digest {entry.digest[:12]} appears twice; a store cannot "
                                 f"hold one object under two entries")
        seen.add(entry.digest)

    pinned = [e for e in items if e.pinned]
    pinned_bytes = sum(e.nbytes for e in pinned)
    if pinned_bytes > budget_bytes:
        raise RetentionError(
            f"pinned prefixes need {pinned_bytes} bytes against a budget of {budget_bytes}. A "
            f"pin that can be evicted is not a pin, so this is refused rather than resolved "
            f"by dropping one of them later")

    keep = [e.digest for e in pinned]
    reasons = {e.digest: "pinned" for e in pinned}
    used = pinned_bytes
    evict = []
    # Most recently used first, so the freshest unpinned entries win the remaining room.
    for entry in sorted((e for e in items if not e.pinned),
                        key=lambda e: (-e.last_used, e.digest)):
        if used + entry.nbytes <= budget_bytes:
            keep.append(entry.digest)
            reasons[entry.digest] = "fits"
            used += entry.nbytes
        else:
            evict.append(entry.digest)
            reasons[entry.digest] = (f"no room: {entry.nbytes} bytes with "
                                     f"{budget_bytes - used} free")
    return Plan(keep=tuple(keep), evict=tuple(evict), kept_bytes=used,
                budget_bytes=budget_bytes, reasons=reasons)


def admit_or_refuse(entries, candidate: Entry, budget_bytes: int) -> Plan:
    """Whether a new prefix can be stored, and what it would cost.

    A candidate too large for the whole budget is refused **before** anything is evicted.
    Sweeping the store to make room for something that still will not fit is the shape of the
    warmer's mistake: work that destroys existing value and produces none.
    """
    problems = candidate.validate()
    if problems:
        raise RetentionError(f"candidate {candidate.digest[:12]}: {'; '.join(problems)}")
    if candidate.nbytes > budget_bytes:
        raise RetentionError(
            f"{candidate.digest[:12]} needs {candidate.nbytes} bytes and the whole budget is "
            f"{budget_bytes}; refusing before evicting anything, since freeing space for "
            f"something that still will not fit destroys value and produces none")
    plan = plan_retention(list(entries) + [candidate], budget_bytes)
    if candidate.digest not in plan.keep:
        # The candidate fits the budget in principle but not around what is already pinned.
        # Raising rather than returning a plan that quietly lists it as evicted: a caller who
        # does not inspect the return value must not be left believing it stored something.
        raise RetentionError(
            f"{candidate.digest[:12]} needs {candidate.nbytes} bytes and only "
            f"{plan.budget_bytes - sum(e.nbytes for e in entries if e.pinned)} can be freed "
            f"without evicting a pin; not admitted")
    return plan
