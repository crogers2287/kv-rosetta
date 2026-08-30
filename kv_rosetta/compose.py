"""Composing a prompt from independently cached regions, and knowing when that is a lie.

Prefix caching answers one question: does this prompt start with bytes we have already
prefilled? That covers a system prompt and nothing else. The moment you want a memory file
that is appended to, or a document retrieved per request, or tool definitions that change
without invalidating everything after them, you need to reuse a region that is **not** at the
front - and the cached values for such a region were computed while it attended to a
different left context than the one you are about to place it in.

That is the whole difficulty, and it is not a detail. A region's K and V encode what it
attended to. Concatenating two independently prefilled regions produces a cache in which the
second region never saw the first. The output stays fluent. It is simply conditioned on
something that did not happen.

The literature offers two answers and this module is built to express both:

* **Prompt Cache** (Gim et al., MLSys 2024) declares the structure up front and gives each
  module its own position ids, accepting the missing cross-attention where modules are
  semantically independent.
* **CacheBlend** (Yao et al., EuroSys 2025) reuses non-prefix caches and then selectively
  recomputes the small fraction of tokens whose values deviate most, recovering full-prefill
  quality for a fraction of the cost.

So a region here records the one fact that decides which case applies: the digest of the
context it was **actually prefilled behind**. When that matches the context it is being placed
behind, reuse is exact and this is ordinary prefix caching. When it does not, the region needs
repair, and this module says so rather than handing back a cache that is quietly conditioned
on the wrong thing. Refusing is the default; repair is something a caller opts into and pays
for.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

#: Roles are advisory - nothing here behaves differently by role - but they are constrained
#: so a store can be grouped and reasoned about. An unknown role is refused rather than
#: accepted as free text, because "memory " and "memmory" would silently become two regions.
ROLES = frozenset({"system", "tools", "memory", "document", "session"})

#: A region prefilled at the very front has no left context. Spelled explicitly so that "not
#: recorded" and "recorded as nothing" cannot be confused.
AT_FRONT = "root"


class ComposeError(ValueError):
    """Raised when a composition cannot be justified. Never a silent concatenation."""


def token_digest(token_ids) -> str:
    """A region's identity: its exact token ids, not its text.

    Text is the wrong identity. Two different tokenizations of one string produce different
    caches, and one tokenization of two different strings can collide after normalisation.
    """
    ids = list(token_ids)
    if not ids:
        raise ComposeError("a region with no tokens has no identity")
    running = hashlib.sha256()
    for value in ids:
        running.update(int(value).to_bytes(8, "little", signed=True))
    return running.hexdigest()


@dataclass(frozen=True)
class Region:
    """One independently cached span of a prompt.

    `prefilled_after` is the load-bearing field: the chain digest of everything that preceded
    this region when its cache was computed, or AT_FRONT if it was prefilled at position 0.
    """

    name: str
    role: str
    token_start: int
    token_end: int
    tokens_sha256: str
    prefilled_after: str = AT_FRONT

    @property
    def n_tokens(self) -> int:
        return self.token_end - self.token_start

    def validate(self) -> list[str]:
        problems = []
        if not self.name:
            problems.append("a region needs a name")
        if self.role not in ROLES:
            problems.append(f"role {self.role!r} is not one of {sorted(ROLES)}")
        if self.n_tokens <= 0:
            problems.append(f"token range {self.token_start}..{self.token_end} is empty")
        if self.token_start < 0:
            problems.append(f"token_start {self.token_start} is negative")
        if len(self.tokens_sha256) != 64:
            problems.append("tokens_sha256 must be a full sha256 hex digest")
        if not self.prefilled_after:
            problems.append("prefilled_after must be recorded; use AT_FRONT for position 0")
        return problems


def chain_digest(previous: str, region: Region) -> str:
    """The identity of everything up to and including `region`.

    Order-sensitive by construction: the same regions in a different order give a different
    digest, which is correct, because the same text in a different order is a different cache.
    """
    running = hashlib.sha256()
    running.update(previous.encode())
    running.update(b"\x1f")
    running.update(region.tokens_sha256.encode())
    return running.hexdigest()


@dataclass(frozen=True)
class Placement:
    """What reusing one region in this composition would actually mean."""

    region: Region
    status: str            # "exact" or "needs_repair"
    expected_context: str
    reason: str = ""

    @property
    def exact(self) -> bool:
        return self.status == "exact"


@dataclass(frozen=True)
class Plan:
    placements: tuple[Placement, ...]
    total_tokens: int
    repair_tokens: int

    @property
    def exact(self) -> bool:
        return self.repair_tokens == 0

    def needing_repair(self) -> tuple[Placement, ...]:
        return tuple(p for p in self.placements if not p.exact)


def plan(regions, *, allow_repair: bool = False) -> Plan:
    """Decide, region by region, whether its cache is valid where it is being placed.

    With `allow_repair` false - the default - a region whose recorded left context does not
    match the one it is being placed behind is refused. That is the case where reuse produces
    a cache conditioned on text that was never there, and the resulting output is wrong in the
    one way that does not look wrong.

    With it true, such regions come back marked `needs_repair` along with the token count, so
    a caller that has a selective-recompute pass can price the work and run it. This module
    does not perform the repair; it establishes which tokens need one.
    """
    ordered = list(regions)
    if not ordered:
        raise ComposeError("nothing to compose")

    seen_names = set()
    expected_start = ordered[0].token_start
    if expected_start != 0:
        raise ComposeError(f"the first region starts at token {expected_start}, not 0; a "
                           f"composition with no beginning cannot be placed")

    placements, context, repair_tokens = [], AT_FRONT, 0
    for index, region in enumerate(ordered):
        problems = region.validate()
        if problems:
            raise ComposeError(f"region[{index}] {region.name!r}: {'; '.join(problems)}")
        if region.name in seen_names:
            raise ComposeError(f"region name {region.name!r} appears twice; names address "
                               f"regions in a store and must be unique within a composition")
        seen_names.add(region.name)
        if region.token_start != expected_start:
            raise ComposeError(
                f"region[{index}] {region.name!r} starts at token {region.token_start} but "
                f"the previous region ends at {expected_start}; a gap or overlap means the "
                f"composed cache does not describe one continuous prompt")
        expected_start = region.token_end

        if region.prefilled_after == context:
            status, reason = "exact", ""
        else:
            status = "needs_repair"
            reason = (f"cached behind {region.prefilled_after[:12]} but being placed behind "
                      f"{context[:12]}; its keys and values encode attention to a different "
                      f"left context")
            repair_tokens += region.n_tokens
        placements.append(Placement(region=region, status=status, expected_context=context,
                                    reason=reason))
        context = chain_digest(context, region)

    if repair_tokens and not allow_repair:
        broken = ", ".join(p.region.name for p in placements if not p.exact)
        raise ComposeError(
            f"{repair_tokens} tokens across [{broken}] were cached behind a different left "
            f"context. Reusing them as they are yields a cache conditioned on text that was "
            f"never present, and the output stays fluent while being wrong. Pass "
            f"allow_repair=True only with a selective-recompute pass that will fix them")

    return Plan(placements=tuple(placements), total_tokens=expected_start,
                repair_tokens=repair_tokens)


def appendable(existing, addition: Region) -> bool:
    """Whether `addition` can be appended to `existing` with no repair at all.

    This is the case worth designing for: a memory file that grows at the end costs nothing,
    because everything already cached keeps the same left context. Insert in the middle
    instead and every later region needs repair - which is why an append-only memory region
    is not a limitation of this design but the reason it works.
    """
    context = AT_FRONT
    for region in existing:
        context = chain_digest(context, region)
    return addition.prefilled_after == context
