from __future__ import annotations

from dataclasses import dataclass


class BudgetError(ValueError):
    pass


@dataclass(frozen=True)
class BudgetConfig:
    max_warm: int = 4
    max_warm_tokens: int = 60000
    max_entries: int = 200
    max_age_seconds: float = 14 * 86400.0

    def __post_init__(self) -> None:
        if (self.max_warm < 0 or self.max_warm_tokens < 0
                or self.max_entries < 0 or self.max_age_seconds < 0):
            raise BudgetError("all BudgetConfig fields must be non-negative")


@dataclass(frozen=True)
class Selection:
    warm: tuple[object, ...]
    skipped: tuple[tuple[object, str], ...]
    total_tokens: int


def select_warm(manifests: tuple[object, ...], cfg: BudgetConfig) -> Selection:
    warm: list[object] = []
    skipped: list[tuple[object, str]] = []
    total_tokens = 0
    for m in manifests:
        est = m.est_tokens
        if est <= 0:
            skipped.append((m, "no tokens"))
            continue
        if total_tokens + est > cfg.max_warm_tokens:
            skipped.append((m, "token budget"))
            continue
        if len(warm) >= cfg.max_warm:
            skipped.append((m, "slot budget"))
            continue
        warm.append(m)
        total_tokens += est
    return Selection(warm=tuple(warm), skipped=tuple(skipped), total_tokens=total_tokens)


def select_prune(manifests: tuple[object, ...], cfg: BudgetConfig, now: float) -> tuple[object, ...]:
    to_delete: list[object] = []
    remaining: list[object] = []
    if cfg.max_age_seconds > 0:
        age_cutoff = now - cfg.max_age_seconds
        for m in manifests:
            if m.mtime < age_cutoff:
                to_delete.append(m)
            else:
                remaining.append(m)
    else:
        remaining.extend(manifests)
    if cfg.max_entries > 0 and len(remaining) > cfg.max_entries:
        remaining.sort(key=lambda m: m.mtime)
        to_delete.extend(remaining[: len(remaining) - cfg.max_entries])
    to_delete.sort(key=lambda m: m.mtime)
    return tuple(to_delete)
