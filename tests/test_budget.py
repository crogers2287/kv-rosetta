from __future__ import annotations

import unittest

from kv_rosetta.daemon.budget import (
    BudgetConfig,
    BudgetError,
    select_prune,
    select_warm,
)


class Stub:
    """Minimal stand-in for a manifest: only est_tokens and mtime are used."""

    def __init__(self, est_tokens: int, mtime: float) -> None:
        self.est_tokens = est_tokens
        self.mtime = mtime


class SelectWarmTests(unittest.TestCase):
    def test_admits_in_order_until_slot_budget(self) -> None:
        cfg = BudgetConfig(max_warm=2, max_warm_tokens=10_000_000)
        manifests = (
            Stub(10, 3.0),
            Stub(10, 2.0),
            Stub(10, 1.0),
            Stub(10, 0.0),
        )
        sel = select_warm(manifests, cfg)
        self.assertEqual(len(sel.warm), 2)
        self.assertEqual(len(sel.skipped), 2)
        self.assertEqual(sel.total_tokens, 20)
        for _, reason in sel.skipped:
            self.assertEqual(reason, "slot budget")

    def test_token_budget_skips_large_but_admits_smaller_later(self) -> None:
        cfg = BudgetConfig(max_warm=10, max_warm_tokens=100)
        manifests = (
            Stub(90, 3.0),
            Stub(90, 2.0),
            Stub(10, 1.0),
        )
        sel = select_warm(manifests, cfg)
        self.assertEqual(len(sel.warm), 2)
        self.assertEqual(sel.total_tokens, 100)
        reasons = [reason for _, reason in sel.skipped]
        self.assertEqual(reasons, ["token budget"])

    def test_every_dropped_manifest_is_reported(self) -> None:
        cfg = BudgetConfig(max_warm=2, max_warm_tokens=25)
        manifests = tuple(Stub(10, float(i)) for i in range(10))
        sel = select_warm(manifests, cfg)
        self.assertEqual(len(sel.warm) + len(sel.skipped), len(manifests))

    def test_zero_tokens_skipped_with_no_tokens_reason(self) -> None:
        cfg = BudgetConfig(max_warm=4, max_warm_tokens=100_000)
        manifests = (
            Stub(0, 3.0),
            Stub(10, 2.0),
        )
        sel = select_warm(manifests, cfg)
        self.assertEqual(len(sel.warm), 1)
        self.assertEqual(len(sel.skipped), 1)
        dropped, reason = sel.skipped[0]
        self.assertEqual(reason, "no tokens")
        self.assertEqual(dropped.est_tokens, 0)


class BudgetConfigTests(unittest.TestCase):
    def test_negative_max_warm_raises(self) -> None:
        with self.assertRaises(BudgetError):
            BudgetConfig(max_warm=-1)

    def test_negative_tokens_raises(self) -> None:
        with self.assertRaises(BudgetError):
            BudgetConfig(max_warm_tokens=-5)

    def test_negative_entries_raises(self) -> None:
        with self.assertRaises(BudgetError):
            BudgetConfig(max_entries=-1)


class SelectPruneTests(unittest.TestCase):
    def test_prune_drops_entries_older_than_max_age(self) -> None:
        cfg = BudgetConfig(max_age_seconds=100, max_entries=0)
        now = 1000.0
        manifests = (
            Stub(10, 500.0),
            Stub(10, 950.0),
            Stub(10, 999.0),
        )
        deleted = select_prune(manifests, cfg, now)
        self.assertEqual(len(deleted), 1)
        self.assertEqual(deleted[0].mtime, 500.0)

    def test_prune_drops_oldest_beyond_max_entries(self) -> None:
        cfg = BudgetConfig(max_age_seconds=0, max_entries=2)
        now = 1000.0
        manifests = (
            Stub(10, 1.0),
            Stub(10, 2.0),
            Stub(10, 3.0),
            Stub(10, 4.0),
        )
        deleted = select_prune(manifests, cfg, now)
        self.assertEqual(len(deleted), 2)
        self.assertEqual(sorted(m.mtime for m in deleted), [1.0, 2.0])

    def test_both_limits_zero_returns_empty(self) -> None:
        cfg = BudgetConfig(max_age_seconds=0, max_entries=0)
        now = 1000.0
        manifests = (
            Stub(10, 1.0),
            Stub(10, 2.0),
        )
        deleted = select_prune(manifests, cfg, now)
        self.assertEqual(deleted, ())


if __name__ == "__main__":
    unittest.main()
