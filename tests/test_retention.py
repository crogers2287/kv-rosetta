"""Which cached prefixes survive a budget.

kvwarm cycled up to 51 prefixes through a server with 2 slots every 900 seconds, and thirteen
of twenty observed cycles were complete re-prefills costing 15 to 84 seconds each: every pass
evicted what the previous pass had built. Fifty-one things do not fit in two slots and no
scheduling makes them. These tests pin the policy that starts from that fact.
"""

import unittest

from kv_rosetta.retention import (
    Entry,
    RetentionError,
    admit_or_refuse,
    plan_retention,
)


def entry(tag, nbytes, last_used, pinned=False):
    return Entry(digest=tag * 64, nbytes=nbytes, last_used=last_used, pinned=pinned)


class KeepTest(unittest.TestCase):
    def test_everything_fitting_is_kept(self):
        found = plan_retention([entry("a", 100, 1.0), entry("b", 100, 2.0)], 500)
        self.assertEqual(len(found.keep), 2)
        self.assertEqual(found.evict, ())
        self.assertEqual(found.headroom, 300)

    def test_the_least_recently_used_is_dropped_first(self):
        found = plan_retention([entry("a", 100, 1.0), entry("b", 100, 9.0)], 100)
        self.assertEqual(found.keep, ("b" * 64,))
        self.assertEqual(found.evict, ("a" * 64,))

    def test_pinned_entries_survive_regardless_of_age(self):
        """A pin that can be evicted is not a pin."""
        found = plan_retention([entry("a", 100, 99.0), entry("b", 100, 1.0, pinned=True)], 100)
        self.assertEqual(found.keep, ("b" * 64,))
        self.assertIn("a" * 64, found.evict)

    def test_every_decision_carries_a_reason(self):
        found = plan_retention([entry("a", 100, 1.0), entry("b", 900, 2.0)], 150)
        self.assertEqual(found.reasons["a" * 64], "fits")
        self.assertIn("no room", found.reasons["b" * 64])

    def test_ties_break_on_digest_so_the_choice_is_stable(self):
        """An unstable eviction makes one bad prefix look like an intermittent fault."""
        first = plan_retention([entry("a", 100, 5.0), entry("b", 100, 5.0)], 100)
        second = plan_retention([entry("b", 100, 5.0), entry("a", 100, 5.0)], 100)
        self.assertEqual(first.keep, second.keep)

    def test_a_zero_budget_keeps_nothing_but_does_not_raise(self):
        found = plan_retention([entry("a", 100, 1.0)], 0)
        self.assertEqual(found.keep, ())
        self.assertEqual(len(found.evict), 1)

    def test_nothing_stored_is_not_an_error(self):
        self.assertEqual(plan_retention([], 500).keep, ())


class RefusalTest(unittest.TestCase):
    def test_pins_exceeding_the_budget_are_refused_not_resolved_later(self):
        with self.assertRaises(RetentionError) as caught:
            plan_retention([entry("a", 300, 1.0, pinned=True),
                            entry("b", 300, 2.0, pinned=True)], 500)
        self.assertIn("not a pin", str(caught.exception))

    def test_a_duplicate_digest_is_refused(self):
        with self.assertRaises(RetentionError) as caught:
            plan_retention([entry("a", 100, 1.0), entry("a", 200, 2.0)], 500)
        self.assertIn("appears twice", str(caught.exception))

    def test_a_negative_budget_is_refused_as_negative(self):
        """The message is asserted: with an empty store the pinned-bytes check also fires on
        a negative budget, so a test that only catches the type passes with this guard gone
        and reports a nonsensical reason."""
        with self.assertRaises(RetentionError) as caught:
            plan_retention([], -1)
        self.assertIn("is negative", str(caught.exception))

    def test_a_malformed_entry_is_refused(self):
        for bad in (Entry("short", 100, 1.0), Entry("a" * 64, 0, 1.0),
                    Entry("a" * 64, -5, 1.0)):
            with self.subTest(entry=bad):
                with self.assertRaises(RetentionError):
                    plan_retention([bad], 500)


class AdmissionTest(unittest.TestCase):
    def test_a_candidate_that_fits_is_admitted(self):
        found = admit_or_refuse([entry("a", 100, 1.0)], entry("b", 100, 2.0), 500)
        self.assertIn("b" * 64, found.keep)
        self.assertIn("a" * 64, found.keep)

    def test_a_candidate_larger_than_the_whole_budget_is_refused_before_evicting(self):
        """Freeing space for something that still will not fit destroys value and produces
        none - the shape of the warmer's mistake."""
        with self.assertRaises(RetentionError) as caught:
            admit_or_refuse([entry("a", 100, 1.0)], entry("b", 900, 2.0), 500)
        self.assertIn("refusing before evicting anything", str(caught.exception))

    def test_admitting_a_fresher_candidate_evicts_a_staler_entry(self):
        found = admit_or_refuse([entry("a", 400, 1.0)], entry("b", 400, 9.0), 500)
        self.assertEqual(found.keep, ("b" * 64,))
        self.assertEqual(found.evict, ("a" * 64,))

    def test_a_candidate_cannot_evict_a_pin(self):
        with self.assertRaises(RetentionError):
            admit_or_refuse([entry("a", 400, 1.0, pinned=True)], entry("b", 400, 9.0), 500)

    def test_a_malformed_candidate_is_refused_as_the_candidate(self):
        """plan_retention would also reject it a moment later, but by then the caller has
        been told the store is malformed rather than their input."""
        with self.assertRaises(RetentionError) as caught:
            admit_or_refuse([], Entry("a" * 64, 0, 1.0), 500)
        self.assertIn("candidate", str(caught.exception))


class WarmerArithmeticTest(unittest.TestCase):
    """The situation that motivated this, in the policy's own terms."""

    def test_fifty_one_prefixes_into_two_slots_keeps_two_and_says_so(self):
        entries = [entry(chr(ord("a") + i % 26) * 1, 100, float(i)) for i in range(51)]
        entries = [Entry(f"{i:064d}", 100, float(i)) for i in range(51)]
        found = plan_retention(entries, 200)
        self.assertEqual(len(found.keep), 2)
        self.assertEqual(len(found.evict), 49)
        self.assertTrue(all("no room" in found.reasons[d] for d in found.evict))

    def test_pinning_the_two_that_matter_makes_the_outcome_deliberate(self):
        entries = [Entry(f"{i:064d}", 100, float(i), pinned=i in (0, 1)) for i in range(51)]
        found = plan_retention(entries, 200)
        self.assertEqual(set(found.keep), {f"{0:064d}", f"{1:064d}"})
        self.assertEqual(found.reasons[f"{0:064d}"], "pinned")
