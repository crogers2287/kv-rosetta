"""Composing a prompt from independently cached regions.

The failure this module exists to prevent does not raise and does not look wrong: reuse a
region that was cached behind a different left context and you get a cache conditioned on text
that was never there. The model keeps writing fluent sentences about the wrong premise. Every
test below is ultimately about refusing that.
"""

import unittest

from kv_rosetta.compose import (
    AT_FRONT,
    ComposeError,
    Region,
    appendable,
    chain_digest,
    plan,
    token_digest,
)


def region(name, role, start, end, tokens=None, after=AT_FRONT):
    return Region(name=name, role=role, token_start=start, token_end=end,
                  tokens_sha256=token_digest(tokens if tokens is not None
                                             else range(start, end)),
                  prefilled_after=after)


def behind(*regions):
    context = AT_FRONT
    for item in regions:
        context = chain_digest(context, item)
    return context


class TokenDigestTest(unittest.TestCase):
    def test_identity_is_the_token_ids_not_the_text(self):
        """Two tokenizations of one string give different caches, so text is the wrong key."""
        self.assertNotEqual(token_digest([1, 2, 3]), token_digest([1, 23]))

    def test_the_same_ids_give_the_same_digest(self):
        self.assertEqual(token_digest([5, 6, 7]), token_digest([5, 6, 7]))

    def test_order_matters(self):
        self.assertNotEqual(token_digest([1, 2]), token_digest([2, 1]))

    def test_negative_ids_are_handled(self):
        """Signed so a sentinel like -1 cannot collide with a large positive id."""
        self.assertNotEqual(token_digest([-1]), token_digest([2 ** 63 - 1]))

    def test_no_tokens_has_no_identity(self):
        with self.assertRaises(ComposeError):
            token_digest([])


class ChainDigestTest(unittest.TestCase):
    def test_the_chain_is_order_sensitive(self):
        a = region("a", "system", 0, 10)
        b = region("b", "memory", 10, 20)
        self.assertNotEqual(behind(a, b), behind(b, a))

    def test_a_separator_prevents_a_boundary_collision(self):
        """Without one, ('ab','c') and ('a','bc') could hash alike."""
        self.assertNotEqual(chain_digest("ab", region("x", "system", 0, 1)),
                            chain_digest("a", region("x", "system", 0, 1)))


class AppendTest(unittest.TestCase):
    """The case worth designing for: memory that grows at the end costs nothing."""

    def test_appending_at_the_end_is_exact(self):
        system = region("system", "system", 0, 100)
        memory = region("memory", "memory", 100, 150, after=behind(system))
        found = plan([system, memory])
        self.assertTrue(found.exact)
        self.assertEqual((found.total_tokens, found.repair_tokens), (150, 0))

    def test_a_third_append_is_still_exact(self):
        system = region("system", "system", 0, 100)
        tools = region("tools", "tools", 100, 180, after=behind(system))
        memory = region("memory", "memory", 180, 260, after=behind(system, tools))
        self.assertTrue(plan([system, tools, memory]).exact)

    def test_appendable_reports_the_same_thing_without_building_a_plan(self):
        system = region("system", "system", 0, 100)
        good = region("memory", "memory", 100, 150, after=behind(system))
        bad = region("memory", "memory", 100, 150)
        self.assertTrue(appendable([system], good))
        self.assertFalse(appendable([system], bad))

    def test_inserting_in_the_middle_invalidates_what_follows(self):
        """Everything after an insertion attended to a left context that no longer exists.

        This is why append-only is not a limitation of the design but the reason it works.
        """
        system = region("system", "system", 0, 100)
        memory = region("memory", "memory", 100, 150, after=behind(system))
        inserted = region("doc", "document", 100, 120, after=behind(system))
        moved = Region(name="memory", role="memory", token_start=120, token_end=170,
                       tokens_sha256=memory.tokens_sha256,
                       prefilled_after=memory.prefilled_after)
        found = plan([system, inserted, moved], allow_repair=True)
        self.assertEqual([p.region.name for p in found.needing_repair()], ["memory"])
        self.assertEqual(found.repair_tokens, 50)


class RefusalTest(unittest.TestCase):
    def test_a_region_cached_behind_a_different_context_is_refused(self):
        system = region("system", "system", 0, 100)
        memory = region("memory", "memory", 100, 150)          # cached at the front
        with self.assertRaises(ComposeError) as caught:
            plan([system, memory])
        self.assertIn("conditioned on text that was never present", str(caught.exception))

    def test_repair_is_opt_in_and_priced(self):
        system = region("system", "system", 0, 100)
        memory = region("memory", "memory", 100, 150)
        found = plan([system, memory], allow_repair=True)
        self.assertFalse(found.exact)
        self.assertEqual(found.repair_tokens, 50)
        self.assertIn("encode attention to a different left context",
                      found.needing_repair()[0].reason)

    def test_a_gap_between_regions_is_refused(self):
        system = region("system", "system", 0, 100)
        memory = region("memory", "memory", 120, 150, after=behind(system))
        with self.assertRaises(ComposeError) as caught:
            plan([system, memory])
        self.assertIn("gap or overlap", str(caught.exception))

    def test_an_overlap_is_refused(self):
        system = region("system", "system", 0, 100)
        memory = region("memory", "memory", 80, 150, after=behind(system))
        with self.assertRaises(ComposeError):
            plan([system, memory])

    def test_a_composition_that_does_not_start_at_zero_is_refused(self):
        with self.assertRaises(ComposeError) as caught:
            plan([region("memory", "memory", 40, 90)])
        self.assertIn("cannot be placed", str(caught.exception))

    def test_a_repeated_region_name_is_refused(self):
        system = region("system", "system", 0, 100)
        twin = region("system", "memory", 100, 150, after=behind(system))
        with self.assertRaises(ComposeError) as caught:
            plan([system, twin])
        self.assertIn("appears twice", str(caught.exception))

    def test_an_unknown_role_is_refused_rather_than_taken_as_free_text(self):
        """'memmory' and 'memory ' would otherwise become two silently different roles."""
        for role in ("memmory", "memory ", "", "Memory"):
            with self.subTest(role=role):
                with self.assertRaises(ComposeError):
                    plan([region("r", role, 0, 10)])

    def test_an_empty_region_is_refused(self):
        bad = Region("r", "system", 10, 10, token_digest([1]), AT_FRONT)
        with self.assertRaises(ComposeError):
            plan([bad])

    def test_a_truncated_digest_is_refused(self):
        bad = Region("r", "system", 0, 10, "abc123", AT_FRONT)
        with self.assertRaises(ComposeError) as caught:
            plan([bad])
        self.assertIn("full sha256", str(caught.exception))

    def test_an_unrecorded_left_context_is_refused(self):
        """Blank is not the same as AT_FRONT; treating it so would assume position zero."""
        bad = Region("r", "system", 0, 10, token_digest([1]), "")
        with self.assertRaises(ComposeError) as caught:
            plan([bad])
        self.assertIn("must be recorded", str(caught.exception))

    def test_a_negative_start_is_refused(self):
        bad = Region("r", "system", -5, 10, token_digest([1]), AT_FRONT)
        with self.assertRaises(ComposeError):
            plan([bad])

    def test_composing_nothing_is_refused(self):
        with self.assertRaises(ComposeError):
            plan([])


if __name__ == "__main__":
    unittest.main()
