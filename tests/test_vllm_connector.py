"""The vLLM bridge, tested without vLLM.

vLLM's seam is in-process, so a connector can only be constructed inside a worker. That makes
it exactly the kind of code that never gets tested, so every decision lives in CanonicalBridge
which imports nothing from vLLM and runs here.

The behaviour that matters is what happens when something does not line up. A connector that
returns a wrong number of matched tokens does not make a response slow, it makes it wrong -
vLLM will trust those tokens as already computed and skip them. So every doubt returns zero,
which is precisely vLLM prefilling as it would without any connector at all.
"""

import unittest

import numpy as np

from kv_rosetta.adapters.vllm_connector import (
    CanonicalBridge,
    ConnectorError,
    ShardIdentity,
)


def paged(pages=4, page_size=8, width=6):
    """A stand-in for vLLM's paged buffer: (num_pages, page_size, ...)."""
    return np.arange(pages * page_size * width, dtype=np.float32).reshape(
        pages, page_size, width)


class ShardIdentityTest(unittest.TestCase):
    def test_a_single_rank_is_a_whole_cache(self):
        self.assertTrue(ShardIdentity().whole)
        self.assertFalse(ShardIdentity(tp_rank=0, tp_size=2).whole)

    def test_an_impossible_shard_is_refused(self):
        for shard in (ShardIdentity(tp_rank=2, tp_size=2), ShardIdentity(tp_size=0),
                      ShardIdentity(tp_rank=-1, tp_size=2)):
            with self.subTest(shard=shard):
                with self.assertRaises(ConnectorError):
                    CanonicalBridge(shard)


class ExtractInjectTest(unittest.TestCase):
    def setUp(self):
        self.bridge = CanonicalBridge()

    def test_extract_gathers_the_requested_slots_in_order(self):
        layer = paged()
        out = self.bridge.extract(layer, [0, 1, 9], n_head=2, head_dim=3)
        self.assertEqual(out.shape, (3, 2, 3))
        flat = layer.reshape(32, 6)
        for position, slot in enumerate([0, 1, 9]):
            np.testing.assert_array_equal(out[position].reshape(-1), flat[slot])

    def test_inject_is_the_inverse_of_extract(self):
        layer = paged()
        slots = [3, 11, 20]
        original = self.bridge.extract(layer, slots, n_head=2, head_dim=3).copy()
        self.bridge.inject(layer, slots, np.zeros_like(original))
        self.bridge.inject(layer, slots, original)
        np.testing.assert_array_equal(
            self.bridge.extract(layer, slots, n_head=2, head_dim=3), original)

    def test_inject_touches_only_the_named_slots(self):
        layer = paged()
        before = layer.copy()
        self.bridge.inject(layer, [5], np.zeros((1, 2, 3), dtype=np.float32))
        flat, was = layer.reshape(32, 6), before.reshape(32, 6)
        for slot in range(32):
            if slot == 5:
                self.assertTrue((flat[slot] == 0).all())
            else:
                np.testing.assert_array_equal(flat[slot], was[slot])

    def test_a_slot_outside_the_buffer_is_refused(self):
        for slots in ([32], [0, 999], [31, 32]):
            with self.subTest(slots=slots):
                with self.assertRaises(ConnectorError) as caught:
                    self.bridge.extract(paged(), slots, n_head=2, head_dim=3)
                self.assertIn("outside", str(caught.exception))

    def test_a_negative_slot_is_refused(self):
        with self.assertRaises(ConnectorError):
            self.bridge.extract(paged(), [-1], n_head=2, head_dim=3)

    def test_geometry_that_does_not_match_the_buffer_is_refused(self):
        with self.assertRaises(ConnectorError) as caught:
            self.bridge.extract(paged(width=6), [0], n_head=4, head_dim=3)
        self.assertIn("does not match the declared geometry", str(caught.exception))

    def test_a_partial_write_is_refused_rather_than_leaving_stale_slots(self):
        with self.assertRaises(ConnectorError) as caught:
            self.bridge.inject(paged(), [0, 1, 2], np.zeros((2, 2, 3), dtype=np.float32))
        self.assertIn("remainder stale", str(caught.exception))

    def test_a_width_mismatch_on_inject_is_refused(self):
        with self.assertRaises(ConnectorError):
            self.bridge.inject(paged(width=6), [0], np.zeros((1, 2, 8), dtype=np.float32))

    def test_a_buffer_without_pages_is_refused(self):
        with self.assertRaises(ConnectorError):
            self.bridge.extract(np.zeros(10, dtype=np.float32), [0], n_head=1, head_dim=1)


class MatchedTokensTest(unittest.TestCase):
    """Zero is the safe answer, and vLLM treats matched tokens as already computed."""

    def setUp(self):
        self.bridge = CanonicalBridge()
        self.base = dict(artifact_shard=ShardIdentity(), artifact_model="m",
                         live_model="m", artifact_tokens=[1, 2, 3],
                         request_tokens=[1, 2, 3, 4, 5])

    def test_an_exact_prefix_is_matched(self):
        self.assertEqual(self.bridge.matched_tokens(**self.base), 3)

    def test_a_different_model_matches_nothing(self):
        self.assertEqual(self.bridge.matched_tokens(**{**self.base, "live_model": "other"}), 0)
        self.assertTrue(self.bridge.stats.refusals)

    def test_an_unknown_model_matches_nothing(self):
        self.assertEqual(self.bridge.matched_tokens(**{**self.base, "artifact_model": ""}), 0)

    def test_a_shard_from_another_rank_matches_nothing(self):
        found = self.bridge.matched_tokens(
            **{**self.base, "artifact_shard": ShardIdentity(tp_rank=1, tp_size=2)})
        self.assertEqual(found, 0)
        self.assertIn("not a whole cache", self.bridge.stats.refusals[-1])

    def test_a_sharded_artifact_is_not_accepted_by_an_unsharded_worker(self):
        found = self.bridge.matched_tokens(
            **{**self.base, "artifact_shard": ShardIdentity(tp_rank=0, tp_size=4)})
        self.assertEqual(found, 0, "a quarter of a cache was treated as a whole one")

    def test_a_diverging_prefix_matches_nothing_not_the_common_part(self):
        """Half a prefix is not a usable cache: vLLM would skip tokens it never computed."""
        found = self.bridge.matched_tokens(
            **{**self.base, "artifact_tokens": [1, 2, 9], "request_tokens": [1, 2, 3, 4]})
        self.assertEqual(found, 0)
        self.assertIn("diverges", self.bridge.stats.refusals[-1])

    def test_an_empty_side_matches_nothing_and_says_why(self):
        """Assert the reason, not just the zero.

        With no artifact tokens the comparison returns zero by a second route anyway, so
        the guard broke no test until the refusal itself was checked - a mutation run
        caught that.
        """
        for key in ("artifact_tokens", "request_tokens"):
            with self.subTest(empty=key):
                bridge = CanonicalBridge()
                self.assertEqual(bridge.matched_tokens(**{**self.base, key: []}), 0)
                self.assertTrue(bridge.stats.refusals, "refused without recording why")
                self.assertIn("no tokens to compare", bridge.stats.refusals[-1])

    def test_a_request_shorter_than_the_artifact_matches_nothing(self):
        found = self.bridge.matched_tokens(
            **{**self.base, "artifact_tokens": [1, 2, 3, 4], "request_tokens": [1, 2]})
        self.assertEqual(found, 0)


class ConnectorConstructionTest(unittest.TestCase):
    def test_building_without_vllm_raises_a_clear_error(self):
        import importlib.util
        if importlib.util.find_spec("vllm") is not None:
            self.skipTest("vLLM is importable here; this checks the absent case")
        from kv_rosetta.adapters.vllm_connector import build_connector
        with self.assertRaises(ConnectorError) as caught:
            build_connector(None, None)
        self.assertIn("not importable", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
