"""transformers as an independent witness: a restored cache must change nothing.

Every other test in this project reads llama.cpp's view of a cache, which makes llama.cpp
both subject and witness. This runs a second implementation of the same maths and asks the
question that matters: after a full export and import, does the model produce the same
logits?

Not "close". Identical. If a round trip perturbs the cache at all, the next token's logits
move, and a cache that changes the answer is not a cache - it is a bug with good latency.

The model is tiny and randomly initialised on purpose. No download, no GPU, deterministic
under a seed, and the property under test is cache handling rather than model quality.
"""

import unittest

import numpy as np

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.cache_utils import DynamicCache
    HAVE_HF = True
except ImportError:                                # pragma: no cover
    HAVE_HF = False

from kv_rosetta.adapters.hf import (
    HFAdapterError,
    HFGeometry,
    describe,
    export_cache,
    import_cache,
)


@unittest.skipUnless(HAVE_HF, "torch and transformers are required")
class CacheRoundTripTest(unittest.TestCase):
    def cache(self, layers=3, heads=2, tokens=5, dim=8, seed=0):
        torch.manual_seed(seed)
        cache = DynamicCache()
        for layer in range(layers):
            cache.update(torch.randn(1, heads, tokens, dim),
                         torch.randn(1, heads, tokens, dim), layer)
        return cache

    def test_export_produces_canonical_ordering(self):
        canonical, geometry = export_cache(self.cache())
        self.assertEqual(canonical.shape, (3, 2, 5, 2, 8))
        self.assertEqual((geometry.n_layer, geometry.n_kv_head, geometry.head_dim),
                         (3, 2, 8))

    def test_the_round_trip_is_bit_exact(self):
        original = self.cache()
        restored = import_cache(export_cache(original)[0])
        for layer in range(3):
            np.testing.assert_array_equal(restored.key_cache[layer].numpy(),
                                          original.key_cache[layer].numpy())
            np.testing.assert_array_equal(restored.value_cache[layer].numpy(),
                                          original.value_cache[layer].numpy())

    def test_transposition_is_not_silently_wrong(self):
        """A head/token transpose survives a round trip but scrambles the cache.

        Shapes alone cannot catch it when head count and token count are equal, so this
        uses different values for each.
        """
        original = self.cache(heads=2, tokens=5)
        canonical, _ = export_cache(original)
        # canonical is (layer, kv, token, head, dim): index by token then head.
        for token in range(5):
            for head in range(2):
                np.testing.assert_array_equal(
                    canonical[0, 0, token, head],
                    original.key_cache[0][0, head, token].numpy())

    def test_a_batched_cache_is_refused(self):
        cache = DynamicCache()
        cache.update(torch.randn(2, 2, 5, 8), torch.randn(2, 2, 5, 8), 0)
        with self.assertRaises(HFAdapterError) as caught:
            export_cache(cache)
        self.assertIn("silently discard", str(caught.exception))

    def test_a_ragged_cache_is_refused(self):
        cache = DynamicCache()
        cache.update(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8), 0)
        cache.update(torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8), 1)
        with self.assertRaises(HFAdapterError) as caught:
            export_cache(cache)
        self.assertIn("no canonical form", str(caught.exception))

    def test_an_empty_cache_is_refused(self):
        with self.assertRaises(HFAdapterError):
            export_cache(DynamicCache())

    def test_a_malformed_canonical_array_is_refused(self):
        for bad in (np.zeros((3, 2, 5, 2)), np.zeros((3, 3, 5, 2, 8))):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(HFAdapterError):
                    import_cache(bad)

    def test_the_geometry_states_rope_is_applied(self):
        """HF caches keys after rotary embedding; a map must strip it before fitting."""
        _canonical, geometry = export_cache(self.cache())
        self.assertEqual(geometry.rope_state, "applied")
        self.assertTrue(HFGeometry(1, 1, 1, rope_state="not_applied").validate())
        self.assertIn("rope_state", describe(geometry))


@unittest.skipUnless(HAVE_HF, "torch and transformers are required")
class LogitEquivalenceTest(unittest.TestCase):
    """The headline claim: a restored cache must not move the next token's logits."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(11)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=64)
        cls.model = LlamaForCausalLM(config).eval()
        cls.prompt = torch.tensor([[3, 14, 15, 9, 26, 5, 35, 8]])
        cls.next_token = torch.tensor([[27]])

    def prefill(self):
        with torch.no_grad():
            out = self.model(self.prompt, use_cache=True)
        return out.past_key_values

    def test_a_restored_cache_gives_bitwise_identical_logits(self):
        cache = self.prefill()
        with torch.no_grad():
            direct = self.model(self.next_token, past_key_values=cache,
                                use_cache=True).logits

        restored = import_cache(export_cache(self.prefill())[0])
        with torch.no_grad():
            through_kvx = self.model(self.next_token, past_key_values=restored,
                                     use_cache=True).logits

        self.assertEqual(direct.shape, through_kvx.shape)
        np.testing.assert_array_equal(
            direct.numpy(), through_kvx.numpy(),
            "a round-tripped cache changed the logits; a cache that changes the answer is "
            "not a cache")

    def test_a_perturbed_cache_does_change_the_logits(self):
        """Proves the previous test can fail - otherwise it shows only that nothing ran."""
        canonical, _ = export_cache(self.prefill())
        canonical[0, 0, 0, 0, 0] += 1.0
        with torch.no_grad():
            direct = self.model(self.next_token, past_key_values=self.prefill(),
                                use_cache=True).logits
            perturbed = self.model(self.next_token,
                                   past_key_values=import_cache(canonical),
                                   use_cache=True).logits
        self.assertFalse(np.array_equal(direct.numpy(), perturbed.numpy()),
                         "a one-element change did not move the logits, so the equality "
                         "test above proves nothing")

    def test_a_truncated_cache_changes_the_logits(self):
        canonical, _ = export_cache(self.prefill())
        with torch.no_grad():
            full = self.model(self.next_token, past_key_values=self.prefill(),
                              use_cache=True).logits
            short = self.model(self.next_token,
                               past_key_values=import_cache(canonical[:, :, :-2]),
                               use_cache=True).logits
        self.assertFalse(np.array_equal(full.numpy(), short.numpy()))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_HF, "torch and transformers are required")
class MalformedCacheTest(unittest.TestCase):
    """One test per refusal, each reaching the guard it names.

    A guard is only real if disabling it makes a test fail. These were written after a
    mutation run showed six of them were decoration: every malformed input was being caught
    by an earlier check, so the later ones had never once fired.
    """

    def tensor(self, *shape):
        return torch.zeros(*shape)

    def legacy_only(self, pairs):
        """A cache exposing to_legacy_cache and nothing else - the 4.4x Cache shape."""
        class LegacyOnly:
            def to_legacy_cache(self):
                return pairs
        return LegacyOnly()

    def split(self, keys, values):
        """A cache with key_cache/value_cache set independently, to desynchronise them."""
        class Split:
            key_cache = keys
            value_cache = values
        return Split()

    def test_a_legacy_only_cache_is_read_through_its_accessor(self):
        pairs = [(self.tensor(1, 2, 5, 8), self.tensor(1, 2, 5, 8))]
        canonical, geometry = export_cache(self.legacy_only(pairs))
        self.assertEqual(canonical.shape, (1, 2, 5, 2, 8))
        self.assertEqual(geometry.n_layer, 1)

    def test_an_object_that_is_no_kind_of_cache_is_refused(self):
        with self.assertRaises(HFAdapterError) as caught:
            export_cache(object())
        self.assertIn("is not a cache", str(caught.exception))

    def test_a_legacy_entry_that_is_not_a_pair_is_refused(self):
        for entry in ((self.tensor(1, 2, 5, 8),),                       # one element
                      (self.tensor(1, 2, 5, 8),) * 3,                   # three
                      self.tensor(1, 2, 5, 8)):                         # not a sequence
            with self.subTest(entry=type(entry).__name__):
                with self.assertRaises(HFAdapterError) as caught:
                    export_cache(self.legacy_only([entry]))
                self.assertIn("(key, value) pair", str(caught.exception))

    def test_a_cache_with_more_keys_than_values_is_refused(self):
        cache = self.split([self.tensor(1, 2, 5, 8)] * 3,
                           [self.tensor(1, 2, 5, 8)] * 2)
        with self.assertRaises(HFAdapterError) as caught:
            export_cache(cache)
        self.assertIn("3 key layers against 2 value layers", str(caught.exception))

    def test_a_tensor_of_the_wrong_rank_is_refused(self):
        """Batch 1 keeps this past the batch guard, so it must reach the rank guard."""
        cache = self.split([self.tensor(1, 5, 8)], [self.tensor(1, 5, 8)])
        with self.assertRaises(HFAdapterError) as caught:
            export_cache(cache)
        self.assertIn("expected 4", str(caught.exception))

    def test_a_key_and_value_of_different_shape_are_refused(self):
        cache = self.split([self.tensor(1, 2, 5, 8)], [self.tensor(1, 2, 5, 4)])
        with self.assertRaises(HFAdapterError) as caught:
            export_cache(cache)
        self.assertIn("!= value", str(caught.exception))

    def test_a_zero_width_head_is_refused(self):
        """Shape-consistent, rank-correct, batch 1 - only the geometry check can catch it."""
        cache = self.split([self.tensor(1, 2, 5, 0)], [self.tensor(1, 2, 5, 0)])
        with self.assertRaises(HFAdapterError) as caught:
            export_cache(cache)
        self.assertIn("head_dim must be positive", str(caught.exception))
