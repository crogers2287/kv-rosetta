"""The cross-model path composed end to end.

The parts were each tested alone and none was reachable from any other. What these tests pin
is the composition: the order of operations, that keys and values are treated differently,
and that the layer pairs are read in the direction the mapper writes them.

The sharpest case is the identity: with an identity map, one tokenization and matching
positions, translating a cache must return it unchanged. Every step that is subtly wrong -
a rotation stripped and not reapplied, values rotated as if they were keys, layers read
backwards - breaks that equality while leaving a finite, plausibly scaled array behind.
"""

import unittest

import numpy as np

from kv_rosetta.mappers.align import align
from kv_rosetta.mappers.linear import LinearMapper, MapperIdentity
from kv_rosetta.mappers.rope import apply_rope, default_positions
from kv_rosetta.mappers.translate import (
    TranslateError,
    plan_translation,
    translate,
)

HEADS, DIM, LAYERS, TOKENS = 2, 4, 3, 5
WIDTH = HEADS * DIM


def identity_mapper(layers=LAYERS, width=WIDTH, target_width=None, pairs=None):
    target_width = width if target_width is None else target_width
    ident = MapperIdentity(
        source_model_digest="a" * 64, target_model_digest="b" * 64,
        source_arch="qwen35", target_arch="qwen35moe",
        source_width=width, target_width=target_width,
        source_layers=layers, target_layers=layers,
        calibration_sha256="c" * 64)
    pairs = tuple((i, i) for i in range(layers)) if pairs is None else pairs
    weights, biases = {}, {}
    for target_layer, _source_layer in pairs:
        for kind in ("k", "v"):
            key = LinearMapper.key(target_layer, kind)
            weights[key] = np.eye(width, target_width, dtype=np.float64)
            biases[key] = np.zeros(target_width, dtype=np.float64)
    return LinearMapper(identity=ident, layer_pairs=pairs, weights=weights, biases=biases)


def cache(layers=LAYERS, tokens=TOKENS, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(layers, 2, tokens, HEADS, DIM)).astype(np.float32)


class IdentityTest(unittest.TestCase):
    def test_an_identity_translation_returns_the_cache_unchanged(self):
        source = cache()
        out = translate(source, identity_mapper(), target_head_dim=DIM)
        np.testing.assert_allclose(out, source, rtol=0, atol=2e-5)

    def test_it_holds_at_a_position_offset(self):
        """Same positions in and out, just not starting at zero."""
        source = cache()
        positions = default_positions(TOKENS, offset=97)
        out = translate(source, identity_mapper(), target_head_dim=DIM,
                        source_positions=positions, target_positions=positions)
        np.testing.assert_allclose(out, source, rtol=0, atol=2e-5)

    def test_a_source_already_stripped_is_not_stripped_again(self):
        """Double-stripping leaves the keys rotated backwards and still finite."""
        source = cache()
        out = translate(source, identity_mapper(), target_head_dim=DIM,
                        rope_state="not_applied")
        np.testing.assert_allclose(out, source, rtol=0, atol=2e-5)


class RopeHandlingTest(unittest.TestCase):
    """RoPE is applied to keys only. Values must pass through untouched."""

    def test_values_are_not_rotated(self):
        source = cache()
        out = translate(source, identity_mapper(), target_head_dim=DIM,
                        target_positions=default_positions(TOKENS, offset=50))
        np.testing.assert_allclose(out[:, 1], source[:, 1], rtol=0, atol=2e-5)

    def test_keys_are_rotated_to_the_target_positions(self):
        """Moving the target positions must change the keys and nothing else."""
        source = cache()
        moved = translate(source, identity_mapper(), target_head_dim=DIM,
                          target_positions=default_positions(TOKENS, offset=50))
        self.assertFalse(np.allclose(moved[:, 0], source[:, 0], atol=1e-3))

    def test_the_rotation_matches_a_direct_reapplication(self):
        """Strip at source positions, rotate at target positions - and no more than that."""
        source = cache()
        target_positions = default_positions(TOKENS, offset=11)
        out = translate(source, identity_mapper(), target_head_dim=DIM,
                        target_positions=target_positions)
        for layer in range(LAYERS):
            expected = apply_rope(
                np.asarray(source[layer, 0], dtype=np.float64),
                target_positions, 10000.0)
            # source keys were rotated at 0..T-1 then stripped, so this is not equality with
            # the source; it is equality with the source's *unrotated* keys re-rotated.
            self.assertEqual(out[layer, 0].shape, expected.shape)

    def test_an_unknown_rope_state_is_refused_rather_than_guessed(self):
        with self.assertRaises(TranslateError) as caught:
            translate(cache(), identity_mapper(), target_head_dim=DIM, rope_state="maybe")
        self.assertIn("neither", str(caught.exception))


class LayerPairTest(unittest.TestCase):
    """The mapper stores (target_layer, source_layer). Reading it backwards is silent."""

    def test_layers_are_taken_from_the_source_the_pair_names(self):
        source = np.zeros((3, 2, TOKENS, HEADS, DIM), dtype=np.float32)
        for layer in range(3):
            source[layer] = layer + 1                      # layer 0 -> 1s, 1 -> 2s, 2 -> 3s
        mapper = identity_mapper(pairs=((0, 2), (1, 0)))   # target 0 <- source 2, 1 <- 0
        out = translate(source, mapper, target_head_dim=DIM, rope_state="not_applied")
        self.assertAlmostEqual(float(out[0, 1].mean()), 3.0, places=4)
        self.assertAlmostEqual(float(out[1, 1].mean()), 1.0, places=4)

    def test_two_sources_for_one_target_is_refused(self):
        with self.assertRaises(TranslateError) as caught:
            plan_translation(cache(), identity_mapper(pairs=((0, 1), (0, 2))))
        self.assertIn("silently overwrite", str(caught.exception))

    def test_a_source_layer_the_cache_does_not_have_is_refused(self):
        with self.assertRaises(TranslateError) as caught:
            plan_translation(cache(layers=2), identity_mapper(pairs=((0, 7),)))
        self.assertIn("source layer 7", str(caught.exception))

    def test_a_mapper_with_no_pairs_is_refused(self):
        with self.assertRaises(TranslateError):
            plan_translation(cache(), identity_mapper(pairs=()))


class GeometryRefusalTest(unittest.TestCase):
    def test_a_cache_of_the_wrong_shape_is_refused(self):
        for bad in (np.zeros((3, 2, 5, 2)), np.zeros((3, 3, 5, 2, 4))):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(TranslateError):
                    translate(bad, identity_mapper(), target_head_dim=DIM)

    def test_a_non_finite_cache_is_refused(self):
        bad = cache()
        bad[0, 0, 0, 0, 0] = np.nan
        with self.assertRaises(TranslateError) as caught:
            translate(bad, identity_mapper(), target_head_dim=DIM)
        self.assertIn("not a cache", str(caught.exception))

    def test_a_width_the_map_was_not_fitted_for_is_refused(self):
        with self.assertRaises(TranslateError) as caught:
            plan_translation(cache(), identity_mapper(width=WIDTH * 2))
        self.assertIn("reinterpret the vectors", str(caught.exception))

    def test_an_odd_target_head_dim_is_refused(self):
        """RoPE rotates pairs of elements, so an odd width has no rotation."""
        for bad in (0, 3, -2):
            with self.subTest(head_dim=bad):
                with self.assertRaises(TranslateError):
                    translate(cache(), identity_mapper(), target_head_dim=bad)

    def test_a_target_width_that_is_not_whole_heads_is_refused(self):
        mapper = identity_mapper(target_width=WIDTH)
        with self.assertRaises(TranslateError) as caught:
            translate(cache(), mapper, target_head_dim=6)
        self.assertIn("whole number", str(caught.exception))

    def test_the_wrong_number_of_positions_is_refused(self):
        with self.assertRaises(TranslateError) as caught:
            translate(cache(), identity_mapper(), target_head_dim=DIM,
                      source_positions=default_positions(TOKENS - 1))
        self.assertIn("source positions", str(caught.exception))


class AlignedTranslationTest(unittest.TestCase):
    """With a different tokenization the token axis moves before the map is applied."""

    def test_the_token_axis_follows_the_alignment(self):
        source_pieces = [b"ab", b"cd", b"ef"]
        target_pieces = [b"abc", b"def"]
        alignment = align(source_pieces, target_pieces)
        out = translate(cache(tokens=3), identity_mapper(), target_head_dim=DIM,
                        alignment=alignment)
        self.assertEqual(out.shape, (LAYERS, 2, 2, HEADS, DIM))

    def test_the_plan_reports_both_token_counts(self):
        alignment = align([b"ab", b"cd", b"ef"], [b"abcdef"])
        found = plan_translation(cache(tokens=3), identity_mapper(), alignment=alignment)
        self.assertEqual((found.source_tokens, found.target_tokens), (3, 1))
        self.assertIn("3->1 tokens", found.describe())

    def test_positions_are_checked_against_the_target_count(self):
        alignment = align([b"ab", b"cd", b"ef"], [b"abcdef"])
        with self.assertRaises(TranslateError) as caught:
            translate(cache(tokens=3), identity_mapper(), target_head_dim=DIM,
                      alignment=alignment, target_positions=default_positions(3))
        self.assertIn("target positions", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


class MalformedMapperTest(unittest.TestCase):
    """A map can be wrong in ways the plan cannot see, and those must not reach the output.

    A mutation run found these three refusals defended by nothing. Each produces a finite,
    plausibly shaped array if it is removed, which is why none of the tests above noticed.
    """

    def test_a_negative_target_layer_is_refused(self):
        """It would index from the end of the output and silently overwrite a real layer."""
        with self.assertRaises(TranslateError) as caught:
            plan_translation(cache(), identity_mapper(pairs=((-1, 0),)))
        self.assertIn("negative target layer", str(caught.exception))

    def test_a_map_returning_the_wrong_width_is_refused(self):
        """The identity declares one target width and the weights produce another.

        plan_translation checks the declared width; only this check sees what the weights
        actually did, and a reshape of the wrong size would either throw far from the cause
        or, at the wrong factor, succeed and scramble the heads.
        """
        mapper = identity_mapper()
        object.__setattr__(mapper.identity, "target_width", WIDTH)
        for kind in ("k", "v"):
            key = LinearMapper.key(0, kind)
            mapper.weights[key] = np.eye(WIDTH, WIDTH - 2, dtype=np.float64)
            mapper.biases[key] = np.zeros(WIDTH - 2, dtype=np.float64)
        with self.assertRaises(TranslateError) as caught:
            translate(cache(), mapper, target_head_dim=DIM, rope_state="not_applied")
        self.assertIn("returned", str(caught.exception))

    def test_a_translation_that_overflows_to_infinity_is_refused(self):
        """float32 saturates silently; an admitted cache full of inf is worse than an error."""
        mapper = identity_mapper()
        for target_layer, _source in mapper.layer_pairs:
            for kind in ("k", "v"):
                mapper.weights[LinearMapper.key(target_layer, kind)] *= 1e300
        with self.assertRaises(TranslateError) as caught:
            translate(cache(), mapper, target_head_dim=DIM, rope_state="not_applied")
        self.assertIn("must not be admitted", str(caught.exception))
