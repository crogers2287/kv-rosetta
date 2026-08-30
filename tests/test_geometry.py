"""Reshaping canonical KV between model geometries.

This module changes shape and nothing else. Its correctness has a definition the
representation map does not: the target runtime either accepts the file or it does not, and
llama.cpp is precise about why - `state_read_data: mismatched layer count (16 instead of 8)`.

Keeping that separate from the quality question is the point. A geometry conversion can be
proved correct while the map is still being argued about, and a loadable file is the
precondition for measuring any map at all.
"""

import unittest

import numpy as np

from kv_rosetta.geometry import (
    Geometry,
    GeometryError,
    convert,
    depth_layer_map,
)


def cache(layers=8, tokens=5, heads=4, dim=256, seed=0):
    return np.random.default_rng(seed).normal(
        size=(layers, 2, tokens, heads, dim)).astype(np.float32)


class LayerMapTest(unittest.TestCase):
    def test_depth_mapping_covers_every_target_layer_once(self):
        pairs = depth_layer_map(Geometry(8, 4, 256), Geometry(16, 4, 256))
        self.assertEqual(sorted(t for t, _s in pairs), list(range(16)))

    def test_the_ends_are_anchored(self):
        """Target 0 takes source 0 and the last target takes the last source; drifting at
        the ends would misalign the whole stack."""
        pairs = dict(depth_layer_map(Geometry(8, 4, 256), Geometry(16, 4, 256)))
        self.assertEqual(pairs[0], 0)
        self.assertEqual(pairs[15], 7)

    def test_it_is_monotonic(self):
        pairs = depth_layer_map(Geometry(8, 4, 256), Geometry(16, 4, 256))
        sources = [s for _t, s in sorted(pairs)]
        self.assertEqual(sources, sorted(sources))

    def test_a_single_target_layer_does_not_divide_by_zero(self):
        self.assertEqual(depth_layer_map(Geometry(8, 4, 256), Geometry(1, 4, 256)),
                         ((0, 0),))


class ShapeTest(unittest.TestCase):
    def test_an_identity_conversion_returns_the_cache_unchanged(self):
        g = Geometry(8, 4, 256)
        source = cache()
        np.testing.assert_array_equal(convert(source, g, g), source)

    def test_expanding_layers_produces_the_target_shape(self):
        out = convert(cache(), Geometry(8, 4, 256), Geometry(16, 4, 256))
        self.assertEqual(out.shape, (16, 2, 5, 4, 256))

    def test_narrowing_heads_averages_adjacent_pairs(self):
        """GQA groups adjacent KV heads, so adjacent source heads are what merge."""
        source = cache()
        out = convert(source, Geometry(8, 4, 256), Geometry(8, 2, 256), head_policy="mean")
        np.testing.assert_allclose(out[0, 0, :, 0, :], source[0, 0, :, 0:2, :].mean(axis=1),
                                   rtol=1e-6)

    def test_selecting_takes_the_first_of_each_group_not_an_average(self):
        source = cache()
        out = convert(source, Geometry(8, 4, 256), Geometry(8, 2, 256), head_policy="select")
        np.testing.assert_array_equal(out[0, 0, :, 1, :], source[0, 0, :, 2, :])

    def test_widening_heads_repeats_them(self):
        source = cache(heads=2)
        out = convert(source, Geometry(8, 2, 256), Geometry(8, 4, 256))
        np.testing.assert_array_equal(out[0, 0, :, 0, :], out[0, 0, :, 1, :])

    def test_an_explicit_layer_map_is_honoured_over_the_depth_default(self):
        source = np.zeros((8, 2, 3, 4, 8), dtype=np.float32)
        for layer in range(8):
            source[layer] = layer
        out = convert(source, Geometry(8, 4, 8), Geometry(2, 4, 8),
                      layer_map=((0, 7), (1, 3)))
        self.assertAlmostEqual(float(out[0].mean()), 7.0)
        self.assertAlmostEqual(float(out[1].mean()), 3.0)


class RefusalTest(unittest.TestCase):
    def test_a_head_dimension_change_is_refused(self):
        """Regrouping heads cannot change the width of one; that is a projection, and
        pretending otherwise would silently truncate or pad every vector."""
        with self.assertRaises(GeometryError) as caught:
            convert(cache(), Geometry(8, 4, 256), Geometry(8, 4, 128))
        self.assertIn("projection rather than a regrouping", str(caught.exception))

    def test_an_uneven_head_grouping_is_refused(self):
        with self.assertRaises(GeometryError) as caught:
            convert(cache(heads=4), Geometry(8, 4, 256), Geometry(8, 3, 256))
        self.assertIn("do not group evenly", str(caught.exception))

    def test_an_uneven_head_expansion_is_refused(self):
        with self.assertRaises(GeometryError):
            convert(cache(heads=2), Geometry(8, 2, 256), Geometry(8, 3, 256))

    def test_a_layer_map_that_misses_a_target_is_refused(self):
        with self.assertRaises(GeometryError) as caught:
            convert(cache(), Geometry(8, 4, 256), Geometry(3, 4, 256),
                    layer_map=((0, 0), (1, 1)))
        self.assertIn("every target layer needs exactly one source", str(caught.exception))

    def test_a_layer_map_naming_the_same_target_twice_is_refused(self):
        with self.assertRaises(GeometryError):
            convert(cache(), Geometry(8, 4, 256), Geometry(2, 4, 256),
                    layer_map=((0, 0), (0, 1)))

    def test_a_layer_map_reading_past_the_source_is_refused(self):
        with self.assertRaises(GeometryError) as caught:
            convert(cache(), Geometry(8, 4, 256), Geometry(1, 4, 256), layer_map=((0, 99),))
        self.assertIn("reads source layer 99", str(caught.exception))

    def test_a_cache_that_disagrees_with_its_declared_geometry_is_refused(self):
        """The declared source is what the layer map is checked against, so a mismatch here
        would apply a valid-looking map to the wrong tensors."""
        with self.assertRaises(GeometryError) as caught:
            convert(cache(layers=4), Geometry(8, 4, 256), Geometry(8, 4, 256))
        self.assertIn("source geometry says", str(caught.exception))

    def test_a_non_canonical_array_is_refused(self):
        for bad in (np.zeros((8, 2, 5, 4)), np.zeros((8, 3, 5, 4, 256))):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(GeometryError):
                    convert(bad, Geometry(8, 4, 256), Geometry(8, 4, 256))

    def test_an_unknown_head_policy_is_refused(self):
        for bad in ("", "first", "MEAN", None):
            with self.subTest(policy=bad):
                with self.assertRaises(GeometryError):
                    convert(cache(), Geometry(8, 4, 256), Geometry(8, 2, 256),
                            head_policy=bad)

    def test_nonsense_geometry_is_refused_as_nonsense(self):
        """The message is asserted, not just the type: a zero-layer source also fails the
        shape check a few lines later, so a test that only catches GeometryError passes with
        this guard removed and reports the wrong reason."""
        for g, field in ((Geometry(0, 4, 256), "layers"),
                         (Geometry(8, 0, 256), "kv_heads"),
                         (Geometry(8, 4, 0), "head_dim")):
            with self.subTest(geometry=g):
                with self.assertRaises(GeometryError) as caught:
                    convert(cache(), g, Geometry(8, 4, 256))
                self.assertIn(f"{field} must be positive", str(caught.exception))

    def test_a_nonsense_target_geometry_is_refused_too(self):
        for g in (Geometry(0, 4, 256), Geometry(8, 0, 256), Geometry(8, 4, 0)):
            with self.subTest(geometry=g):
                with self.assertRaises(GeometryError) as caught:
                    convert(cache(), Geometry(8, 4, 256), g)
                self.assertIn("must be positive", str(caught.exception))


class HonestyTest(unittest.TestCase):
    def test_every_output_value_came_from_the_input(self):
        """This module reshapes and never invents. A conversion that produced values not
        derived from the source would be a map wearing a reshape's clothes."""
        source = cache(layers=4, tokens=3, heads=2, dim=4)
        out = convert(source, Geometry(4, 2, 4), Geometry(8, 2, 4))
        self.assertTrue(set(np.unique(out)).issubset(set(np.unique(source))))
