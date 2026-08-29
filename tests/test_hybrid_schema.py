"""The canonical hybrid schema must refuse anything it cannot faithfully represent.

The layout inventory showed a llama.cpp state file is not self-describing: head split, layer
identity, RoPE state, and the recurrent layer map all come from the source GGUF. This schema
records them so the target does not need that GGUF. An artifact missing them is not
"underspecified" - it is unusable, and these tests pin that as a refusal.
"""

import json
import unittest

from kv_rosetta.hybrid_schema import (
    CANONICAL_LAYOUT,
    SCHEMA,
    CheckpointRecord,
    HybridState,
    RecurrentLayerMap,
    SchemaError,
    SourceGeometry,
)


def geometry(**overrides) -> SourceGeometry:
    base = dict(architecture="qwen35", n_layer=4, n_head_kv=(4, 4, 4, 4),
                n_embd_head_k=128, n_embd_head_v=128, n_embd_r=512, n_embd_s=128,
                has_cell_ext=False, rope_state="applied", rope_theta=1000000.0,
                model_weights_sha256="a" * 64, gguf_content_digest="b" * 64)
    base.update(overrides)
    return SourceGeometry(**base)


def state(**overrides) -> HybridState:
    base = dict(geometry=geometry(),
                layer_map=RecurrentLayerMap((0, 2, 3)),
                checkpoints=(CheckpointRecord(n_tokens=2044, pos_min=0, pos_max=2043,
                                              recurrent_segments=("ckpt0.r",)),),
                attention_segments=("k", "v"),
                recurrent_segments=("r", "s"))
    base.update(overrides)
    return HybridState(**base)


class HybridSchemaTest(unittest.TestCase):
    def test_a_complete_state_validates(self):
        self.assertEqual(state().validate(), [])
        state().require_valid()

    def test_round_trip_through_json_is_faithful(self):
        original = state()
        restored = HybridState.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored, original)
        self.assertEqual(restored.validate(), [])

    # -- the fields the source file cannot supply ------------------------------------

    def test_missing_geometry_is_refused(self):
        problems = state(geometry=None).validate()
        self.assertTrue(any("original GGUF" in p for p in problems))

    def test_an_ambiguous_head_split_is_refused(self):
        self.assertTrue(any("ambiguous" in p for p in
                            state(geometry=geometry(n_embd_head_k=0)).validate()))
        self.assertTrue(any("ambiguous" in p for p in
                            state(geometry=geometry(n_head_kv=(4, 4))).validate()))

    def test_an_unstated_rope_state_is_refused(self):
        for bad in ("", "maybe", "true", None):
            with self.subTest(rope_state=bad):
                problems = state(geometry=geometry(rope_state=bad)).validate()
                self.assertTrue(any("rope_state" in p for p in problems))

    def test_an_unknown_architecture_is_refused(self):
        self.assertTrue(any("architecture" in p for p in
                            state(geometry=geometry(architecture="")).validate()))

    def test_geometry_must_be_tied_to_a_model_digest(self):
        for field_name in ("model_weights_sha256", "gguf_content_digest"):
            with self.subTest(field=field_name):
                problems = state(geometry=geometry(**{field_name: "short"})).validate()
                self.assertTrue(any(field_name in p for p in problems))

    # -- the recurrent layer map -------------------------------------------------------

    def test_recurrent_state_without_a_layer_map_is_refused(self):
        problems = state(layer_map=None).validate()
        self.assertTrue(any("no layer map" in p for p in problems))

    def test_a_layer_map_out_of_range_is_refused(self):
        problems = state(layer_map=RecurrentLayerMap((0, 99))).validate()
        self.assertTrue(any("outside" in p for p in problems))

    def test_a_layer_map_with_duplicates_is_refused(self):
        self.assertTrue(any("repeats" in p for p in
                            state(layer_map=RecurrentLayerMap((1, 1))).validate()))

    def test_a_layer_map_out_of_order_is_refused(self):
        self.assertTrue(any("ascending" in p for p in
                            state(layer_map=RecurrentLayerMap((3, 1))).validate()))

    def test_a_model_without_recurrent_state_needs_no_layer_map(self):
        plain = state(geometry=geometry(n_embd_r=0), layer_map=None,
                      recurrent_segments=())
        self.assertEqual(plain.validate(), [])

    # -- checkpoints -------------------------------------------------------------------

    def test_a_hybrid_state_without_checkpoints_is_refused(self):
        self.assertTrue(any("without at least one" in p
                            for p in state(checkpoints=()).validate()))

    def test_a_checkpoint_carrying_draft_state_is_refused(self):
        bad = CheckpointRecord(n_tokens=10, pos_min=0, pos_max=9,
                               recurrent_segments=("r",), has_draft_state=True)
        self.assertTrue(any("draft or speculative" in p
                            for p in state(checkpoints=(bad,)).validate()))

    def test_a_checkpoint_carrying_speculative_state_is_refused(self):
        bad = CheckpointRecord(n_tokens=10, pos_min=0, pos_max=9,
                               recurrent_segments=("r",), has_speculative_state=True)
        self.assertTrue(any("draft or speculative" in p
                            for p in state(checkpoints=(bad,)).validate()))

    def test_a_checkpoint_without_recurrent_segments_is_refused(self):
        bad = CheckpointRecord(n_tokens=10, pos_min=0, pos_max=9, recurrent_segments=())
        self.assertTrue(any("names no recurrent segments" in p
                            for p in state(checkpoints=(bad,)).validate()))

    def test_impossible_checkpoint_positions_are_refused(self):
        bad = CheckpointRecord(n_tokens=10, pos_min=5, pos_max=1,
                               recurrent_segments=("r",))
        self.assertTrue(state(checkpoints=(bad,)).validate())

    # -- layout ------------------------------------------------------------------------

    def test_a_non_canonical_layout_is_refused(self):
        problems = state(layout="layer,kv,token,dim,head").validate()
        self.assertTrue(any("normalised at decode" in p for p in problems))
        self.assertEqual(CANONICAL_LAYOUT, "layer,kv,token,head,dim")

    def test_a_wrong_schema_version_is_refused(self):
        self.assertTrue(any("is not" in p for p in state(schema="kvx-hybrid/0").validate()))
        self.assertEqual(SCHEMA, "kvx-hybrid/1")

    # -- malformed input ---------------------------------------------------------------

    def test_require_valid_raises_rather_than_returning(self):
        with self.assertRaises(SchemaError):
            state(geometry=None).require_valid()

    def test_from_dict_refuses_non_mappings(self):
        for bad in ([], "x", 3, None):
            with self.subTest(value=bad):
                with self.assertRaises(SchemaError):
                    HybridState.from_dict(bad)

    def test_from_dict_refuses_unreadable_geometry(self):
        payload = state().to_dict()
        del payload["geometry"]["n_layer"]
        with self.assertRaises(SchemaError) as caught:
            HybridState.from_dict(payload)
        self.assertIn("geometry is not readable", str(caught.exception))

    def test_from_dict_refuses_unreadable_checkpoints(self):
        payload = state().to_dict()
        payload["checkpoints"][0]["n_tokens"] = "many"
        with self.assertRaises(SchemaError):
            HybridState.from_dict(payload)

    def test_from_dict_refuses_an_unreadable_layer_map(self):
        payload = state().to_dict()
        payload["layer_map"]["source_layers"] = ["x"]
        with self.assertRaises(SchemaError):
            HybridState.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
