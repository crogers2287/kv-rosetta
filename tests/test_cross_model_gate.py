"""The same-geometry cross-model gate's preconditions.

This runner exists because research-findings §20 answered the cross-GEOMETRY question and
was then quoted as if it had answered the cross-MODEL question. The two differ, and the
tests here pin the parts that decide whether a foreign number means anything: the geometry
precondition, the payload-scrambling noise control, and the verdict's refusal to admit
anything when a control misbehaved.
"""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_model_gate import (GateError, geometry_of, require_same_geometry,
                              scramble_payload, verdict)


def _geo(**over):
    base = {"arch": "qwen3", "n_layer": 36, "n_head_kv": 8, "key_length": 128,
            "value_length": 128, "head_dim_derived": False, "rope_freq_base": 1e6}
    base.update(over)
    return base


def _leg(cache_n=100, text_match=True, tokens_match=True, top1=1.0, vs_own=None,
         forced_vs_native=None):
    return {"cache_n": cache_n, "text_matches_native": text_match,
            "tokens_match_native": tokens_match,
            "divergence_vs_native": {"top1_agreement": top1},
            "divergence_vs_identity": {"top1_agreement": top1 if vs_own is None else vs_own,
                                       "max_abs_logprob_delta": 0.1},
            "forced_vs_identity": {"top1_agreement": top1 if vs_own is None else vs_own,
                                   "max_abs_logprob_delta": 0.1},
            "forced_vs_native": {"top1_agreement": forced_vs_native},
            "text_matches_identity": text_match}


class GeometryPrecondition(unittest.TestCase):
    def test_identical_geometry_is_accepted(self):
        require_same_geometry(_geo(), _geo())

    def test_a_differing_field_is_refused_and_named(self):
        with self.assertRaises(GateError) as caught:
            require_same_geometry(_geo(), _geo(n_head_kv=2))
        self.assertIn("n_head_kv", str(caught.exception))

    def test_the_refusal_points_at_the_answered_question(self):
        # Running a mismatched pair here would re-ask §20 under a name that implies a
        # different question.
        with self.assertRaises(GateError) as caught:
            require_same_geometry(_geo(), _geo(n_layer=10))
        self.assertIn("§20", str(caught.exception))

    def test_head_dim_is_never_compared_as_null(self):
        # qwen2 omits attention.key_length. Left as None, two models with different head
        # dimensions would pass the check by comparing None to None.
        import struct as _s
        def kv_string(k, v):
            return (_s.pack("<Q", len(k)) + k.encode() + _s.pack("<I", 8)
                    + _s.pack("<Q", len(v)) + v.encode())
        def kv_u32(k, v):
            return _s.pack("<Q", len(k)) + k.encode() + _s.pack("<I", 4) + _s.pack("<I", v)
        blob = bytearray(b"GGUF" + _s.pack("<IQQ", 3, 0, 3))
        blob += kv_string("general.architecture", "qwen2")
        blob += kv_u32("qwen2.embedding_length", 2048)
        blob += kv_u32("qwen2.attention.head_count", 16)
        path = Path(tempfile.mkdtemp()) / "m.gguf"
        path.write_bytes(bytes(blob))
        geo = geometry_of(str(path))
        self.assertEqual(geo["key_length"], 128)
        self.assertTrue(geo["head_dim_derived"])

    def _gguf(self, pairs, strings=(("general.architecture", "qwen2"),)):
        import struct as _s
        blob = bytearray(b"GGUF" + _s.pack("<IQQ", 3, 0, len(pairs) + len(strings)))
        for k, v in strings:
            blob += (_s.pack("<Q", len(k)) + k.encode() + _s.pack("<I", 8)
                     + _s.pack("<Q", len(v)) + v.encode())
        for k, v in pairs:
            blob += _s.pack("<Q", len(k)) + k.encode() + _s.pack("<I", 4) + _s.pack("<I", v)
        path = Path(tempfile.mkdtemp()) / "m.gguf"
        path.write_bytes(bytes(blob))
        return str(path)

    def test_an_indivisible_embedding_is_refused_rather_than_floored(self):
        # 5120/24 floors to 213 against a declared key_length of 256. Flooring silently
        # produces a plausible head_dim and two models would then "match" on a wrong one.
        path = self._gguf([("qwen2.embedding_length", 5120),
                           ("qwen2.attention.head_count", 24)])
        with self.assertRaises(GateError) as caught:
            geometry_of(path)
        self.assertIn("not divisible", str(caught.exception))

    def test_a_model_with_no_derivable_head_dim_is_refused(self):
        path = self._gguf([("qwen2.block_count", 36)])
        with self.assertRaises(GateError) as caught:
            geometry_of(path)
        self.assertIn("no head dimension", str(caught.exception))

    def test_rope_base_is_part_of_the_geometry(self):
        # Same shapes but different theta means the cached keys were rotated differently,
        # which the shape fields alone would not catch.
        with self.assertRaises(GateError):
            require_same_geometry(_geo(), _geo(rope_freq_base=5e5))


class GeometryRead(unittest.TestCase):
    def test_a_gguf_without_an_architecture_is_refused(self):
        # Every geometry field is looked up under the architecture prefix, so without it
        # the whole geometry reads back as None and two unrelated models would compare
        # equal on a table full of nulls.
        def s(text):
            raw = text.encode()
            return struct.pack("<Q", len(raw)) + raw
        blob = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
        blob += s("general.name") + struct.pack("<I", 8) + s("nameless")
        path = Path(tempfile.mkdtemp()) / "model.gguf"
        path.write_bytes(bytes(blob))
        with self.assertRaises(GateError) as caught:
            geometry_of(str(path))
        self.assertIn("no architecture", str(caught.exception))


class NoiseControl(unittest.TestCase):
    """The noise control must restore successfully and still carry garbage.

    A first version overwrote everything after the token header, which destroyed the
    per-layer type ids and row sizes too. The server answered 400, and a rejected noise
    control bounds nothing - from outside it is identical to never having tried. So these
    tests build a real GGSQ body and check that structure survives while values do not.
    """

    def _state(self, n_tokens=3, n_layer=2, cell_count=4):
        from tests.test_ggsq_decoder import build_attention
        body = build_attention(cell_count=cell_count, n_layer=n_layer)
        path = Path(tempfile.mkdtemp()) / "s.state"
        path.write_bytes(b"qsgg" + struct.pack("<I", 3) + struct.pack("<I", n_tokens)
                         + b"\x00" * (4 * n_tokens) + body)
        return path

    def test_every_structural_field_survives_scrambling(self):
        # The parser must still find the same spans in the scrambled copy: if it cannot,
        # neither can the server, and the control would be refused rather than restored.
        from kv_rosetta.adapters import ggsq_envelope, llamacpp_ggsq
        src = self._state()
        dest = src.with_name("noise.state")
        scramble_payload(src, dest)
        spans = {}
        for path in (src, dest):
            raw = path.read_bytes()
            env = ggsq_envelope.parse_file_envelope(raw)
            with open(path, "rb") as handle:
                sec = llamacpp_ggsq.read_attention_section(handle, env.body_offset,
                                                           len(raw), has_cell_ext=False)
            spans[path.name] = [(s.layer_index, s.kind, s.type_id, s.row_size,
                                 s.offset, s.nbytes) for s in sec.spans]
        self.assertEqual(spans[src.name], spans[dest.name])

    def test_the_tensor_data_is_actually_replaced(self):
        src = self._state()
        dest = src.with_name("noise.state")
        info = scramble_payload(src, dest)
        self.assertEqual(info["spans_scrambled"], 4)      # 2 layers x (k, v)
        self.assertGreater(info["payload_bytes_scrambled"], 0)
        self.assertNotEqual(src.read_bytes(), dest.read_bytes())

    def test_a_non_ggsq_file_is_refused_by_magic(self):
        # The message matters: the parser wrapper below would also refuse this file, so
        # asserting only the class would pass with the magic check deleted.
        src = Path(tempfile.mkdtemp()) / "x.state"
        src.write_bytes(b"KVX1" + b"\x00" * 32)
        with self.assertRaises(GateError) as caught:
            scramble_payload(src, src.with_name("n.state"))
        self.assertIn("not a GGSQ state file", str(caught.exception))

    def test_a_body_whose_values_cannot_be_located_is_refused(self):
        # If the spans cannot be found the copy would be byte-identical to the foreign
        # file, silently turning the noise leg into a second foreign leg that agrees with
        # it perfectly - the most convincing possible wrong answer.
        from tests.test_ggsq_decoder import build_attention
        body = build_attention(cell_count=4, n_layer=2)[:40]      # truncated mid-section
        src = Path(tempfile.mkdtemp()) / "s.state"
        src.write_bytes(b"qsgg" + struct.pack("<I", 3) + struct.pack("<I", 3)
                        + b"\x00" * 12 + body)
        with self.assertRaises(GateError) as caught:
            scramble_payload(src, src.with_name("n.state"))
        self.assertIn("cannot locate the cache values", str(caught.exception))


class VerdictControls(unittest.TestCase):
    def _legs(self, **over):
        legs = {"identity": _leg(),
                "noise": _leg(text_match=False, vs_own=0.0),
                "foreign": _leg()}
        legs.update(over)
        return legs

    def test_healthy_controls_allow_a_threshold_result(self):
        v = verdict(self._legs(), min_top1=0.99)
        self.assertTrue(v["controls_ok"])
        self.assertTrue(v["meets_threshold"])

    def test_the_score_is_against_the_targets_own_restore(self):
        # Restoring and prefilling are different computations, and they already disagree
        # for a model reading its OWN cache. Scoring the foreign leg against a cold
        # prefill would charge it for a difference that is not about whose cache it is.
        legs = self._legs(foreign=_leg(top1=0.3, vs_own=1.0))
        v = verdict(legs, min_top1=0.99)
        self.assertEqual(v["foreign_top1_vs_own_restore"], 1.0)
        self.assertTrue(v["meets_threshold"])

    def test_noise_matching_native_invalidates_the_run(self):
        # The §20 failure: if noise reproduces the native output, nothing was restored in
        # any leg and all three were cold prefills.
        v = verdict(self._legs(noise=_leg(text_match=True, vs_own=0.0)), min_top1=0.99)
        self.assertFalse(v["controls_ok"])
        self.assertIn("cold prefill", " ".join(v["problems"]))

    def test_noise_that_agrees_with_the_own_restore_invalidates_the_run(self):
        # If scrambled values still track the real cache, the floor is not a floor and a
        # high foreign score means nothing.
        v = verdict(self._legs(noise=_leg(text_match=False, vs_own=0.9)), min_top1=0.99)
        self.assertFalse(v["controls_ok"])
        self.assertIn("not a floor", " ".join(v["problems"]))

    def test_controls_that_reused_nothing_invalidate_the_run(self):
        v = verdict(self._legs(identity=_leg(cache_n=0)), min_top1=0.99)
        self.assertFalse(v["controls_ok"])
        v = verdict(self._legs(noise=_leg(cache_n=0, text_match=False, vs_own=0.0)),
                    min_top1=0.99)
        self.assertFalse(v["controls_ok"])

    def test_a_good_foreign_score_cannot_pass_on_broken_controls(self):
        # The number being attractive is exactly when this matters.
        v = verdict(self._legs(identity=_leg(cache_n=0), foreign=_leg(vs_own=1.0)),
                    min_top1=0.99)
        self.assertFalse(v["meets_threshold"])

    def test_the_baseline_is_reported_next_to_the_absolute_threshold(self):
        # Restore-versus-prefill on this host agrees on 0.969-0.977 of positions for a
        # model reading its OWN cache, so a 0.99 absolute threshold is unreachable for
        # reasons that have nothing to do with whose cache it is.
        legs = self._legs(identity=_leg(forced_vs_native=0.97),
                          foreign=_leg(vs_own=0.98))
        v = verdict(legs, min_top1=0.99)
        self.assertEqual(v["baseline_top1"], 0.97)
        self.assertTrue(v["threshold_exceeds_baseline"])
        self.assertTrue(v["at_or_above_baseline"])
        self.assertFalse(v["meets_threshold"])

    def test_a_foreign_result_below_the_baseline_is_marked_as_such(self):
        legs = self._legs(identity=_leg(forced_vs_native=0.977),
                          foreign=_leg(vs_own=0.930))
        v = verdict(legs, min_top1=0.99)
        self.assertFalse(v["at_or_above_baseline"])

    def test_a_low_foreign_score_does_not_meet_the_threshold(self):
        v = verdict(self._legs(foreign=_leg(vs_own=0.4)), min_top1=0.99)
        self.assertTrue(v["controls_ok"])
        self.assertFalse(v["meets_threshold"])


if __name__ == "__main__":
    unittest.main()


class MatchedCaches(unittest.TestCase):
    def test_equal_cell_counts_are_accepted(self):
        from cross_model_gate import require_matched_caches
        require_matched_caches({"n_saved": 3088}, {"n_saved": 3088})

    def test_a_longer_identity_cache_is_refused(self):
        # The control must differ from the subject in authorship only.
        from cross_model_gate import GateError as GE, require_matched_caches
        with self.assertRaises(GE) as caught:
            require_matched_caches({"n_saved": 3088}, {"n_saved": 3119})
        self.assertIn("3119", str(caught.exception))


class TeacherForcedComparison(unittest.TestCase):
    """Free generation is a cliff; these pin the protocol that avoids it.

    Measured, not assumed: over 128 freely generated tokens even a model restoring its OWN
    cache agreed with its own cold prefill on 0.23 of positions. That is a fact about
    autoregressive cascade, not about the cache, and it made the free-generation numbers
    uninterpretable in both directions.
    """

    def test_identical_scores_agree_completely(self):
        from cross_model_gate import compare_forced
        scores = [{1: -0.1, 2: -2.0}, {3: -0.3, 4: -1.0}]
        result = compare_forced(scores, scores)
        self.assertEqual(result["top1_agreement"], 1.0)
        self.assertEqual(result["max_abs_logprob_delta"], 0.0)
        self.assertEqual(result["positions"], 2)

    def test_one_disagreement_does_not_cascade(self):
        # The whole point: position 0 differing must cost exactly one position, because
        # every position is scored against the same fixed prefix.
        from cross_model_gate import compare_forced
        left = [{1: -0.1, 2: -2.0}, {3: -0.3, 4: -1.0}, {5: -0.2, 6: -1.0}]
        right = [{1: -2.0, 2: -0.1}, {3: -0.3, 4: -1.0}, {5: -0.2, 6: -1.0}]
        self.assertAlmostEqual(compare_forced(left, right)["top1_agreement"], 2 / 3)

    def test_the_mean_delta_is_reported_beside_the_max(self):
        # One outlier token dominates the max. A sweep needs a quantity that moves
        # smoothly, or a real trend reads as noise.
        from cross_model_gate import compare_forced
        left = [{1: 0.0, 2: 0.0}, {1: 0.0, 2: 0.0}]
        right = [{1: 0.0, 2: 4.0}, {1: 0.0, 2: 0.0}]
        result = compare_forced(left, right)
        self.assertEqual(result["max_abs_logprob_delta"], 4.0)
        self.assertEqual(result["mean_abs_logprob_delta"], 1.0)   # 4.0 over 4 shared

    def test_the_mean_delta_is_none_when_nothing_is_shared(self):
        from cross_model_gate import compare_forced
        result = compare_forced([{1: -0.1}], [{2: -0.1}])
        self.assertIsNone(result["mean_abs_logprob_delta"])

    def test_comparison_stops_at_the_shorter_run(self):
        from cross_model_gate import compare_forced
        result = compare_forced([{1: -0.1}], [{1: -0.1}, {2: -0.1}])
        self.assertEqual(result["positions"], 1)

    def test_a_position_with_no_vector_is_refused(self):
        # An empty vector compares equal to any other empty vector, so a leg that silently
        # returned none would score as perfect agreement.
        from cross_model_gate import GateError as GE, teacher_forced

        class _Reader:
            def post(self, path, body):
                return {"completion_probabilities": []}

        with self.assertRaises(GE) as caught:
            teacher_forced(_Reader(), [1, 2], [3], 0)
        self.assertIn("no probability vector", str(caught.exception))

    def test_no_positions_reports_none_rather_than_a_perfect_score(self):
        # An empty comparison must never read as agreement.
        from cross_model_gate import compare_forced
        result = compare_forced([], [])
        self.assertEqual(result["positions"], 0)
        self.assertIsNone(result["top1_agreement"])


class CellExtensionSizing(unittest.TestCase):
    """has_cell_ext without its size silently parses as if there were no extension.

    The parser defaults cell_ext_size to 0, so passing only the flag reproduces the
    no-extension parse and desynchronises one cell in. On a real qwen35 state that
    surfaced as "cell 1 claims 2523 sequence ids" -- a corruption message for a missing
    argument, which is the hardest kind of bug to read.
    """

    def _state(self, cell_ext=b""):
        from tests.test_ggsq_decoder import build_attention
        body = build_attention(cell_count=4, n_layer=2, cell_ext=cell_ext)
        path = Path(tempfile.mkdtemp()) / "s.state"
        path.write_bytes(b"qsgg" + struct.pack("<I", 3) + struct.pack("<I", 3)
                         + b"\x00" * 12 + body)
        return path

    def test_a_state_with_cell_extensions_scrambles_cleanly(self):
        from kv_rosetta.sizing import BYTES_PER_CELL_EXT
        src = self._state(cell_ext=b"\x01" * BYTES_PER_CELL_EXT)
        dest = src.with_name("noise.state")
        info = scramble_payload(src, dest, has_cell_ext=True)
        self.assertEqual(info["spans_scrambled"], 4)
        self.assertNotEqual(src.read_bytes(), dest.read_bytes())

    def test_omitting_the_extension_on_such_a_state_is_refused(self):
        # Not silently mis-parsed: the guard turns a desync into a refusal.
        from kv_rosetta.sizing import BYTES_PER_CELL_EXT
        src = self._state(cell_ext=b"\x01" * BYTES_PER_CELL_EXT)
        with self.assertRaises(GateError) as caught:
            scramble_payload(src, src.with_name("n.state"), has_cell_ext=False)
        self.assertIn("cannot locate the cache values", str(caught.exception))

    def test_a_state_without_extensions_is_unaffected(self):
        src = self._state()
        info = scramble_payload(src, src.with_name("noise.state"), has_cell_ext=False)
        self.assertEqual(info["spans_scrambled"], 4)
