"""The partial-depth splice: byte placement and the layout precondition.

A splice copies bytes by offset. If two files disagree on any span's position or type the
copy lands in the wrong place and produces a file that is structurally valid, restorable,
and silently wrong. These pin the check that prevents that, and the splice's endpoints.
"""

import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cross_model_gate import GateError
from kvx_splice import foreign_layers, require_spliceable, splice


@dataclass
class _Span:
    layer_index: int
    kind: str
    type_id: int = 1
    row_size: int = 8
    offset: int = 0
    nbytes: int = 4
    transposed: bool = False


def _spans(n_layer=3):
    out, offset = [], 0
    for layer in range(n_layer):
        for kind in ("k", "v"):
            out.append(_Span(layer_index=layer, kind=kind, offset=offset, nbytes=4))
            offset += 4
    return out


class LayoutPrecondition(unittest.TestCase):
    def test_identical_layouts_are_accepted(self):
        require_spliceable(_spans(), _spans(), 24, 24)

    def test_a_size_mismatch_is_refused(self):
        with self.assertRaises(GateError) as caught:
            require_spliceable(_spans(), _spans(), 24, 28)
        self.assertIn("differ in size", str(caught.exception))

    def test_a_span_count_mismatch_is_refused(self):
        with self.assertRaises(GateError):
            require_spliceable(_spans(3), _spans(2), 24, 24)

    def test_a_differing_offset_is_refused_and_named(self):
        # The dangerous case: same count, same sizes, one span moved. The splice would
        # still produce a restorable file.
        other = _spans()
        other[2] = replace(other[2], offset=other[2].offset + 4)
        with self.assertRaises(GateError) as caught:
            require_spliceable(_spans(), other, 24, 24)
        self.assertIn("offset", str(caught.exception))

    def test_a_differing_quantisation_type_is_refused(self):
        other = _spans()
        other[0] = replace(other[0], type_id=8)
        with self.assertRaises(GateError) as caught:
            require_spliceable(_spans(), other, 24, 24)
        self.assertIn("type_id", str(caught.exception))

    def test_a_differing_transposition_is_refused(self):
        # v_trans changes how values are laid out entirely; equal offsets would not save it.
        other = _spans()
        other[1] = replace(other[1], transposed=True)
        with self.assertRaises(GateError) as caught:
            require_spliceable(_spans(), other, 24, 24)
        self.assertIn("transposed", str(caught.exception))


class Splice(unittest.TestCase):
    def setUp(self):
        self.spans = _spans(3)
        self.foreign = bytes(range(0x10, 0x10 + 24))
        self.own = bytes(range(0xA0, 0xA0 + 24))

    def test_no_layers_is_the_targets_own_file_unchanged(self):
        # One endpoint of the sweep must be a known quantity.
        blob, replaced = splice(self.foreign, self.own, self.spans, set())
        self.assertEqual(blob, self.own)
        self.assertEqual(replaced, 0)

    def test_every_layer_is_the_foreign_file(self):
        blob, replaced = splice(self.foreign, self.own, self.spans, {0, 1, 2})
        self.assertEqual(blob, self.foreign)
        self.assertEqual(replaced, 6)

    def test_a_partial_splice_takes_exactly_the_named_layers(self):
        blob, replaced = splice(self.foreign, self.own, self.spans, {0})
        self.assertEqual(replaced, 2)                       # layer 0's k and v
        self.assertEqual(blob[0:8], self.foreign[0:8])      # layer 0 from foreign
        self.assertEqual(blob[8:], self.own[8:])            # layers 1-2 untouched

    def test_the_late_end_takes_the_top_of_the_stack(self):
        blob, replaced = splice(self.foreign, self.own, self.spans, {2})
        self.assertEqual(replaced, 2)
        self.assertEqual(blob[:16], self.own[:16])
        self.assertEqual(blob[16:], self.foreign[16:])

    def test_the_splice_never_changes_length(self):
        for count in range(0, 4):
            blob, _ = splice(self.foreign, self.own, self.spans,
                             set(range(count)))
            self.assertEqual(len(blob), len(self.own))


class ForeignLayerSelection(unittest.TestCase):
    def test_early_takes_from_the_bottom(self):
        self.assertEqual(foreign_layers(3, 36, "early"), {0, 1, 2})

    def test_late_takes_from_the_top(self):
        self.assertEqual(foreign_layers(3, 36, "late"), {33, 34, 35})

    def test_both_ends_agree_at_the_endpoints(self):
        # Zero and n_layer must name the same sets either way, or the two sweeps are not
        # anchored to the same reference and cannot be compared.
        for count in (0, 36):
            self.assertEqual(foreign_layers(count, 36, "early"),
                             foreign_layers(count, 36, "late"))

    def test_a_negative_count_is_refused(self):
        with self.assertRaises(GateError):
            foreign_layers(-1, 36, "early")

    def test_a_count_past_the_stack_is_refused(self):
        with self.assertRaises(GateError):
            foreign_layers(37, 36, "early")

    def test_an_unknown_direction_is_refused(self):
        with self.assertRaises(GateError) as caught:
            foreign_layers(3, 36, "middle")
        self.assertIn("early", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
