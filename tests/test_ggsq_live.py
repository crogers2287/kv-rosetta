"""Decode a real llama.cpp slot file. Skipped unless one is provided.

Every other decoder test uses fixtures I built from the writer's field order, which means
they share my reading of it. This one reads bytes llama.cpp actually produced, so it can
falsify that reading. It is opt-in because it needs a model to generate the artifact:

    KVX_LIVE_STATE=/path/to/slot.bin KVX_LIVE_MODEL=/path/to/model.gguf \\
        python3 -m unittest tests.test_ggsq_live
"""

import os
import unittest
from pathlib import Path

from kv_rosetta import gguf
from kv_rosetta.adapters import ggsq_envelope
from kv_rosetta.adapters.llamacpp_ggsq import (
    materialise,
    read_attention_section,
    row_size,
)

STATE = os.environ.get("KVX_LIVE_STATE", "")
MODEL = os.environ.get("KVX_LIVE_MODEL", "")


@unittest.skipUnless(STATE and Path(STATE).is_file() and MODEL and Path(MODEL).is_file(),
                     "set KVX_LIVE_STATE and KVX_LIVE_MODEL to a real slot file and model")
class LiveDecodeTest(unittest.TestCase):
    """Geometry comes from the GGUF, never from the decoder's own output."""

    @classmethod
    def setUpClass(cls):
        cls.path = Path(STATE)
        cls.size = cls.path.stat().st_size
        meta = gguf.read_metadata(MODEL, ("block_count", "head_count", "embedding_length"))
        arch = gguf.architecture(MODEL)
        cls.n_layer = int(meta[f"{arch}.block_count"])
        heads = int(meta[f"{arch}.attention.head_count"])
        cls.n_head_kv = int(meta[f"{arch}.attention.head_count_kv"])
        cls.head_dim = int(meta[f"{arch}.embedding_length"]) // heads
        cls.hybrid = not gguf.supports_prefix_reuse(arch)[0]

    def section(self, handle):
        head = handle.read(12)
        head += handle.read(ggsq_envelope.header_size(head) - len(head))
        envelope = ggsq_envelope.parse_file_envelope(head)
        return envelope, read_attention_section(handle, envelope.body_offset, self.size,
                                                has_cell_ext=False)

    def test_the_attention_section_parses_and_matches_the_model(self):
        with open(self.path, "rb") as handle:
            _envelope, section = self.section(handle)
        self.assertEqual(section.n_layer, self.n_layer,
                         "layer count disagrees with the model's block_count")
        self.assertEqual(len(section.spans), self.n_layer * 2)

    def test_the_row_stride_matches_geometry_computed_from_the_gguf(self):
        with open(self.path, "rb") as handle:
            _envelope, section = self.section(handle)
        keys = [s for s in section.spans if s.kind == "k"]
        expected = row_size(keys[0].type_id, self.n_head_kv * self.head_dim)
        self.assertEqual(keys[0].row_size, expected,
                         "the stride in the file disagrees with n_head_kv x head_dim")

    def test_a_non_hybrid_body_is_consumed_exactly(self):
        """The sharpest structural check available: nothing left over."""
        if self.hybrid:
            self.skipTest("a hybrid body continues into recurrent state after attention")
        with open(self.path, "rb") as handle:
            _envelope, section = self.section(handle)
        self.assertEqual(section.end_offset, self.size,
                         f"{self.size - section.end_offset} bytes unexplained after the "
                         f"attention section")

    def test_tensors_materialise_to_finite_values(self):
        import numpy as np

        with open(self.path, "rb") as handle:
            _envelope, section = self.section(handle)
            tensor = materialise(handle, section.spans[0], n_head=self.n_head_kv,
                                 head_dim=self.head_dim, cell_count=section.cell_count)
        self.assertEqual(tensor.shape, (section.cell_count, self.n_head_kv, self.head_dim))
        self.assertTrue(bool(np.isfinite(tensor).all()), "non-finite values in the cache")
        self.assertGreater(float(np.abs(tensor).max()), 0.0, "the tensor is all zeros")
