"""Refusals in the rejected-restore safety runner.

A 400 is not safety. The runner's job is to prove nothing reached the slot on the way to
that 400, and its value is in refusing to render that verdict when it cannot be causal.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from slot_poisoning import PoisonError, compare, load_reproducible, stage_foreign


def _record(label="cuda-nvidia-hybrid", reproducible=True):
    return {"label": label, "verdict": {"reproducible": reproducible,
                                        "distinct_texts": 1 if reproducible else 3}}


def _write(record):
    path = Path(tempfile.mkdtemp()) / "det.json"
    path.write_text(json.dumps(record))
    return str(path)


def _run(text="alpha", tokens=(1, 2), vectors=None):
    import hashlib
    return {"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_ids": list(tokens),
            "vectors": vectors if vectors is not None else [{1: -0.1}, {2: -0.2}]}


class DeterminismProofRequired(unittest.TestCase):
    def test_a_record_for_another_configuration_is_refused(self):
        # A proof about a different reader says nothing about this one, and the labels are
        # the only thing tying the proof to the configuration under test.
        with self.assertRaises(PoisonError) as caught:
            load_reproducible(_write(_record(label="vulkan-amd-dense")),
                              "cuda-nvidia-hybrid")
        self.assertIn("not 'cuda-nvidia-hybrid'", str(caught.exception))

    def test_a_non_reproducible_reader_is_refused(self):
        # On a reader that answers identical work differently, a post-rejection difference
        # cannot be attributed to the restore rather than to the reader.
        with self.assertRaises(PoisonError) as caught:
            load_reproducible(_write(_record(reproducible=False)), "cuda-nvidia-hybrid")
        self.assertIn("not reproducible", str(caught.exception))

    def test_a_matching_reproducible_record_is_accepted(self):
        proof = load_reproducible(_write(_record()), "cuda-nvidia-hybrid")
        self.assertTrue(proof["verdict"]["reproducible"])


class ForeignStaging(unittest.TestCase):
    def test_a_missing_foreign_state_is_refused(self):
        # Otherwise the server answers with its own file-not-found refusal, which is
        # indistinguishable from the model-mismatch refusal under test.
        slots = tempfile.mkdtemp()
        with self.assertRaises(PoisonError) as caught:
            stage_foreign(str(Path(slots) / "absent.state"), slots)
        self.assertIn("does not exist", str(caught.exception))

    def test_the_artifact_is_copied_into_the_slot_directory(self):
        src = Path(tempfile.mkdtemp()) / "foreign.state"
        src.write_bytes(b"qsgg" + b"\x00" * 32)
        slots = tempfile.mkdtemp()
        stage_foreign(str(src), slots)
        self.assertEqual((Path(slots) / "foreign.state").read_bytes(), src.read_bytes())


class PristineComparison(unittest.TestCase):
    def test_identical_runs_are_pristine(self):
        self.assertTrue(compare(_run(), _run())["pristine"])

    def test_changed_text_is_not_pristine(self):
        self.assertFalse(compare(_run(), _run(text="beta"))["pristine"])

    def test_changed_tokens_are_not_pristine(self):
        self.assertFalse(compare(_run(), _run(tokens=(1, 3)))["pristine"])

    def test_same_text_but_changed_vectors_is_not_pristine(self):
        # The poisoning that matters most is the one that leaves the text intact: a
        # perturbed distribution behind an unchanged answer is exactly what a text-only
        # check would wave through.
        moved = _run(vectors=[{1: -0.9}, {2: -0.2}])
        result = compare(_run(), moved)
        self.assertTrue(result["text_matches"])
        self.assertTrue(result["tokens_match"])
        self.assertFalse(result["vectors_match"])
        self.assertFalse(result["pristine"])


if __name__ == "__main__":
    unittest.main()
