"""The preflight's refusals, tested without a GPU.

The runner exists because a restored-versus-cold verdict is only causal when the reader
answers identical uncached work identically. Its value is entirely in what it refuses, so
those refusals are what these tests pin.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from reader_determinism import PreflightError, check_run, summarise


def _run(text="alpha", tokens=(1, 2, 3), slot=0, cache_n=0, vectors=None):
    return {
        "id_slot": slot,
        "cache_n": cache_n,
        "prompt_n": 430,
        "text": text,
        "text_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
        "token_ids": list(tokens),
        "vectors": vectors if vectors is not None else [{1: -0.1}] * len(tokens),
    }


class RunLevelRefusals(unittest.TestCase):
    def test_a_run_that_reused_cache_is_not_a_cold_sample(self):
        with self.assertRaises(PreflightError) as caught:
            check_run(_run(cache_n=17), 0)
        self.assertIn("not a cold sample", str(caught.exception))

    def test_empty_probability_vectors_are_refused(self):
        # Two runs of empty vectors compare equal, which is the exact vacuous comparison
        # this record exists to prevent.
        with self.assertRaises(PreflightError) as caught:
            check_run(_run(vectors=[]), 0)
        self.assertIn("empty probability vectors", str(caught.exception))

    def test_individually_empty_vectors_are_refused(self):
        with self.assertRaises(PreflightError):
            check_run(_run(vectors=[{1: -0.1}, {}, {1: -0.2}]), 0)

    def test_vector_count_must_match_token_count(self):
        with self.assertRaises(PreflightError) as caught:
            check_run(_run(tokens=(1, 2, 3), vectors=[{1: -0.1}]), 0)
        self.assertIn("vectors for", str(caught.exception))

    def test_no_tokens_is_refused(self):
        # Vectors present so the empty-vector guard cannot fire first, and the message is
        # asserted because the count check below would otherwise raise the same class.
        with self.assertRaises(PreflightError) as caught:
            check_run(_run(tokens=(), vectors=[{1: -0.1}]), 0)
        self.assertIn("no token ids", str(caught.exception))


class SetLevelVerdict(unittest.TestCase):
    def test_identical_runs_are_reproducible(self):
        verdict = summarise([_run() for _ in range(6)])
        self.assertTrue(verdict["reproducible"])
        self.assertEqual(verdict["distinct_texts"], 1)

    def test_differing_text_is_not_reproducible(self):
        runs = [_run() for _ in range(5)] + [_run(text="beta")]
        verdict = summarise(runs)
        self.assertFalse(verdict["reproducible"])
        self.assertEqual(verdict["distinct_texts"], 2)

    def test_identical_text_but_differing_vectors_is_not_reproducible(self):
        # The text can agree while the distribution behind it does not. Admitting on text
        # alone would allowlist a reader whose margins are moving underneath.
        runs = [_run() for _ in range(5)]
        drifted = _run()
        drifted["vectors"] = [{1: -0.9}] * len(drifted["token_ids"])
        verdict = summarise(runs + [drifted])
        self.assertFalse(verdict["reproducible"])
        self.assertEqual(verdict["distinct_texts"], 1)
        self.assertEqual(verdict["distinct_probability_vectors"], 2)

    def test_too_few_runs_is_refused(self):
        with self.assertRaises(PreflightError) as caught:
            summarise([_run() for _ in range(5)])
        self.assertIn("fewer than", str(caught.exception))

    def test_runs_routed_to_different_slots_are_refused(self):
        # A set spread across slots is not six repetitions of one configuration, and slot
        # routing has already produced a wrong measurement on this project once.
        runs = [_run() for _ in range(5)] + [_run(slot=2)]
        with self.assertRaises(PreflightError) as caught:
            summarise(runs)
        self.assertIn("more than one slot", str(caught.exception))

    def test_one_bad_run_refuses_the_whole_set(self):
        # Never dropped and never averaged over: a set containing a non-cold run cannot be
        # partially salvaged into a verdict.
        runs = [_run() for _ in range(5)] + [_run(cache_n=3)]
        with self.assertRaises(PreflightError):
            summarise(runs)


class ReaderLifecycle(unittest.TestCase):
    def test_argv_sets_a_slot_save_path(self):
        # Without it llama.cpp answers /slots actions with 501, the per-run erase fails,
        # and no run is cold. This cost a full three-configuration preflight once.
        import tempfile
        from reader_determinism import Reader
        slots = tempfile.mkdtemp()
        argv = Reader("/bin/true", "m.gguf", 1, Path(slots) / "l", [], 512, slots).argv()
        self.assertIn("--slot-save-path", argv)
        self.assertEqual(argv[argv.index("--slot-save-path") + 1], slots.rstrip("/") + "/")

    def test_a_reader_that_exits_before_serving_is_refused(self):
        # Without this check the health loop polls a dead process until its timeout and
        # reports "did not become healthy", hiding the real cause and the log path.
        import tempfile
        from reader_determinism import Reader
        out = Path(tempfile.mkdtemp()) / "dead.log"
        reader = Reader("/bin/false", "/nonexistent.gguf", 1, out, [], 512, str(out.parent))
        with self.assertRaises(PreflightError) as caught:
            reader.start(timeout=30)
        self.assertIn("exited with", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
