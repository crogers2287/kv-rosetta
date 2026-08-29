"""The cross-backend runner's refusal to compare builds it cannot show are the same.

This guard exists because of a real false negative: a state file written by a current build
was restored into a llama.cpp compiled months earlier, which rejected it for the state-file
version. The run looked exactly like "a cache does not survive a change of backend", and it
was recorded that way until the server log was read.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cross_backend_gate import (  # noqa: E402
    logprob_divergence,
    parse_revision,
    require_one_revision,
    source_revision,
)


class RevisionParseTest(unittest.TestCase):
    def test_it_reads_the_commit_out_of_a_real_banner(self):
        self.assertEqual(
            parse_revision("version: 0.3.0-dev (build 151, commit ca3d5a3e1)\n"
                           "built with GNU 13.3.0 for Linux x86_64"),
            "ca3d5a3")

    def test_abbreviations_of_different_length_are_one_revision(self):
        """git abbreviates to whatever is unambiguous in that tree, so the widths differ.

        Comparing the raw strings reported two builds of the same commit as a mismatch and
        refused a run that should have gone ahead.
        """
        self.assertEqual(parse_revision("commit ca3d5a3"),
                         parse_revision("commit ca3d5a3e1"))

    def test_different_commits_stay_different(self):
        self.assertNotEqual(parse_revision("commit ca3d5a3e1"),
                            parse_revision("commit ca3d5b7f2"))

    def test_a_banner_with_no_commit_yields_nothing(self):
        for text in ("", "version: 0.3.0-dev", "commit unknown", "commit ca3d5"):
            with self.subTest(text=text):
                self.assertEqual(parse_revision(text), "")

    def test_a_binary_that_does_not_exist_yields_nothing(self):
        self.assertEqual(source_revision(Path("/nonexistent/llama-server")), "")

    def test_a_binary_that_cannot_run_yields_nothing(self):
        """A HIP build on a host without ROCm exits non-zero; that is not a revision."""
        self.assertEqual(source_revision(Path("/dev/null")), "")


class RevisionGuardTest(unittest.TestCase):
    def test_matching_revisions_pass_and_are_returned(self):
        self.assertEqual(require_one_revision({"hip": "ca3d5a3", "cpu": "ca3d5a3"}),
                         "ca3d5a3")

    def test_differing_revisions_stop_the_run(self):
        with self.assertRaises(SystemExit) as caught:
            require_one_revision({"hip": "ca3d5a3", "cpu": "9f1e0d2"})
        self.assertIn("not from one source revision", str(caught.exception))

    def test_an_unreadable_revision_stops_the_run(self):
        """Unknown must not pass as matching - that is the whole point of the guard."""
        for revisions in ({"hip": "", "cpu": ""}, {"hip": "ca3d5a3", "cpu": ""}):
            with self.subTest(revisions=revisions):
                with self.assertRaises(SystemExit) as caught:
                    require_one_revision(revisions)
                self.assertIn("not a matching one", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


def completion(*positions):
    """A llama.cpp completion response, in the shape probs() actually reads.

    The alternatives live under `top_logprobs`, each carrying an `id` and a `logprob`. An
    earlier harness read `top_probs`/`prob`, which the default response contract does not
    populate, and compared two lists of empty dicts - a parity check that passed while
    comparing nothing.
    """
    return {"completion_probabilities": [
        {"top_logprobs": [{"id": t, "logprob": v} for t, v in pos.items()]}
        for pos in positions]}


class LogprobDivergenceTest(unittest.TestCase):
    """A bare "the logprobs differ" cannot separate a kernel difference from a broken cache.

    Both directions of the first real HIP/Vulkan run reported logprobs_match false while
    producing identical tokens and identical text. Without a magnitude that reads as either
    "expected floating-point noise" or "the restore is wrong", and they need opposite
    responses.
    """

    def test_identical_completions_diverge_by_nothing(self):
        one = completion({1: -0.5, 2: -1.5})
        found = logprob_divergence(one, completion({1: -0.5, 2: -1.5}))
        self.assertEqual(found["max_abs_logprob_delta"], 0.0)
        self.assertEqual(found["top1_agreement"], 1.0)
        self.assertTrue(found["identical"])

    def test_it_reports_the_largest_difference_not_the_last(self):
        found = logprob_divergence(completion({1: -0.5}, {1: -1.0}),
                                   completion({1: -0.5}, {1: -1.25}))
        self.assertAlmostEqual(found["max_abs_logprob_delta"], 0.25)
        self.assertEqual(found["positions"], 2)

    def test_small_noise_and_a_wrong_cache_do_not_look_alike(self):
        reference = completion({1: -0.1, 2: -2.0})
        noise = logprob_divergence(reference, completion({1: -0.1000001, 2: -2.0}))
        wrong = logprob_divergence(reference, completion({1: -3.0, 2: -0.2}))
        self.assertLess(noise["max_abs_logprob_delta"], 1e-5)
        self.assertEqual(noise["top1_agreement"], 1.0)
        self.assertGreater(wrong["max_abs_logprob_delta"], 1.0)
        self.assertEqual(wrong["top1_agreement"], 0.0)

    def test_a_token_present_in_only_one_vector_is_not_a_small_difference(self):
        found = logprob_divergence(completion({1: -0.1}), completion({99: -0.1}))
        self.assertEqual(found["max_abs_logprob_delta"], float("inf"))

    def test_comparing_nothing_is_not_agreement(self):
        found = logprob_divergence(completion(), completion())
        self.assertEqual(found["positions"], 0)
        self.assertFalse(found["identical"])
        self.assertIsNone(found["top1_agreement"])

    def test_the_compared_count_is_reported_so_a_short_match_cannot_pass_as_full(self):
        found = logprob_divergence(completion({1: -0.1}),
                                   completion({1: -0.1}, {2: -0.2}, {3: -0.3}))
        self.assertEqual(found["positions"], 1)
        self.assertEqual(found["max_abs_logprob_delta"], 0.0)
        self.assertFalse(found["identical"])
