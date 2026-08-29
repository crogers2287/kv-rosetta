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
