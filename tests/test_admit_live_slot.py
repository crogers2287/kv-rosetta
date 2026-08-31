"""Guards for admitting a live slot. Traffic that bypasses cfrproxy has no manifest, so
this is the only path to an attachment for it -- and admitting a partial or trivial slot
would publish an artifact that never matches anything."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from admit_live_slot import (  # noqa: E402
    LiveCaptureError, prefix_fingerprint, require_tokens, wait_for_idle,
)


class FingerprintTests(unittest.TestCase):
    def test_is_stable_for_the_same_tokens(self):
        self.assertEqual(prefix_fingerprint([7, 8, 9]), prefix_fingerprint([7, 8, 9]))

    def test_differs_when_a_token_changes(self):
        self.assertNotEqual(prefix_fingerprint([7, 8, 9]), prefix_fingerprint([7, 8, 10]))

    def test_order_matters(self):
        self.assertNotEqual(prefix_fingerprint([1, 2]), prefix_fingerprint([2, 1]))

    def test_is_64_hex_characters_like_every_other_prefix_id(self):
        fp = prefix_fingerprint([1, 2, 3])
        self.assertEqual(len(fp), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in fp))

    def test_refuses_an_empty_sequence(self):
        with self.assertRaises(LiveCaptureError) as cm:
            prefix_fingerprint([])
        self.assertIn("empty token sequence", str(cm.exception))


class SlotGuardTests(unittest.TestCase):
    def test_accepts_a_slot_above_the_minimum(self):
        require_tokens(9000, 8000)

    def test_refuses_a_slot_below_the_minimum(self):
        with self.assertRaises(LiveCaptureError) as cm:
            require_tokens(120, 8000)
        self.assertIn("below the 8000", str(cm.exception))

    def test_returns_the_slot_once_it_goes_idle(self):
        seq = [[{"id": 0, "is_processing": True}], [{"id": 0, "is_processing": False,
                                                     "n_prompt_tokens": 42}]]
        got = wait_for_idle(lambda p: seq.pop(0), 0, timeout=30, poll=0)
        self.assertEqual(got["n_prompt_tokens"], 42)

    def test_refuses_when_the_slot_does_not_exist(self):
        with self.assertRaises(LiveCaptureError) as cm:
            wait_for_idle(lambda p: [{"id": 1, "is_processing": False}], 0, timeout=30, poll=0)
        self.assertIn("no slot 0", str(cm.exception))

    def test_refuses_a_non_positive_timeout(self):
        with self.assertRaises(LiveCaptureError) as cm:
            wait_for_idle(lambda p: [], 0, timeout=0)
        self.assertIn("timeout 0 is not positive", str(cm.exception))

    def test_gives_up_rather_than_admitting_a_busy_slot(self):
        with self.assertRaises(LiveCaptureError) as cm:
            wait_for_idle(lambda p: [{"id": 0, "is_processing": True}], 0,
                          timeout=0.01, poll=0)
        self.assertIn("still busy", str(cm.exception))
