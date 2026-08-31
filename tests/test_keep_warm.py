"""Guards for the page-cache keeper. A bad range would silently warm the wrong bytes."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from keep_warm import KeepWarmError, check_range, run, touch_once  # noqa: E402


class KeepWarmGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.bin"
        self.path.write_bytes(b"x" * 4096)

    def test_valid_range_returns_file_size(self):
        self.assertEqual(check_range(self.path, 0, 4096), 4096)

    def test_touch_makes_the_range_resident_without_error(self):
        self.assertGreaterEqual(touch_once(self.path, 0, 4096), 0.0)

    def test_refuses_a_range_past_the_end(self):
        with self.assertRaises(KeepWarmError) as cm:
            check_range(self.path, 4000, 1000)
        self.assertIn("runs past the end", str(cm.exception))

    def test_refuses_a_missing_file(self):
        with self.assertRaises(KeepWarmError) as cm:
            check_range(self.path.with_name("absent.bin"), 0, 1)
        self.assertIn("is not a file", str(cm.exception))

    def test_refuses_a_negative_offset(self):
        with self.assertRaises(KeepWarmError) as cm:
            check_range(self.path, -1, 10)
        self.assertIn("is negative", str(cm.exception))

    def test_refuses_a_non_positive_length(self):
        with self.assertRaises(KeepWarmError) as cm:
            check_range(self.path, 0, 0)
        self.assertIn("is not positive", str(cm.exception))

    def test_refuses_a_non_positive_interval(self):
        with self.assertRaises(KeepWarmError) as cm:
            run(self.path, 0, 4096, 0, once=True, log=lambda m: None)
        self.assertIn("interval 0 is not positive", str(cm.exception))

    def test_once_returns_after_a_single_pass(self):
        seen = []
        run(self.path, 0, 4096, 60, once=True, log=seen.append)
        self.assertTrue(any("touched" in m for m in seen))
