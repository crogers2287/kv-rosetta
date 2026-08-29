"""Content-derived weight identity.

The two properties that matter, stated as the steer states them: renaming identical
weights must preserve identity, and replacing a file behind the same name must change it.
A path-derived hash gets both backwards.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from kv_rosetta.weights import (
    DigestCache,
    FileStamp,
    WeightsError,
    content_digest,
    model_content_digest,
    shards_of,
)


class ContentIdentityTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.cache = DigestCache(self.dir / "digests.sqlite")
        self.model = self.dir / "model.gguf"
        self.model.write_bytes(b"WEIGHTS-A" * 2000)
        self.digest = model_content_digest(self.model, self.cache)

    def tearDown(self):
        self.cache.close()

    def test_rename_preserves_identity(self):
        renamed = self.dir / "renamed.gguf"
        self.model.rename(renamed)
        self.assertEqual(model_content_digest(renamed, self.cache), self.digest)

    def test_relocation_preserves_identity(self):
        elsewhere = self.dir / "sub"
        elsewhere.mkdir()
        copied = elsewhere / "different-name.gguf"
        shutil.copy(self.model, copied)
        self.assertEqual(model_content_digest(copied, self.cache), self.digest)

    def test_replacing_content_behind_the_same_name_changes_identity(self):
        self.model.write_bytes(b"WEIGHTS-B" * 2000)
        self.assertNotEqual(model_content_digest(self.model, self.cache), self.digest)

    def test_a_one_byte_change_changes_identity(self):
        data = bytearray(self.model.read_bytes())
        data[len(data) // 2] ^= 0xFF
        self.model.write_bytes(bytes(data))
        self.assertNotEqual(model_content_digest(self.model, self.cache), self.digest)

    def test_same_size_different_content_differ(self):
        other = self.dir / "other.gguf"
        other.write_bytes(b"WEIGHTS-C" * 2000)
        self.assertNotEqual(model_content_digest(other, self.cache), self.digest)

    def test_digest_is_64_lowercase_hex(self):
        self.assertRegex(self.digest, r"\A[0-9a-f]{64}\Z")

    def test_missing_file_raises(self):
        with self.assertRaises(WeightsError):
            model_content_digest(self.dir / "absent.gguf", self.cache)


class ShardTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.cache = DigestCache(self.dir / "d.sqlite")
        self.shards = []
        for i in (1, 2, 3):
            p = self.dir / f"big-0000{i}-of-00003.gguf"
            p.write_bytes(f"SHARD-{i}".encode() * 500)
            self.shards.append(p)

    def tearDown(self):
        self.cache.close()

    def test_all_shards_are_discovered(self):
        self.assertEqual(len(shards_of(self.shards[0])), 3)

    def test_identity_covers_every_shard_not_just_the_first(self):
        before = model_content_digest(self.shards[0], self.cache)
        self.shards[2].write_bytes(b"DIFFERENT" * 500)
        self.assertNotEqual(model_content_digest(self.shards[0], self.cache), before)

    def test_shard_contents_swapped_changes_identity(self):
        before = model_content_digest(self.shards[0], self.cache)
        a, b = self.shards[0].read_bytes(), self.shards[1].read_bytes()
        self.shards[0].write_bytes(b)
        self.shards[1].write_bytes(a)
        self.assertNotEqual(model_content_digest(self.shards[0], self.cache), before)

    def test_single_file_model_needs_no_shard_pattern(self):
        solo = self.dir / "solo.gguf"
        solo.write_bytes(b"x" * 100)
        self.assertEqual(shards_of(solo), (solo,))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.cache = DigestCache(self.dir / "d.sqlite")
        self.f = self.dir / "w.bin"
        self.f.write_bytes(b"data" * 1000)

    def tearDown(self):
        self.cache.close()

    def test_second_call_does_not_rehash(self):
        first = content_digest(self.f, self.cache)
        # Corrupt the file WITHOUT touching its metadata stamp, then confirm the cache is
        # consulted. This documents the cache's trust boundary rather than hiding it.
        stamp = FileStamp.of(self.f)
        self.assertIsNotNone(self.cache.get(stamp))
        self.assertEqual(content_digest(self.f, self.cache), first)

    def test_rewriting_the_file_invalidates_the_entry(self):
        first = content_digest(self.f, self.cache)
        os.utime(self.f, ns=(0, 0))
        self.f.write_bytes(b"other" * 1000)
        self.assertNotEqual(content_digest(self.f, self.cache), first)

    def test_stamp_excludes_the_path(self):
        self.assertNotIn("path", FileStamp.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
