"""An admitted object must stop being admitted the moment anything about it changes.

The store lets the request path skip rehashing a 577 MiB payload. That is only safe if
resolution refuses everything the rehash would have caught. These are the refusals, written
before the store was used for any measurement: a name, a mode bit, a prior hash, or a
manifest is not admission.

Every guard here was mutation-checked - each one disabled in turn, confirming at least one
test fails. Two rounds were needed: the first pass showed the symlink and traversal tests
passing whether or not O_NOFOLLOW, the store confinement, and the address-format check were
present, because the pinned-fact and missing-manifest checks caught those cases first. Tests
that isolate each guard were added.

O_NOFOLLOW and the realpath confinement remain mutually redundant: either alone refuses a
substituted symlink, so neither is individually necessary and no single-mutation test can
separate them. Removing BOTH is caught. That is defence in depth, and it is recorded as such
rather than claimed as two independently proven guards.
"""

import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path

from kv_rosetta.admitted_store import (
    MANIFEST_SUFFIX,
    OBJECT_SUFFIX,
    AdmissionError,
    AdmittedStore,
)

PAYLOAD = b"raw ggsq sequence state and its checkpoint appendix" * 40


class AdmittedStoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.store = AdmittedStore(self.dir / "store")
        self.raw = self.dir / "raw.bin"
        self.raw.write_bytes(PAYLOAD)
        self.obj = self.store.admit(self.raw, {"prompt_token_count": 2048,
                                               "compound_tuple": "ggsq/3+sckp/1"})

    def object_path(self) -> Path:
        return self.store.root / f"{self.obj.digest}{OBJECT_SUFFIX}"

    def manifest_path(self) -> Path:
        return self.store.root / f"{self.obj.digest}{MANIFEST_SUFFIX}"

    # -- the happy path -------------------------------------------------------------

    def test_a_freshly_admitted_object_resolves(self):
        resolved = self.store.resolve(self.obj.digest)
        self.assertEqual(resolved.digest, self.obj.digest)
        self.assertEqual(resolved.manifest["prompt_token_count"], 2048)
        self.assertTrue(self.store.verify_bytes(self.obj.digest))

    def test_the_address_is_a_full_sha256(self):
        self.assertEqual(len(self.obj.digest), 64)
        self.assertIn(self.obj.digest, self.obj.path.name)

    def test_the_manifest_is_bound_to_the_content_address(self):
        self.assertEqual(self.obj.manifest["raw_sha256"], self.obj.digest)
        self.assertEqual(self.obj.manifest["raw_size"], len(PAYLOAD))

    # -- the store itself must be private -------------------------------------------

    def test_a_group_or_world_accessible_store_is_refused(self):
        for mode in (0o750, 0o707, 0o777):
            with self.subTest(mode=oct(mode)):
                os.chmod(self.store.root, mode)
                with self.assertRaises(AdmissionError) as caught:
                    self.store.resolve(self.obj.digest)
                self.assertIn("prove nothing", str(caught.exception))
                os.chmod(self.store.root, 0o700)

    def test_admission_creates_a_private_store(self):
        fresh = AdmittedStore(self.dir / "fresh")
        self.assertEqual(stat.S_IMODE(fresh.root.stat().st_mode), 0o700)

    # -- identity drift ---------------------------------------------------------------

    def test_replacement_with_different_bytes_is_refused(self):
        time.sleep(0.01)
        self.object_path().write_bytes(PAYLOAD + b"tampered")
        with self.assertRaises(AdmissionError) as caught:
            self.store.resolve(self.obj.digest)
        self.assertIn("no longer matches", str(caught.exception))

    def test_replacement_with_identical_size_is_still_refused(self):
        # Same length, so only inode/ctime/mtime can catch it - which is the point of
        # pinning more than size.
        time.sleep(0.01)
        data = bytearray(PAYLOAD)
        data[10] ^= 0xFF
        self.object_path().write_bytes(bytes(data))
        with self.assertRaises(AdmissionError):
            self.store.resolve(self.obj.digest)

    def test_a_new_inode_at_the_same_name_is_refused(self):
        other = self.store.root / "decoy.tmp"
        other.write_bytes(PAYLOAD)
        os.replace(other, self.object_path())
        with self.assertRaises(AdmissionError) as caught:
            self.store.resolve(self.obj.digest)
        self.assertIn("inode", str(caught.exception))

    def test_truncation_is_refused(self):
        with open(self.object_path(), "r+b") as handle:
            handle.truncate(len(PAYLOAD) // 2)
        with self.assertRaises(AdmissionError):
            self.store.resolve(self.obj.digest)

    def test_an_extra_hard_link_is_refused(self):
        # Another name for the same inode means another writer can reach the bytes.
        os.link(self.object_path(), self.store.root / "second-name")
        with self.assertRaises(AdmissionError) as caught:
            self.store.resolve(self.obj.digest)
        self.assertIn("nlink", str(caught.exception))

    # -- symlinks and traversal --------------------------------------------------------

    def test_a_symlink_in_place_of_the_object_is_refused(self):
        elsewhere = self.dir / "elsewhere.bin"
        elsewhere.write_bytes(PAYLOAD)
        self.object_path().unlink()
        os.symlink(elsewhere, self.object_path())
        with self.assertRaises(AdmissionError):
            self.store.resolve(self.obj.digest)

    def test_a_symlink_pointing_inside_the_store_is_still_refused(self):
        real = self.store.root / "real.bin"
        real.write_bytes(PAYLOAD)
        self.object_path().unlink()
        os.symlink(real, self.object_path())
        with self.assertRaises(AdmissionError):
            self.store.resolve(self.obj.digest)

    def test_a_symlink_is_refused_even_when_its_target_matches_the_pinned_facts(self):
        """Isolates the symlink and confinement guards from the pinned-fact check.

        The earlier symlink tests pass whether or not O_NOFOLLOW and the realpath
        confinement are present, because the link target has a different inode and the
        pinned facts catch it. A mutation run proved that: removing O_NOFOLLOW failed no
        test. Here the manifest is rewritten to pin the TARGET's facts, so the pinned-fact
        check would accept it and only the link-specific guards can refuse.
        """
        outside = self.dir / "outside.bin"
        outside.write_bytes(PAYLOAD)
        self.object_path().unlink()
        os.symlink(outside, self.object_path())
        st = outside.stat()
        manifest = dict(self.obj.manifest)
        manifest["file_facts"] = {
            "device": st.st_dev, "inode": st.st_ino, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns, "nlink": st.st_nlink}
        self.manifest_path().write_text(json.dumps(manifest))
        with self.assertRaises(AdmissionError) as caught:
            self.store.resolve(self.obj.digest)
        self.assertRegex(str(caught.exception), "outside the store|cannot open")

    def test_an_address_escaping_the_store_is_refused_before_opening(self):
        """Isolates the address-format check from the missing-manifest refusal.

        A traversing address otherwise fails only because no manifest is found under that
        name, which would also be true of a harmless typo. This gives the traversal a
        manifest, so the address check is the only thing left to refuse it.
        """
        target = self.dir / "outside-target.state"
        target.write_bytes(PAYLOAD)
        escaping = "../outside-target"
        (self.store.root / f"{escaping}{MANIFEST_SUFFIX}").write_text(
            json.dumps({"raw_sha256": escaping, "file_facts": {}}))
        with self.assertRaises(AdmissionError) as caught:
            self.store.resolve(escaping)
        self.assertIn("not a full sha256 content address", str(caught.exception))

    def test_a_traversing_address_is_refused(self):
        for bad in ("../etc/passwd", "a" * 63, "A" * 64, "zz" + "0" * 62, ""):
            with self.subTest(address=bad[:16]):
                with self.assertRaises(AdmissionError):
                    self.store.resolve(bad)

    # -- manifests ---------------------------------------------------------------------

    def test_a_missing_manifest_means_not_admitted(self):
        self.manifest_path().unlink()
        with self.assertRaises(AdmissionError) as caught:
            self.store.resolve(self.obj.digest)
        self.assertIn("not admitted", str(caught.exception))

    def test_a_substituted_manifest_is_refused(self):
        foreign = dict(self.obj.manifest, raw_sha256="b" * 64)
        self.manifest_path().write_text(json.dumps(foreign))
        with self.assertRaises(AdmissionError) as caught:
            self.store.resolve(self.obj.digest)
        self.assertIn("substituted manifest", str(caught.exception))

    def test_an_unparseable_manifest_is_refused(self):
        self.manifest_path().write_text("{not json")
        with self.assertRaises(AdmissionError):
            self.store.resolve(self.obj.digest)

    def test_a_manifest_without_pinned_facts_is_refused(self):
        stripped = dict(self.obj.manifest)
        stripped.pop("file_facts")
        self.manifest_path().write_text(json.dumps(stripped))
        with self.assertRaises(AdmissionError) as caught:
            self.store.resolve(self.obj.digest)
        self.assertIn("pins no file facts", str(caught.exception))

    # -- admission-time refusals --------------------------------------------------------

    def test_a_declared_digest_that_disagrees_with_the_bytes_is_refused(self):
        with self.assertRaises(AdmissionError) as caught:
            self.store.admit(self.raw, {"raw_sha256": "c" * 64})
        self.assertIn("does not match the bytes written", str(caught.exception))

    def test_a_declared_size_that_disagrees_is_refused(self):
        with self.assertRaises(AdmissionError):
            self.store.admit(self.raw, {"raw_size": 7})

    def test_a_failed_admission_leaves_nothing_resolvable(self):
        before = sorted(p.name for p in self.store.root.iterdir())
        with self.assertRaises(AdmissionError):
            self.store.admit(self.raw, {"raw_size": 7})
        self.assertEqual(sorted(p.name for p in self.store.root.iterdir()), before,
                         "a failed admission left an object behind")

    def test_an_object_without_admission_cannot_be_resolved(self):
        # An arbitrary file dropped into the store, named as if it were admitted.
        import hashlib
        rogue = PAYLOAD + b"never admitted"
        digest = hashlib.sha256(rogue).hexdigest()
        (self.store.root / f"{digest}{OBJECT_SUFFIX}").write_bytes(rogue)
        with self.assertRaises(AdmissionError) as caught:
            self.store.resolve(digest)
        self.assertIn("not admitted", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
