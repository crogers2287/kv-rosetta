from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from kv_rosetta.store import (
    IdentityError,
    Record,
    Store,
    StoreError,
    default_root,
    fingerprint,
    model_key,
)


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = Store(root=self.root)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    # ---- helpers ----

    def _write_source(self, name: str, data: bytes) -> Path:
        src = self.root / "_sources" / name
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(data)
        return src

    def _fp(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        system_sha256: str = "s",
        tools_sha256: str = "t",
    ) -> str:
        payload = (
            provider + "\x00" + model + "\x00" + system_sha256 + "\x00" + tools_sha256
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _put(
        self,
        fp: str,
        provider: str = "openai",
        model: str = "gpt-4",
        tier: int = 1,
        l0_sha256: str = "l0",
        data: bytes = b"data",
        now: float | None = None,
    ) -> Record:
        src = self._write_source(f"{fp[:8]}.kvx", data)
        return self.store.put(
            fingerprint=fp,
            provider=provider,
            model=model,
            tier=tier,
            l0_sha256=l0_sha256,
            source=src,
            now=now,
        )
        return src

    # ---- fingerprint / model_key ----

    def test_fingerprint_stable_and_length(self) -> None:
        fp1 = fingerprint("openai", "gpt-4", "aaa", "bbb")
        fp2 = fingerprint("openai", "gpt-4", "aaa", "bbb")
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in fp1))

    def test_fingerprint_changes_on_any_field(self) -> None:
        base = fingerprint("openai", "gpt-4", "aaa", "bbb")
        self.assertNotEqual(base, fingerprint("anthropic", "gpt-4", "aaa", "bbb"))
        self.assertNotEqual(base, fingerprint("openai", "gpt-3.5", "aaa", "bbb"))
        self.assertNotEqual(base, fingerprint("openai", "gpt-4", "xxx", "bbb"))
        self.assertNotEqual(base, fingerprint("openai", "gpt-4", "aaa", "yyy"))

    def test_model_key_stable_and_length(self) -> None:
        mk1 = model_key("openai", "gpt-4")
        mk2 = model_key("openai", "gpt-4")
        self.assertEqual(mk1, mk2)
        self.assertEqual(len(mk1), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in mk1))

    def test_model_key_changes_on_input(self) -> None:
        self.assertNotEqual(model_key("openai", "gpt-4"), model_key("anthropic", "gpt-4"))
        self.assertNotEqual(model_key("openai", "gpt-4"), model_key("openai", "gpt-3.5"))

    def test_default_root_uses_env(self) -> None:
        env_path = self.root / "envstore"
        old = os.environ.get("KVROSETTA_STORE")
        os.environ["KVROSETTA_STORE"] = str(env_path)
        try:
            self.assertEqual(default_root(), env_path)
        finally:
            if old is None:
                os.environ.pop("KVROSETTA_STORE", None)
            else:
                os.environ["KVROSETTA_STORE"] = old

    # ---- store construction ----

    def test_root_mode_0700(self) -> None:
        mode = self.root.stat().st_mode & 0o777
        self.assertEqual(mode, 0o700)

    def test_index_file_created(self) -> None:
        self.assertTrue((self.root / "index.sqlite").exists())

    # ---- put / get ----

    def test_put_get_roundtrip(self) -> None:
        data = b"hello kv-cache"
        src = self._write_source("a.kvx", data)
        fp = self._fp()
        rec = self.store.put(
            fingerprint=fp,
            provider="openai",
            model="gpt-4",
            tier=1,
            l0_sha256="deadbeef",
            source=src,
        )
        self.assertIsInstance(rec, Record)
        self.assertTrue(Path(rec.path).exists())
        self.assertEqual(rec.nbytes, len(data))
        self.assertEqual(rec.nbytes, Path(rec.path).stat().st_size)

        got = self.store.get(fp)
        self.assertIsNotNone(got)
        self.assertEqual(got.fingerprint, fp)
        self.assertEqual(got.model_key, model_key("openai", "gpt-4"))
        self.assertEqual(got.provider, "openai")
        self.assertEqual(got.model, "gpt-4")
        self.assertEqual(got.tier, 1)
        self.assertEqual(got.l0_sha256, "deadbeef")

    def test_artifact_lands_at_expected_path(self) -> None:
        fp = self._fp()
        mk = model_key("openai", "gpt-4")
        self._put(fp)
        expected = self.root / mk / (fp + ".kvx")
        self.assertTrue(expected.exists())
        self.assertEqual(self.store.path_for("openai", "gpt-4", fp), expected)

    def test_put_twice_keeps_original_created(self) -> None:
        now = 1000.0
        fp = self._fp()
        r1 = self._put(fp, now=now)
        self.assertEqual(r1.created, now)
        self.assertEqual(r1.last_used, now)

        self.store.touch(fp, now=now + 50)

        r2 = self._put(fp, tier=2, now=now + 100)
        self.assertEqual(r2.created, now)
        self.assertEqual(r2.last_used, now + 100)
        self.assertEqual(r2.tier, 2)

    def test_get_unknown_returns_none(self) -> None:
        self.assertIsNone(self.store.get(self._fp(provider="nope", model="nope")))

    def test_get_missing_file_deletes_row(self) -> None:
        fp = self._fp()
        rec = self._put(fp)
        Path(rec.path).unlink()
        self.assertIsNone(self.store.get(fp))
        self.assertEqual(len(self.store.list()), 0)

    def test_delete(self) -> None:
        fp = self._fp()
        rec = self._put(fp)
        self.assertTrue(Path(rec.path).exists())
        self.assertTrue(self.store.delete(fp))
        self.assertFalse(Path(rec.path).exists())
        self.assertFalse(self.store.delete(fp))
        self.assertEqual(len(self.store.list()), 0)

    def test_put_missing_source_raises(self) -> None:
        with self.assertRaises(StoreError):
            self.store.put(
                fingerprint=self._fp(),
                provider="openai",
                model="gpt-4",
                tier=1,
                l0_sha256="l0",
                source=self.root / "does_not_exist.kvx",
            )

    # ---- list ----

    def test_list_limit_desc(self) -> None:
        fps = []
        for i in range(3):
            fp = self._fp(system_sha256=str(i))
            fps.append(fp)
            self._put(fp, now=1000.0 + i)
        limited = self.store.list(limit=2)
        self.assertEqual(len(limited), 2)
        self.assertEqual(limited[0].fingerprint, fps[2])
        self.assertEqual(limited[1].fingerprint, fps[1])
        self.assertGreater(limited[0].last_used, limited[1].last_used)

    def test_list_by_model_key(self) -> None:
        fp1 = self._fp(model="gpt-4")
        fp2 = self._fp(model="gpt-3.5")
        self._put(fp1, model="gpt-4", now=1000.0)
        self._put(fp2, model="gpt-3.5", now=1001.0)
        rows = self.store.list(model_key=model_key("openai", "gpt-4"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].fingerprint, fp1)

    def test_list_empty(self) -> None:
        self.assertEqual(self.store.list(), ())
        self.assertEqual(self.store.list(model_key="anything"), ())

    # ---- prune ----

    def test_prune_max_entries(self) -> None:
        fps = []
        for i in range(3):
            fp = self._fp(system_sha256=str(i))
            fps.append(fp)
            self._put(fp, now=1000.0 + i)
        deleted = self.store.prune(max_entries=1)
        self.assertEqual(len(deleted), 2)
        self.assertEqual(deleted[0].fingerprint, fps[0])
        self.assertEqual(deleted[1].fingerprint, fps[1])
        survivors = self.store.list()
        self.assertEqual(len(survivors), 1)
        self.assertEqual(survivors[0].fingerprint, fps[2])

    def test_prune_max_age(self) -> None:
        old_fp = self._fp(system_sha256="old")
        new_fp = self._fp(system_sha256="new")
        self._put(old_fp, now=100.0)
        self._put(new_fp, now=200.0)
        deleted = self.store.prune(max_age_seconds=10, now=205.0)
        self.assertEqual(len(deleted), 1)
        self.assertEqual(deleted[0].fingerprint, old_fp)
        survivors = self.store.list()
        self.assertEqual(len(survivors), 1)
        self.assertEqual(survivors[0].fingerprint, new_fp)

    def test_prune_max_bytes(self) -> None:
        fps = []
        for i in range(3):
            fp = self._fp(system_sha256=str(i))
            fps.append(fp)
            self._put(fp, data=b"0123456789", now=1000.0 + i)
        deleted = self.store.prune(max_bytes=15)
        self.assertEqual(len(deleted), 2)
        self.assertEqual(deleted[0].fingerprint, fps[0])
        self.assertEqual(deleted[1].fingerprint, fps[1])
        self.assertEqual(self.store.total_bytes(), 10)

    def test_prune_order_age_then_entries_then_bytes(self) -> None:
        old_fp = self._fp(system_sha256="old")
        mid_fp = self._fp(system_sha256="mid")
        new_fp = self._fp(system_sha256="new")
        self._put(old_fp, now=100.0)
        self._put(mid_fp, now=200.0)
        self._put(new_fp, now=300.0)
        deleted = self.store.prune(max_age_seconds=50, max_entries=1, now=250.0)
        fps_deleted = {r.fingerprint for r in deleted}
        self.assertIn(old_fp, fps_deleted)
        self.assertIn(mid_fp, fps_deleted)
        self.assertNotIn(new_fp, fps_deleted)
        survivors = self.store.list()
        self.assertEqual(len(survivors), 1)
        self.assertEqual(survivors[0].fingerprint, new_fp)

    def test_prune_no_limits_returns_empty(self) -> None:
        self._put(self._fp(), now=1000.0)
        self.assertEqual(self.store.prune(), ())

    # ---- persistence across stores ----

    def test_second_store_sees_rows(self) -> None:
        fp = self._fp()
        self._put(fp, now=1000.0)
        store2 = Store(root=self.root)
        try:
            got = store2.get(fp)
            self.assertIsNotNone(got)
            self.assertEqual(got.fingerprint, fp)
            self.assertEqual(len(store2.list()), 1)
        finally:
            store2.close()

    def test_double_open_same_root_safe(self) -> None:
        store2 = Store(root=self.root)
        try:
            fp = self._fp()
            self._put(fp, now=1000.0)
            self.assertEqual(len(store2.list()), 1)
        finally:
            store2.close()

    # ---- context manager ----

    def test_context_manager(self) -> None:
        fp = self._fp()
        with Store(root=self.root) as s:
            self._put(fp, now=1000.0)
            self.assertEqual(len(s.list()), 1)
        self.assertIsNone(s._conn)


if __name__ == "__main__":
    unittest.main()
