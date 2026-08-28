import json
import os
import tempfile
import unittest
from pathlib import Path

from kv_rosetta.daemon.watcher import (
    PrefixManifest,
    WatcherError,
    changed_since,
    load_manifests,
)


def _manifest(provider="acme", model="m1", last_prompt_tokens=1000,
              system_bytes=100, tools="[]"):
    return {
        "schema": 1,
        "provider": provider,
        "model": model,
        "client": "client-a",
        "scope": "global",
        "fingerprint": "fp-" + provider + "-" + model,
        "system_sha256": "s" * 64,
        "tools_sha256": "t" * 64,
        "system": "You are a helpful assistant.",
        "tools": tools,
        "system_bytes": system_bytes,
        "tool_count": 0,
        "first_seen": "2024-01-01T00:00:00Z",
        "last_seen": "2024-01-02T00:00:00Z",
        "last_prompt_tokens": last_prompt_tokens,
        "last_cache_source": "prompt",
        "last_cache_reason": "warm",
    }


def _write(root, relpath, data, mtime):
    path = Path(root) / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    os.utime(path, (mtime, mtime))
    return path


class WatcherLoadTests(unittest.TestCase):
    def test_two_valid_found_newest_first(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p1 = _write(root, "m1/client-a/global/1.json",
                        _manifest(provider="acme", model="m1"), 1000.0)
            p2 = _write(root, "m2/client-a/global/2.json",
                        _manifest(provider="acme", model="m2"), 2000.0)
            result = load_manifests(root)
            self.assertEqual(len(result), 2)
            self.assertTrue(all(isinstance(m, PrefixManifest) for m in result))
            self.assertEqual(result[0].path, p2)
            self.assertEqual(result[0].data["model"], "m2")
            self.assertEqual(result[1].path, p1)
            self.assertEqual(result[1].data["model"], "m1")

    def test_identity_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root, "m1/client-a/global/_identity.json",
                   _manifest(), 1000.0)
            _write(root, "m1/client-a/global/real.json",
                   _manifest(), 1000.0)
            result = load_manifests(root)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].path.name, "real.json")

    def test_invalid_json_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = Path(root) / "m1/client-a/global/bad.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{ this is not valid json ")
            os.utime(path, (1000.0, 1000.0))
            result = load_manifests(root)
            self.assertEqual(result, ())

    def test_schema_2_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bad = _manifest()
            bad["schema"] = 2
            _write(root, "m1/client-a/global/bad.json", bad, 1000.0)
            _write(root, "m1/client-a/global/good.json", _manifest(), 1000.0)
            result = load_manifests(root)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].data["schema"], 1)

    def test_json_list_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = Path(root) / "m1/client-a/global/list.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([1, 2, 3]))
            os.utime(path, (1000.0, 1000.0))
            result = load_manifests(root)
            self.assertEqual(result, ())

    def test_provider_filter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root, "acme/client-a/global/1.json",
                   _manifest(provider="acme", model="m1"), 1000.0)
            _write(root, "other/client-a/global/2.json",
                   _manifest(provider="other", model="m2"), 1000.0)
            result = load_manifests(root, provider="acme")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].data["provider"], "acme")

    def test_model_filter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root, "acme/client-a/global/1.json",
                   _manifest(provider="acme", model="m1"), 1000.0)
            _write(root, "acme/client-a/global/2.json",
                   _manifest(provider="acme", model="m2"), 1000.0)
            result = load_manifests(root, model="m2")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].data["model"], "m2")

    def test_provider_and_model_filter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root, "acme/client-a/global/1.json",
                   _manifest(provider="acme", model="m1"), 1000.0)
            _write(root, "acme/client-a/global/2.json",
                   _manifest(provider="acme", model="m2"), 1000.0)
            _write(root, "other/client-a/global/3.json",
                   _manifest(provider="other", model="m1"), 1000.0)
            result = load_manifests(root, provider="acme", model="m1")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].data["provider"], "acme")
            self.assertEqual(result[0].data["model"], "m1")

    def test_missing_root(self):
        result = load_manifests(Path("/definitely/not/here/xyz"))
        self.assertEqual(result, ())

    def test_prefix_manifest_properties(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root, "m1/client-a/global/1.json",
                   _manifest(provider="acme", model="gpt-x"), 1000.0)
            (manifest,) = load_manifests(root)
            self.assertEqual(manifest.fingerprint, "fp-acme-gpt-x")
            self.assertEqual(manifest.provider, "acme")
            self.assertEqual(manifest.model, "gpt-x")
            self.assertEqual(manifest.mtime, 1000.0)

    def test_est_tokens_prefers_last_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root, "m1/client-a/global/1.json",
                   _manifest(last_prompt_tokens=4242), 1000.0)
            (manifest,) = load_manifests(root)
            self.assertEqual(manifest.est_tokens, 4242)

    def test_est_tokens_byte_estimate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # system_bytes=68, tools="[]" (len 2): (68 + 2) / 3.5 = 20
            _write(root, "m1/client-a/global/1.json",
                   _manifest(last_prompt_tokens=0, system_bytes=68, tools="[]"),
                   1000.0)
            (manifest,) = load_manifests(root)
            self.assertEqual(manifest.est_tokens, 20)

    def test_changed_since(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p1 = _write(root, "m1/client-a/global/1.json", _manifest(model="m1"), 1000.0)
            p2 = _write(root, "m1/client-a/global/2.json", _manifest(model="m2"), 2000.0)
            p3 = _write(root, "m1/client-a/global/3.json", _manifest(model="m3"), 3000.0)
            result = load_manifests(root)
            changed = changed_since(result, 2500.0)
            self.assertEqual(len(changed), 1)
            self.assertEqual(changed[0].path, p3)

    def test_changed_since_preserves_order(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root, "m1/client-a/global/1.json", _manifest(model="m1"), 1000.0)
            _write(root, "m1/client-a/global/2.json", _manifest(model="m2"), 2000.0)
            _write(root, "m1/client-a/global/3.json", _manifest(model="m3"), 3000.0)
            _write(root, "m1/client-a/global/4.json", _manifest(model="m4"), 4000.0)
            result = load_manifests(root)
            changed = changed_since(result, 2500.0)
            self.assertEqual([m.data["model"] for m in changed], ["m4", "m3"])


class WatcherErrorTests(unittest.TestCase):
    def test_watcher_error_is_value_error(self):
        self.assertTrue(issubclass(WatcherError, ValueError))


if __name__ == "__main__":
    unittest.main()
