"""llama-swap aliases must resolve to the canonical id before any loaded check or lookup.

Three of the four request-time misses in the first live hour were requests for
`qwen38-27b-kvx-3090`, an alias of the serving `qwen38-27b-3090-agg`. /running reports
only canonical ids and the store keys artifacts by them, so the raw name was refused as
"not loaded" while the model was answering the very request.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from kv_rosetta.daemon.server import Fallback, Sidecar, SidecarConfig

from tests.test_sidecar import FakeSwap

CONFIG = """
models:
  qwen38-27b-3090-agg:
    aliases: [qwen38-27b-kvx-3090, "27b", hermes-v7]
    cmd: whatever
  tiel-kvx-w6800:
    cmd: whatever
  no-aliases-key:
    cmd: whatever
"""


class AliasTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Path(self.tmp.name) / "config.yaml"
        self.cfg.write_text(CONFIG)

    def build(self, loaded=("qwen38-27b-3090-agg",), swap_config=None):
        swap = FakeSwap(loaded=loaded)
        sidecar = Sidecar(SidecarConfig(port=0, manifest_root="/nonexistent",
                                        swap_config=str(swap_config or self.cfg)))
        import kv_rosetta.daemon.server as module
        original = module.urllib.request.urlopen
        module.urllib.request.urlopen = swap.open
        self.addCleanup(setattr, module.urllib.request, "urlopen", original)
        return sidecar, swap


class ResolutionTests(AliasTestCase):
    def test_alias_resolves_to_canonical(self):
        sidecar, _ = self.build()
        self.assertEqual(sidecar.canonical("qwen38-27b-kvx-3090"), "qwen38-27b-3090-agg")
        self.assertEqual(sidecar.canonical("27b"), "qwen38-27b-3090-agg")

    def test_canonical_and_unknown_ids_pass_through(self):
        sidecar, _ = self.build()
        self.assertEqual(sidecar.canonical("qwen38-27b-3090-agg"), "qwen38-27b-3090-agg")
        self.assertEqual(sidecar.canonical("tiel-kvx-w6800"), "tiel-kvx-w6800")
        self.assertEqual(sidecar.canonical("never-heard-of-it"), "never-heard-of-it")

    def test_missing_config_fails_open(self):
        sidecar, _ = self.build(swap_config=Path(self.tmp.name) / "absent.yaml")
        self.assertEqual(sidecar.canonical("27b"), "27b")

    def test_unparseable_config_fails_open(self):
        self.cfg.write_text("models: [this is: not: valid")
        sidecar, _ = self.build()
        self.assertEqual(sidecar.canonical("27b"), "27b")

    def test_map_is_reread_when_the_file_changes(self):
        sidecar, _ = self.build()
        self.assertEqual(sidecar.canonical("27b"), "qwen38-27b-3090-agg")
        self.cfg.write_text(CONFIG.replace('"27b", ', ""))
        import os
        os.utime(self.cfg, (1_000_000_000, 1_000_000_000))      # force a new mtime
        self.assertEqual(sidecar.canonical("27b"), "27b")


class LoadedCheckTests(AliasTestCase):
    def test_an_alias_of_a_loaded_model_is_loaded(self):
        # The live failure: refused as "not loaded" while llama-swap was serving it.
        sidecar, _ = self.build(loaded=("qwen38-27b-3090-agg",))
        sidecar.require_loaded("qwen38-27b-kvx-3090")          # must not raise
        self.assertTrue(sidecar.upstream_base("hermes-v7").endswith("/upstream/hermes-v7"))

    def test_an_alias_of_an_unloaded_model_is_still_refused(self):
        sidecar, _ = self.build(loaded=("tiel-kvx-w6800",))
        with self.assertRaises(Fallback):
            sidecar.require_loaded("27b")


def _artifact(model, ids, fingerprint):
    return SimpleNamespace(manifest={"runtime_model": model, "prompt_token_ids": list(ids),
                                     "prefix_fingerprint": fingerprint},
                           digest=fingerprint, path=None)


class _Store:
    def __init__(self, objs): self._objs, self.root = objs, "/nonexistent"
    def list_objects(self): return list(self._objs)


class _Adapter:
    def __init__(self, ids): self.ids = ids
    def apply_template(self, messages, tools=None): return "rendered"
    def tokenize(self, text): return list(self.ids)
    def _get(self, path): return [{"id": 0, "is_processing": False, "n_prompt_tokens": 0}]


class LookupTests(AliasTestCase):
    def test_find_artifact_matches_the_store_under_the_canonical_id(self):
        sidecar, _ = self.build()
        fp = "f" * 64
        sidecar.store = lambda: _Store([_artifact("qwen38-27b-3090-agg", [1, 2, 3], fp)])
        self.assertIsNotNone(sidecar.find_artifact(fp, "qwen38-27b-kvx-3090"))
        self.assertIsNotNone(sidecar.find_artifact(fp, "27b"))

    def test_restore_for_prompt_via_alias_finds_canonical_artifacts(self):
        sidecar, _ = self.build(loaded=("qwen38-27b-3090-agg",))
        ids = list(range(1, 41))
        fp = "c" * 64
        sidecar.store = lambda: _Store([_artifact("qwen38-27b-3090-agg", ids[:30], fp)])
        with mock.patch.object(sidecar, "ensure", return_value={}) as ensure:
            out = sidecar.restore_for_prompt("qwen38-27b-kvx-3090",
                                             [{"role": "user", "content": "hi"}],
                                             adapter=_Adapter(ids))
        self.assertTrue(out["restored"], out)
        self.assertEqual(out["covers_tokens"], 30)
        self.assertEqual((out["requested"], out["model"]),
                         ("qwen38-27b-kvx-3090", "qwen38-27b-3090-agg"))
        self.assertEqual(ensure.call_args.args[1], "qwen38-27b-3090-agg")
