import json
import tempfile
import unittest
from pathlib import Path

import kv_rosetta


# The kvx/0.1 manifest shape used by tests/test_kv_rosetta.py. Kept literal here on
# purpose: this file exists to prove that manifest evolution does not disturb it.
def manifest(schema="kvx/0.1", **overrides):
    value = {
        "schema": schema,
        "model": {"architecture": "qwen", "weights_id": "qwen-a"},
        "prompt": {
            "tokenizer_id": "tok-a",
            "token_ids_sha256": "1" * 64,
            "token_count": 1024,
        },
        "kv": {
            "layers": 40,
            "heads": 8,
            "head_dim": 128,
            "dtype": "f16",
            "layout": "layer,kv,token,head,dim",
            "byte_order": "little",
        },
        "position": {"kind": "rope", "theta": 1000000},
        "producer": {
            "runtime": "llama.cpp",
            "backend": "cuda",
            "device_arch": "sm_86",
        },
    }
    for section, patch in overrides.items():
        value[section].update(patch)
    return value


# Recorded from the pre-0.2 module (kv_rosetta.py at commit 7675906) against the
# manifest above. A change here means an artifact already written to disk would no
# longer be recognised as the same cache. It must never be edited to make a test pass.
GOLDEN = "6e9e1fa7e16699f781403a2555844cde142a1bed4746dc0087c25f300b7b39d2"


class ManifestV02Test(unittest.TestCase):
    def abi(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(value))
            return kv_rosetta.load(path)[1]

    def test_v01_manifest_still_loads(self):
        abi = self.abi(manifest())
        self.assertEqual(abi.layers, 40)
        self.assertTrue(abi.rope_applied)
        self.assertEqual(abi.kv_tokens, 0)

    def test_v02_manifest_loads(self):
        abi = self.abi(manifest(
            schema="kvx/0.2",
            position={"rope_applied": False},
            kv={"tokens": 1024},
        ))
        self.assertFalse(abi.rope_applied)
        self.assertEqual(abi.kv_tokens, 1024)

    def test_unknown_schema_rejected(self):
        with self.assertRaises(kv_rosetta.ManifestError):
            self.abi(manifest(schema="kvx/9.9"))

    def test_golden_fingerprint_unchanged(self):
        self.assertEqual(self.abi(manifest()).fingerprint(), GOLDEN)

    def test_rope_applied_not_in_fingerprint(self):
        applied = self.abi(manifest(schema="kvx/0.2", position={"rope_applied": True}))
        stripped = self.abi(manifest(schema="kvx/0.2", position={"rope_applied": False}))
        self.assertEqual(applied.fingerprint(), stripped.fingerprint())

    def test_kv_tokens_not_in_fingerprint(self):
        few = self.abi(manifest(schema="kvx/0.2", kv={"tokens": 8}))
        many = self.abi(manifest(schema="kvx/0.2", kv={"tokens": 200000}))
        self.assertEqual(few.fingerprint(), many.fingerprint())

    def test_bad_rope_applied_type(self):
        with self.assertRaises(kv_rosetta.ManifestError):
            self.abi(manifest(schema="kvx/0.2", position={"rope_applied": "yes"}))

    def test_negative_kv_tokens_rejected(self):
        with self.assertRaises(kv_rosetta.ManifestError):
            self.abi(manifest(schema="kvx/0.2", kv={"tokens": -1}))

    def test_schema_constants(self):
        self.assertEqual(kv_rosetta.SCHEMA, "kvx/0.1")
        self.assertEqual(kv_rosetta.SCHEMA_LATEST, "kvx/0.2")
        self.assertIn("kvx/0.1", kv_rosetta.ACCEPTED_SCHEMAS)
        self.assertIn("kvx/0.2", kv_rosetta.ACCEPTED_SCHEMAS)


if __name__ == "__main__":
    unittest.main()
