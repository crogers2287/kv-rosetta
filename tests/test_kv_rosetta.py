import json
import tempfile
import unittest
from pathlib import Path

import kv_rosetta


def manifest(*, backend="cuda", architecture="qwen", weights="qwen-a",
             tokenizer="tok-a", layers=40, heads=8, head_dim=128,
             position="rope"):
    return {
        "schema": "kvx/0.1",
        "model": {"architecture": architecture, "weights_id": weights},
        "prompt": {
            "tokenizer_id": tokenizer,
            "token_ids_sha256": "1" * 64,
            "token_count": 1024,
        },
        "kv": {
            "layers": layers,
            "heads": heads,
            "head_dim": head_dim,
            "dtype": "f16",
            "layout": "layer,kv,token,head,dim",
            "byte_order": "little",
        },
        "position": {"kind": position, "theta": 1000000},
        "producer": {
            "runtime": "llama.cpp",
            "backend": backend,
            "device_arch": "sm_86" if backend == "cuda" else "gfx1030",
        },
    }


class KVXTest(unittest.TestCase):
    def abi(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(value))
            return kv_rosetta.load(path)[1]

    def test_hardware_backend_does_not_change_identity(self):
        self.assertEqual(
            self.abi(manifest(backend="cuda")).fingerprint(),
            self.abi(manifest(backend="hip")).fingerprint(),
        )

    def test_matched_kv_cross_model_is_linear_candidate(self):
        source = self.abi(manifest())
        target = self.abi(manifest(weights="qwen-b", layers=64))
        self.assertEqual(kv_rosetta.compatibility(source, target)["mode"],
                         "linear-candidate")

    def test_cross_arch_and_tokenizer_uses_learned_path(self):
        source = self.abi(manifest())
        target = self.abi(manifest(
            architecture="tiel-moe", weights="tiel", tokenizer="tok-b",
            layers=48, heads=4, head_dim=256,
        ))
        plan = kv_rosetta.compatibility(source, target)
        self.assertEqual(plan["mode"], "learned-candidate")
        self.assertIn("byte-span token alignment", plan["requirements"])
        self.assertIn("head/dimension projector", plan["requirements"])

    def test_bad_layout_is_rejected(self):
        value = manifest()
        value["kv"]["layout"] = "whatever"
        with self.assertRaises(kv_rosetta.ManifestError):
            self.abi(value)


if __name__ == "__main__":
    unittest.main()
