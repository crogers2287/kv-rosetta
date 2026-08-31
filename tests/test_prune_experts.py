"""Every guard in scripts/prune_experts.py must be defended by a failing test.

A mis-sliced warmer is worse than no warmer: it would load, serve, and quietly
produce a cache the target model reads as its own. So each refusal is asserted
on its message, not merely on the exception class -- an unrelated later failure
raising ValueError must not be able to satisfy these tests.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path.home() / "llama.cpp" / "gguf-py"))

try:
    import numpy as np
    from gguf import GGUFWriter, GGUFReader
    import prune_experts as pe
except Exception as exc:  # pragma: no cover - environment without gguf-py
    raise unittest.SkipTest(f"gguf-py unavailable: {exc}")

ARCH = "qwen4exp"


def build(path, *, experts=8, used=4, expert_axis0=None, router_axis0=None,
          arch=ARCH, with_expert_count=True, with_geometry=True, with_arch=True,
          with_tensors=True, expert_tensors=True, dims=(4, 3)):
    d, f = dims
    w = GGUFWriter(str(path), arch if with_arch else "")
    if with_arch is False:
        # GGUFWriter always writes general.architecture; emulate a shard that
        # carries only tensors by writing it and deleting it from the reader side.
        pass
    if with_expert_count:
        w.add_uint32(f"{arch}.expert_count", experts)
    w.add_uint32(f"{arch}.expert_used_count", used)
    if with_geometry:
        w.add_uint32(f"{arch}.block_count", 2)
        w.add_uint32(f"{arch}.embedding_length", d)
        w.add_uint32(f"{arch}.attention.head_count_kv", 2)
        w.add_uint32(f"{arch}.ssm.inner_size", 16)
    if with_tensors:
        e0 = expert_axis0 if expert_axis0 is not None else experts
        r0 = router_axis0 if router_axis0 is not None else experts
        if expert_tensors:
            w.add_tensor("blk.0.ffn_down_exps.weight",
                         np.arange(e0 * d * f, dtype=np.float32).reshape(e0, d, f))
            w.add_tensor("blk.0.ffn_gate_inp.weight",
                         np.arange(r0 * d, dtype=np.float32).reshape(r0, d))
        w.add_tensor("blk.0.attn_k.weight", np.ones((d, d), dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file()
    w.write_tensors_to_file(); w.close()
    return path


class PruneGuardTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.src = build(self.dir / "src.gguf")
        self.dst = self.dir / "out.gguf"

    def test_round_trip_keeps_first_n_experts_and_leaves_others_alone(self):
        pe.prune([self.src], self.dst, 2)
        got = {t.name: t.data for t in GGUFReader(str(self.dst)).tensors}
        src = {t.name: t.data for t in GGUFReader(str(self.src)).tensors}
        self.assertTrue(np.array_equal(got["blk.0.ffn_down_exps.weight"],
                                       src["blk.0.ffn_down_exps.weight"][:2]))
        self.assertTrue(np.array_equal(got["blk.0.ffn_gate_inp.weight"],
                                       src["blk.0.ffn_gate_inp.weight"][:2]))
        self.assertTrue(np.array_equal(got["blk.0.attn_k.weight"], src["blk.0.attn_k.weight"]))

    def test_cache_geometry_is_preserved(self):
        pe.prune([self.src], self.dst, 2)
        before = pe.cache_geometry(GGUFReader(str(self.src)), ARCH)
        after = pe.cache_geometry(GGUFReader(str(self.dst)), ARCH)
        self.assertEqual(before, after)

    def test_used_count_clamps_to_expert_count(self):
        pe.prune([self.src], self.dst, 2)
        r = GGUFReader(str(self.dst))
        self.assertEqual(int(r.fields[f"{ARCH}.expert_used_count"].contents()), 2)

    def test_used_count_is_kept_when_it_already_fits(self):
        pe.prune([self.src], self.dst, 6)
        r = GGUFReader(str(self.dst))
        self.assertEqual(int(r.fields[f"{ARCH}.expert_used_count"].contents()), 4)

    def test_refuses_more_experts_than_the_file_has(self):
        with self.assertRaises(ValueError) as cm:
            pe.prune([self.src], self.dst, 99)
        self.assertIn("only has 8", str(cm.exception))

    def test_refuses_zero_experts(self):
        with self.assertRaises(ValueError) as cm:
            pe.prune([self.src], self.dst, 0)
        self.assertIn("must be >= 1", str(cm.exception))

    def test_refuses_file_without_expert_count(self):
        src = build(self.dir / "dense.gguf", with_expert_count=False)
        with self.assertRaises(ValueError) as cm:
            pe.prune([src], self.dst, 2)
        self.assertIn("not an MoE file", str(cm.exception))

    def test_refuses_when_expert_dimension_is_not_outermost(self):
        src = build(self.dir / "odd.gguf", expert_axis0=7)
        with self.assertRaises(ValueError) as cm:
            pe.prune([src], self.dst, 2)
        self.assertIn("expert dimension is not outermost", str(cm.exception))

    def test_refuses_when_no_expert_tensors_are_present(self):
        src = build(self.dir / "notensors.gguf", with_tensors=False)
        with self.assertRaises(ValueError) as cm:
            pe.prune([src], self.dst, 2)
        self.assertIn("no tensors found", str(cm.exception))

    def test_refuses_a_file_holding_no_expert_or_router_tensors(self):
        src = build(self.dir / "attnonly.gguf", expert_tensors=False)
        with self.assertRaises(ValueError) as cm:
            pe.prune([src], self.dst, 2)
        self.assertIn("no expert or router tensors", str(cm.exception))

    def test_refuses_if_the_written_file_changed_cache_geometry(self):
        """The post-write re-read is the last line of defence; it must actually bite."""
        real, calls = pe.cache_geometry, []
        def fake(reader, arch):
            calls.append(1)
            return {"block_count": 999} if len(calls) > 1 else real(reader, arch)
        pe.cache_geometry = fake
        self.addCleanup(setattr, pe, "cache_geometry", real)
        with self.assertRaises(ValueError) as cm:
            pe.prune([self.src], self.dst, 2)
        self.assertIn("cache geometry changed", str(cm.exception))

    def test_refuses_if_expert_count_was_not_written(self):
        real, calls = pe.expert_count, []
        def fake(reader, arch):
            calls.append(1)
            return 8 if len(calls) > 1 else real(reader, arch)
        pe.expert_count = fake
        self.addCleanup(setattr, pe, "expert_count", real)
        with self.assertRaises(ValueError) as cm:
            pe.prune([self.src], self.dst, 2)
        self.assertIn("expert_count was not written", str(cm.exception))

    def test_refuses_when_no_shard_carries_the_architecture(self):
        reader = GGUFReader(str(self.src))
        del reader.fields["general.architecture"]
        with self.assertRaises(ValueError) as cm:
            pe.metadata_shard([reader])
        self.assertIn("no shard carries general.architecture", str(cm.exception))

    def test_refuses_a_file_with_no_cache_geometry_fields(self):
        reader = GGUFReader(str(self.src))
        with self.assertRaises(ValueError) as cm:
            pe.cache_geometry(reader, "not_an_arch")
        self.assertIn("no cache-geometry fields", str(cm.exception))

    def test_architecture_refuses_when_missing(self):
        reader = GGUFReader(str(self.src))
        del reader.fields["general.architecture"]
        with self.assertRaises(ValueError) as cm:
            pe.architecture(reader)
        self.assertIn("no general.architecture", str(cm.exception))

    def test_split_keys_are_dropped_so_output_is_a_single_file(self):
        pe.prune([self.src], self.dst, 2)
        r = GGUFReader(str(self.dst))
        for key in pe.SPLIT_KEYS:
            self.assertNotIn(key, r.fields)

    def test_metadata_shard_finds_a_later_shard(self):
        tensors_only = GGUFReader(str(self.src))
        del tensors_only.fields["general.architecture"]
        real = GGUFReader(str(self.src))
        self.assertIs(pe.metadata_shard([tensors_only, real]), real)


if __name__ == "__main__":
    unittest.main()
