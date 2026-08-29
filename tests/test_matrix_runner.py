"""The economic verdict must be computed from the timers that belong to the path it names.

The runner previously added the raw-endpoint tail completion to the adapter import total.
Those two states should behave alike, but substituting a neighbouring measurement reports a
number for a path that was never timed - and no amount of parity checking makes a borrowed
timer into a measurement of the thing it stands in for.
"""

import importlib.util
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "production_matrix", REPO / "scripts" / "production_matrix.py")
matrix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matrix)


def leg(*, import_s=3.0, adapter_tail=0.4, raw_tail=99.0, cold=1.0, phases=None,
        reported=None):
    phases = {"container_verify": 1.0, "preflight": 0.5, "staging": 1.0,
              "runtime_restore": 0.3, "reuse_probe": 0.1,
              "pristine_restore": 0.1} if phases is None else phases
    return {
        "cold": {"wall_s": cold},
        "tail_completion_wall_s": raw_tail,
        "adapter": {
            "import_seconds_end_to_end": import_s,
            "adapter_tail_completion_wall_s": adapter_tail,
            "import_reported_seconds": sum(phases.values()) if reported is None else reported,
            "import_phases": phases,
        },
    }


class VerdictTest(unittest.TestCase):
    def test_verdict_uses_the_timed_post_adapter_completion(self):
        verdict = matrix.compute_verdict(leg(import_s=3.0, adapter_tail=0.4, raw_tail=99.0))
        self.assertAlmostEqual(verdict["adapter_import_plus_tail_s"], 3.4)
        self.assertEqual(verdict["adapter_tail_completion_s"], 0.4)

    def test_changing_only_the_raw_tail_cannot_change_the_verdict(self):
        a = matrix.compute_verdict(leg(raw_tail=0.001))
        b = matrix.compute_verdict(leg(raw_tail=1000.0))
        self.assertEqual(a["adapter_import_plus_tail_s"], b["adapter_import_plus_tail_s"])
        self.assertEqual(a["restore_is_cheaper"], b["restore_is_cheaper"])
        self.assertNotEqual(a["raw_endpoint_tail_s"], b["raw_endpoint_tail_s"])

    def test_an_untimed_adapter_tail_refuses_rather_than_substituting(self):
        broken = leg()
        del broken["adapter"]["adapter_tail_completion_wall_s"]
        with self.assertRaises(RuntimeError) as caught:
            matrix.compute_verdict(broken)
        self.assertIn("refusing to substitute", str(caught.exception))

    def test_a_refused_import_has_no_verdict(self):
        empty = leg()
        empty["adapter"]["import_seconds_end_to_end"] = None
        self.assertIsNone(matrix.compute_verdict(empty))

    def test_cheaper_is_decided_against_the_cold_request(self):
        self.assertTrue(matrix.compute_verdict(
            leg(import_s=0.4, adapter_tail=0.1, cold=1.0))["restore_is_cheaper"])
        self.assertFalse(matrix.compute_verdict(
            leg(import_s=4.0, adapter_tail=0.1, cold=1.0))["restore_is_cheaper"])


class PhaseReconciliationTest(unittest.TestCase):
    def test_every_second_is_assigned_within_tolerance(self):
        phases = {"container_verify": 1.0, "preflight": 0.5, "staging": 1.0,
                  "runtime_restore": 0.3, "reuse_probe": 0.1, "pristine_restore": 0.1}
        rec = matrix.reconcile_phases(3.0, phases)
        self.assertTrue(rec["reconciled"], rec)
        self.assertAlmostEqual(rec["unclassified_s"], 0.0)

    def test_an_unclassified_remainder_is_reported_not_absorbed(self):
        rec = matrix.reconcile_phases(3.6, {"staging": 1.0, "runtime_restore": 0.3})
        self.assertFalse(rec["reconciled"])
        self.assertAlmostEqual(rec["unclassified_s"], 2.3)

    def test_the_verdict_carries_the_reconciliation(self):
        verdict = matrix.compute_verdict(leg(reported=3.6))
        self.assertIn("phase_reconciliation", verdict)
        self.assertFalse(verdict["phase_reconciliation"]["reconciled"])


class RetainedRecordTest(unittest.TestCase):
    """The retained 256 record must stay readable by the parameterized schema."""

    def test_the_sealed_256_record_still_parses(self):
        record = json.loads((REPO / "bench" / "production-27b-matrix.json").read_text())
        for key in ("repo_commit", "runner_sha256", "legs", "prompt_tokens"):
            self.assertIn(key, record)
        patched = record["legs"]["patched"]
        for key in ("cold", "warm_after_restore", "adapter", "binary_digests"):
            self.assertIn(key, patched)
        self.assertIn("import_phases", patched["adapter"])
        self.assertEqual(patched["warm_after_restore"]["cache_n"], 252)


if __name__ == "__main__":
    unittest.main()


class RecordedDtypeTest(unittest.TestCase):
    """The advertised cache types must reach the record, not just the adapter.

    The runner read `type_k`, which nothing populates, while the adapter reads
    `target_cache_type_k`. The result was a record showing an empty cache dtype beside a
    successful import that only happens when the dtype is non-empty - a record contradicting
    itself.
    """

    def test_the_runner_reads_the_advertised_key(self):
        source = (REPO / "scripts" / "production_matrix.py").read_text()
        self.assertIn('props.get("target_cache_type_k"', source)
        self.assertIn('props.get("target_cache_type_v"', source)

    def test_the_runner_does_not_fall_back_to_weight_quantization(self):
        source = (REPO / "scripts" / "production_matrix.py").read_text()
        marker = source.index('"kv_dtype_k"')
        window = source[marker:marker + 400]
        self.assertNotIn("model_ftype", window.split('"model_ftype":')[0],
                         "weight quantization must not be a fallback for cache dtype")


class ReconciliationIsRequiredTest(unittest.TestCase):
    """An unreconciled record must fail the run, not merely report a remainder.

    The 256 gate reported 0.612 s unclassified after the support check moved ahead of the
    identity work, leaving that window outside every named phase. The runner printed the
    gap and carried on, so the defect reached a committed record.
    """

    def test_the_runner_raises_on_an_unreconciled_record(self):
        source = (REPO / "scripts" / "production_matrix.py").read_text()
        marker = source.index('phases account for')
        window = source[marker:marker + 900]
        self.assertIn('rec["reconciled"]', window)
        self.assertIn("raise RuntimeError", window)


class StorageEvidenceTest(unittest.TestCase):
    """A persistent-storage record must identify the mount, not the pathname.

    On this host /mnt/storage is a FUSE-mounted SATA volume while the NVMe is mounted at /,
    so a name-based guess would have labelled the wrong device.
    """

    def setUp(self):
        _spec2 = importlib.util.spec_from_file_location(
            "admitted_store_gate", REPO / "scripts" / "admitted_store_gate.py")
        self.gate = importlib.util.module_from_spec(_spec2)
        _spec2.loader.exec_module(self.gate)

    def evidence(self, path):
        return self.gate.storage_evidence(Path(path), Path(__file__))

    def test_memory_backed_targets_are_refused(self):
        for path in ("/dev/shm", "/run"):
            if not Path(path).is_dir():
                continue
            with self.subTest(path=path):
                found = self.evidence(path)
                if found["filesystem"] not in self.gate.MEMORY_BACKED:
                    continue
                with self.assertRaises(SystemExit) as caught:
                    self.gate.require_persistent(found)
                self.assertIn("memory-backed", str(caught.exception))

    def test_an_unresolved_target_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.gate.require_persistent({"resolved_path": "/nowhere", "filesystem": "",
                                          "mount_source": ""})
        self.assertIn("refusing to call an unresolved target persistent",
                      str(caught.exception))

    def test_overlay_is_refused(self):
        with self.assertRaises(SystemExit):
            self.gate.require_persistent({"resolved_path": "/x", "filesystem": "overlay",
                                          "mount_source": "overlay"})

    def test_evidence_names_the_mount_source_and_backing_device(self):
        found = self.evidence(REPO)
        for key in ("resolved_path", "stat_device_id", "mount_source", "mount_target",
                    "filesystem", "mount_options", "available_bytes", "backing_device",
                    "rotational", "same_mount_as_model"):
            self.assertIn(key, found)
        self.assertTrue(found["mount_source"], "no mount source identified")
