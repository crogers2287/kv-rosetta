"""What the MTP/speculative checkpoint gate refuses, and why each refusal exists.

The runner's whole value is in the runs it declines to score. A verdict about speculative
checkpoint state is worth having only if the harness can show that it restored anything at
all, that the instance it used as a baseline is the already-proven target-only case, that
the two instances are the same weights, and that the reader answers identical uncached work
identically. Each of those is a guard, and each guard is pinned here by its message rather
than by its exception class - deleting one of these guards usually leaves a LATER guard
raising the same class from a different path, and `assertRaises` alone would pass (REQ-066).

None of this needs a GPU or a server. The live path is exercised against a fake endpoint
that speaks the same three routes llama.cpp does and writes a real GGSQ state file, so the
composition - save, scramble, restore, score - is covered offline too.
"""

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtp_speculative_gate import (  # noqa: E402
    MIN_FORCED_POSITIONS, Endpoint, SpecGateError, active_state_classes, architecture_of,
    assemble, measure_instance, preflight_instance, require_checkpoint_saved,
    require_continuation, require_determinism_runs, require_distinct_endpoints,
    require_forced_positions, require_identical_tokenisation, require_live_determinism,
    require_record_matches_live, require_saved_state, require_serves_model,
    require_speculative_active, require_target_only, verdict, wall_clock,
)

FORCED = MIN_FORCED_POSITIONS


# --- fixtures --------------------------------------------------------------------------

def _gguf(strings=(("general.architecture", "qwen2"),)):
    blob = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(strings)))
    for key, value in strings:
        blob += (struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 8)
                 + struct.pack("<Q", len(value)) + value.encode())
    path = Path(tempfile.mkdtemp()) / "model.gguf"
    path.write_bytes(bytes(blob))
    return path


def _state_bytes(n_tokens=3, n_layer=2, cell_count=4):
    from tests.test_ggsq_decoder import build_attention
    body = build_attention(cell_count=cell_count, n_layer=n_layer)
    return (b"qsgg" + struct.pack("<I", 3) + struct.pack("<I", n_tokens)
            + b"\x00" * (4 * n_tokens) + body)


def _run(text="alpha", tokens=None, slot=0, cache_n=0):
    tokens = list(tokens if tokens is not None else range(1, FORCED + 1))
    return {"id_slot": slot, "cache_n": cache_n, "prompt_n": 430, "seconds": 1.0,
            "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_ids": tokens, "vectors": [{1: -0.1}] * len(tokens)}


def _cmp(top1=1.0, positions=FORCED):
    return {"positions": positions, "top1_agreement": top1,
            "max_abs_logprob_delta": 0.0, "mean_abs_logprob_delta": 0.0,
            "shared_tokens": positions, "tokens_only_in_one": 0}


def _leg(cache_n=500, top1=1.0, seconds=0.5, refused=None, text_match=True,
         positions=FORCED):
    return {"cache_n": cache_n, "prompt_n": 4, "prompt_ms": 40.0, "seconds": seconds,
            "restore_refused": refused, "text_matches_cold": text_match,
            "forced_vs_cold": _cmp(top1, positions),
            "restored": {"n_restored": 500, "n_read": 1024}}


def _instance(identity_top1=1.0, noise_top1=0.0, **over):
    legs = {
        "cold": {"cache_n": 0, "prompt_n": 430, "prompt_ms": 900.0, "seconds": 1.0,
                 "restore_refused": None, "text_matches_cold": True,
                 "forced_vs_cold": None, "restored": None},
        "identity": _leg(top1=identity_top1),
        "noise": _leg(top1=noise_top1, text_match=False),
    }
    legs.update(over)
    return legs


def _measured(spec_within=1.0, cross=1.0, cross_cold=1.0, **over):
    measured = {
        "spec": _instance(identity_top1=spec_within),
        "nospec": _instance(),
        "cross": {"spec_restored_vs_nospec_cold": _cmp(cross),
                  "spec_cold_vs_nospec_cold": _cmp(cross_cold)},
    }
    measured.update(over)
    return measured


def _verdict(measured=None, **kw):
    options = {"min_top1": 0.99, "control_min_top1": 1.0, "noise_max_top1": 0.5}
    options.update(kw)
    return verdict(measured if measured is not None else _measured(), **options)


# --- preconditions on the two endpoints ------------------------------------------------

class EndpointPreconditions(unittest.TestCase):
    def test_two_different_urls_are_accepted(self):
        require_distinct_endpoints("http://127.0.0.1:1", "http://127.0.0.1:2")

    def test_one_url_passed_twice_is_refused(self):
        # A typo would make every cross-instance comparison a run against itself and
        # report perfect agreement - the most convincing possible wrong answer.
        with self.assertRaises(SpecGateError) as caught:
            require_distinct_endpoints("http://h:8080", "http://h:8080/")
        self.assertIn("same server", str(caught.exception))

    def test_the_harness_cannot_start_a_server(self):
        # The speculative launch is the variable under test and belongs to the operator.
        endpoint = Endpoint("http://127.0.0.1:9/", "spec")
        self.assertEqual(endpoint.url, "http://127.0.0.1:9")
        self.assertFalse(hasattr(endpoint, "start"))
        self.assertFalse(hasattr(endpoint, "argv"))


class ServedModel(unittest.TestCase):
    def test_the_declared_model_must_be_the_one_given(self):
        model = _gguf()
        require_serves_model({"model_path": str(model)}, "http://h", model)

    def test_a_server_that_declares_no_model_is_refused(self):
        # The message matters: with this guard gone the comparison below raises the same
        # class about "serves None", so the class alone proves nothing.
        model = _gguf()
        with self.assertRaises(SpecGateError) as caught:
            require_serves_model({}, "http://h", model)
        self.assertIn("does not report which model", str(caught.exception))

    def test_a_server_running_other_weights_is_refused(self):
        model = _gguf()
        with self.assertRaises(SpecGateError) as caught:
            require_serves_model({"model_path": "/elsewhere/other.gguf"}, "http://h", model)
        self.assertIn("must run the same weights", str(caught.exception))


class StateClasses(unittest.TestCase):
    def test_a_declared_list_is_returned(self):
        self.assertEqual(active_state_classes(
            {"active_checkpoint_state_classes": ["target", "speculative"]}, "u"),
            ["target", "speculative"])

    def test_an_unreported_field_is_refused(self):
        # Deleting this guard falls through to the empty-list guard, which raises the same
        # class with a different message, so the message is what is asserted.
        with self.assertRaises(SpecGateError) as caught:
            active_state_classes({}, "http://h")
        self.assertIn("list of strings", str(caught.exception))

    def test_a_list_of_non_strings_is_refused(self):
        with self.assertRaises(SpecGateError) as caught:
            active_state_classes({"active_checkpoint_state_classes": [1, 2]}, "http://h")
        self.assertIn("list of strings", str(caught.exception))

    def test_an_empty_list_is_not_read_as_target_only(self):
        with self.assertRaises(SpecGateError) as caught:
            active_state_classes({"active_checkpoint_state_classes": []}, "http://h")
        self.assertIn("empty", str(caught.exception))

    def test_the_subject_must_require_more_than_target_state(self):
        # Otherwise the run measures §33's already-proven target-only case and files the
        # result under a name that claims it is about speculative state.
        with self.assertRaises(SpecGateError) as caught:
            require_speculative_active(["target"], "http://h")
        self.assertIn("already-proven target-only", str(caught.exception))

    def test_the_subject_with_speculative_state_is_accepted(self):
        require_speculative_active(["target", "speculative"], "http://h")

    def test_the_control_must_be_target_only(self):
        # A control that also carries speculative state contains the effect being
        # measured, so the baseline it provides is not a baseline.
        with self.assertRaises(SpecGateError) as caught:
            require_target_only(["target", "draft"], "http://h")
        self.assertIn("second subject rather than a control", str(caught.exception))

    def test_a_target_only_control_is_accepted(self):
        require_target_only(["target"], "http://h")


class DeterminismProof(unittest.TestCase):
    def _record(self, **over):
        attestation = {"build_info": "b1-abc", "model_path": "/m.gguf", "n_ctx": 8192}
        attestation.update(over)
        return {"label": "spec-on", "verdict": {"reproducible": True},
                "attestation": attestation}

    def _props(self, **over):
        props = {"build_info": "b1-abc", "model_path": "/m.gguf",
                 "default_generation_settings": {"n_ctx": 8192}}
        props.update(over)
        return props

    def test_a_record_matching_the_live_server_is_accepted(self):
        require_record_matches_live(self._record(), self._props(), "spec-on")

    def test_a_record_from_another_build_is_refused(self):
        # The label is a string the operator typed; on its own it will certify a record
        # made against a build that does not carry checkpoint persistence at all.
        with self.assertRaises(SpecGateError) as caught:
            require_record_matches_live(self._record(build_info="b0-old"), self._props(),
                                        "spec-on")
        self.assertIn("build_info", str(caught.exception))

    def test_a_record_for_other_weights_is_refused(self):
        with self.assertRaises(SpecGateError) as caught:
            require_record_matches_live(self._record(model_path="/other.gguf"),
                                        self._props(), "spec-on")
        self.assertIn("model_path", str(caught.exception))

    def test_a_record_at_another_context_size_is_refused(self):
        with self.assertRaises(SpecGateError) as caught:
            require_record_matches_live(self._record(n_ctx=4096), self._props(), "spec-on")
        self.assertIn("n_ctx", str(caught.exception))

    def test_the_sample_size_may_not_be_shrunk(self):
        # reader_determinism.main clamps with min(args.runs, MIN_RUNS) so a short run still
        # renders a record. Here the number is the licence for everything downstream.
        with self.assertRaises(SpecGateError) as caught:
            require_determinism_runs(2)
        self.assertIn("has not been shown to repeat itself", str(caught.exception))

    def test_the_protocol_sample_size_is_accepted(self):
        require_determinism_runs(6)

    def test_a_reproducible_live_instance_is_accepted(self):
        summary = require_live_determinism([_run() for _ in range(6)], "spec-on")
        self.assertTrue(summary["reproducible"])

    def test_a_live_instance_that_answers_differently_is_refused(self):
        # The record cannot cover this: reader_determinism starts its own server, so no
        # record was ever produced by the process now serving --spec-url.
        runs = [_run(text="alpha") for _ in range(5)] + [_run(text="beta")]
        with self.assertRaises(SpecGateError) as caught:
            require_live_determinism(runs, "spec-on")
        self.assertIn("identical uncached work", str(caught.exception))


class PromptAndPositions(unittest.TestCase):
    def test_identical_token_sequences_are_accepted(self):
        require_identical_tokenisation([1, 2, 3], [1, 2, 3])

    def test_differing_tokenisation_is_refused(self):
        # compare_forced lines vectors up by index; it cannot know the prefixes differed.
        with self.assertRaises(SpecGateError) as caught:
            require_identical_tokenisation([1, 2, 3], [1, 9, 3])
        self.assertIn("first differing at index 1", str(caught.exception))

    def test_a_prefix_of_the_other_is_refused(self):
        with self.assertRaises(SpecGateError) as caught:
            require_identical_tokenisation([1, 2, 3], [1, 2])
        self.assertIn("not scoring the same positions", str(caught.exception))

    def test_too_few_forced_positions_is_refused(self):
        with self.assertRaises(SpecGateError) as caught:
            require_forced_positions(4)
        self.assertIn("moves in steps of", str(caught.exception))

    def test_the_minimum_is_accepted(self):
        require_forced_positions(MIN_FORCED_POSITIONS)

    def test_a_short_cold_run_cannot_supply_the_continuation(self):
        # Silently scoring fewer positions than requested changes the measurement without
        # changing the flag that named it.
        with self.assertRaises(SpecGateError) as caught:
            require_continuation([1, 2, 3], 48)
        self.assertIn("fewer than the 48", str(caught.exception))

    def test_the_continuation_is_truncated_to_what_was_asked_for(self):
        self.assertEqual(require_continuation([1, 2, 3, 4], 3), [1, 2, 3])


class SavedArtifact(unittest.TestCase):
    def test_a_present_state_file_is_accepted(self):
        slots = Path(tempfile.mkdtemp())
        (slots / "s.state").write_bytes(b"qsgg")
        self.assertEqual(require_saved_state(slots / "s.state", "spec"),
                         slots / "s.state")

    def test_a_state_file_this_process_cannot_see_is_refused(self):
        # The server reports a successful save whatever --spec-slots points at; the noise
        # control would then be built from a stale file, or from none.
        slots = Path(tempfile.mkdtemp())
        with self.assertRaises(SpecGateError) as caught:
            require_saved_state(slots / "absent.state", "spec")
        self.assertIn("not the one the server writes to", str(caught.exception))

    def test_a_save_that_carried_checkpoints_is_accepted(self):
        require_checkpoint_saved({"n_saved": 500, "n_checkpoints_saved": 2}, "spec")

    def test_a_save_that_reports_no_checkpoint_count_is_refused(self):
        # Message asserted: with this guard gone, int(None) raises TypeError from the next
        # line, which is a different class and would make assertRaises misleading either
        # way.
        with self.assertRaises(SpecGateError) as caught:
            require_checkpoint_saved({"n_saved": 500}, "spec")
        self.assertIn("did not report n_checkpoints_saved", str(caught.exception))

    def test_a_plain_sequence_state_is_refused(self):
        # On a build with no checkpoint persistence both instances save a plain ggsq body,
        # both agree perfectly, and the verdict would read "proven" about a state class
        # that was never in the file.
        with self.assertRaises(SpecGateError) as caught:
            require_checkpoint_saved({"n_saved": 500, "n_checkpoints_saved": 0}, "spec")
        self.assertIn("plain sequence state", str(caught.exception))


class ArchitectureRead(unittest.TestCase):
    def test_the_architecture_is_read_from_the_gguf(self):
        self.assertEqual(architecture_of(str(_gguf())), "qwen2")

    def test_a_gguf_without_an_architecture_is_refused(self):
        # writes_cell_ext needs it; guessing desynchronises the noise control's parse one
        # cell in, which presents as file corruption rather than as a bad argument (§35).
        path = _gguf(strings=(("general.name", "nameless"),))
        with self.assertRaises(SpecGateError) as caught:
            architecture_of(str(path))
        self.assertIn("declares no architecture", str(caught.exception))


# --- the verdict -----------------------------------------------------------------------

class VerdictControls(unittest.TestCase):
    def test_healthy_controls_and_matching_output_prove_the_restore(self):
        result = _verdict()
        self.assertTrue(result["controls_ok"])
        self.assertEqual(result["speculative_restore"], "proven")
        self.assertTrue(result["at_or_above_within_baseline"])
        self.assertTrue(result["at_or_above_cross_baseline"])

    def test_a_control_instance_that_reused_nothing_yields_no_verdict(self):
        measured = _measured()
        measured["nospec"]["identity"]["cache_n"] = 0
        result = _verdict(measured)
        self.assertFalse(result["controls_ok"])
        self.assertIsNone(result["speculative_restore"])
        self.assertIn("control instance reused nothing", " ".join(result["problems"]))

    def test_a_control_instance_that_did_not_reproduce_yields_no_verdict(self):
        # If the already-proven target-only case fails under this harness, a speculative
        # result cannot be attributed to the speculative state.
        measured = _measured()
        measured["nospec"]["identity"]["forced_vs_cold"] = _cmp(0.90)
        result = _verdict(measured)
        self.assertFalse(result["controls_ok"])
        self.assertIn("no baseline", " ".join(result["problems"]))

    def test_noise_that_reused_nothing_yields_no_verdict(self):
        for instance in ("spec", "nospec"):
            measured = _measured()
            measured[instance]["noise"]["cache_n"] = 0
            result = _verdict(measured)
            self.assertFalse(result["controls_ok"])
            self.assertIn("bounds nothing", " ".join(result["problems"]))

    def test_noise_that_looks_right_yields_no_verdict(self):
        # The §20 failure: if scrambled values still agree, nothing was being restored and
        # every leg was a cold prefill.
        for instance in ("spec", "nospec"):
            measured = _measured()
            measured[instance]["noise"]["forced_vs_cold"] = _cmp(0.95)
            result = _verdict(measured)
            self.assertFalse(result["controls_ok"])
            self.assertIn("floor is not a floor", " ".join(result["problems"]))

    def test_a_perfect_subject_cannot_pass_on_broken_controls(self):
        # The number being attractive is exactly when this matters.
        measured = _measured()
        measured["spec"]["noise"]["forced_vs_cold"] = _cmp(1.0)
        result = _verdict(measured)
        self.assertIsNone(result["speculative_restore"])
        self.assertIsNone(result["meets_threshold"])

    def test_a_refused_restore_is_not_evidence(self):
        measured = _measured()
        measured["spec"]["identity"] = _leg(refused="400: model mismatch")
        result = _verdict(measured)
        self.assertFalse(result["controls_ok"])
        self.assertIn("refusal is not evidence", " ".join(result["problems"]))

    def test_a_subject_that_reused_nothing_is_neither_proven_nor_refuted(self):
        measured = _measured()
        measured["spec"]["identity"]["cache_n"] = None
        result = _verdict(measured)
        self.assertIsNone(result["speculative_restore"])
        self.assertIn("neither proof nor refutation", " ".join(result["problems"]))

    def test_legs_scored_over_different_lengths_are_not_comparable(self):
        measured = _measured()
        measured["spec"]["identity"]["forced_vs_cold"] = _cmp(1.0, positions=FORCED - 1)
        result = _verdict(measured)
        self.assertFalse(result["controls_ok"])
        self.assertIn("different numbers of positions", " ".join(result["problems"]))

    def test_a_missing_leg_yields_no_verdict(self):
        measured = _measured()
        del measured["spec"]["noise"]
        result = _verdict(measured)
        self.assertFalse(result["controls_ok"])
        self.assertIn("has no noise leg", " ".join(result["problems"]))


class VerdictBaselines(unittest.TestCase):
    def test_the_subject_is_judged_against_the_measured_baselines(self):
        # Restoring and prefilling are different computations. §28 shipped an absolute
        # 0.99 threshold that a model reading its OWN cache could not reach, so a failure
        # against it said nothing about the cache. Here the control's own number is the
        # bar, and the absolute threshold is reported next to it.
        measured = _measured(spec_within=0.97)
        measured["nospec"]["identity"]["forced_vs_cold"] = _cmp(0.97)
        result = _verdict(measured, control_min_top1=0.95)
        self.assertEqual(result["control_top1_vs_own_cold"], 0.97)
        self.assertTrue(result["at_or_above_within_baseline"])
        self.assertTrue(result["threshold_exceeds_baseline"])
        self.assertFalse(result["meets_threshold"])
        self.assertEqual(result["speculative_restore"], "proven")

    def test_a_subject_below_the_within_baseline_is_refuted(self):
        result = _verdict(_measured(spec_within=0.80))
        self.assertTrue(result["controls_ok"])
        self.assertEqual(result["speculative_restore"], "refuted")
        self.assertFalse(result["at_or_above_within_baseline"])

    def test_the_cross_comparison_is_scored_against_cold_versus_cold(self):
        # Two launches of one model can disagree cold; charging the restore for a
        # difference that exists without any restore is the §28 mistake.
        result = _verdict(_measured(cross=0.90, cross_cold=0.90))
        self.assertEqual(result["speculative_restore"], "proven")
        self.assertEqual(result["cold_top1_spec_vs_nospec"], 0.90)

    def test_a_restore_that_diverges_from_the_plain_instance_is_refuted(self):
        result = _verdict(_measured(cross=0.70, cross_cold=0.99))
        self.assertEqual(result["speculative_restore"], "refuted")
        self.assertFalse(result["at_or_above_cross_baseline"])


class WallClock(unittest.TestCase):
    def test_reuse_and_wall_clock_are_reported_together(self):
        # §34: a Gemma run reused all 578 of its tokens and was slower than prefilling
        # them. cache_n answers "was the cache used", not "was it worth using".
        measured = _measured()
        measured["spec"]["cold"]["seconds"] = 0.383
        measured["spec"]["identity"]["seconds"] = 0.417
        clock = wall_clock(measured)["spec"]
        self.assertEqual(clock["restored_cache_n"], 500)
        self.assertLess(clock["speedup"], 1.0)
        self.assertTrue(clock["restore_was_slower"])

    def test_a_genuine_speedup_is_not_flagged(self):
        measured = _measured()
        measured["spec"]["cold"]["seconds"] = 2.0
        measured["spec"]["identity"]["seconds"] = 0.5
        clock = wall_clock(measured)["spec"]
        self.assertEqual(clock["speedup"], 4.0)
        self.assertFalse(clock["restore_was_slower"])

    def test_the_verdict_carries_the_wall_clock(self):
        self.assertIn("spec", _verdict()["wall_clock"])


# --- the live path, against a fake endpoint ---------------------------------------------

class FakeEndpoint:
    """Speaks the three routes this runner uses, and writes a real GGSQ file on save.

    Restoring the identity artifact reproduces the cold answer; restoring the scrambled
    one does not. That is the behaviour a healthy run must show, and building it here means
    the save/scramble/restore/score composition is covered without a GPU.
    """

    def __init__(self, slots: Path, label: str = "fake", cells: int = 500):
        self.url, self.label, self.slots, self.cells = "http://fake", label, slots, cells
        self.restored = None
        self.calls = []

    def _vectors(self, count):
        token = 7 if self.restored != "noise" else 9
        return [{"id": token, "top_logprobs": [{"id": token, "logprob": -0.1},
                                               {"id": 5, "logprob": -2.0}]}
                for _ in range(count)]

    def post(self, path, body, timeout=900):
        self.calls.append(path)
        if path == "/tokenize":
            return {"tokens": list(range(1, 41))}
        if "action=erase" in path:
            self.restored = None
            return {}
        if "action=save" in path:
            (self.slots / body["filename"]).write_bytes(_state_bytes())
            return {"n_saved": self.cells, "n_written": 4096, "n_checkpoints_saved": 2}
        if "action=restore" in path:
            self.restored = "noise" if "noise" in body["filename"] else "identity"
            return {"n_restored": self.cells, "n_read": 4096}
        if path == "/completion":
            count = body["n_predict"]
            return {"content": "COLD" if self.restored != "noise" else "GARBLED",
                    "timings": {"cache_n": 0 if self.restored is None else self.cells,
                                "prompt_n": 430, "prompt_ms": 900.0},
                    "completion_probabilities": self._vectors(count)}
        raise AssertionError(f"unexpected route {path}")


class LivePath(unittest.TestCase):
    def _measure(self):
        slots = Path(tempfile.mkdtemp())
        endpoint = FakeEndpoint(slots, "fake")
        cold = preflight_instance(endpoint, "prompt", slot=0, predict=FORCED, runs=6)
        continuation = require_continuation(cold["cold"]["token_ids"], FORCED)
        return endpoint, measure_instance(
            endpoint, prompt="prompt", prompt_ids=[1, 2, 3], continuation=continuation,
            slot=0, predict=FORCED, presave_predict=2, slots=slots, arch="qwen2",
            cold=cold["cold"])

    def test_a_healthy_instance_measures_three_legs(self):
        endpoint, measured = self._measure()
        self.assertEqual(sorted(measured["legs"]), ["cold", "identity", "noise"])
        self.assertEqual(measured["legs"]["identity"]["cache_n"], 500)
        self.assertEqual(measured["legs"]["noise"]["cache_n"], 500)
        self.assertEqual(measured["legs"]["identity"]["forced_vs_cold"]["top1_agreement"],
                         1.0)
        self.assertEqual(measured["legs"]["noise"]["forced_vs_cold"]["top1_agreement"],
                         0.0)

    def test_the_noise_leg_restores_a_different_file_from_the_identity_leg(self):
        # A noise control that is byte-identical to the subject is a second subject, and
        # agrees with it perfectly.
        _, measured = self._measure()
        self.assertGreater(measured["noise"]["payload_bytes_scrambled"], 0)
        self.assertGreater(measured["noise"]["structural_bytes_preserved"], 0)

    def test_the_slot_is_re_restored_before_the_teacher_forced_pass(self):
        # The free-generation completion has already advanced the slot; scoring on it
        # would measure a cache that no longer holds only what was restored.
        endpoint, _ = self._measure()
        restores = [call for call in endpoint.calls if "action=restore" in call]
        self.assertEqual(len(restores), 4)      # identity and noise, twice each

    def test_two_measured_instances_assemble_into_a_provable_verdict(self):
        _, spec = self._measure()
        _, nospec = self._measure()
        measured = assemble(spec, nospec)
        result = _verdict(measured)
        self.assertTrue(result["controls_ok"])
        self.assertEqual(result["speculative_restore"], "proven")

    def test_a_save_without_checkpoints_stops_the_measurement(self):
        slots = Path(tempfile.mkdtemp())
        endpoint = FakeEndpoint(slots, "fake")
        original = endpoint.post

        def post(path, body, timeout=900):
            result = original(path, body, timeout)
            if "action=save" in path:
                result["n_checkpoints_saved"] = 0
            return result

        endpoint.post = post
        cold = preflight_instance(endpoint, "prompt", slot=0, predict=FORCED, runs=6)
        with self.assertRaises(SpecGateError) as caught:
            measure_instance(endpoint, prompt="prompt", prompt_ids=[1, 2, 3],
                             continuation=cold["cold"]["token_ids"][:FORCED], slot=0,
                             predict=FORCED, presave_predict=2, slots=slots,
                             arch="qwen2", cold=cold["cold"])
        self.assertIn("plain sequence state", str(caught.exception))


class RecordShape(unittest.TestCase):
    def test_the_verdict_is_json_serialisable(self):
        # The runner writes its record after every expensive leg has already run.
        json.dumps(_verdict(), default=str)


if __name__ == "__main__":
    unittest.main()
