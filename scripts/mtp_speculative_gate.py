#!/usr/bin/env python3
"""Is restoring a checkpoint that carries speculative/draft state CORRECT? RESEARCH ONLY.

`llamacpp_http` refuses a checkpoint whose launch reports

    active_checkpoint_state_classes       = ['target', 'speculative']
    supports_speculative_checkpoint_state = false

because `proven_state_classes` holds `{"target"}` and nothing had measured the rest. That
refusal is correct as long as the question is open. This runner is what closes it, in the
only direction that can be closed by measurement: it does not ask whether the runtime says
the state is supported, it asks whether the restored cache produces the answer a cold
prefill produces.

Two servers, both already running and started by the operator. This script starts nothing
and touches no GPU:

    spec    speculative decoding ON. Its checkpoints carry the state class in question.
            This is the SUBJECT.
    nospec  the same model and weights with speculative decoding OFF, so its launch is
            target-only - the case §33 already proved restores and reuses. This is the
            CONTROL INSTANCE.

Both restore their own cache under an identical protocol, so a difference on the spec
instance can be attributed to the speculative state rather than to the harness, the model,
the prompt or the host. Two comparisons come out of that:

    within   spec restored vs spec cold prefill, against the same comparison measured on
             the control instance as the baseline. Restoring and prefilling are different
             computations and already disagree slightly for a model reading its own cache
             (§28: 0.977; §35 on a checkpoint-persisting build: 1.000), so the control's
             own number - not an absolute constant - is what the subject is judged against.
    cross    spec restored vs nospec cold prefill, against spec cold vs nospec cold as the
             baseline. Two launches of one model can disagree cold; charging the restore
             for a difference that exists without any restore is the mistake §28's verdict
             docstring records.

Everything else here is protocol this repo already established and this runner reuses
rather than re-derives:

* Teacher-forced scoring (`cross_model_gate.teacher_forced` / `compare_forced`). Free
  generation is a cliff - over 128 freely generated tokens a model restoring its OWN cache
  agreed with its own cold prefill on 0.23 of positions, which is a fact about
  autoregressive cascade and not about the cache.
* Mandatory controls. The control instance's identity leg must reuse AND reproduce; a
  noise leg on BOTH instances (`cross_model_gate.scramble_payload`: payload destroyed,
  structure intact) must reuse and come out WRONG. Noise that looks right means nothing
  was restored anywhere and every leg was a cold prefill - the failure that invalidated
  two runs in §20. No verdict is rendered unless every control behaved.
* Reader determinism as a precondition (`reader_determinism`). A verdict comparing runs on
  a reader that has not been shown reproducible is measuring the reader. A passing record
  is required for each instance the way `slot_poisoning.load_reproducible` requires one,
  AND the preflight is re-run live against each URL - because a record is produced by a
  separately launched server and therefore cannot speak for a launch with speculative
  decoding enabled, which is precisely the variable under test.
* cache_n AND wall clock, reported together and never collapsed. §34: a Gemma run reused
  every one of 578 tokens and was slower than prefilling them.

Fail closed. Every ambiguity is a refusal; there is no default verdict.

Exit codes: 0 the speculative restore is proven, 1 it is refuted, 2 no verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_model_gate import (  # noqa: E402
    compare_forced, require_matched_caches, run_completion, scramble_payload,
    teacher_forced, tokenize,
)
from production_matrix import sha256_file  # noqa: E402
from reader_determinism import (  # noqa: E402
    MIN_RUNS, cold_run, digest_text, summarise,
)
from slot_poisoning import load_reproducible  # noqa: E402

from kv_rosetta import gguf  # noqa: E402
from kv_rosetta.sizing import writes_cell_ext  # noqa: E402

#: Top-1 agreement over N positions moves in steps of 1/N. Below this the subject and the
#: control cannot be separated by anything finer than a coin flip, and a "proven" from such
#: a run would be an artifact of the resolution rather than a measurement.
MIN_FORCED_POSITIONS = 16

#: The state class this project has proven behaviourally, mirroring
#: `LlamaCppHttpAdapter.proven_state_classes`. A launch that requires only this is the
#: control, not the subject.
PROVEN_STATE_CLASSES = frozenset({"target"})

#: Attestation fields that must agree between a determinism record and the live server it
#: is offered as proof for.
BOUND_ATTESTATION_FIELDS = ("build_info", "model_path", "n_ctx")


class SpecGateError(RuntimeError):
    """A refusal or an invalid run. Never downgraded to a warning."""


class Endpoint:
    """A server the operator started. This class cannot start or stop one.

    `reader_determinism.Reader` owns a subprocess, and every runner in this repo that uses
    it launches its own llama-server. This measurement deliberately cannot: the speculative
    launch is the variable under test and the operator owns it. Only `.post` and `.get` are
    provided, which is all `cold_run`, `teacher_forced` and `run_completion` call, so those
    functions are reused unchanged against a remote endpoint.
    """

    def __init__(self, url: str, label: str):
        self.url = url.rstrip("/")
        self.label = label

    def post(self, path: str, body: dict, timeout: int = 900):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def get(self, path: str, timeout: int = 60):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as response:
            return json.load(response)


# --- preconditions -------------------------------------------------------------------

def require_distinct_endpoints(spec_url: str, nospec_url: str) -> None:
    """The two URLs must not be the same server.

    Handed one URL twice, every cross-instance comparison becomes a run against itself and
    reports perfect agreement - the most convincing possible wrong answer, produced by a
    typo. The state-class checks below would also catch it, but only after a full
    measurement and with a message about state classes rather than about the arguments.
    """
    if spec_url.rstrip("/") == nospec_url.rstrip("/"):
        raise SpecGateError(
            f"--spec-url and --nospec-url are the same server ({spec_url}); the "
            f"cross-instance comparison would be a run against itself and would agree "
            f"perfectly no matter what the speculative state did")


def architecture_of(model: str) -> str:
    """The GGUF's declared architecture, which decides the cell-extension layout.

    `writes_cell_ext` needs it, and getting it wrong desynchronises the noise control's
    parse one cell in - which presents as file corruption rather than as a bad argument
    (§35).
    """
    arch = gguf.read_metadata(model).get("general.architecture")
    if not arch:
        raise SpecGateError(f"{model} declares no architecture, so whether its cells "
                            f"carry a 12-byte llama_kv_cell_ext cannot be read")
    return str(arch)


def require_serves_model(props: dict, url: str, model: Path) -> None:
    """The live server must be serving the exact weights this run was pointed at.

    Two things ride on this. The cell-extension layout used to build the noise control is
    read from the GGUF at `--model`, so an unrelated file there scrambles at the wrong
    offsets. And because BOTH instances are checked against the same path, this is also
    what makes the two instances the same model - without which every cross-instance
    number is a cross-model measurement wearing this runner's name.
    """
    declared = props.get("model_path")
    if not declared:
        raise SpecGateError(
            f"{url} does not report which model it loaded, so the weights behind it "
            f"cannot be bound to {model}; refusing rather than assuming they match")
    if Path(str(declared)).resolve() != model.resolve():
        raise SpecGateError(
            f"{url} serves {declared}, not {model}; the two instances must run the same "
            f"weights or the comparison measures the models rather than the checkpoint")


def active_state_classes(props: dict, url: str) -> list[str]:
    """Which checkpoint state classes this launch requires, refusing anything unusable."""
    value = props.get("active_checkpoint_state_classes")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SpecGateError(
            f"{url} does not report active_checkpoint_state_classes as a list of strings "
            f"(got {value!r}); which state a checkpoint carries would have to be assumed")
    if not value:
        raise SpecGateError(
            f"{url} reports an empty active_checkpoint_state_classes; an empty list is "
            f"not the same claim as ['target'] and this runner will not read it as one")
    return value


def require_speculative_active(classes: list[str], url: str) -> None:
    """The subject must actually require the state class under test."""
    beyond = sorted(set(classes) - PROVEN_STATE_CLASSES)
    if not beyond:
        raise SpecGateError(
            f"{url} requires only {sorted(classes)} checkpoint state, which is the "
            f"already-proven target-only case; pointing --spec-url at it would measure "
            f"§33 again and report it as a speculative result")


def require_target_only(classes: list[str], url: str) -> None:
    """The control instance must be the proven case, or it controls for nothing."""
    beyond = sorted(set(classes) - PROVEN_STATE_CLASSES)
    if beyond:
        raise SpecGateError(
            f"{url} requires {beyond} checkpoint state as well as target, so it is a "
            f"second subject rather than a control; the baseline it is supposed to "
            f"provide would already contain the effect being measured")


def require_record_matches_live(record: dict, props: dict, label: str) -> None:
    """Bind a determinism record to the server it is offered as proof for.

    `load_reproducible` checks that the record is labelled for this configuration and that
    it passed. A label is a string the operator typed, so on its own it will happily
    certify a record made against a different build or different weights.
    """
    attestation = record.get("attestation", {})
    live = {"build_info": props.get("build_info"),
            "model_path": props.get("model_path"),
            "n_ctx": props.get("default_generation_settings", {}).get("n_ctx")}
    differing = {field: (attestation.get(field), live[field])
                 for field in BOUND_ATTESTATION_FIELDS
                 if attestation.get(field) != live[field]}
    if differing:
        raise SpecGateError(
            f"the determinism record labelled {label!r} attests to a different server "
            f"than the one at the URL: {differing}. A proof about another build, another "
            f"model or another context size proves nothing about this launch")


def require_determinism_runs(runs: int) -> None:
    """The live preflight may not be shrunk below the retained protocol's sample size.

    `reader_determinism.main` passes `min(args.runs, MIN_RUNS)` so that a short run still
    renders a verdict. That is the right call for a preflight whose output is a record a
    human reads; it is the wrong one here, where the number silently becomes the licence
    for everything downstream.
    """
    if runs < MIN_RUNS:
        raise SpecGateError(
            f"{runs} determinism runs is fewer than the {MIN_RUNS} the retained protocol "
            f"requires; a reader that repeats itself twice has not been shown to repeat "
            f"itself")


def require_live_determinism(runs: list[dict], label: str) -> dict:
    """Re-prove determinism against the live launch, not only against a record.

    The record cannot cover this. `reader_determinism` starts its own server, so no record
    can have been produced by the process now serving `--spec-url`, and speculative
    decoding is exactly the sort of runtime behaviour - draft proposals accepted or
    rejected per step - that could make identical uncached work answer differently. If it
    does, every comparison below is measuring the sampler.
    """
    summary = summarise(runs, min_runs=MIN_RUNS)
    if not summary["reproducible"]:
        raise SpecGateError(
            f"{label} does not answer identical uncached work identically ({summary}); a "
            f"restored-versus-cold verdict on this instance would be reporting its own "
            f"variance")
    return summary


def require_identical_tokenisation(spec_ids: list[int], nospec_ids: list[int]) -> None:
    """One prompt must become one token sequence on both instances.

    Teacher forcing scores position i of a fixed continuation against a fixed prefix. If
    the two instances tokenise that prefix differently they are not scoring the same
    positions, and `compare_forced` would compare them anyway - it lines vectors up by
    index and has no way to know.
    """
    if spec_ids != nospec_ids:
        first = next((i for i, (a, b) in enumerate(zip(spec_ids, nospec_ids)) if a != b),
                     min(len(spec_ids), len(nospec_ids)))
        raise SpecGateError(
            f"the two instances tokenise the prompt differently: {len(spec_ids)} tokens "
            f"against {len(nospec_ids)}, first differing at index {first}; they are not "
            f"scoring the same positions")


def require_forced_positions(wanted: int) -> None:
    if wanted < MIN_FORCED_POSITIONS:
        raise SpecGateError(
            f"{wanted} teacher-forced positions is below the {MIN_FORCED_POSITIONS} "
            f"minimum; top-1 agreement then moves in steps of 1/{wanted} and the subject "
            f"could not be separated from the control by anything finer")


def require_continuation(tokens: list[int], wanted: int) -> list[int]:
    """The continuation every leg is scored on, taken once from a cold run."""
    if len(tokens) < wanted:
        raise SpecGateError(
            f"the cold run produced {len(tokens)} tokens, fewer than the {wanted} "
            f"teacher-forced positions requested; scoring the legs on a shorter "
            f"continuation than asked for would quietly change the measurement")
    return tokens[:wanted]


def require_saved_state(path: Path, label: str) -> Path:
    """The file the server said it wrote must be where this process can read it.

    `--{spec,nospec}-slots` is the server's own --slot-save-path. Pointed elsewhere, the
    server still reports a successful save, the noise control is then built from a stale
    file or fails on a missing one, and the restore that follows reads whatever the server
    finds under that name instead.
    """
    if not path.is_file():
        raise SpecGateError(
            f"{label} reported a successful save but {path} does not exist; the slots "
            f"directory given here is not the one the server writes to, so the noise "
            f"control cannot be built from the file that was actually saved")
    return path


def require_checkpoint_saved(saved: dict, label: str) -> None:
    """The artifact must actually carry checkpoint state.

    Without this the run degenerates into a sequence-state comparison and reports it under
    a name that claims otherwise. On a build with no checkpoint persistence both instances
    would save a plain ggsq body, both would agree perfectly, and the verdict would read
    "proven" about a state class that was never in the file.
    """
    count = saved.get("n_checkpoints_saved")
    if count is None:
        raise SpecGateError(
            f"{label} did not report n_checkpoints_saved, so whether the artifact "
            f"carries any checkpoint state is unknown; this runner will not assume it")
    if int(count) <= 0:
        raise SpecGateError(
            f"{label} saved {count} checkpoints; the artifact is a plain sequence state "
            f"and a verdict from it would say nothing about checkpoint state of any class")


# --- measurement ---------------------------------------------------------------------

def preflight_instance(endpoint: Endpoint, prompt: str, *, slot: int, predict: int,
                       runs: int) -> dict:
    """Cold determinism runs against a live URL, and the cold reference they leave behind.

    `reader_determinism.cold_run` erases, displaces the slot with unrelated text and erases
    again, because erasing alone left the previous prefix reusable. Reused verbatim.
    """
    samples = [cold_run(endpoint, prompt, slot, predict) for _ in range(runs)]
    summary = require_live_determinism(samples, endpoint.label)
    return {"determinism": summary, "runs": samples, "cold": samples[-1]}


def restore_leg(endpoint: Endpoint, filename: str, *, prompt: str, prompt_ids: list[int],
                continuation: list[int], slot: int, predict: int, cold: dict) -> dict:
    """Restore one artifact, then score it two ways: free generation and teacher forced.

    Free generation supplies cache_n and wall clock - the two numbers §34 showed must be
    read together, because a Gemma run reused all 578 of its tokens and was slower than
    prefilling them. Teacher forcing supplies the agreement, because free generation past a
    few tokens measures the cascade instead of the cache.

    The slot is erased and the artifact restored a second time before the teacher-forced
    pass: the free-generation completion above has already advanced the slot.
    """
    endpoint.post(f"/slots/{slot}?action=erase", {})
    try:
        restored = endpoint.post(f"/slots/{slot}?action=restore", {"filename": filename})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:400]
        return {"restore_refused": body, "cache_n": None, "prompt_n": None,
                "seconds": None, "prompt_ms": None, "text": None,
                "text_matches_cold": False, "forced_vs_cold": None, "restored": None}
    started = time.time()
    response = run_completion(endpoint, prompt, slot, predict)
    seconds = round(time.time() - started, 3)
    timings = response["timings"]
    endpoint.post(f"/slots/{slot}?action=erase", {})
    endpoint.post(f"/slots/{slot}?action=restore", {"filename": filename})
    forced = teacher_forced(endpoint, prompt_ids, continuation, slot)
    return {
        "restore_refused": None,
        "restored": {k: restored.get(k) for k in ("n_restored", "n_read")},
        "cache_n": timings.get("cache_n"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "seconds": seconds,
        "text": response["content"],
        "text_sha256": digest_text(response["content"]),
        "text_matches_cold": response["content"] == cold["text"],
        "forced": forced,
    }


def measure_instance(endpoint: Endpoint, *, prompt: str, prompt_ids: list[int],
                     continuation: list[int], slot: int, predict: int,
                     presave_predict: int, slots: Path, arch: str, cold: dict) -> dict:
    """Cold, identity and noise legs for one instance, under one protocol.

    The identity artifact is saved after a short completion rather than after the full
    `--predict` run, so that both instances save a cache of the same length; §28 recorded
    what happens otherwise - the control differed from the subject in length as well as in
    the variable under test and isolated nothing. The length is checked between instances
    by `require_matched_caches`, not assumed from the flag.
    """
    endpoint.post(f"/slots/{slot}?action=erase", {})
    cold_forced = teacher_forced(endpoint, prompt_ids, continuation, slot)

    endpoint.post(f"/slots/{slot}?action=erase", {})
    run_completion(endpoint, prompt, slot, presave_predict)
    identity_name = f"{endpoint.label}-identity.state"
    saved = endpoint.post(f"/slots/{slot}?action=save", {"filename": identity_name})
    require_checkpoint_saved(saved, endpoint.label)
    state = require_saved_state(slots / identity_name, endpoint.label)

    # The scrambler replaces the attention section's value spans and nothing else, so the
    # token header, the per-layer type ids and row sizes, and any SCKP appendix survive.
    # That is the conservative direction: an intact appendix can only make the noise leg
    # look MORE correct, and a noise leg that looks correct refuses the run rather than
    # passing it.
    noise_name = f"{endpoint.label}-noise.state"
    noise_info = scramble_payload(state, slots / noise_name,
                                  has_cell_ext=writes_cell_ext(arch))

    legs = {"cold": {"cache_n": cold["cache_n"], "prompt_n": cold["prompt_n"],
                     "seconds": cold["seconds"], "text": cold["text"],
                     "text_sha256": cold["text_sha256"], "forced": cold_forced}}
    for name, filename in (("identity", identity_name), ("noise", noise_name)):
        leg = restore_leg(endpoint, filename, prompt=prompt, prompt_ids=prompt_ids,
                          continuation=continuation, slot=slot, predict=predict,
                          cold=cold)
        if leg.get("forced") is not None:
            leg["forced_vs_cold"] = compare_forced(leg["forced"], cold_forced)
        legs[name] = leg
    return {"label": endpoint.label, "saved": saved, "noise": noise_info, "legs": legs}


def assemble(spec: dict, nospec: dict) -> dict:
    """Everything the verdict reads, derived from the two instances' legs.

    Kept separate from the live measurement so the verdict can be exercised on fixtures.
    """
    spec_legs, nospec_legs = spec["legs"], nospec["legs"]
    spec_identity_forced = spec_legs["identity"].get("forced")
    nospec_cold_forced = nospec_legs["cold"]["forced"]
    cross = {}
    if spec_identity_forced is not None:
        cross["spec_restored_vs_nospec_cold"] = compare_forced(spec_identity_forced,
                                                               nospec_cold_forced)
    cross["spec_cold_vs_nospec_cold"] = compare_forced(spec_legs["cold"]["forced"],
                                                       nospec_cold_forced)
    strip = ("cache_n", "prompt_n", "prompt_ms", "seconds", "restore_refused",
             "text_matches_cold", "forced_vs_cold", "restored")
    return {
        "spec": {name: {k: leg.get(k) for k in strip} for name, leg in spec_legs.items()},
        "nospec": {name: {k: leg.get(k) for k in strip}
                   for name, leg in nospec_legs.items()},
        "cross": cross,
    }


def wall_clock(measured: dict) -> dict:
    """cache_n next to seconds, for both instances. Reported, never a verdict input.

    §34: reuse is not speedup. The same Gemma reused every one of 578 tokens and took 417ms
    against a 383ms cold prefill, then gained 3.03x at 7,363. A runner that reported only
    cache_n would have recorded the first of those as a success.
    """
    out = {}
    for instance in ("spec", "nospec"):
        legs = measured.get(instance, {})
        cold_seconds = (legs.get("cold") or {}).get("seconds")
        restored = (legs.get("identity") or {}).get("seconds")
        speedup = None
        if cold_seconds and restored:
            speedup = round(cold_seconds / restored, 3)
        out[instance] = {
            "cold_seconds": cold_seconds,
            "restored_seconds": restored,
            "cold_prompt_ms": (legs.get("cold") or {}).get("prompt_ms"),
            "restored_prompt_ms": (legs.get("identity") or {}).get("prompt_ms"),
            "restored_cache_n": (legs.get("identity") or {}).get("cache_n"),
            "speedup": speedup,
            "restore_was_slower": bool(speedup is not None and speedup < 1.0),
        }
    return out


def _top1(leg: dict | None, key: str = "forced_vs_cold"):
    return ((leg or {}).get(key) or {}).get("top1_agreement")


def _positions(comparison: dict | None):
    return (comparison or {}).get("positions")


def verdict(measured: dict, *, min_top1: float, control_min_top1: float,
            noise_max_top1: float) -> dict:
    """Render proven, refuted, or nothing at all.

    The subject is judged against measured baselines rather than against constants. An
    absolute threshold is reported alongside, and flagged when it sits above the baseline,
    because §28 shipped a 0.99 threshold that a model reading its OWN cache could not reach
    - so a failure against it said nothing about the cache.
    """
    spec = measured.get("spec", {})
    nospec = measured.get("nospec", {})
    cross = measured.get("cross", {})
    problems: list[str] = []

    for instance, legs in (("spec", spec), ("nospec", nospec)):
        for leg in ("cold", "identity", "noise"):
            if leg not in legs:
                problems.append(f"{instance} instance has no {leg} leg")
    for instance, legs in (("spec", spec), ("nospec", nospec)):
        for leg in ("identity", "noise"):
            refusal = (legs.get(leg) or {}).get("restore_refused")
            if refusal:
                problems.append(f"{instance} {leg} restore was refused ({refusal[:120]}); "
                                f"a refusal is not evidence about correctness")

    # The control instance: same model, same prompt, same protocol, target-only launch.
    control = nospec.get("identity") or {}
    if control.get("cache_n") in (0, None):
        problems.append("the control instance reused nothing restoring its own cache, so "
                        "the harness itself is not restoring anything here and a result "
                        "on the speculative instance could not be attributed")
    control_top1 = _top1(control)
    if control_top1 is None:
        problems.append("the control instance produced no teacher-forced comparison")
    elif control_top1 < control_min_top1:
        problems.append(
            f"the control instance restoring its own cache reproduced only "
            f"{control_top1:.3f} of its cold prefill, below the required "
            f"{control_min_top1}; with the proven target-only case failing there is no "
            f"baseline against which a speculative result would mean anything")

    for instance, legs in (("spec", spec), ("nospec", nospec)):
        noise = legs.get("noise") or {}
        if noise.get("cache_n") in (0, None):
            problems.append(f"the {instance} noise control reused nothing, so it bounds "
                            f"nothing and every leg on that instance may have been a cold "
                            f"prefill")
        noise_top1 = _top1(noise)
        if noise_top1 is None:
            problems.append(f"the {instance} noise control produced no teacher-forced "
                            f"comparison")
        elif noise_top1 > noise_max_top1:
            problems.append(
                f"the {instance} noise control agreed with its own cold prefill on "
                f"{noise_top1:.3f} of positions with its values scrambled; the payload was "
                f"not being read and the floor is not a floor")

    subject = spec.get("identity") or {}
    if subject.get("cache_n") in (0, None):
        problems.append("the speculative instance reused nothing from its restored "
                        "checkpoint; there is no restored cache to judge, which is neither "
                        "proof nor refutation")

    # Every leg must have been scored over the same positions. compare_forced truncates to
    # the shorter of the two score sets and reports the count, so a leg that came back
    # short would otherwise be compared on a prefix and reported as if it were not.
    counted = {_positions(legs.get(leg, {}).get("forced_vs_cold"))
               for legs in (spec, nospec) for leg in ("identity", "noise")}
    counted |= {_positions(comparison) for comparison in cross.values()}
    counted.discard(None)
    if len(counted) > 1:
        problems.append(f"the legs were scored over different numbers of positions "
                        f"({sorted(counted)}); they are not comparable")
    if counted and min(counted) < MIN_FORCED_POSITIONS:
        problems.append(f"the legs were scored over {min(counted)} positions, below the "
                        f"{MIN_FORCED_POSITIONS} minimum")

    within = _top1(subject)
    across = (cross.get("spec_restored_vs_nospec_cold") or {}).get("top1_agreement")
    cross_baseline = (cross.get("spec_cold_vs_nospec_cold") or {}).get("top1_agreement")
    if within is None or across is None or cross_baseline is None:
        problems.append("a headline comparison is missing, so nothing can be concluded")

    result = {
        "controls_ok": not problems,
        "problems": problems,
        "spec_reused": subject.get("cache_n"),
        # The headline pair.
        "spec_top1_vs_own_cold": within,
        "spec_top1_vs_nospec_cold": across,
        # The baselines they are judged against, both measured in this run.
        "control_top1_vs_own_cold": control_top1,
        "cold_top1_spec_vs_nospec": cross_baseline,
        "noise_top1_spec": _top1(spec.get("noise")),
        "noise_top1_nospec": _top1(nospec.get("noise")),
        "spec_text_matches_own_cold": subject.get("text_matches_cold"),
        "min_top1": min_top1,
        "wall_clock": wall_clock(measured),
    }
    if problems:
        result.update({"at_or_above_within_baseline": None,
                       "at_or_above_cross_baseline": None,
                       "meets_threshold": None,
                       "threshold_exceeds_baseline": None,
                       "speculative_restore": None})
        return result

    at_within = within >= control_top1
    at_cross = across >= cross_baseline
    result.update({
        "at_or_above_within_baseline": at_within,
        "at_or_above_cross_baseline": at_cross,
        "meets_threshold": within >= min_top1 and across >= min_top1,
        # An absolute threshold above the baseline is unreachable for reasons that have
        # nothing to do with the speculative state, and is reported as such rather than
        # read as a failure.
        "threshold_exceeds_baseline": min_top1 > min(control_top1, cross_baseline),
        # Reported, never auto-promoted: a passing run is a research result about one
        # model at one length, not an entry in proven_state_classes.
        "speculative_restore": "proven" if (at_within and at_cross) else "refuted",
    })
    return result


# --- driver ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="the GGUF BOTH servers must be serving")
    ap.add_argument("--spec-url", required=True,
                    help="an already-running server with speculative decoding ON")
    ap.add_argument("--spec-slots", required=True, help="that server's --slot-save-path")
    ap.add_argument("--spec-label", required=True,
                    help="names the speculative reader configuration")
    ap.add_argument("--spec-determinism-record", required=True,
                    help="a passing scripts/reader_determinism.py record for it")
    ap.add_argument("--nospec-url", required=True,
                    help="an already-running server with speculative decoding OFF")
    ap.add_argument("--nospec-slots", required=True, help="that server's --slot-save-path")
    ap.add_argument("--nospec-label", required=True)
    ap.add_argument("--nospec-determinism-record", required=True)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--predict", type=int, default=32)
    ap.add_argument("--presave-predict", type=int, default=8,
                    help="tokens generated before the cache is saved; more than one so "
                         "the draft path is actually exercised before the checkpoint")
    ap.add_argument("--forced-positions", type=int, default=48)
    ap.add_argument("--determinism-runs", type=int, default=MIN_RUNS)
    ap.add_argument("--prompt-repeat", type=int, default=220)
    ap.add_argument("--min-top1", type=float, default=0.99)
    ap.add_argument("--control-min-top1", type=float, default=1.0,
                    help="what the target-only instance must reach restoring its own cache")
    ap.add_argument("--noise-max-top1", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    require_forced_positions(args.forced_positions)
    require_determinism_runs(args.determinism_runs)
    require_distinct_endpoints(args.spec_url, args.nospec_url)

    model = Path(args.model)
    arch = architecture_of(str(model))

    spec = Endpoint(args.spec_url, args.spec_label)
    nospec = Endpoint(args.nospec_url, args.nospec_label)
    spec_props, nospec_props = spec.get("/props"), nospec.get("/props")

    require_serves_model(spec_props, spec.url, model)
    require_serves_model(nospec_props, nospec.url, model)
    require_speculative_active(active_state_classes(spec_props, spec.url), spec.url)
    require_target_only(active_state_classes(nospec_props, nospec.url), nospec.url)

    spec_proof = load_reproducible(args.spec_determinism_record, args.spec_label)
    require_record_matches_live(spec_proof, spec_props, args.spec_label)
    nospec_proof = load_reproducible(args.nospec_determinism_record, args.nospec_label)
    require_record_matches_live(nospec_proof, nospec_props, args.nospec_label)

    prompt = ("You are a meticulous systems engineer working on a portable KV cache "
              "format. " * args.prompt_repeat) + "\nState the single most important invariant."

    require_identical_tokenisation(tokenize(spec, prompt), tokenize(nospec, prompt))
    prompt_ids = tokenize(nospec, prompt)

    nospec_pre = preflight_instance(nospec, prompt, slot=args.slot, predict=args.predict,
                                    runs=args.determinism_runs)
    spec_pre = preflight_instance(spec, prompt, slot=args.slot, predict=args.predict,
                                  runs=args.determinism_runs)
    # One continuation, taken from the control's cold run, scored at the same positions on
    # every leg of both instances.
    continuation = require_continuation(nospec_pre["cold"]["token_ids"],
                                        args.forced_positions)

    common = dict(prompt=prompt, prompt_ids=prompt_ids, continuation=continuation,
                  slot=args.slot, predict=args.predict,
                  presave_predict=args.presave_predict, arch=arch)
    nospec_measured = measure_instance(nospec, slots=Path(args.nospec_slots),
                                       cold=nospec_pre["cold"], **common)
    spec_measured = measure_instance(spec, slots=Path(args.spec_slots),
                                     cold=spec_pre["cold"], **common)
    require_matched_caches(spec_measured["saved"], nospec_measured["saved"])

    measured = assemble(spec_measured, nospec_measured)
    record = {
        "model": str(model), "model_sha256": sha256_file(model), "architecture": arch,
        "prompt_sha256": digest_text(prompt), "prompt_chars": len(prompt),
        "forced_positions": len(continuation),
        "instances": {
            "spec": {"url": spec.url, "label": spec.label, "slots": args.spec_slots,
                     "props": spec_props, "determinism": spec_pre["determinism"],
                     "determinism_record": args.spec_determinism_record,
                     "saved": spec_measured["saved"], "noise": spec_measured["noise"]},
            "nospec": {"url": nospec.url, "label": nospec.label,
                       "slots": args.nospec_slots, "props": nospec_props,
                       "determinism": nospec_pre["determinism"],
                       "determinism_record": args.nospec_determinism_record,
                       "saved": nospec_measured["saved"],
                       "noise": nospec_measured["noise"]},
        },
        "cold_runs": {"spec": spec_pre["runs"], "nospec": nospec_pre["runs"]},
        "legs": {"spec": spec_measured["legs"], "nospec": nospec_measured["legs"]},
        "measured": measured,
    }
    record["verdict"] = verdict(measured, min_top1=args.min_top1,
                                control_min_top1=args.control_min_top1,
                                noise_max_top1=args.noise_max_top1)
    Path(args.out).write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record["verdict"], indent=1))
    print(f"wrote {args.out}")
    if not record["verdict"]["controls_ok"]:
        return 2
    return 0 if record["verdict"]["speculative_restore"] == "proven" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SpecGateError, RuntimeError) as exc:
        # A refusal is an outcome, not a crash, and it exits 2 - the same code as a run
        # whose controls misbehaved, because both mean "no verdict".
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
