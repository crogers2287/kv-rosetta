#!/usr/bin/env python3
"""Can one model read a cache another model wrote? RESEARCH ONLY.

The rejected result in research-findings §20 was cross-*geometry*: 16 layers and 4 KV heads
into 10 layers and 2 KV heads, which needs a learned translation and failed. It says nothing
about two models whose cache rows are already the same shape, where no translation is
involved and llama.cpp will simply accept the file - because the state format records shape,
not model identity.

That case is what this runner measures. It is the honest form of "model agnostic": not one
file for every model, but one file for every model that lays its cache out identically.

Three controls, all mandatory, because this measurement can lie in both directions:

  identity  the target restoring its OWN cache. Must reuse and reproduce the native output.
            If it does not, the harness is broken and the foreign number means nothing.
  noise     the foreign file with its payload scrambled and its token header intact, so it
            restores and is reused but carries garbage. Must NOT reproduce the native
            output. If noise "passes", nothing was actually being restored and every leg
            was a cold prefill - the exact failure that invalidated two runs in §20.
  native    the target with no cache at all. The reference every leg is scored against.

A foreign leg is reported, never admitted, unless both controls behaved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_backend_gate import logprob_divergence  # noqa: E402
from production_matrix import N_PROBS, binary_digests, free_port, probs, sha256_file, toks  # noqa: E402
from reader_determinism import Reader, digest_text  # noqa: E402

from kv_rosetta import gguf  # noqa: E402
from kv_rosetta.adapters import ggsq_envelope, llamacpp_ggsq  # noqa: E402
from kv_rosetta.sizing import BYTES_PER_CELL_EXT, writes_cell_ext  # noqa: E402

GGSQ_MAGIC = b"qsgg"
PREAMBLE = 12


class GateError(RuntimeError):
    """A refusal or an invalid run. Never downgraded to a warning."""


def geometry_of(path: str) -> dict:
    """The fields that decide whether two caches are the same shape."""
    md = gguf.read_metadata(path)
    arch = md.get("general.architecture")
    if not arch:
        raise GateError(f"{path} declares no architecture")
    def g(key):
        return md.get(f"{arch}.{key}")

    # Many architectures omit attention.key_length and derive head_dim from
    # embedding_length / head_count instead. Left as None those fields compare None to
    # None, so two models with different head dimensions would pass the geometry check on
    # a pair of nulls - the same vacuous comparison this project keeps finding. Derived
    # when absent, and the derivation is recorded so the two cases stay distinguishable.
    key_length, value_length = g("attention.key_length"), g("attention.value_length")
    derived = False
    embd, n_head = g("embedding_length"), g("attention.head_count")
    if key_length is None and embd and n_head:
        if embd % n_head:
            raise GateError(
                f"{path}: embedding_length {embd} is not divisible by head_count "
                f"{n_head}, so head_dim cannot be derived and is not declared")
        key_length = value_length = embd // n_head
        derived = True
    if key_length is None:
        raise GateError(f"{path} declares no head dimension and none can be derived")
    return {
        "arch": arch,
        "n_layer": g("block_count"),
        "n_head_kv": g("attention.head_count_kv"),
        "key_length": key_length,
        "value_length": value_length,
        "head_dim_derived": derived,
        "rope_freq_base": g("rope.freq_base"),
    }


def require_same_geometry(a: dict, b: dict) -> None:
    """Refuse a pair whose caches are not the same shape.

    Not a safety check on the server - llama.cpp refuses those itself - but a check on the
    claim. A mismatched pair is the §20 experiment, already run and already answered, and
    running it again under this name would confuse two different questions.
    """
    differing = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    if differing:
        raise GateError(
            f"the two models do not share a KV geometry: {differing}. That is the "
            f"cross-geometry case, which needs a learned mapping and is answered in "
            f"research-findings §20; this runner measures the same-geometry case.")


def scramble_payload(src: Path, dest: Path, *, has_cell_ext: bool = False,
                     cell_ext_size: int = BYTES_PER_CELL_EXT) -> dict:
    """Copy a state file with every structural field intact and only the tensor data destroyed.

    A first version overwrote everything after the token header, which also destroyed the
    per-layer type ids and row sizes, and the server rejected the file with a 400. A
    rejected noise control bounds nothing: from outside it is indistinguishable from a
    restore that was never attempted, so the run had no floor at all.

    The spans come from the repo's own GGSQ parser rather than from arithmetic here, so the
    bytes replaced are exactly the ones the server will read as cache values.
    """
    raw = bytearray(src.read_bytes())
    if bytes(raw[:4]) != GGSQ_MAGIC:
        raise GateError(f"{src} is not a GGSQ state file")
    # A body this parser cannot read is one whose value bytes cannot be located, so the
    # copy would be byte-identical to the foreign file and the noise leg would silently
    # become a second foreign leg. Refused as a gate error rather than leaking the parser's.
    try:
        envelope = ggsq_envelope.parse_file_envelope(bytes(raw))
        with open(src, "rb") as handle:
            # cell_ext_size must accompany has_cell_ext. The parser defaults it to 0, so
            # passing the flag alone silently reproduces the has_cell_ext=False parse and
            # desynchronises one cell in -- which surfaces as "cell 1 claims 2523 sequence
            # ids" rather than as a missing argument.
            section = llamacpp_ggsq.read_attention_section(
                handle, envelope.body_offset, len(raw), has_cell_ext=has_cell_ext,
                cell_ext_size=cell_ext_size if has_cell_ext else 0)
    except ValueError as exc:
        raise GateError(f"cannot locate the cache values in {src}: {exc}") from exc
    scrambled = 0
    for span in section.spans:
        memoryview(raw)[span.offset:span.offset + span.nbytes] = os.urandom(span.nbytes)
        scrambled += span.nbytes
    dest.write_bytes(bytes(raw))
    return {"tokens_preserved": len(envelope.token_ids),
            "spans_scrambled": len(section.spans),
            "payload_bytes_scrambled": scrambled,
            "structural_bytes_preserved": len(raw) - scrambled}


def require_matched_caches(saved_a: dict, saved_b: dict) -> None:
    """The identity cache must hold the same number of cells as the foreign one.

    An earlier version saved the identity cache after the 32-token native run and the
    foreign cache after a one-token run, so the control carried 31 extra cells the subject
    did not. It then differed from the subject in length as well as authorship and
    isolated nothing. (Correcting it left the measured numbers unchanged, so the flaw was
    real but not what was driving the result - recorded because the reverse would have been
    easy to assume.)
    """
    if saved_a.get("n_saved") != saved_b.get("n_saved"):
        raise GateError(
            f"identity cache holds {saved_b.get('n_saved')} cells against the foreign "
            f"file's {saved_a.get('n_saved')}; the control differs from the subject in "
            f"length as well as authorship and would not isolate anything")


def tokenize(reader: Reader, text: str) -> list[int]:
    return reader.post("/tokenize", {"content": text})["tokens"]


def teacher_forced(reader: Reader, prompt_ids: list[int], continuation: list[int],
                   slot: int) -> list[dict]:
    """Score each continuation position against the SAME prefix for every leg.

    Free generation is a cliff: one divergent token changes the next input and everything
    after it, so a free-generation comparison past a few tokens measures the cascade rather
    than the cache. Measured here rather than argued - over 128 freely generated tokens
    even a model restoring its OWN cache agreed with its own cold prefill on 0.23 of
    positions, which is not a fact about the cache at all.

    Teacher forcing feeds the identical prefix at every position, so a disagreement at
    position i cannot contaminate position i+1. This is the protocol gate.py already
    declares as its default, for this reason.
    """
    scores = []
    for index in range(len(continuation)):
        response = reader.post("/completion", {
            "prompt": prompt_ids + continuation[:index],
            "n_predict": 1, "temperature": 0.0, "seed": 1,
            "cache_prompt": True, "id_slot": slot, "n_probs": N_PROBS})
        vectors = probs(response)
        if not vectors:
            raise GateError(f"no probability vector at teacher-forced position {index}")
        scores.append(vectors[0])
    return scores


def compare_forced(left: list[dict], right: list[dict]) -> dict:
    """Position-by-position agreement between two teacher-forced score sets."""
    compared = min(len(left), len(right))
    if compared == 0:
        return {"positions": 0, "top1_agreement": None, "max_abs_logprob_delta": None}
    agreed, worst, shared, only_one, total = 0, 0.0, 0, 0, 0.0
    for a, b in zip(left[:compared], right[:compared]):
        if not a or not b:
            continue
        if max(a, key=a.get) == max(b, key=b.get):
            agreed += 1
        for token in set(a) | set(b):
            if token in a and token in b:
                delta = abs(a[token] - b[token])
                worst = max(worst, delta)
                total += delta
                shared += 1
            else:
                only_one += 1
    # The mean alongside the max on purpose. Top-1 agreement over a few dozen positions
    # moves in steps of 1/positions and reads as noise; the max is one worst-case token and
    # is nearly as jumpy. The mean over every shared alternative is the smooth quantity, and
    # a sweep needs one of those to show a trend at all.
    return {"positions": compared, "top1_agreement": agreed / compared,
            "max_abs_logprob_delta": worst,
            "mean_abs_logprob_delta": (total / shared) if shared else None,
            "shared_tokens": shared, "tokens_only_in_one": only_one}


def run_completion(reader: Reader, prompt: str, slot: int, predict: int) -> dict:
    r = reader.post("/completion", {
        "prompt": prompt, "n_predict": predict, "temperature": 0.0, "seed": 1,
        "cache_prompt": True, "id_slot": slot, "n_probs": N_PROBS})
    return r


def summarise(response, native) -> dict:
    t = response["timings"]
    out = {
        "cache_n": t.get("cache_n"),
        "prompt_n": t.get("prompt_n"),
        "text": response["content"],
        "text_sha256": digest_text(response["content"]),
        "tokens_match_native": toks(response) == toks(native),
        "text_matches_native": response["content"] == native["content"],
    }
    out["divergence_vs_native"] = logprob_divergence(response, native)
    return out


def verdict(legs: dict, *, min_top1: float) -> dict:
    """Score the foreign cache against the target's OWN restore, not against a cold prefill.

    An earlier version required the identity control to reproduce the cold-prefill output
    exactly and failed a run because it did not. That was the wrong question: restoring a
    cache and prefilling the prompt are different computations, and on this host they
    already disagree on 3 of 32 tokens for a model reading its own cache. Comparing a
    foreign cache to a cold prefill therefore charges it for a difference that has nothing
    to do with whose cache it is.

    The apples-to-apples comparison is foreign-restore against own-restore - both restores,
    differing only in which model wrote the bytes. That is the row §17 found meaningful for
    cross-backend transfer, and the same reasoning applies here.

    Controls still gate everything: the identity leg must actually reuse, and the noise leg
    must actually reuse AND come out wrong. Noise that looks right means nothing was
    restored in any leg.
    """
    identity, noise, foreign = legs["identity"], legs["noise"], legs["foreign"]
    problems = []
    if identity["cache_n"] in (0, None):
        problems.append("identity control reused nothing, so no leg restored anything")
    if noise["cache_n"] in (0, None):
        problems.append("noise control reused nothing, so it does not bound anything")
    if noise["text_matches_native"]:
        problems.append("noise control reproduced the native output, so the payload was "
                        "not being used and every leg was effectively a cold prefill")
    noise_top1 = (noise.get("divergence_vs_identity") or {}).get("top1_agreement")
    if noise_top1 is not None and noise_top1 > 0.5:
        problems.append(f"noise control agreed with the target's own restore on "
                        f"{noise_top1:.2f} of tokens, so scrambled values are not "
                        f"distinguishable here and the floor is not a floor")

    noise_forced = (noise.get("forced_vs_identity") or {}).get("top1_agreement")
    if noise_forced is not None and noise_forced > 0.5:
        problems.append(f"noise agreed with the target's own restore on {noise_forced:.2f} "
                        f"of teacher-forced positions; the floor is not a floor")
    vs_own = foreign.get("forced_vs_identity") or foreign.get("divergence_vs_identity") or {}
    top1 = vs_own.get("top1_agreement")
    # The identity leg against a cold prefill is this machine's own reproducibility floor:
    # the same model, the same weights, the same prompt, differing only by restore-versus-
    # prefill arithmetic. Measured at 0.969-0.977 over 128 teacher-forced positions, which
    # is BELOW the 0.99 absolute threshold this gate shipped with - so that threshold was
    # unreachable even for a model reading its own cache, and a foreign result could never
    # have passed it for reasons having nothing to do with the cache. Reported alongside so
    # an absolute number is never read without it.
    baseline = (identity.get("forced_vs_native") or {}).get("top1_agreement")
    return {
        "controls_ok": not problems,
        "problems": problems,
        "foreign_reused": foreign["cache_n"],
        # The headline: foreign cache vs the target reading its own cache.
        "foreign_top1_vs_own_restore": top1,
        "foreign_max_delta_vs_own_restore": vs_own.get("max_abs_logprob_delta"),
        "foreign_text_matches_own_restore": foreign.get("text_matches_identity"),
        # Context, not the verdict.
        "identity_forced_top1_vs_native": (identity.get("forced_vs_native") or {}).get("top1_agreement"),
        "noise_forced_top1_vs_own": noise_forced,
        "identity_free_top1_vs_native": (identity["divergence_vs_native"] or {}).get("top1_agreement"),
        "noise_top1_vs_own_restore": noise_top1,
        "min_top1": min_top1,
        "baseline_top1": baseline,
        # Whether the foreign cache is as good as this machine can do at all. The absolute
        # threshold below is kept, but it is meaningless above the baseline.
        "at_or_above_baseline": bool(not problems and top1 is not None
                                     and baseline is not None and top1 >= baseline),
        "threshold_exceeds_baseline": bool(baseline is not None and min_top1 > baseline),
        # Reported, never auto-promoted: a passing number is a research result about one
        # pair at one length, not an allowlist entry.
        "meets_threshold": bool(not problems and top1 is not None and top1 >= min_top1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model-a", required=True, help="the model that WRITES the cache")
    ap.add_argument("--model-b", required=True, help="the model that READS it")
    ap.add_argument("--slots", required=True)
    ap.add_argument("--args-a", default="")
    ap.add_argument("--args-b", default="")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--predict", type=int, default=32)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--prompt-repeat", type=int, default=220)
    ap.add_argument("--min-top1", type=float, default=0.99)
    ap.add_argument("--forced-positions", type=int, default=48,
                    help="teacher-forced positions scored per leg")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    geo_a, geo_b = geometry_of(args.model_a), geometry_of(args.model_b)
    require_same_geometry(geo_a, geo_b)

    prompt = ("You are a meticulous systems engineer working on a portable KV cache "
              "format. " * args.prompt_repeat) + "\nState the single most important invariant."
    slots = Path(args.slots)
    slots.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)

    # --- writer -----------------------------------------------------------------------
    a = Reader(args.binary, args.model_a, free_port(), out.with_suffix(".a.log"),
               args.args_a.split(), args.n_ctx, str(slots))
    a.start()
    try:
        a.post(f"/slots/{args.slot}?action=erase", {})
        run_completion(a, prompt, args.slot, 1)
        saved_a = a.post(f"/slots/{args.slot}?action=save", {"filename": "from-a.state"})
    finally:
        a.stop()

    # has_cell_ext is an architecture property and is not discoverable from the file;
    # guessing it desynchronises the parse and the scrambled offsets would land on
    # structural fields again.
    noise_info = scramble_payload(slots / "from-a.state", slots / "noise.state",
                                  has_cell_ext=writes_cell_ext(geo_a["arch"]))

    # --- reader -----------------------------------------------------------------------
    b = Reader(args.binary, args.model_b, free_port(), out.with_suffix(".b.log"),
               args.args_b.split(), args.n_ctx, str(slots))
    b.start()
    legs = {}
    try:
        responses, forced = {}, {}
        b.post(f"/slots/{args.slot}?action=erase", {})
        native = run_completion(b, prompt, args.slot, args.predict)
        prompt_ids = tokenize(b, prompt)
        # The continuation is fixed once, from the target's own uncached run, and every leg
        # is scored against those same positions.
        continuation = toks(native)[:args.forced_positions]
        b.post(f"/slots/{args.slot}?action=erase", {})
        native_forced = teacher_forced(b, prompt_ids, continuation, args.slot)

        # The identity cache must be saved the same way the foreign one was: after a
        # one-token completion, not after the 32-token native run. Saving it from the
        # native run leaves 31 generated tokens in the slot that the foreign file does not
        # have, so the two restores differ in cache length as well as in authorship - and
        # the control stops controlling for anything. Caught when the FOREIGN leg scored
        # better against the cold prefill than the identity leg did.
        b.post(f"/slots/{args.slot}?action=erase", {})
        run_completion(b, prompt, args.slot, 1)
        saved_b = b.post(f"/slots/{args.slot}?action=save", {"filename": "from-b.state"})
        require_matched_caches(saved_a, saved_b)

        for name, filename in (("identity", "from-b.state"),
                               ("foreign", "from-a.state"),
                               ("noise", "noise.state")):
            b.post(f"/slots/{args.slot}?action=erase", {})
            try:
                restored = b.post(f"/slots/{args.slot}?action=restore", {"filename": filename})
                refused = None
            except Exception as exc:
                body = exc.read().decode()[:400] if hasattr(exc, "read") else str(exc)
                legs[name] = {"refused": body, "cache_n": None, "prompt_n": None,
                              "text_matches_native": False, "tokens_match_native": False,
                              "divergence_vs_native": None}
                continue
            response = run_completion(b, prompt, args.slot, args.predict)
            responses[name] = response
            leg = {"restored": {k: restored.get(k) for k in ("n_restored", "n_read")},
                   "refused": refused, **summarise(response, native)}
            # Teacher-forced pass on the same restored cache: re-restore first, because the
            # free-generation completion above already advanced the slot.
            b.post(f"/slots/{args.slot}?action=erase", {})
            b.post(f"/slots/{args.slot}?action=restore", {"filename": filename})
            forced[name] = teacher_forced(b, prompt_ids, continuation, args.slot)
            legs[name] = leg
        # Both restores, differing only in who wrote the bytes.
        own = responses.get("identity")
        if own is not None:
            for name in ("foreign", "noise"):
                if name in responses:
                    legs[name]["divergence_vs_identity"] = logprob_divergence(
                        responses[name], own)
                    legs[name]["text_matches_identity"] = (
                        responses[name]["content"] == own["content"])
        for name, scores in forced.items():
            legs[name]["forced_vs_native"] = compare_forced(scores, native_forced)
            if "identity" in forced:
                legs[name]["forced_vs_identity"] = compare_forced(scores, forced["identity"])
    finally:
        b.stop()

    record = {
        "binary": args.binary,
        "binary_digests": binary_digests(Path(args.binary)),
        "model_a": args.model_a, "model_a_sha256": sha256_file(Path(args.model_a)),
        "model_b": args.model_b, "model_b_sha256": sha256_file(Path(args.model_b)),
        "geometry": geo_a,
        "prompt_sha256": digest_text(prompt),
        "saved_a": {k: saved_a.get(k) for k in ("n_saved", "n_written")},
        "saved_b": {k: saved_b.get(k) for k in ("n_saved", "n_written")},
        "noise": noise_info,
        "forced_positions": len(continuation),
        "native_forced": native_forced,
        "native": {"text": native["content"], "cache_n": native["timings"].get("cache_n"),
                   "prompt_n": native["timings"].get("prompt_n"),
                   "token_ids": toks(native), "vectors": probs(native)},
        "legs": legs,
    }
    record["verdict"] = verdict(legs, min_top1=args.min_top1)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["verdict"], indent=1))
    print(f"wrote {out}")
    return 0 if record["verdict"]["controls_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
