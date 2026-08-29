#!/usr/bin/env python3
"""Move one cache file between two compute backends, both directions. RESEARCH ONLY.

The project's central claim is that a cache written on one backend is usable on another.
This tests it directly: save a slot on backend A, restore it on backend B, and require the
prefix to be reused with identical output - then do the reverse.

The runner refuses unless both binaries are built from the same llama.cpp source revision.
That check exists because its absence already produced a false negative: a first attempt
restored a v3 state file into a binary compiled months earlier, which rejected it for the
version rather than the backend, and looked exactly like "cross-backend does not work".
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production_matrix import (  # noqa: E402
    N_PROBS,
    Server,
    binary_digests,
    free_port,
    probs,
    require_clean_worktree,
    sha256_file,
    toks,
)

from kv_rosetta import gguf  # noqa: E402


class BackendServer(Server):
    """A Server whose backend is chosen by launch flags and environment."""

    def __init__(self, *args, ngl: int = 99, env: dict | None = None,
                 extra_args: tuple[str, ...] = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.ngl = ngl
        self.extra_env = env or {}
        self.extra_args = tuple(extra_args)

    def build_argv(self) -> list[str]:
        argv = super().build_argv()
        argv[argv.index("-ngl") + 1] = str(self.ngl)
        for flag in ("--split-mode", "--tensor-split", "-fa"):
            if flag in argv:                       # backend-specific, not portable
                at = argv.index(flag)
                del argv[at:at + 2]
        # Selecting a device is backend-specific: HIP reads HIP_VISIBLE_DEVICES from the
        # environment, Vulkan has no equivalent and needs --device on the command line.
        return argv + list(self.extra_args)

    def _start_once(self) -> int:
        import os
        original = dict(os.environ)
        os.environ.update({k: str(v) for k, v in self.extra_env.items()})
        try:
            return super()._start_once()
        finally:
            os.environ.clear()
            os.environ.update(original)


#: llama.cpp prints "version: 0.3.0-dev (build 151, commit ca3d5a3e1)" to stderr.
COMMIT_RE = re.compile(r"commit\s+([0-9a-f]{7,40})")

#: Git abbreviates to whatever length is unambiguous, so the same commit can print as
#: ca3d5a3 from one build tree and ca3d5a3e1 from another. Comparing the raw strings called
#: two builds of one revision a mismatch and refused a run that should have gone ahead.
REVISION_PREFIX = 7


def parse_revision(text: str) -> str:
    """The commit out of a llama.cpp --version banner, normalised to a fixed prefix.

    Returns "" when there is no commit to read, which the caller must treat as a refusal
    rather than as a match - an unknown revision is the case this guard exists for.
    """
    found = COMMIT_RE.search(text)
    return found.group(1)[:REVISION_PREFIX] if found else ""


def source_revision(binary: Path) -> str:
    """The llama.cpp revision a binary was built from, asked of the binary itself.

    An earlier version scraped `strings` output for a build tag. That read the two builds as
    different revisions when they were the same commit, because the tag is not where the
    commit reliably lives. Asking --version is what the binary itself reports, and a binary
    that cannot run - a HIP build on a host with no ROCm - yields "" and stops the run,
    which is the correct outcome rather than a guess.
    """
    if not binary.is_file():
        return ""
    try:
        done = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return parse_revision(f"{done.stdout}\n{done.stderr}")


def require_one_revision(revisions: dict[str, str]) -> str:
    """Stop unless every binary reports the same, known revision.

    A version mismatch is rejected by llama.cpp for the state-file format, not for the
    backend, and the rejection is indistinguishable from "cross-backend does not work".
    That already happened once here, against a build several months old, and it read as a
    negative result. An unknown revision is refused for the same reason: it cannot be shown
    to match, so it is not allowed to pass as one.
    """
    if not all(revisions.values()):
        raise SystemExit(
            f"refusing to run: could not read a source revision for every binary "
            f"({revisions}). An unreadable revision is not a matching one.")
    if len(set(revisions.values())) != 1:
        raise SystemExit(
            f"refusing to run: the binaries are not from one source revision "
            f"({revisions}). A version mismatch is rejected for the format, not the "
            f"backend, and would look exactly like a cross-backend failure.")
    return next(iter(revisions.values()))


def transfer(name: str, writer: dict, reader: dict, model: str, slots: Path,
             log_dir: Path, prompt_tokens: int) -> dict:
    """Write a cache on one backend, restore it on the other, and require reuse."""
    filename = f"xb-{name}.bin"
    first = BackendServer(writer["binary"], model, str(slots), free_port(),
                          log_dir / f"{name}-write.log", n_ctx=prompt_tokens + 512,
                          ngl=writer["ngl"], env=writer.get("env"),
                          extra_args=writer.get("args", ()))
    first.start()
    try:
        text = "The naturalist recorded the following observations in detail. " * 24
        ids = first.post("/tokenize", {"content": text})["tokens"][:prompt_tokens]
        request = {"prompt": ids, "n_predict": 8, "temperature": 0.0, "top_k": 1,
                   "n_probs": N_PROBS, "cache_prompt": True, "id_slot": 0}
        first.post("/slots/0?action=erase", {})
        native = first.post("/completion", request)
        if native["timings"]["cache_n"] != 0:
            raise RuntimeError(f"{name}: writer reused a cache on a cold run")
        first.post("/slots/0?action=erase", {})
        first.post("/completion", dict(request, n_predict=0))
        saved = first.post("/slots/0?action=save", {"filename": filename})
        artifact = slots / filename
        digest = sha256_file(artifact)
        writer_props = first.props()
    finally:
        first.stop()

    second = BackendServer(reader["binary"], model, str(slots), free_port(),
                           log_dir / f"{name}-read.log", n_ctx=prompt_tokens + 512,
                           ngl=reader["ngl"], env=reader.get("env"),
                           extra_args=reader.get("args", ()))
    second.start()
    try:
        ids2 = second.post("/tokenize", {"content": text})["tokens"][:prompt_tokens]
        if ids2 != ids:
            raise RuntimeError(f"{name}: tokenization differs between backends")
        request2 = dict(request, prompt=ids2)
        second.post("/slots/0?action=erase", {})
        control = second.post("/completion", request2)
        if control["timings"]["cache_n"] != 0:
            raise RuntimeError(f"{name}: reader reused before any restore")
        second.post("/slots/0?action=erase", {})
        restored = second.post("/slots/0?action=restore", {"filename": filename})
        warm = second.post("/completion", request2)
        reader_props = second.props()
    finally:
        second.stop()

    if sha256_file(artifact) != digest:
        raise RuntimeError(f"{name}: the artifact changed between backends")
    timings = warm["timings"]
    return {
        "direction": name,
        "writer": {"binary": writer["binary"], "backend": writer["label"],
                   "build_info": writer_props.get("build_info"),
                   "digests": binary_digests(Path(writer["binary"]))},
        "reader": {"binary": reader["binary"], "backend": reader["label"],
                   "build_info": reader_props.get("build_info"),
                   "digests": binary_digests(Path(reader["binary"]))},
        "artifact_sha256": digest,
        "artifact_bytes": artifact.stat().st_size,
        "prompt_tokens": len(ids),
        "save_response": {k: v for k, v in saved.items() if k != "timings"},
        "restore_response": {k: v for k, v in restored.items() if k != "timings"},
        "control_cache_n": control["timings"]["cache_n"],
        "cache_n": timings["cache_n"],
        "prompt_n": timings["prompt_n"],
        "reused_fraction": timings["cache_n"] / len(ids),
        "content_matches": warm["content"] == native["content"],
        "tokens_match": toks(warm) == toks(native),
        "logprobs_match": probs(warm) == probs(native),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend-a", required=True, help="binary:label:ngl")
    ap.add_argument("--backend-b", required=True)
    ap.add_argument("--env-a", default="")
    ap.add_argument("--env-b", default="")
    ap.add_argument("--args-a", default="", help="extra server flags, space separated")
    ap.add_argument("--args-b", default="")
    ap.add_argument("--slots", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--out", default="bench/cross-backend.json")
    args = ap.parse_args()
    repo_commit = require_clean_worktree()

    def parse(spec: str, env: str) -> dict:
        binary, label, ngl = spec.split(":")
        pairs = dict(p.split("=", 1) for p in env.split(",") if "=" in p)
        return {"binary": binary, "label": label, "ngl": int(ngl), "env": pairs,
                "args": ()}

    a, b = parse(args.backend_a, args.env_a), parse(args.backend_b, args.env_b)
    a["args"], b["args"] = tuple(args.args_a.split()), tuple(args.args_b.split())
    llama_revision = require_one_revision(
        {a["label"]: source_revision(Path(a["binary"])),
         b["label"]: source_revision(Path(b["binary"]))})

    slots = Path(args.slots)
    slots.mkdir(parents=True, exist_ok=True)
    log_dir = slots.parent / "xbackend-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, writer, reader in ((f"{a['label']}_to_{b['label']}", a, b),
                                 (f"{b['label']}_to_{a['label']}", b, a)):
        print(f"=== {name} ===", flush=True)
        found = transfer(name, writer, reader, args.model, slots, log_dir,
                         args.prompt_tokens)
        results.append(found)
        print(f"  reused {found['cache_n']}/{found['prompt_tokens']} "
              f"({found['reused_fraction']*100:.1f}%), content match "
              f"{found['content_matches']}, tokens match {found['tokens_match']}", flush=True)

    record = {
        "kind": "cross-backend-cache-transfer",
        "warning": "RESEARCH ONLY. Same model, same cache dtype, different compute backend.",
        "repo_commit": repo_commit,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "llama_source_revision": llama_revision,
        "model_path": args.model,
        "architecture": gguf.architecture(args.model),
        "prompt_tokens": args.prompt_tokens,
        "directions": results,
        "both_directions_reuse": all(r["cache_n"] > 0 for r in results),
        "all_outputs_match": all(r["content_matches"] and r["tokens_match"]
                                 for r in results),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"  both directions reuse: {record['both_directions_reuse']}")
    print(f"  all outputs match:     {record['all_outputs_match']}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
