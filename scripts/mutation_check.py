#!/usr/bin/env python3
"""Disable each guard in a module and require a test to notice. RESEARCH TOOL.

A test that passes whether or not the code under it works is worse than no test, because it
is counted. This project has found eighteen of them that way, and the dominant pattern is
never a missing assertion - it is an earlier check shadowing the one being tested, so the
malformed input never reaches the guard the test names.

Two things this refuses to do:

* Run without a green baseline. If the suite is already failing, every mutation is "caught"
  and the whole report is noise that reads like rigour.
* Trust bytecode. Rewriting one file six times in as many seconds can leave a stale .pyc in
  place, and a mutation that never loaded reports as survived. That produced a false
  "vacuous" here and cost a round of chasing a guard that was fine.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path


def guard_lines(source: str) -> list[tuple[int, str]]:
    """Every `if` that governs a raise, as (0-based line index, text).

    Found by walking the parsed tree rather than by matching text, so an `if` inside a
    string or a comment is not offered as a guard to disable.
    """
    found: dict[int, str] = {}
    lines = source.splitlines()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If):
            continue
        if any(isinstance(inner, ast.Raise) for inner in node.body):
            found[node.lineno - 1] = lines[node.lineno - 1].strip()
    return sorted(found.items())


TIMEOUT = 600


def run(tests: list[str], env: dict) -> subprocess.CompletedProcess | None:
    """Run the suite once. ``None`` means it hung.

    A disabled guard can turn a refusal into a non-terminating loop rather than a wrong
    answer -- container.extract_payload spins forever on a truncated file without its
    ``if not block`` check, because read() keeps returning b"" and the remaining count
    never decreases. Without a timeout the audit itself hangs at that mutation and reports
    nothing at all for the module.
    """
    try:
        return subprocess.run([sys.executable, "-B", "-m", "unittest", *tests],
                              capture_output=True, text=True, env=env, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None


def clear_bytecode(root: Path) -> None:
    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="module whose guards to disable")
    ap.add_argument("--tests", required=True, nargs="+", help="unittest target(s)")
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    source = Path(args.source)
    original = source.read_text()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    clear_bytecode(repo)
    baseline = run(args.tests, env)
    if baseline is None:
        print(f"refusing to run: the unmutated suite did not finish in {TIMEOUT}s.")
        return 2
    if baseline.returncode != 0:
        print("refusing to run: the suite fails before any mutation, so every guard would "
              "report as defended.\n" + baseline.stderr[-2000:])
        return 2

    guards = guard_lines(original)
    if not guards:
        print(f"no guards found in {source}")
        return 2

    lines = original.splitlines()
    survived = []
    try:
        for index, text in guards:
            mutated = list(lines)
            indent = len(mutated[index]) - len(mutated[index].lstrip())
            mutated[index] = " " * indent + "if False:"
            source.write_text("\n".join(mutated) + "\n")
            clear_bytecode(repo)
            result = run(args.tests, env)
            # A hang is a detection: the suite did not pass. Reported separately so it is
            # never read as a clean assertion failure.
            hung = result is None
            caught = hung or result.returncode != 0
            label = "HUNG   " if hung else ("CAUGHT " if caught else "SURVIVED")
            print(f"{label}  L{index + 1}: {text[:72]}")
            if not caught:
                survived.append(f"L{index + 1}: {text}")
    finally:
        source.write_text(original)
        clear_bytecode(repo)

    print(f"\n{len(guards) - len(survived)}/{len(guards)} guards defended by a test")
    for item in survived:
        print(f"  undefended: {item}")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
