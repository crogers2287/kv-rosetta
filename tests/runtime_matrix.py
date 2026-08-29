"""Which llama.cpp runtime a test needs, and how to tell which one is connected.

Hybrid tests come in two mutually exclusive kinds: some assert the UNPATCHED failure (the
negative control), some assert PATCHED behaviour. Run either against the wrong binary and it
fails for a reason that has nothing to do with the code under test - the negative control
"failing" against a patched server is correct behaviour, not a regression.

So every such test declares what it needs and skips otherwise, and the declaration is
checked against evidence from the running binary rather than against a version string, a
filename, or an artifact's size.

Detection is by FORMAT EVIDENCE: the patched save handler appends a payload tagged with the
four-byte magic "SCKP" (SLOT_CKPT_MAGIC = 0x504b4353) after the llama state. Its presence in
a real saved slot file is a direct observation that checkpoints were persisted. Artifact size
is not evidence: a large file only means a large cache.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

#: SLOT_CKPT_MAGIC from the patch, little-endian, which spells the ASCII it was chosen for.
SCKP_MAGIC = (0x504B4353).to_bytes(4, "little")   # b"SCKP"

PATCHED = "patched"
UNPATCHED = "unpatched"
UNKNOWN = "unknown"


def _post(url: str, path: str, payload: dict, timeout: int = 600):
    request = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def slot_file_has_checkpoints(path: Path | str) -> bool:
    """True when a saved slot file carries a persisted checkpoint payload.

    Scans for the magic rather than trusting an offset: the payload is appended after the
    llama state, whose length depends on the model and the prompt.
    """
    path = Path(path)
    if not path.is_file():
        return False
    found = False
    tail = b""
    with open(path, "rb") as handle:
        while True:
            block = handle.read(4 << 20)
            if not block:
                break
            if SCKP_MAGIC in tail + block:
                found = True
                break
            tail = block[-3:]          # a magic split across a chunk boundary
    return found


def checkpoint_protocol(url: str) -> dict:
    """The runtime's advertised checkpoint-persistence protocol, or {} when absent.

    This is the authoritative signal: a machine-readable statement of BEHAVIOUR from the
    running server. An architecture name, a commit, a filename, a strings(1) match or an
    artifact size all describe the build instead, and none of them may enable a capability.
    """
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/props", timeout=30) as response:
            props = json.loads(response.read())
    except Exception:
        return {}
    if not props.get("slot_checkpoint_persistence"):
        return {}
    return {
        "format": props.get("slot_checkpoint_format", ""),
        "sequence_state_version": props.get("sequence_state_version"),
        "target": bool(props.get("supports_target_checkpoint_state")),
        "draft": bool(props.get("supports_draft_checkpoint_state")),
        "speculative": bool(props.get("supports_speculative_checkpoint_state")),
    }


def detect_runtime(url: str, slots: Path | str, slot: int = 0,
                   probe_tokens: int = 64) -> str:
    """Classify the connected runtime.

    Prefers the advertised protocol; falls back to scanning a saved slot for the SCKP magic
    only when the runtime advertises nothing, which is the case for a build carrying the
    upstream patch alone. UNKNOWN is returned rather than guessed when the server cannot be
    probed - a test that cannot establish which binary it is talking to must skip.
    """
    if checkpoint_protocol(url):
        return PATCHED
    try:
        text = "In the year 1892 the naturalist recorded. " * 20
        ids = _post(url, "/tokenize", {"content": text})["tokens"][:probe_tokens]
        if not ids:
            return UNKNOWN
        _post(url, f"/slots/{slot}?action=erase", {})
        _post(url, "/completion", {"prompt": ids, "n_predict": 1, "temperature": 0.0,
                                   "cache_prompt": True, "id_slot": slot})
        name = "kvx-runtime-probe.bin"
        _post(url, f"/slots/{slot}?action=save", {"filename": name})
        probe = Path(slots) / name
        try:
            return PATCHED if slot_file_has_checkpoints(probe) else UNPATCHED
        finally:
            probe.unlink(missing_ok=True)
    except Exception:
        return UNKNOWN


def require_runtime(test_case, url: str, slots: Path | str, expected: str) -> str:
    """Skip unless the connected runtime is the one this test is about."""
    actual = detect_runtime(url, slots)
    if actual != expected:
        test_case.skipTest(
            f"this test asserts {expected} behaviour but the runtime at {url} is {actual}; "
            f"running it here would report a difference in binaries as a code failure")
    return actual
