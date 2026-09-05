"""Localhost sidecar that restores known prefixes into models that are already loaded.

This exists to replace kvwarm, and the most important thing it does is something kvwarm does
NOT do: it never causes a model to load.

kvwarm wakes a model just to identify it - `/upstream/<model>/props` makes llama-swap load
the target - then re-prefills every known prefix on a timer. On this host that evicted the
models actually in use. Replacing recompute with restore would not have fixed it, because a
restore needs the model resident too. The fix is to be demand-driven.

So there is deliberately no scheduled warm loop and no target-model list. The only way a
prefix gets restored is a caller asking for one, for a model that llama-swap already reports
as running. Every path that cannot be served returns a fallback reason instead, and the
caller prefills natively - which is exactly what would have happened without this service.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from kv_rosetta.adapters.base import AdapterError

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 1 << 20


@dataclass(frozen=True)
class SidecarConfig:
    host: str = "127.0.0.1"
    port: int = 8431
    swap: str = "http://127.0.0.1:9069"
    manifest_root: str = "~/.cfrproxy/cache"
    store_root: str | None = None
    request_timeout: float = 30.0
    #: llama-swap's own config, the only place its aliases are recorded. /running and the
    #: store speak canonical ids; requests arrive under aliases. Read locally, never fetched.
    swap_config: str = "~/llama-swap/config.yaml"


@dataclass
class Stats:
    restores_served: int = 0
    fallbacks: int = 0
    refusals: int = 0
    errors: int = 0
    models_woken: int = 0          # must stay zero; a nonzero value is a defect
    started: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"restores_served": self.restores_served, "fallbacks": self.fallbacks,
                "refusals": self.refusals, "errors": self.errors,
                "models_woken": self.models_woken,
                "uptime_s": time.time() - self.started}


#: Shortest shared prefix worth restoring for. A restore writes hundreds of MB into a slot
#: and probes it (seconds); the fleet prefills at roughly 500-1,500 tok/s, so below about
#: a thousand tokens the prefill is the cheaper path.
MIN_USEFUL_LCP = 1024
# REQ-112: cfrproxy's probe timed out (3 s) while the restore finished behind it, then a retry
# restored the same prefix again into ANOTHER idle slot. A restore answered within this window
# for the same (model, prefix) is reported as already done, from /slots, without touching one.
RECENT_RESTORE_S = 30.0
# REQ-113: a seed whose rendered prompt is already covered by a held artifact, bar the
# one-token user turn and assistant header it ends in, is not seeded again.
SEED_TAIL_SLACK = 32
# REQ-114: a live restore pre-empts a background seed on the same runtime. The seed's
# connection is dropped (llama-server cancels the prefill) and the restore waits this long
# for the slot to come free before answering "busy: seeding" so the router can go elsewhere.
SEED_YIELD_WAIT_S = 5.0


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Fallback(Exception):
    """A prefix cannot be served. Not an error: the caller prefills, as it always would."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Sidecar:
    """Resolves and restores prefixes for already-loaded models. Never loads one."""

    def __init__(self, config: SidecarConfig) -> None:
        self._require_loopback(config.host)
        self.config = config
        self.stats = Stats()
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None

    @staticmethod
    def _require_loopback(host: str) -> None:
        """This service can cause a model to restore state; it must not be reachable off-box."""
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError(f"host {host!r} is not an IP address; refusing to bind") from exc
        if not address.is_loopback:
            raise ValueError(f"refusing to bind to {host}: this endpoint mutates model slot "
                             f"state, so exposing it beyond loopback would hand that to the "
                             f"network")

    # -- llama-swap, read-only ---------------------------------------------------------

    # -- aliases -----------------------------------------------------------------------

    def _alias_map(self) -> dict[str, str]:
        """alias -> canonical llama-swap model id, from llama-swap's own config.

        llama-swap resolves aliases itself and reports only canonical ids on /running, and
        the store keys artifacts by the canonical id capture saw there. A request arrives
        under whatever alias the client used -- `qwen38-27b-kvx-3090`, `27b`, `hermes-v7`
        -- so a raw comparison refused every one of them as "not loaded" while the model
        was serving: three of the four request-time misses in the first live hour.

        No HTTP surface exposes the mapping (/v1/models lists aliases and canonicals with
        identical metadata and no link), so the file is read locally. Cached by mtime;
        fails OPEN to an empty map, which is exactly the pre-existing behaviour.
        """
        # Tolerate a sidecar built without __init__ (tests do this): no config, no map.
        configured = getattr(getattr(self, "config", None), "swap_config", "") or ""
        if not configured:                    # no config named: nothing to resolve
            return {}
        path = Path(configured).expanduser()
        try:
            stamp = path.stat().st_mtime
        except OSError:
            return {}
        cached = getattr(self, "_alias_cache", None)
        if cached and cached[0] == stamp:
            return cached[1]
        try:
            import yaml
            doc = yaml.safe_load(path.read_text()) or {}
            mapping: dict[str, str] = {}
            for canonical, entry in (doc.get("models") or {}).items():
                for alias in (entry or {}).get("aliases") or []:
                    mapping[str(alias)] = str(canonical)
        except Exception:                     # unreadable config: behave as before
            mapping = {}
        self._alias_cache = (stamp, mapping)
        return mapping

    def canonical(self, model: str) -> str:
        """The id llama-swap and the store use for `model`; unchanged if it is not an alias."""
        return self._alias_map().get(model, model)

    def running_models(self) -> list[str]:
        """Models llama-swap reports as loaded.

        /running is a status endpoint: it reports what is loaded without loading anything.
        Nothing here may ever call /upstream/<model>/..., because that is the call that
        wakes a model, and waking models is the behaviour this service exists to remove.
        """
        url = f"{self.config.swap.rstrip('/')}/running"
        try:
            with urllib.request.urlopen(url, timeout=self.config.request_timeout) as reply:
                payload = json.loads(reply.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise Fallback(f"cannot read loaded models from llama-swap: {exc}") from exc
        entries = payload.get("running", []) if isinstance(payload, dict) else []
        return [str(e.get("model", "")) for e in entries
                if isinstance(e, dict) and e.get("state") == "ready" and e.get("model")]

    def require_loaded(self, model: str) -> None:
        """Refuse unless the model is already resident. This is the whole safety property."""
        if self.canonical(model) not in self.running_models():
            raise Fallback(f"model {model!r} is not loaded; refusing to wake it - prefill "
                           f"natively instead")

    def upstream_base(self, model: str) -> str:
        """The one place an upstream URL is constructed, and only for a loaded model.

        Restoring requires talking to the model's own server, and through llama-swap that
        means /upstream/<model>/. That is the same path kvwarm used to wake models - the
        difference is not the URL, it is that this one cannot be reached without first
        proving the model is already resident. A blanket ban on the path would make restore
        impossible; the gate is what makes it safe.
        """
        self.require_loaded(model)
        if "/" in model or model.startswith("."):
            raise Fallback(f"model name {model!r} is not a plain identifier")
        return f"{self.config.swap.rstrip('/')}/upstream/{model}"

    # -- prefixes ----------------------------------------------------------------------

    def known_prefixes(self) -> list[dict[str, Any]]:
        from kv_rosetta.daemon.watcher import load_manifests

        root = Path(self.config.manifest_root).expanduser()
        if not root.is_dir():
            return []
        try:
            found = load_manifests(root)
        except Exception as exc:                      # watcher raises its own error type
            raise Fallback(f"cannot read prefix manifests: {exc}") from exc
        return [{"fingerprint": m.fingerprint, "provider": m.provider, "model": m.model,
                 "est_tokens": m.est_tokens} for m in found]

    # -- the one action ----------------------------------------------------------------

    def store(self):
        from kv_rosetta.admitted_store import AdmittedStore

        if self.config.store_root is None:
            raise Fallback("no admitted-state store is configured")
        return AdmittedStore(Path(self.config.store_root).expanduser(), create=False)

    def find_artifact(self, fingerprint: str, model: str):
        """An admitted object for this prefix and this model, or None.

        Matching is on the recorded prefix fingerprint and runtime model. The cache ABI is
        re-checked during the restore itself, against the live runtime rather than against
        what the manifest claims.
        """
        model = self.canonical(model)
        for obj in self.store().list_objects():
            manifest = obj.manifest
            if manifest.get("prefix_fingerprint") == fingerprint and \
                    manifest.get("runtime_model") == model:
                return obj
        return None

    def ensure(self, fingerprint: str, model: str, slot: int = 0,
               cancelled=None) -> dict[str, Any]:
        """Restore a prefix into a loaded model, or explain why the caller should prefill."""
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or \
                any(c not in "0123456789abcdef" for c in fingerprint):
            raise Fallback("fingerprint is not a 64-character lowercase hex digest")
        model = self.canonical(model)             # the store and /running speak this id
        base = self.upstream_base(model)          # proves the model is loaded first
        found = self.find_artifact(fingerprint, model)
        if found is None:
            raise Fallback(f"no admitted artifact for prefix {fingerprint[:12]} on "
                           f"{model!r}; prefill natively")

        from kv_rosetta.adapters.admitted_path import AdmittedPath
        from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter

        adapter = LlamaCppHTTPAdapter(base, str(self.store().root))
        token_ids = list(found.manifest.get("prompt_token_ids") or [])
        if not token_ids:
            raise Fallback(
                f"artifact {found.digest[:12]} records no token ids, so reuse cannot be "
                f"verified; prefill natively rather than trusting an unverified restore")
        # Ask the runtime what it can do before writing anything into a slot. A build
        # without the context-checkpoint patch accepts a hybrid restore and reuses none of
        # it, reporting the same n_restored as one that reuses everything - so a silent
        # uselessness becomes an explicit fallback here instead of a mystery downstream.
        declared = found.manifest.get("requirements")
        if declared:
            from kv_rosetta.requirements import Requirements, check
            try:
                props = adapter.props()
            except Exception as exc:              # a runtime that will not answer /props
                raise Fallback(f"could not read runtime capabilities from {model!r}: {exc}; "
                               f"prefill natively rather than restore blind") from exc
            # llama.cpp puts no model identity in /props; the adapter derives it from the
            # weights file, so it has to be supplied rather than looked up.
            try:
                runtime_identity = adapter.model_identity(model).weights_sha256
            except Exception:                     # identity is checked, not assumed present
                runtime_identity = ""
            problems = check(Requirements(**declared), props,
                             runtime_identity=runtime_identity)
            if problems:
                raise Fallback(f"artifact {found.digest[:12]} cannot be restored into "
                               f"{model!r}: {'; '.join(problems)}")
        report = AdmittedPath(adapter, self.store()).restore(
            found.digest, model=model, token_ids=token_ids, slot=slot, cancelled=cancelled)
        if not report.ok:
            raise Fallback(f"restore refused: {report.reason}")
        return {"restored": True, "digest": found.digest, "cache_n": report.cache_n,
                "prompt_n": report.prompt_n, "seconds": report.seconds,
                "phases": report.phases, "mode": "admitted_direct_restore"}

    # -- restore at request time ---------------------------------------------------------

    def restore_for_prompt(self, model: str, messages: list[dict[str, Any]],
                           tools: list[dict[str, Any]] | None = None, *,
                           template_fields: dict[str, Any] | None = None,
                           adapter: Any | None = None, dry_run: bool = False,
                           cancelled=None) -> dict[str, Any]:
        """Put the attachment that best matches an incoming prompt into a slot, now.

        `dry_run` (REQ-113) stops after the scan: render, tokenize, match, answer
        `would_restore` -- nothing is read from or written to a slot. ~0.4 s at 70k
        tokens, so a router can count a local model as "prefix cached" before choosing it.

        The load-time restore only ever fills an EMPTY slot, and after a model's first
        request its slots are never empty again: llama.cpp keeps the last conversation's
        cache in each one. So on a busy fleet a new conversation never met a restore --
        llama.cpp evicted a slot and prefilled the whole prompt cold (measured: 30,335 and
        7,399-token first requests both `cached: 0`). This is the request-time half.

        The prefix is rendered and tokenized by the runtime that will serve it, because the
        prefix a request presents is the chat-templated string, and the template is the
        runtime's. The attachment chosen is the one whose stored token ids are the LONGEST
        prefix of that sequence. The slot chosen is an idle one -- empty if any, otherwise
        the one this sidecar restored into least recently -- which is the slot llama.cpp
        would evict for this request anyway, so a warm session is never turned cold that
        was not about to be.

        Never wakes a model (upstream_base refuses an unloaded one) and never touches a
        busy slot. Returns a dict that always carries `restored`; a miss is an answer, not
        an error, because the caller is about to forward the request either way.
        """
        started = time.time()
        requested = model
        model = self.canonical(model)             # store keys and /running are canonical
        if not messages:
            return {"restored": False, "reason": "no messages to match"}
        try:
            base = self.upstream_base(model)
        except Fallback as exc:
            return {"restored": False, "reason": str(exc)}
        if adapter is None:
            from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter
            adapter = LlamaCppHTTPAdapter(base, str(self.store().root))
        stages: dict[str, float] = {}
        mark = time.time()
        try:
            # Every field the template reads must reach the render, or the head diverges
            # in its first tokens (measured: 3 shared tokens without `reasoning_effort`).
            prompt = adapter.apply_template(messages, tools, extra=template_fields)
            stages["render"] = round(time.time() - mark, 3); mark = time.time()
            ids = adapter.tokenize(prompt)
            stages["tokenize"] = round(time.time() - mark, 3); mark = time.time()
        except Exception as exc:
            return {"restored": False, "reason": f"runtime could not render the prompt: {exc}"}
        if not ids:
            return {"restored": False, "reason": "prompt tokenized to nothing"}

        # Longest COMMON prefix, not "stored is a prefix of the prompt". Capture saves whole
        # conversations at idle, so every stored artifact is longer than the first request
        # of the next conversation from the same harness -- the strict rule could never
        # match (measured: a 21,629-token first request against 25,958..42,726-token
        # artifacts, miss). llama.cpp itself reuses get_common_prefix() and then restores
        # the latest context checkpoint at or before it, so the shared head is what a
        # restore is worth. Below MIN_USEFUL_LCP a restore costs more than the prefill it
        # saves. Tie-break toward the smaller artifact: less to write into the slot.
        best = None
        mark = time.time()
        for obj in self.store().list_objects():
            man = obj.manifest or {}
            if man.get("runtime_model") != model:
                continue
            stored = list(man.get("prompt_token_ids") or [])
            shared = _common_prefix_len(stored, ids)
            if shared < MIN_USEFUL_LCP:
                continue
            key = (shared, -len(stored))
            if best is None or key > best[0]:
                best = (key, str(man.get("prefix_fingerprint") or ""), obj, len(stored))
        if best is None:
            stages["scan"] = round(time.time() - mark, 3)
            return {"restored": False, "would_restore": False, "stages": stages,
                    "prompt_tokens": len(ids), "model": model, "requested": requested,
                    "reason": f"no attachment shares at least {MIN_USEFUL_LCP} tokens with "
                              f"this prompt ({len(ids)} tokens)"}
        (shared, _), fingerprint, _, covers = best
        stages["scan"] = round(time.time() - mark, 3); mark = time.time()
        if dry_run:
            return {"restored": False, "would_restore": True, "covers_tokens": covers,
                    "shared_tokens": shared, "prefix": fingerprint[:12],
                    "prompt_tokens": len(ids), "model": model, "requested": requested,
                    "stages": stages, "seconds": round(time.time() - started, 3)}

        # A background seed on this runtime yields to a live turn (REQ-114).
        seeding = getattr(self, "_seeds", {}).get(model)
        if seeding is not None:
            seeding["yielded"] = True
            try:
                seeding["abort"]()
            except Exception:
                pass
            deadline = time.time() + SEED_YIELD_WAIT_S
            while True:
                try:
                    slots = adapter._get("/slots")
                except Exception as exc:
                    return {"restored": False, "reason": f"could not read slots: {exc}"}
                held = next((s for s in slots if int(s["id"]) == seeding["slot"]), None)
                if held is None or not held.get("is_processing"):
                    break
                if time.time() >= deadline:
                    stages["slots"] = round(time.time() - mark, 3)
                    return {"restored": False, "busy": "seeding", "stages": stages,
                            "reason": f"busy: seeding slot {seeding['slot']} "
                                      f"({seeding['tokens']:,} tokens) did not yield in "
                                      f"{SEED_YIELD_WAIT_S:.0f}s"}
                time.sleep(0.25)
            stages["yield"] = round(time.time() - mark, 3); mark = time.time()
        try:
            slots = adapter._get("/slots")
        except Exception as exc:
            return {"restored": False, "reason": f"could not read slots: {exc}"}
        stages["slots"] = round(time.time() - mark, 3); mark = time.time()

        # Already restored for this prefix moments ago? The caller gave up on that answer
        # (or is retrying); the slot is either serving that request now or still holding
        # the restored cache. Say so instead of restoring again into a second slot.
        recent = getattr(self, "_recent_restores", None)
        if recent is None:
            recent = self._recent_restores = {}
        hit = recent.get((model, fingerprint))
        if hit and time.time() - hit["at"] <= RECENT_RESTORE_S:
            held = next((s for s in slots if int(s["id"]) == hit["slot"]), None)
            if held is not None and (held.get("is_processing") or
                                     int(held.get("n_prompt_tokens") or 0) >= hit["covers"]):
                stages["ensure"] = 0.0
                return {"restored": True, "already": True, "covers_tokens": hit["covers"],
                        "slot": hit["slot"], "prefix": fingerprint[:12],
                        "prompt_tokens": len(ids), "model": model, "requested": requested,
                        "shared_tokens": shared, "stages": stages,
                        "seconds": round(time.time() - started, 3)}
        idle = [s for s in slots if not s.get("is_processing")]
        if not idle:
            return {"restored": False, "reason": "every slot is busy", "stages": stages}
        memo = getattr(self, "_slot_last_used", None)
        if memo is None:
            memo = self._slot_last_used = {}
        empty = [s for s in idle if int(s.get("n_prompt_tokens") or 0) == 0]
        pool = empty or idle
        slot = int(min(pool, key=lambda s: (memo.get((model, int(s["id"])), 0.0),
                                            int(s["id"])))["id"])

        if cancelled is not None and cancelled():
            return {"restored": False, "reason": "caller gone before the restore", "slot": slot,
                    "prefix": fingerprint[:12], "stages": stages}
        try:
            result = self.ensure(fingerprint, model, slot, cancelled=cancelled)
        except Fallback as exc:
            stages["ensure"] = round(time.time() - mark, 3)
            return {"restored": False, "reason": f"refused: {exc}", "slot": slot,
                    "prefix": fingerprint[:12], "stages": stages}
        stages["ensure"] = round(time.time() - mark, 3)
        memo[(model, slot)] = time.time()
        recent[(model, fingerprint)] = {"slot": slot, "at": time.time(), "covers": covers}
        return {"restored": True, "covers_tokens": covers, "slot": slot, "stages": stages,
                "prefix": fingerprint[:12], "prompt_tokens": len(ids),
                "model": model, "requested": requested, "shared_tokens": shared,
                "seconds": round(time.time() - started, 3), **result}

    # -- seed a standing prefix (REQ-113) ---------------------------------------------------

    admitter = None          # set by `serve`: (model, basename, saved, *, pin) -> str

    def seed(self, model: str, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None, *,
             template_fields: dict[str, Any] | None = None, user_turn: str = "seed",
             adapter: Any | None = None) -> dict[str, Any]:
        """Prefill a known static prefix into an idle slot, capture it, admit it, pin it.

        The first conversation a harness opens on a model it has never served finds no
        attachment and pays the whole prefill (measured: Claude Code on ornith, 67k tokens,
        79 s, then the client gave up). The artifact only exists after that victim, and
        capture churn evicts it. This makes the artifact exist first.

        The caller sends the exact body it would forward (system, tools, template fields)
        so the render is the runtime's own; a user turn is appended when the prefix ends
        without one, because this build checkpoints hybrid caches at user-turn starts and
        a prompt with no turn gets no checkpoint. Never wakes a model, never touches a busy
        slot, and does nothing when a held artifact already covers the prompt.
        """
        started = time.time()
        stages: dict[str, float] = {}
        requested = model
        model = self.canonical(model)
        if not messages:
            return {"seeded": False, "reason": "no messages to seed"}
        try:
            base = self.upstream_base(model)
        except Fallback as exc:
            return {"seeded": False, "reason": str(exc)}
        if self.admitter is None:
            return {"seeded": False, "reason": "this sidecar has no admitter; run `serve` "
                                               "with a store root"}
        if adapter is None:
            from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter
            adapter = LlamaCppHTTPAdapter(base, str(self.store().root))
        msgs = list(messages)
        if str(msgs[-1].get("role")) != "user":
            msgs.append({"role": "user", "content": user_turn})

        probe = self.restore_for_prompt(model, msgs, tools, template_fields=template_fields,
                                        adapter=adapter, dry_run=True)
        stages["probe"] = round(time.time() - started, 3)
        need = int(probe.get("prompt_tokens") or 0)
        if not need:
            return {"seeded": False, "reason": probe.get("reason", "prompt rendered to nothing"),
                    "stages": stages}
        if probe.get("would_restore") and \
                int(probe.get("shared_tokens") or 0) >= need - SEED_TAIL_SLACK:
            return {"seeded": False, "already": True, "prefix": probe.get("prefix"),
                    "covers_tokens": probe.get("covers_tokens"), "prompt_tokens": need,
                    "model": model, "requested": requested, "stages": stages,
                    "reason": f"already held: {probe.get('prefix')} shares "
                              f"{probe.get('shared_tokens'):,} of {need:,} tokens"}

        mark = time.time()
        try:
            slots = adapter._get("/slots")
        except Exception as exc:
            return {"seeded": False, "reason": f"could not read slots: {exc}", "stages": stages}
        idle = [s for s in slots if not s.get("is_processing")]
        if not idle:
            return {"seeded": False, "reason": "every slot is busy", "stages": stages}
        memo = getattr(self, "_slot_last_used", None)
        if memo is None:
            memo = self._slot_last_used = {}
        # No request is forcing an eviction here, so take the idle slot with the LEAST to
        # lose: fewest cached tokens, then least recently restored into. (The first live
        # seed took slot 0 by id and turned a 57,840-token idle session cold over a
        # 798-token one.)
        slot = int(min(idle, key=lambda s: (int(s.get("n_prompt_tokens") or 0),
                                            memo.get((model, int(s["id"])), 0.0),
                                            int(s["id"])))["id"])

        body: dict[str, Any] = {"model": model, "messages": msgs, "max_tokens": 1,
                                "stream": False, "id_slot": slot, "cache_prompt": True,
                                **(template_fields or {})}
        if tools:
            body["tools"] = tools
        seeds = getattr(self, "_seeds", None)
        if seeds is None:
            seeds = self._seeds = {}
        if model in seeds:
            return {"seeded": False, "reason": f"a seed is already running on {model!r} "
                                               f"(slot {seeds[model]['slot']})", "stages": stages}
        prefill = getattr(adapter, "prefill", None) or \
            (lambda b: adapter._post("/v1/chat/completions", b))
        entry = {"slot": slot, "tokens": need, "started": time.time(), "yielded": False,
                 "abort": getattr(adapter, "abort_prefill", lambda: False)}
        seeds[model] = entry
        try:
            prefill(body)
        except Exception as exc:
            if entry["yielded"]:
                return {"seeded": False, "yielded": True, "slot": slot, "stages": stages,
                        "reason": "yielded to a live restore-for-prompt on this runtime"}
            return {"seeded": False, "reason": f"prefill failed: {str(exc)[:200]}",
                    "slot": slot, "stages": stages}
        finally:
            seeds.pop(model, None)
        stages["prefill"] = round(time.time() - mark, 3); mark = time.time()

        # The slot must hold what was rendered; n_prompt_tokens is the server's own count.
        try:
            held = next(s for s in adapter._get("/slots") if int(s["id"]) == slot)
        except Exception as exc:
            return {"seeded": False, "reason": f"could not re-read slot {slot}: {exc}",
                    "slot": slot, "stages": stages}
        n = int(held.get("n_prompt_tokens") or 0)
        if n < need - 8:
            return {"seeded": False, "slot": slot, "stages": stages,
                    "reason": f"slot {slot} holds {n:,} tokens after the prefill, expected "
                              f"{need:,}; not captured"}
        name = f"seed-{model}-slot{slot}-{n}.state"
        try:
            saved = adapter._post(f"/slots/{slot}?action=save", {"filename": name})
            stages["save"] = round(time.time() - mark, 3); mark = time.time()
            admitted = self.admitter(model, name, saved, pin=True)
            stages["admit"] = round(time.time() - mark, 3)
        except Exception as exc:
            return {"seeded": False, "slot": slot, "stages": stages,
                    "reason": f"capture/admit failed: {str(exc)[:200]}"}
        memo[(model, slot)] = time.time()
        return {"seeded": True, "slot": slot, "tokens": n, "admitted": str(admitted),
                "model": model, "requested": requested, "stages": stages,
                "seconds": round(time.time() - started, 3)}

    # -- lifecycle ---------------------------------------------------------------------

    def serve_forever(self) -> None:
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self.config.host, self.config.port), handler)
        self._server.daemon_threads = True
        log.info("kv-rosetta sidecar on %s:%d, swap=%s",
                 self.config.host, self.port, self.config.swap)
        self._server.serve_forever()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else self.config.port


def _make_handler(sidecar: Sidecar):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _caller_gone(self) -> bool:
            """True once the client has closed its side (REQ-114). Non-blocking."""
            import select
            import socket as _socket
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if not readable:
                    return False
                return self.connection.recv(1, _socket.MSG_PEEK) == b""
            except (OSError, ValueError):
                return True

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ValueError(f"request body of {length} bytes exceeds the limit")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("request body is not a JSON object")
            return value

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            try:
                self._send(status, payload)
            except (BrokenPipeError, ConnectionResetError):
                # REQ-112: the caller's probe timeout expired before the verdict. The work
                # is done and memoised (see RECENT_RESTORE_S); its retry gets it instantly.
                print(f"[{self.path.split('?', 1)[0]}] caller hung up before the verdict "
                      f"({payload.get('seconds', 0):.2f}s): "
                      f"{'restored' if payload.get('restored') else payload.get('reason', payload.get('error', '?'))}",
                      flush=True)

        def _handle(self, route: str, action) -> None:
            try:
                self._reply(200, action())
            except Fallback as exc:
                with sidecar._lock:
                    sidecar.stats.fallbacks += 1
                self._reply(200, {"ok": False, "fallback": True, "reason": exc.reason,
                                 "action": "prefill_natively"})
            except AdapterError as exc:
                with sidecar._lock:
                    sidecar.stats.refusals += 1
                self._reply(409, {"ok": False, "refused": True, "reason": str(exc)})
            except ValueError as exc:
                self._reply(400, {"ok": False, "error": str(exc)})
            except Exception as exc:                  # never leak a traceback to a caller
                log.exception("unhandled error on %s", route)
                with sidecar._lock:
                    sidecar.stats.errors += 1
                self._reply(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self) -> None:
            route = self.path.split("?", 1)[0].rstrip("/") or "/"
            if route == "/v1/health":
                self._handle(route, lambda: {"ok": True, "swap": sidecar.config.swap})
            elif route == "/v1/runtimes":
                self._handle(route, lambda: {"ok": True, "loaded": sidecar.running_models()})
            elif route == "/v1/prefixes":
                self._handle(route, lambda: {"ok": True,
                                             "prefixes": sidecar.known_prefixes()})
            elif route == "/v1/stats":
                self._handle(route, lambda: {"ok": True, **sidecar.stats.as_dict()})
            else:
                self._send(404, {"ok": False, "error": f"unknown route {route}"})

        def do_POST(self) -> None:
            route = self.path.split("?", 1)[0].rstrip("/") or "/"
            if route == "/v1/restore-for-prompt":
                # Always 200 with a `restored` verdict: the caller is about to forward the
                # request whatever the answer, so a miss is information, not an error.
                def prompt_action() -> dict[str, Any]:
                    body = self._body()
                    # Template-affecting fields ride along from the real request; the
                    # adapter keeps only the ones it knows change the render.
                    template_fields = {k: body[k] for k in (
                        "reasoning_effort", "chat_template_kwargs", "enable_thinking",
                        "reasoning_format", "thinking") if k in body}
                    result = sidecar.restore_for_prompt(
                        str(body.get("model", "")),
                        list(body.get("messages") or []),
                        list(body.get("tools") or []) or None,
                        template_fields=template_fields or None,
                        dry_run=bool(body.get("dry_run")),
                        cancelled=self._caller_gone)
                    if result.get("restored"):
                        with sidecar._lock:
                            sidecar.stats.restores_served += 1
                    # One line per call, so the daemon log is evidence of what cfrproxy
                    # asked and what was answered -- the first live check had to be read
                    # off the proxy's trace table because nothing here recorded the call.
                    if body.get("dry_run"):
                        verdict = (f"probe: would restore {result.get('covers_tokens'):,} "
                                   f"(shared {result.get('shared_tokens'):,})"
                                   if result.get("would_restore")
                                   else f"probe miss: {result.get('reason')}")
                    else:
                        verdict = (f"{'already ' if result.get('already') else ''}restored "
                                   f"{result.get('covers_tokens'):,} tokens into slot "
                                   f"{result.get('slot')}" if result.get("restored")
                                   else f"miss: {result.get('reason')}")
                    timing = " ".join(f"{k}={v:.2f}s" for k, v in
                                      (result.get("stages") or {}).items())
                    phases = " ".join(f"{k}={v:.2f}s" for k, v in
                                      (result.get("phases") or {}).items() if v >= 0.05)
                    print(f"[restore-for-prompt] {body.get('model')}: {verdict} | "
                          f"{result.get('seconds', 0):.2f}s {timing}"
                          f"{' | ensure: ' + phases if phases else ''}", flush=True)
                    return {"ok": True, **result}

                self._handle(route, prompt_action)
                return
            if route == "/v1/seed":
                def seed_action() -> dict[str, Any]:
                    body = self._body()
                    template_fields = {k: body[k] for k in (
                        "reasoning_effort", "chat_template_kwargs", "enable_thinking",
                        "reasoning_format", "thinking") if k in body}
                    result = sidecar.seed(
                        str(body.get("model", "")),
                        list(body.get("messages") or []),
                        list(body.get("tools") or []) or None,
                        template_fields=template_fields or None,
                        user_turn=str(body.get("user_turn") or "seed"))
                    timing = " ".join(f"{k}={v:.2f}s" for k, v in
                                      (result.get("stages") or {}).items())
                    verdict = (f"seeded {result.get('tokens'):,} tokens into slot "
                               f"{result.get('slot')} -> {result.get('admitted')}"
                               if result.get("seeded") else
                               f"{'yielded' if result.get('yielded') else 'not seeded'}: "
                               f"{result.get('reason')}")
                    print(f"[seed] {body.get('model')}: {verdict} | "
                          f"{result.get('seconds', 0):.2f}s {timing}", flush=True)
                    return {"ok": True, **result}
                self._handle(route, seed_action)
                return
            if route != "/v1/ensure":
                self._send(404, {"ok": False, "error": f"unknown route {route}"})
                return

            def action() -> dict[str, Any]:
                body = self._body()
                result = sidecar.ensure(str(body.get("fingerprint", "")),
                                        str(body.get("model", "")),
                                        int(body.get("slot", 0)))
                with sidecar._lock:
                    sidecar.stats.restores_served += 1
                return {"ok": True, **result}

            self._handle(route, action)

        def do_PUT(self) -> None:
            self._send(405, {"ok": False, "error": "method not allowed"})

        do_DELETE = do_PUT
        do_PATCH = do_PUT

    return Handler


def build_server(config: SidecarConfig | None = None) -> Sidecar:
    return Sidecar(config or SidecarConfig())
