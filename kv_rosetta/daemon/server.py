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
        if model not in self.running_models():
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
        for obj in self.store().list_objects():
            manifest = obj.manifest
            if manifest.get("prefix_fingerprint") == fingerprint and \
                    manifest.get("runtime_model") == model:
                return obj
        return None

    def ensure(self, fingerprint: str, model: str, slot: int = 0) -> dict[str, Any]:
        """Restore a prefix into a loaded model, or explain why the caller should prefill."""
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or \
                any(c not in "0123456789abcdef" for c in fingerprint):
            raise Fallback("fingerprint is not a 64-character lowercase hex digest")
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
            found.digest, model=model, token_ids=token_ids, slot=slot)
        if not report.ok:
            raise Fallback(f"restore refused: {report.reason}")
        return {"restored": True, "digest": found.digest, "cache_n": report.cache_n,
                "prompt_n": report.prompt_n, "seconds": report.seconds,
                "phases": report.phases, "mode": "admitted_direct_restore"}

    # -- restore at request time ---------------------------------------------------------

    def restore_for_prompt(self, model: str, messages: list[dict[str, Any]],
                           tools: list[dict[str, Any]] | None = None, *,
                           adapter: Any | None = None) -> dict[str, Any]:
        """Put the attachment that best matches an incoming prompt into a slot, now.

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
        if not messages:
            return {"restored": False, "reason": "no messages to match"}
        try:
            base = self.upstream_base(model)
        except Fallback as exc:
            return {"restored": False, "reason": str(exc)}
        if adapter is None:
            from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter
            adapter = LlamaCppHTTPAdapter(base, str(self.store().root))
        try:
            prompt = adapter.apply_template(messages, tools)
            ids = adapter.tokenize(prompt)
        except Exception as exc:
            return {"restored": False, "reason": f"runtime could not render the prompt: {exc}"}
        if not ids:
            return {"restored": False, "reason": "prompt tokenized to nothing"}

        best = None
        for obj in self.store().list_objects():
            man = obj.manifest or {}
            if man.get("runtime_model") != model:
                continue
            stored = man.get("prompt_token_ids") or []
            n = len(stored)
            if n == 0 or n > len(ids) or ids[:n] != list(stored):
                continue
            if best is None or n > best[0]:
                best = (n, str(man.get("prefix_fingerprint") or ""), obj)
        if best is None:
            return {"restored": False,
                    "reason": f"no attachment is a prefix of this prompt ({len(ids)} tokens)"}
        covers, fingerprint, _ = best

        try:
            slots = adapter._get("/slots")
        except Exception as exc:
            return {"restored": False, "reason": f"could not read slots: {exc}"}
        idle = [s for s in slots if not s.get("is_processing")]
        if not idle:
            return {"restored": False, "reason": "every slot is busy"}
        memo = getattr(self, "_slot_last_used", None)
        if memo is None:
            memo = self._slot_last_used = {}
        empty = [s for s in idle if int(s.get("n_prompt_tokens") or 0) == 0]
        pool = empty or idle
        slot = int(min(pool, key=lambda s: (memo.get((model, int(s["id"])), 0.0),
                                            int(s["id"])))["id"])

        try:
            result = self.ensure(fingerprint, model, slot)
        except Fallback as exc:
            return {"restored": False, "reason": f"refused: {exc}", "slot": slot,
                    "prefix": fingerprint[:12]}
        memo[(model, slot)] = time.time()
        return {"restored": True, "covers_tokens": covers, "slot": slot,
                "prefix": fingerprint[:12], "prompt_tokens": len(ids),
                "seconds": round(time.time() - started, 3), **result}

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

        def _handle(self, route: str, action) -> None:
            try:
                self._send(200, action())
            except Fallback as exc:
                with sidecar._lock:
                    sidecar.stats.fallbacks += 1
                self._send(200, {"ok": False, "fallback": True, "reason": exc.reason,
                                 "action": "prefill_natively"})
            except AdapterError as exc:
                with sidecar._lock:
                    sidecar.stats.refusals += 1
                self._send(409, {"ok": False, "refused": True, "reason": str(exc)})
            except ValueError as exc:
                self._send(400, {"ok": False, "error": str(exc)})
            except Exception as exc:                  # never leak a traceback to a caller
                log.exception("unhandled error on %s", route)
                with sidecar._lock:
                    sidecar.stats.errors += 1
                self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

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
                    result = sidecar.restore_for_prompt(
                        str(body.get("model", "")),
                        list(body.get("messages") or []),
                        list(body.get("tools") or []) or None)
                    if result.get("restored"):
                        with sidecar._lock:
                            sidecar.stats.restores_served += 1
                    # One line per call, so the daemon log is evidence of what cfrproxy
                    # asked and what was answered -- the first live check had to be read
                    # off the proxy's trace table because nothing here recorded the call.
                    verdict = (f"restored {result.get('covers_tokens'):,} tokens into slot "
                               f"{result.get('slot')}" if result.get("restored")
                               else f"miss: {result.get('reason')}")
                    print(f"[restore-for-prompt] {body.get('model')}: {verdict}", flush=True)
                    return {"ok": True, **result}

                self._handle(route, prompt_action)
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
