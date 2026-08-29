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

    def ensure(self, fingerprint: str, model: str, slot: int = 0) -> dict[str, Any]:
        """Restore a prefix into a loaded model, or explain why the caller should prefill."""
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or \
                any(c not in "0123456789abcdef" for c in fingerprint):
            raise Fallback("fingerprint is not a 64-character lowercase hex digest")
        self.require_loaded(model)
        # Restoring requires an admitted artifact for this exact model and runtime. Until
        # the store is wired to a live adapter, say so plainly rather than pretending.
        raise Fallback(
            f"no admitted artifact for {fingerprint[:12]} on {model!r}; prefill natively. "
            f"The admitted-store transfer seam is proven for a same-model, same-runtime "
            f"restore but is not yet connected to this endpoint.")

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
