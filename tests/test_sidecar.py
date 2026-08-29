"""The sidecar must never cause a model to load. Everything else is secondary.

kvwarm evicted models in use because identifying a model required waking it:
`/upstream/<model>/props` makes llama-swap load the target. This service replaces that
behaviour, so the property that matters is not how fast it restores - it is that no code
path here can reach an endpoint that wakes anything.

The interceptor below fails the test if any request touches /upstream/, which is the exact
call kvwarm makes.
"""

import json
import threading
import unittest
import urllib.error
import urllib.request

from kv_rosetta.daemon.server import Fallback, Sidecar, SidecarConfig

WAKING_PATHS = ("/upstream/",)


class FakeSwap:
    """Stands in for llama-swap and records every path asked for."""

    def __init__(self, loaded=("model-a",), broken=False, states=None):
        self.loaded = list(loaded)
        self.states = states or {}
        self.broken = broken
        self.paths: list[str] = []

    def open(self, url, timeout=None):
        path = url.split("9069", 1)[-1] if "9069" in url else url
        self.paths.append(path)
        if any(w in path for w in WAKING_PATHS):
            raise AssertionError(f"the sidecar requested {path}, which wakes a model")
        if self.broken:
            raise urllib.error.URLError("swap is down")
        payload = json.dumps({"running": [
            {"model": m, "state": self.states.get(m, "ready")}
            for m in self.loaded]}).encode()

        class Reply:
            def read(self_inner):
                return payload

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return Reply()


class SidecarTestCase(unittest.TestCase):
    def build(self, **kwargs):
        swap = FakeSwap(**kwargs)
        sidecar = Sidecar(SidecarConfig(port=0, manifest_root="/nonexistent"))
        import kv_rosetta.daemon.server as module
        self._original = module.urllib.request.urlopen
        module.urllib.request.urlopen = swap.open
        self.addCleanup(setattr, module.urllib.request, "urlopen", self._original)
        return sidecar, swap


class NeverWakeAModelTest(SidecarTestCase):
    """The property the whole service exists for."""

    def test_listing_runtimes_does_not_touch_a_waking_endpoint(self):
        sidecar, swap = self.build(loaded=("a", "b"))
        self.assertEqual(sidecar.running_models(), ["a", "b"])
        self.assertTrue(swap.paths, "no request was made at all")
        for path in swap.paths:
            self.assertNotIn("/upstream/", path)

    def test_ensure_on_an_unloaded_model_falls_back_without_contacting_it(self):
        sidecar, swap = self.build(loaded=("model-a",))
        with self.assertRaises(Fallback) as caught:
            sidecar.ensure("a" * 64, "model-b")
        self.assertIn("refusing to wake it", caught.exception.reason)
        for path in swap.paths:
            self.assertNotIn("model-b", path,
                             "the sidecar named an unloaded model in a request")

    def test_no_code_path_builds_an_upstream_url(self):
        """A source check, because a future edit could add one no test exercises.

        Docstrings are excluded deliberately - this module explains the upstream call in
        prose precisely so nobody reintroduces it, and a plain substring search would flag
        that explanation while missing a URL assembled from parts.
        """
        import ast
        from pathlib import Path

        tree = ast.parse(Path("kv_rosetta/daemon/server.py").read_text())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    docstrings.add(doc)
        literals = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and n.value not in docstrings]
        offending = [v for v in literals if "upstream" in v]
        self.assertEqual(offending, [],
                         f"the sidecar builds an upstream URL, which wakes a model: "
                         f"{offending}")

    def test_a_model_still_loading_does_not_count_as_loaded(self):
        """A model mid-load is not resident; sending it work is what evicts things.

        The fake swap previously reported every model as ready, so removing the state
        filter broke no test.
        """
        sidecar, _ = self.build(loaded=("busy",), states={"busy": "loading"})
        self.assertEqual(sidecar.running_models(), [])
        with self.assertRaises(Fallback) as caught:
            sidecar.ensure("f" * 64, "busy")
        self.assertIn("refusing to wake it", caught.exception.reason)

    def test_the_woken_counter_stays_zero(self):
        sidecar, _ = self.build()
        with self.assertRaises(Fallback):
            sidecar.ensure("b" * 64, "not-loaded")
        self.assertEqual(sidecar.stats.models_woken, 0)


class BindingTest(unittest.TestCase):
    def test_a_routable_bind_address_is_refused(self):
        for host in ("0.0.0.0", "192.168.1.10", "::"):
            with self.subTest(host=host):
                with self.assertRaises(ValueError) as caught:
                    Sidecar(SidecarConfig(host=host))
                self.assertIn("loopback", str(caught.exception))

    def test_a_non_address_host_is_refused(self):
        with self.assertRaises(ValueError):
            Sidecar(SidecarConfig(host="example.com"))

    def test_loopback_addresses_are_accepted(self):
        for host in ("127.0.0.1", "127.0.0.5", "::1"):
            with self.subTest(host=host):
                Sidecar(SidecarConfig(host=host))


class FallbackTest(SidecarTestCase):
    def test_a_malformed_fingerprint_falls_back_for_that_reason(self):
        """Assert the specific reason.

        Without this, the test passed whether or not the format check existed: a valid
        fingerprint reaches the "no admitted artifact" fallback, so both paths raise the
        same exception type and the assertion could not tell them apart.
        """
        sidecar, _ = self.build()
        for bad in ("", "short", "A" * 64, "g" * 64):
            with self.subTest(fingerprint=bad[:8]):
                with self.assertRaises(Fallback) as caught:
                    sidecar.ensure(bad, "model-a")
                self.assertIn("64-character lowercase hex", caught.exception.reason)

    def test_an_unreachable_swap_falls_back_rather_than_erroring(self):
        sidecar, _ = self.build(broken=True)
        with self.assertRaises(Fallback) as caught:
            sidecar.running_models()
        self.assertIn("cannot read loaded models", caught.exception.reason)

    def test_a_loaded_model_without_an_artifact_falls_back_honestly(self):
        sidecar, _ = self.build(loaded=("model-a",))
        with self.assertRaises(Fallback) as caught:
            sidecar.ensure("c" * 64, "model-a")
        self.assertIn("no admitted artifact", caught.exception.reason)

    def test_a_missing_manifest_root_returns_no_prefixes(self):
        sidecar, _ = self.build()
        self.assertEqual(sidecar.known_prefixes(), [])


class HttpSurfaceTest(SidecarTestCase):
    def serve(self, **kwargs):
        sidecar, swap = self.build(**kwargs)
        thread = threading.Thread(target=sidecar.serve_forever, daemon=True)
        thread.start()
        for _ in range(200):
            if sidecar._server is not None:
                break
            threading.Event().wait(0.01)
        self.addCleanup(sidecar.shutdown)
        return sidecar, swap

    def get(self, sidecar, path):
        url = f"http://127.0.0.1:{sidecar.port}{path}"
        with self._original(url, timeout=5) as reply:
            return reply.status, json.loads(reply.read())

    def post(self, sidecar, path, payload):
        url = f"http://127.0.0.1:{sidecar.port}{path}"
        request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with self._original(request, timeout=5) as reply:
                return reply.status, json.loads(reply.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health_and_runtimes_answer(self):
        sidecar, _ = self.serve(loaded=("model-a",))
        status, body = self.get(sidecar, "/v1/health")
        self.assertEqual((status, body["ok"]), (200, True))
        status, body = self.get(sidecar, "/v1/runtimes")
        self.assertEqual(body["loaded"], ["model-a"])

    def test_ensure_returns_a_fallback_not_an_error(self):
        sidecar, _ = self.serve(loaded=("model-a",))
        status, body = self.post(sidecar, "/v1/ensure",
                                 {"fingerprint": "d" * 64, "model": "model-b"})
        self.assertEqual(status, 200, "a fallback is a normal answer, not an error")
        self.assertTrue(body["fallback"])
        self.assertEqual(body["action"], "prefill_natively")

    def test_an_unknown_route_is_a_json_404(self):
        sidecar, _ = self.serve()
        try:
            self.get(sidecar, "/v1/nope")
            self.fail("expected 404")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)
            self.assertIn("unknown route", json.loads(exc.read())["error"])

    def test_a_disallowed_method_is_a_json_405(self):
        sidecar, _ = self.serve()
        request = urllib.request.Request(
            f"http://127.0.0.1:{sidecar.port}/v1/ensure", method="DELETE")
        try:
            self._original(request, timeout=5)
            self.fail("expected 405")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 405)

    def test_malformed_json_is_a_400_not_a_traceback(self):
        sidecar, _ = self.serve()
        request = urllib.request.Request(
            f"http://127.0.0.1:{sidecar.port}/v1/ensure", data=b"{not json",
            headers={"Content-Type": "application/json"})
        try:
            self._original(request, timeout=5)
            self.fail("expected 400")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            self.assertIn("error", json.loads(exc.read()))

    def test_stats_report_the_woken_counter(self):
        sidecar, _ = self.serve(loaded=("model-a",))
        self.post(sidecar, "/v1/ensure", {"fingerprint": "e" * 64, "model": "nope"})
        _, body = self.get(sidecar, "/v1/stats")
        self.assertEqual(body["models_woken"], 0)
        self.assertGreaterEqual(body["fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
