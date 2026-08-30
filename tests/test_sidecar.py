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
from pathlib import Path
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

    def test_the_upstream_url_is_built_in_exactly_one_place(self):
        """Restoring needs the upstream path, so it cannot be banned - it is gated.

        A blanket ban would make restore impossible. Instead every upstream URL must come
        from upstream_base(), which proves the model is loaded first. This counts the
        construction sites so a second, ungated one cannot appear unnoticed. Docstrings are
        excluded: the module explains the hazard in prose on purpose.
        """
        import ast
        from pathlib import Path

        tree = ast.parse(Path("kv_rosetta/daemon/server.py").read_text())
        docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))}
        literals = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and n.value not in docstrings]
        sites = [v for v in literals if "upstream" in v]
        self.assertEqual(len(sites), 1,
                         f"expected exactly one gated upstream construction, found {sites}")

    def test_the_upstream_gate_refuses_an_unloaded_model(self):
        sidecar, swap = self.build(loaded=("loaded-one",))
        with self.assertRaises(Fallback) as caught:
            sidecar.upstream_base("other-one")
        self.assertIn("refusing to wake it", caught.exception.reason)
        for path in swap.paths:
            self.assertNotIn("/upstream/", path,
                             "an upstream URL was requested for an unloaded model")

    def test_the_upstream_gate_returns_a_url_for_a_loaded_model(self):
        sidecar, _ = self.build(loaded=("loaded-one",))
        self.assertTrue(sidecar.upstream_base("loaded-one").endswith("/upstream/loaded-one"))

    def test_a_model_name_that_could_traverse_is_refused(self):
        sidecar, _ = self.build(loaded=("../etc/passwd", ".hidden"))
        for name in ("../etc/passwd", ".hidden"):
            with self.subTest(model=name):
                with self.assertRaises(Fallback) as caught:
                    sidecar.upstream_base(name)
                self.assertIn("plain identifier", caught.exception.reason)

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

    def test_an_unconfigured_store_falls_back_rather_than_erroring(self):
        sidecar, _ = self.build(loaded=("model-a",))
        with self.assertRaises(Fallback) as caught:
            sidecar.ensure("c" * 64, "model-a")
        self.assertIn("no admitted-state store", caught.exception.reason)

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


class StoreLookupTest(SidecarTestCase):
    """Finding an artifact by prefix, and refusing to restore what cannot be verified."""

    def store_with(self, **manifest):
        import tempfile
        from kv_rosetta.admitted_store import AdmittedStore

        directory = Path(tempfile.mkdtemp())
        store = AdmittedStore(directory / "store")
        raw = directory / "raw.bin"
        raw.write_bytes(b"state bytes" * 64)
        store.admit(raw, dict(manifest))
        return store

    def build_with_store(self, store, **kwargs):
        sidecar, swap = self.build(**kwargs)
        object.__setattr__(sidecar.config, "store_root", str(store.root))
        return sidecar, swap

    def test_an_artifact_is_found_by_prefix_and_model(self):
        store = self.store_with(prefix_fingerprint="a" * 64, runtime_model="model-a",
                                prompt_token_ids=[1, 2, 3])
        sidecar, _ = self.build_with_store(store, loaded=("model-a",))
        found = sidecar.find_artifact("a" * 64, "model-a")
        self.assertIsNotNone(found)
        self.assertEqual(found.manifest["prompt_token_ids"], [1, 2, 3])

    def test_a_different_model_does_not_match(self):
        store = self.store_with(prefix_fingerprint="a" * 64, runtime_model="model-a",
                                prompt_token_ids=[1])
        sidecar, _ = self.build_with_store(store, loaded=("model-b",))
        self.assertIsNone(sidecar.find_artifact("a" * 64, "model-b"))

    def test_a_different_prefix_does_not_match(self):
        store = self.store_with(prefix_fingerprint="a" * 64, runtime_model="model-a",
                                prompt_token_ids=[1])
        sidecar, _ = self.build_with_store(store, loaded=("model-a",))
        self.assertIsNone(sidecar.find_artifact("b" * 64, "model-a"))

    def test_an_artifact_without_token_ids_is_refused_not_restored(self):
        """Reuse must be verifiable, so an artifact that cannot be probed is not used.

        Without the tokens the only evidence of success would be the runtime's own restore
        response - the claim this project exists to refuse.
        """
        store = self.store_with(prefix_fingerprint="a" * 64, runtime_model="model-a")
        sidecar, _ = self.build_with_store(store, loaded=("model-a",))
        with self.assertRaises(Fallback) as caught:
            sidecar.ensure("a" * 64, "model-a")
        self.assertIn("reuse cannot be verified", caught.exception.reason)

    def test_a_missing_artifact_falls_back_before_any_restore(self):
        store = self.store_with(prefix_fingerprint="a" * 64, runtime_model="model-a",
                                prompt_token_ids=[1])
        sidecar, swap = self.build_with_store(store, loaded=("model-a",))
        with self.assertRaises(Fallback) as caught:
            sidecar.ensure("f" * 64, "model-a")
        self.assertIn("no admitted artifact", caught.exception.reason)
        self.assertFalse([p for p in swap.paths if "action=restore" in p])

    def test_a_refused_restore_becomes_a_fallback_not_a_success(self):
        """Reaches the restore and has it fail, which no other test does.

        Every other path falls back before restoring, so the check on report.ok broke no
        test - a mutation run caught that. Here the store, model and tokens all line up and
        the restore itself refuses.
        """
        store = self.store_with(prefix_fingerprint="a" * 64, runtime_model="model-a",
                                prompt_token_ids=[1, 2, 3])
        sidecar, _ = self.build_with_store(store, loaded=("model-a",))

        import kv_rosetta.adapters.admitted_path as module
        from kv_rosetta.adapters.admitted_path import AdmittedRestoreReport

        original = module.AdmittedPath.restore
        module.AdmittedPath.restore = lambda self, digest, **kw: AdmittedRestoreReport(
            ok=False, reason="cache ABI mismatch", digest=digest)
        try:
            with self.assertRaises(Fallback) as caught:
                sidecar.ensure("a" * 64, "model-a")
        finally:
            module.AdmittedPath.restore = original
        self.assertIn("restore refused", caught.exception.reason)
        self.assertIn("cache ABI mismatch", caught.exception.reason)

    def test_a_successful_restore_reports_the_reused_tokens(self):
        store = self.store_with(prefix_fingerprint="a" * 64, runtime_model="model-a",
                                prompt_token_ids=[1, 2, 3, 4])
        sidecar, _ = self.build_with_store(store, loaded=("model-a",))

        import kv_rosetta.adapters.admitted_path as module
        from kv_rosetta.adapters.admitted_path import AdmittedRestoreReport

        original = module.AdmittedPath.restore
        module.AdmittedPath.restore = lambda self, digest, **kw: AdmittedRestoreReport(
            ok=True, digest=digest, cache_n=3, prompt_n=1, seconds=0.4,
            phases={"runtime_restore": 0.3})
        try:
            result = sidecar.ensure("a" * 64, "model-a")
        finally:
            module.AdmittedPath.restore = original
        self.assertTrue(result["restored"])
        self.assertEqual((result["cache_n"], result["prompt_n"]), (3, 1))
        self.assertEqual(result["mode"], "admitted_direct_restore")


class RequirementGateTest(unittest.TestCase):
    """The sidecar asks what the runtime can do before writing to a slot.

    Without this, a hybrid artifact restored into a build lacking the context-checkpoint patch
    is accepted and reuses nothing, reporting the same n_restored as one that reuses
    everything. The caller sees a slow response and no error at all.
    """

    def sidecar_with(self, manifest_extra, props):
        """A sidecar whose store holds one artifact and whose adapter reports `props`."""
        import types
        from kv_rosetta.daemon import server as mod

        found = types.SimpleNamespace(
            digest="d" * 64,
            manifest={"prompt_token_ids": [1, 2, 3], "runtime_model": "m", **manifest_extra})

        class Adapter:
            def __init__(self, *a, **k):
                pass

            def props(self, refresh=False):
                if props is None:
                    raise RuntimeError("server did not answer")
                return props

        sidecar = mod.Sidecar.__new__(mod.Sidecar)
        sidecar.upstream_base = lambda model: "http://127.0.0.1:1/upstream/m"
        sidecar.find_artifact = lambda fp, model: found
        sidecar.store = lambda: types.SimpleNamespace(root="/nonexistent")
        return sidecar, mod, Adapter

    def ensure(self, manifest_extra, props):
        from unittest import mock
        sidecar, mod, Adapter = self.sidecar_with(manifest_extra, props)
        with mock.patch("kv_rosetta.adapters.llamacpp_http.LlamaCppHTTPAdapter", Adapter):
            return sidecar.ensure("a" * 64, "m")

    def test_a_runtime_missing_the_patch_is_refused_before_any_restore(self):
        from kv_rosetta.daemon.server import Fallback
        needs = {"sequence_state_version": 3, "needs_checkpoint_persistence": True,
                 "checkpoint_format": "sckp/1"}
        with self.assertRaises(Fallback) as caught:
            self.ensure({"requirements": needs}, {"sequence_state_version": 3})
        self.assertIn("252 of 256", str(caught.exception))

    def test_a_runtime_that_will_not_answer_props_is_refused_rather_than_trusted(self):
        from kv_rosetta.daemon.server import Fallback
        needs = {"sequence_state_version": 3, "needs_checkpoint_persistence": True}
        with self.assertRaises(Fallback) as caught:
            self.ensure({"requirements": needs}, None)
        self.assertIn("restore blind", str(caught.exception))

    def test_an_artifact_with_no_recorded_requirements_does_not_consult_the_runtime(self):
        """Artifacts written before this field existed must not become unrestorable.

        Asserted by making props() fail if it is reached at all, rather than by inspecting a
        later error - the restore itself goes on to fail here for unrelated reasons and its
        message would prove nothing either way.
        """
        from unittest import mock
        called = []

        class Adapter:
            def __init__(self, *a, **k):
                pass

            def props(self, refresh=False):
                called.append(True)
                raise AssertionError("props must not be consulted with no requirements")

        sidecar, mod, _ = self.sidecar_with({}, {})
        with mock.patch("kv_rosetta.adapters.llamacpp_http.LlamaCppHTTPAdapter", Adapter):
            with self.assertRaises(Exception):
                sidecar.ensure("a" * 64, "m")
        self.assertEqual(called, [])
