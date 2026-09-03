"""Request-time restore: the half of KVX that works on a busy fleet.

The load-time restore fills an EMPTY slot. After a model's first request no slot is ever
empty again, so a new conversation met no restore and prefilled cold. These tests pin the
request-time path: render with the runtime, tokenize with the runtime, pick the LONGEST
matching attachment, pick the slot llama.cpp would evict anyway, restore, and answer with
a verdict rather than an error.
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from kv_rosetta.daemon.server import Fallback

from tests.test_sidecar import SidecarTestCase


class StubAdapter:
    """Renders and tokenizes deterministically; reports the slots it is told to."""

    def __init__(self, slots, *, prompt="rendered", ids=None, fail_render=False):
        self.slots = slots
        self.prompt = prompt
        self.ids = ids if ids is not None else list(range(1, 41))
        self.fail_render = fail_render
        self.calls: list[str] = []

    def apply_template(self, messages, tools=None):
        self.calls.append("apply_template")
        if self.fail_render:
            raise RuntimeError("no template")
        return self.prompt

    def tokenize(self, text):
        self.calls.append("tokenize")
        return list(self.ids)

    def _get(self, path):
        self.calls.append(path)
        return self.slots


def artifact(model, ids, fingerprint):
    return SimpleNamespace(manifest={"runtime_model": model, "prompt_token_ids": list(ids),
                                     "prefix_fingerprint": fingerprint},
                           digest=fingerprint, path=None)


class StubStore:
    def __init__(self, objs):
        self._objs = objs
        self.root = "/nonexistent"

    def list_objects(self):
        return list(self._objs)


MSGS = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}]
FP_SHORT = "a" * 64
FP_LONG = "b" * 64


class RestoreForPromptTest(SidecarTestCase):
    def _sidecar(self, objs, loaded=("model-a",)):
        sidecar, swap = self.build(loaded=loaded)
        sidecar.store = lambda: StubStore(objs)
        return sidecar, swap

    def test_picks_the_longest_matching_attachment(self):
        ids = list(range(1, 41))
        sidecar, _ = self._sidecar([artifact("model-a", ids[:10], FP_SHORT),
                                    artifact("model-a", ids[:30], FP_LONG)])
        adapter = StubAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 0}], ids=ids)
        with mock.patch.object(sidecar, "ensure", return_value={"digest": FP_LONG}) as ensure:
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        self.assertTrue(out["restored"], out)
        self.assertEqual(out["covers_tokens"], 30)
        self.assertEqual(ensure.call_args.args[0], FP_LONG)
        # rendered and tokenized by the runtime, in that order, before any lookup
        self.assertEqual(adapter.calls[:2], ["apply_template", "tokenize"])

    def test_an_attachment_that_is_not_a_prefix_does_not_match(self):
        ids = list(range(1, 41))
        wrong = ids[:20]
        wrong[5] = 999
        sidecar, _ = self._sidecar([artifact("model-a", wrong, FP_SHORT)])
        adapter = StubAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 0}], ids=ids)
        with mock.patch.object(sidecar, "ensure") as ensure:
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        self.assertFalse(out["restored"])
        self.assertIn("no attachment is a prefix", out["reason"])
        ensure.assert_not_called()

    def test_an_attachment_longer_than_the_prompt_does_not_match(self):
        ids = list(range(1, 11))
        sidecar, _ = self._sidecar([artifact("model-a", list(range(1, 41)), FP_LONG)])
        adapter = StubAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 0}], ids=ids)
        with mock.patch.object(sidecar, "ensure") as ensure:
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        self.assertFalse(out["restored"])
        ensure.assert_not_called()

    def test_another_models_attachment_is_ignored(self):
        ids = list(range(1, 41))
        sidecar, _ = self._sidecar([artifact("model-b", ids[:30], FP_LONG)])
        adapter = StubAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 0}], ids=ids)
        with mock.patch.object(sidecar, "ensure") as ensure:
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        self.assertFalse(out["restored"])
        ensure.assert_not_called()

    def test_prefers_an_empty_idle_slot(self):
        ids = list(range(1, 41))
        sidecar, _ = self._sidecar([artifact("model-a", ids[:30], FP_LONG)])
        adapter = StubAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 500},
                               {"id": 1, "is_processing": False, "n_prompt_tokens": 0}], ids=ids)
        with mock.patch.object(sidecar, "ensure", return_value={}) as ensure:
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        self.assertEqual(out["slot"], 1)
        self.assertEqual(ensure.call_args.args[2], 1)

    def test_with_no_empty_slot_takes_the_least_recently_restored(self):
        # This is the busy-fleet case the endpoint exists for. Slot 1 was restored into
        # more recently than slot 0, so slot 0 is what llama.cpp would evict.
        ids = list(range(1, 41))
        sidecar, _ = self._sidecar([artifact("model-a", ids[:30], FP_LONG)])
        sidecar._slot_last_used = {("model-a", 0): 10.0, ("model-a", 1): 20.0}
        adapter = StubAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 500},
                               {"id": 1, "is_processing": False, "n_prompt_tokens": 700}], ids=ids)
        with mock.patch.object(sidecar, "ensure", return_value={}):
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        self.assertEqual(out["slot"], 0)
        # and the memo now records this restore, so next time slot 1 is the older one
        self.assertGreater(sidecar._slot_last_used[("model-a", 0)], 20.0)

    def test_never_touches_a_busy_slot(self):
        ids = list(range(1, 41))
        sidecar, _ = self._sidecar([artifact("model-a", ids[:30], FP_LONG)])
        adapter = StubAdapter([{"id": 0, "is_processing": True, "n_prompt_tokens": 0},
                               {"id": 1, "is_processing": True, "n_prompt_tokens": 0}], ids=ids)
        with mock.patch.object(sidecar, "ensure") as ensure:
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        self.assertFalse(out["restored"])
        self.assertIn("busy", out["reason"])
        ensure.assert_not_called()

    def test_never_wakes_an_unloaded_model(self):
        # upstream_base refuses a model llama-swap does not report as running; FakeSwap
        # asserts if anything waking is requested.
        sidecar, swap = self._sidecar([], loaded=("model-a",))
        adapter = StubAdapter([])
        out = sidecar.restore_for_prompt("model-z", MSGS, adapter=adapter)
        self.assertFalse(out["restored"])
        self.assertEqual(adapter.calls, [])          # never even rendered

    def test_a_refused_restore_is_a_verdict_not_an_error(self):
        ids = list(range(1, 41))
        sidecar, _ = self._sidecar([artifact("model-a", ids[:30], FP_LONG)])
        adapter = StubAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 0}], ids=ids)
        with mock.patch.object(sidecar, "ensure", side_effect=Fallback("tail contract violated")):
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        self.assertFalse(out["restored"])
        self.assertIn("refused: tail contract violated", out["reason"])

    def test_a_render_failure_is_a_verdict_not_an_error(self):
        sidecar, _ = self._sidecar([])
        adapter = StubAdapter([], fail_render=True)
        out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        self.assertFalse(out["restored"])
        self.assertIn("could not render", out["reason"])

    def test_empty_messages_is_a_miss(self):
        sidecar, _ = self._sidecar([])
        out = sidecar.restore_for_prompt("model-a", [], adapter=StubAdapter([]))
        self.assertFalse(out["restored"])


class RouteTest(SidecarTestCase):
    def test_the_route_answers_200_with_a_verdict(self):
        import threading
        import urllib.request
        sidecar, _ = self.build(loaded=("model-a",))
        sidecar.store = lambda: StubStore([])
        with mock.patch.object(sidecar, "restore_for_prompt",
                               return_value={"restored": False, "reason": "stubbed"}) as rfp:
            t = threading.Thread(target=sidecar.serve_forever, daemon=True)
            t.start()
            self.addCleanup(sidecar.shutdown)
            for _ in range(200):
                if sidecar._server is not None:
                    break
                threading.Event().wait(0.01)
            body = json.dumps({"model": "model-a", "messages": MSGS,
                               "tools": [{"type": "function"}]}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{sidecar.port}/v1/restore-for-prompt", data=body,
                headers={"Content-Type": "application/json"})
            with self._original(req, timeout=5) as r:
                self.assertEqual(r.status, 200)
                out = json.loads(r.read())
        self.assertEqual(out["ok"], True)
        self.assertEqual(out["restored"], False)
        self.assertEqual(rfp.call_args.args[0], "model-a")
        self.assertEqual(rfp.call_args.args[2], [{"type": "function"}])
