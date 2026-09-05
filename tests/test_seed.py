"""REQ-113: seeded, pinned harness prefixes and the read-only probe.

The first conversation a harness opens on a model it has never served pays the whole
prefill (Claude Code on ornith: 67k tokens, 79 s, client gave up), the artifact exists only
after that victim, and capture churn evicts it. Seed makes it exist first; pin keeps it;
the probe lets a router know before choosing the model.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from kv_rosetta.daemon import capture
from kv_rosetta.daemon.server import SEED_TAIL_SLACK

from tests.test_restore_for_prompt import MSGS, StubStore, artifact
from tests.test_sidecar import SidecarTestCase


class PinTests(unittest.TestCase):
    def _store(self, n):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name); objs = []
        for i in range(n):
            digest = f"{i:064x}"
            path = root / f"{digest}.state"; path.write_bytes(b"x")
            (root / f"{digest}.manifest.json").write_text("{}")
            import os; os.utime(path, (1_000_000 + i, 1_000_000 + i))     # i = newer
            objs.append(SimpleNamespace(digest=digest, path=path,
                                        manifest={"runtime_model": "m"}))
        return SimpleNamespace(root=root, list_objects=lambda: list(objs)), objs

    def test_a_pinned_artifact_is_neither_counted_nor_evicted(self):
        store, objs = self._store(6)
        capture.pin_artifact(store, objs[0].digest)              # the OLDEST one
        dropped = capture.prune_model_artifacts(store, "m", keep=2)
        self.assertEqual(dropped, 3)                             # 5 unpinned -> keep 2
        self.assertTrue(objs[0].path.exists(), "pinned must survive")
        self.assertTrue(objs[5].path.exists() and objs[4].path.exists())
        self.assertFalse(objs[1].path.exists())

    def test_unpin_lets_it_go_again(self):
        store, objs = self._store(3)
        capture.pin_artifact(store, objs[0].digest)
        self.assertTrue(capture.unpin_artifact(store, objs[0].digest))
        self.assertFalse(capture.unpin_artifact(store, objs[0].digest))
        capture.prune_model_artifacts(store, "m", keep=1)
        self.assertFalse(objs[0].path.exists())


class SeedAdapter:
    """Renders deterministically; a prefill fills the chosen slot with `ids` tokens."""

    def __init__(self, slots, ids, *, after_prefill=None):
        self.slots, self.ids = slots, ids
        self.after_prefill = after_prefill
        self.posts: list[tuple[str, dict]] = []

    def apply_template(self, messages, tools=None, extra=None):
        self.messages, self.extra = messages, extra
        return "rendered"

    def tokenize(self, text):
        return list(self.ids)

    def _get(self, path):
        return self.slots

    def _post(self, path, payload):
        self.posts.append((path, payload))
        if path == "/v1/chat/completions":
            slot = next(s for s in self.slots if s["id"] == payload["id_slot"])
            slot["n_prompt_tokens"] = (self.after_prefill if self.after_prefill is not None
                                       else len(self.ids))
            return {"choices": []}
        if "action=save" in path:
            return {"n_saved": len(self.ids), "n_written": 10, "filename": payload["filename"]}
        raise AssertionError(path)


class SeedTests(SidecarTestCase):
    def _sidecar(self, objs=(), loaded=("model-a",), admitter="default"):
        sidecar, _ = self.build(loaded=loaded)
        sidecar.store = lambda: StubStore(list(objs))
        self.admits = []
        if admitter == "default":
            sidecar.admitter = lambda model, name, saved, *, pin=False: (
                self.admits.append((model, name, pin)) or f"deadbeef0000 covering {len(self.ids):,} tokens")
        return sidecar

    ids = list(range(1, 3001))

    def test_seeds_an_idle_slot_then_captures_admits_and_pins(self):
        sidecar = self._sidecar()
        slots = [{"id": 0, "is_processing": True, "n_prompt_tokens": 500},
                 {"id": 1, "is_processing": False, "n_prompt_tokens": 0}]
        adapter = SeedAdapter(slots, self.ids)
        out = sidecar.seed("model-a", [MSGS[0]], tools=[{"type": "function"}],
                           template_fields={"reasoning_effort": "medium"}, adapter=adapter)
        self.assertTrue(out["seeded"], out)
        self.assertEqual((out["slot"], out["tokens"]), (1, 3000))
        # a user turn was appended: hybrid checkpoints land at user-turn starts
        self.assertEqual(adapter.messages[-1]["role"], "user")
        prefill = dict(adapter.posts[0][1])
        self.assertEqual((adapter.posts[0][0], prefill["id_slot"], prefill["max_tokens"],
                          prefill["reasoning_effort"], prefill["tools"]),
                         ("/v1/chat/completions", 1, 1, "medium", [{"type": "function"}]))
        self.assertIn("action=save", adapter.posts[1][0])
        self.assertEqual(self.admits, [("model-a", "seed-model-a-slot1-3000.state", True)])
        self.assertEqual(set(out["stages"]), {"probe", "prefill", "save", "admit"})

    def test_takes_the_idle_slot_with_the_least_to_lose(self):
        # Live: slot 0 held an idle 57,840-token session, slot 1 held 798; the seed took 0.
        sidecar = self._sidecar()
        slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 57840},
                 {"id": 1, "is_processing": False, "n_prompt_tokens": 798},
                 {"id": 2, "is_processing": True, "n_prompt_tokens": 0}]
        out = sidecar.seed("model-a", [MSGS[0]], adapter=SeedAdapter(slots, self.ids))
        self.assertEqual(out["slot"], 1, out)

    def test_a_prefix_already_held_is_not_seeded_again(self):
        held = artifact("model-a", self.ids[:3000 - SEED_TAIL_SLACK + 1], "b" * 64)
        sidecar = self._sidecar([held])
        adapter = SeedAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 0}], self.ids)
        out = sidecar.seed("model-a", [MSGS[0]], adapter=adapter)
        self.assertEqual((out["seeded"], out.get("already")), (False, True))
        self.assertEqual(adapter.posts, [], "no prefill, no save")

    def test_a_shorter_held_prefix_does_not_block_the_seed(self):
        held = artifact("model-a", self.ids[:1500], "b" * 64)
        sidecar = self._sidecar([held])
        adapter = SeedAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 0}], self.ids)
        self.assertTrue(sidecar.seed("model-a", [MSGS[0]], adapter=adapter)["seeded"])

    def test_never_touches_a_busy_slot_or_wakes_a_model(self):
        sidecar = self._sidecar()
        adapter = SeedAdapter([{"id": 0, "is_processing": True, "n_prompt_tokens": 9}], self.ids)
        out = sidecar.seed("model-a", [MSGS[0]], adapter=adapter)
        self.assertEqual((out["seeded"], out["reason"]), (False, "every slot is busy"))
        self.assertEqual(adapter.posts, [])
        sidecar = self._sidecar(loaded=("other",))
        out = sidecar.seed("model-a", [MSGS[0]], adapter=adapter)
        self.assertFalse(out["seeded"]); self.assertEqual(adapter.posts, [])

    def test_a_slot_that_did_not_take_the_prefill_is_not_captured(self):
        sidecar = self._sidecar()
        adapter = SeedAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 0}],
                              self.ids, after_prefill=100)
        out = sidecar.seed("model-a", [MSGS[0]], adapter=adapter)
        self.assertFalse(out["seeded"]); self.assertIn("holds 100", out["reason"])
        self.assertEqual(len(adapter.posts), 1, "no save after a short prefill")
        self.assertEqual(self.admits, [])

    def test_without_an_admitter_nothing_is_prefilled(self):
        sidecar = self._sidecar(admitter=None)
        sidecar.admitter = None
        adapter = SeedAdapter([{"id": 0, "is_processing": False, "n_prompt_tokens": 0}], self.ids)
        out = sidecar.seed("model-a", [MSGS[0]], adapter=adapter)
        self.assertFalse(out["seeded"]); self.assertEqual(adapter.posts, [])


class ProbeTests(SidecarTestCase):
    def test_dry_run_answers_would_restore_without_reading_or_writing_a_slot(self):
        from tests.test_restore_for_prompt import StubAdapter
        sidecar, _ = self.build(loaded=("model-a",))
        sidecar.store = lambda: StubStore([artifact("model-a", range(1, 3001), "c" * 64)])
        adapter = StubAdapter([{"id": 0, "is_processing": True, "n_prompt_tokens": 1}])
        with mock.patch.object(sidecar, "ensure") as ensure:
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter, dry_run=True)
        self.assertEqual((out["restored"], out["would_restore"], out["covers_tokens"],
                          out["shared_tokens"]), (False, True, 3000, 3000))
        self.assertNotIn("/slots", adapter.calls)
        ensure.assert_not_called()
        miss = sidecar.restore_for_prompt("model-a", MSGS, dry_run=True,
                                          adapter=StubAdapter([], ids=list(range(9, 30))))
        self.assertEqual(miss["would_restore"], False)


class YieldingSeedAdapter(SeedAdapter):
    """A prefill that blocks until aborted, marking its slot busy meanwhile."""

    def __init__(self, slots, ids, *, abortable=True):
        super().__init__(slots, ids)
        import threading
        self.blocked = threading.Event()
        self.started = threading.Event()
        self.abortable = abortable
        self.aborted = False

    def prefill(self, body):
        self.posts.append(("/v1/chat/completions", body))
        slot = next(s for s in self.slots if s["id"] == body["id_slot"])
        slot["is_processing"] = True
        self.started.set()
        self.blocked.wait(10)
        if self.aborted:
            slot["is_processing"] = False
            raise RuntimeError("connection shut down")
        slot["is_processing"] = False
        slot["n_prompt_tokens"] = len(self.ids)
        return {"choices": []}

    def abort_prefill(self):
        if not self.abortable:
            return False
        self.aborted = True
        self.blocked.set()
        return True


class SeedYieldTests(SidecarTestCase):
    """REQ-114: a live turn always beats a background seed on the same runtime."""

    ids = list(range(1, 3001))

    def _sidecar(self, objs):
        sidecar, _ = self.build(loaded=("model-a",))
        sidecar.store = lambda: StubStore(list(objs))
        sidecar.admitter = lambda model, name, saved, *, pin=False: "x"
        return sidecar

    def test_a_live_restore_aborts_the_seed_and_proceeds(self):
        import threading
        held = artifact("model-a", self.ids[:2000], "d" * 64)
        sidecar = self._sidecar([held])
        slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 0},
                 {"id": 1, "is_processing": False, "n_prompt_tokens": 0}]
        adapter = YieldingSeedAdapter(slots, self.ids)
        seed_out = {}
        seed_msgs = [{"role": "system", "content": "a different, unheld head"}]
        seeder = threading.Thread(target=lambda: seed_out.update(
            sidecar.seed("model-a", seed_msgs, adapter=adapter)))
        seeder.start()
        self.assertTrue(adapter.started.wait(5), "seed never began its prefill")
        with mock.patch.object(sidecar, "ensure", return_value={}) as ensure:
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        seeder.join(5)
        self.assertTrue(adapter.aborted, "the seed's connection must be dropped")
        self.assertEqual((seed_out.get("seeded"), seed_out.get("yielded")), (False, True))
        self.assertTrue(out["restored"], out)
        self.assertIn("yield", out["stages"])
        ensure.assert_called_once()
        self.assertNotIn("model-a", sidecar._seeds)

    def test_a_seed_that_will_not_yield_makes_the_restore_answer_busy_fast(self):
        import threading
        from kv_rosetta.daemon import server as module
        held = artifact("model-a", self.ids[:2000], "d" * 64)
        sidecar = self._sidecar([held])
        slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 0}]
        adapter = YieldingSeedAdapter(slots, self.ids, abortable=False)
        seeder = threading.Thread(target=lambda: sidecar.seed(
            "model-a", [{"role": "system", "content": "unheld"}], adapter=adapter))
        seeder.start()
        self.assertTrue(adapter.started.wait(5))
        with mock.patch.object(module, "SEED_YIELD_WAIT_S", 0.3), \
                mock.patch.object(sidecar, "ensure") as ensure:
            out = sidecar.restore_for_prompt("model-a", MSGS, adapter=adapter)
        adapter.blocked.set(); seeder.join(5)
        self.assertEqual((out["restored"], out["busy"]), (False, "seeding"))
        self.assertIn("did not yield", out["reason"])
        ensure.assert_not_called()

    def test_a_second_seed_on_the_same_runtime_is_refused_while_one_runs(self):
        import threading
        sidecar = self._sidecar([])
        slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 0},
                 {"id": 1, "is_processing": False, "n_prompt_tokens": 0}]
        adapter = YieldingSeedAdapter(slots, self.ids)
        seeder = threading.Thread(target=lambda: sidecar.seed(
            "model-a", [{"role": "system", "content": "one"}], adapter=adapter))
        seeder.start(); self.assertTrue(adapter.started.wait(5))
        out = sidecar.seed("model-a", [{"role": "system", "content": "two"}],
                           adapter=SeedAdapter(slots, self.ids))
        adapter.blocked.set(); seeder.join(5)
        self.assertFalse(out["seeded"]); self.assertIn("already running", out["reason"])


class CallerGoneTest(unittest.TestCase):
    def _handler(self, conn):
        from kv_rosetta.daemon.server import _make_handler
        Handler = _make_handler(SimpleNamespace(_lock=mock.MagicMock(), stats=SimpleNamespace()))
        h = Handler.__new__(Handler); h.connection = conn
        return h

    def test_an_open_idle_connection_is_not_gone(self):
        import socket
        a, b = socket.socketpair(); self.addCleanup(a.close); self.addCleanup(b.close)
        self.assertFalse(self._handler(a)._caller_gone())

    def test_a_closed_peer_is_gone(self):
        import socket
        a, b = socket.socketpair(); self.addCleanup(a.close)
        b.close()
        self.assertTrue(self._handler(a)._caller_gone())
