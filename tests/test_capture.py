"""Automatic capture: save warm slots before they are lost.

A cache is only missed once it is gone, so waiting to be asked means never capturing. The
risk is looking like the recompute warmer that was removed, and the distinction is that this
sends no requests: it reads slot status and saves state that already exists on models that
are already loaded.
"""

import unittest

from kv_rosetta.daemon.capture import (
    Candidate, DEFAULT_MIN_TOKENS, choose_candidates, require_loaded_only,
    slot_is_capturable,
)


def _slot(idx, tokens, busy=False):
    return {"id": idx, "n_prompt_tokens": tokens, "is_processing": busy}


class SlotEligibility(unittest.TestCase):
    def test_a_large_idle_slot_is_capturable(self):
        self.assertTrue(slot_is_capturable(_slot(0, 40000)))

    def test_a_busy_slot_is_never_captured(self):
        # Mid-request state describes a prompt still being built; saving it captures a
        # partial prefix that would later restore as though it were complete.
        self.assertFalse(slot_is_capturable(_slot(0, 40000, busy=True)))

    def test_a_small_slot_is_not_worth_an_artifact(self):
        self.assertFalse(slot_is_capturable(_slot(0, 200)))
        self.assertFalse(slot_is_capturable(_slot(0, DEFAULT_MIN_TOKENS - 1)))

    def test_the_threshold_is_inclusive(self):
        self.assertTrue(slot_is_capturable(_slot(0, DEFAULT_MIN_TOKENS)))

    def test_a_slot_with_no_token_count_is_not_captured(self):
        self.assertFalse(slot_is_capturable({"id": 0, "is_processing": False}))


class Candidates(unittest.TestCase):
    def test_the_largest_context_comes_first(self):
        # The biggest context is the one whose loss costs most to rebuild.
        got = choose_candidates({"m": [_slot(0, 9000), _slot(1, 70000)]})
        self.assertEqual([c.tokens for c in got], [70000, 9000])

    def test_busy_and_small_slots_are_excluded(self):
        got = choose_candidates({"m": [_slot(0, 70000, busy=True), _slot(1, 10)]})
        self.assertEqual(got, [])

    def test_an_unchanged_slot_is_not_written_again(self):
        already = frozenset({("m", 70000)})
        self.assertEqual(choose_candidates({"m": [_slot(0, 70000)]}, already=already), [])

    def test_a_grown_slot_is_captured_again(self):
        # The context moved on; the previous artifact no longer covers it.
        already = frozenset({("m", 70000)})
        got = choose_candidates({"m": [_slot(0, 82000)]}, already=already)
        self.assertEqual([c.tokens for c in got], [82000])

    def test_several_models_are_all_considered(self):
        got = choose_candidates({"a": [_slot(0, 30000)], "b": [_slot(0, 50000)]})
        self.assertEqual([c.model for c in got], ["b", "a"])


class NeverWakes(unittest.TestCase):
    """The single rule separating capture from the warmer this project removed."""

    def test_a_model_that_is_not_loaded_is_dropped(self):
        cands = [Candidate("parked", 0, 90000), Candidate("live", 0, 50000)]
        got = require_loaded_only(cands, frozenset({"live"}))
        self.assertEqual([c.model for c in got], ["live"])

    def test_nothing_loaded_means_nothing_to_capture(self):
        self.assertEqual(require_loaded_only([Candidate("m", 0, 90000)], frozenset()), [])


class LoopBehaviour(unittest.TestCase):
    """The loop, with the runtime faked. No network, no server."""

    def _loop(self, running, slots, saves=None, fail=()):
        from kv_rosetta.daemon.capture import CaptureLoop
        loop = CaptureLoop("http://swap", "/store", min_tokens=4096)
        calls = []

        def fake(url, payload=None, timeout=900):
            calls.append(url)
            if url.endswith("/running"):
                return {"running": running}
            if url.endswith("/slots"):
                model = url.split("/upstream/")[1].split("/")[0]
                return slots.get(model, [])
            if "action=save" in url:
                if any(f in url for f in fail):
                    raise RuntimeError("save refused by runtime")
                return (saves or {}).get("default", {"n_saved": 1, "n_written": 10})
            raise AssertionError(url)

        loop._json = fake
        loop.calls = calls
        return loop

    def test_only_ready_models_are_polled(self):
        # A loading model has no settled state, and asking is how a warmer starts.
        loop = self._loop([{"model": "a", "state": "ready"},
                           {"model": "b", "state": "starting"}],
                          {"a": [_slot(0, 50000)], "b": [_slot(0, 90000)]})
        done = loop.tick()
        self.assertEqual([c.model for c in done], ["a"])
        self.assertFalse(any("/upstream/b/" in c for c in loop.calls))

    def test_a_runtime_without_a_slot_api_is_skipped_not_fatal(self):
        # vLLM answers 404 on /slots; that is nothing to capture, not an error.
        from kv_rosetta.daemon.capture import CaptureLoop
        loop = CaptureLoop("http://swap", "/store")
        def fake(url, payload=None, timeout=900):
            if url.endswith("/running"):
                return {"running": [{"model": "vllm-model", "state": "ready"}]}
            raise RuntimeError("404 Not Found")
        loop._json = fake
        self.assertEqual(loop.tick(), [])

    def test_the_same_context_is_not_saved_twice(self):
        loop = self._loop([{"model": "a", "state": "ready"}], {"a": [_slot(0, 50000)]})
        self.assertEqual(len(loop.tick()), 1)
        self.assertEqual(loop.tick(), [])
        self.assertEqual(loop.captured, 1)

    def test_a_grown_context_is_saved_again(self):
        slots = {"a": [_slot(0, 50000)]}
        loop = self._loop([{"model": "a", "state": "ready"}], slots)
        loop.tick()
        slots["a"] = [_slot(0, 61000)]
        self.assertEqual([c.tokens for c in loop.tick()], [61000])

    def test_a_refused_save_is_counted_and_not_remembered_as_done(self):
        # Otherwise a transient failure would permanently skip that context.
        loop = self._loop([{"model": "a", "state": "ready"}], {"a": [_slot(0, 50000)]},
                          fail=("action=save",))
        self.assertEqual(loop.tick(), [])
        self.assertEqual(loop.refused, 1)
        loop2_seen = loop._seen
        self.assertEqual(loop2_seen, set())

    def test_nothing_is_ever_sent_to_a_completion_endpoint(self):
        # The property that separates this from the recompute warmer.
        loop = self._loop([{"model": "a", "state": "ready"}], {"a": [_slot(0, 50000)]})
        loop.tick()
        self.assertFalse(any("completion" in c or "chat" in c for c in loop.calls))
        self.assertFalse(any("action=restore" in c or "action=erase" in c
                             for c in loop.calls))
