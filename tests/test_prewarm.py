"""Pre-warm refusals.

A cache attachment is per model, so a cold prefill is never caused by another model owning
the cache - it means nobody warmed this one. Warming ahead of time removes the cost, and the
risks are all in doing it carelessly: waking a parked model (the behaviour kvwarm was removed
for) or saving the wrong slot (which publishes an artifact describing other text).
"""

import json
import unittest

from kv_rosetta.prewarm import (
    PrewarmError, choose_slot, replay_body, require_loaded,
)


def _slots(*specs):
    return [{"id": i, "n_prompt_tokens": n, "is_processing": busy}
            for i, (n, busy) in enumerate(specs)]


class ReplayBody(unittest.TestCase):
    def test_the_prefix_is_composed_as_the_corpus_records_it(self):
        body = replay_body({"system": "you are careful", "tools": None}, "m")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "you are careful"})
        # The "." turn makes the request valid without extending the prefix being cached.
        self.assertEqual(body["messages"][1]["content"], ".")
        self.assertEqual(body["temperature"], 0)

    def test_tools_are_rendered_as_functions(self):
        tools = json.dumps([{"name": "search", "description": "d", "parameters": {}}])
        body = replay_body({"system": "s", "tools": tools}, "m")
        self.assertEqual(body["tools"][0]["function"]["name"], "search")

    def test_tools_may_be_a_list_already(self):
        body = replay_body({"system": "s", "tools": [{"name": "x"}]}, "m")
        self.assertEqual(body["tools"][0]["function"]["name"], "x")

    def test_a_manifest_without_a_system_prefix_is_refused(self):
        for bad in (None, "", "   ", 7):
            with self.assertRaises(PrewarmError) as caught:
                replay_body({"system": bad}, "m")
            self.assertIn("no system prefix", str(caught.exception))


class RefusesToWake(unittest.TestCase):
    def test_a_loaded_model_is_warmed(self):
        require_loaded(["tiel-kvx-w6800"], "tiel-kvx-w6800")

    def test_a_parked_model_is_left_alone(self):
        # The distinction between a pre-warm and the recompute warmer that was removed.
        with self.assertRaises(PrewarmError) as caught:
            require_loaded(["other"], "qwen38-27b-w6800")
        self.assertIn("refusing to wake a parked model", str(caught.exception))

    def test_the_refusal_names_what_is_loaded(self):
        with self.assertRaises(PrewarmError) as caught:
            require_loaded(["other"], "missing")
        self.assertIn("other", str(caught.exception))

    def test_waking_can_be_asked_for_explicitly(self):
        require_loaded([], "qwen38-27b-w6800", allow_wake=True)


class ChooseSlot(unittest.TestCase):
    def test_the_slot_holding_the_replay_is_chosen(self):
        choice = choose_slot(_slots((6169, False), (12298, False)), 12298)
        self.assertEqual(choice.slot_id, 1)
        self.assertEqual(choice.held_tokens, 12298)

    def test_a_busy_slot_is_never_chosen_even_if_it_matches(self):
        # Live traffic is never disturbed, and a busy slot's contents are not ours to save.
        with self.assertRaises(PrewarmError):
            choose_slot(_slots((12298, True),), 12298)

    def test_a_near_miss_is_refused_rather_than_saved(self):
        # The live-wiring failure: a stale 6,169-token slot for a 12,298-token replay.
        # Saving it publishes an artifact whose bytes describe a different prompt.
        with self.assertRaises(PrewarmError) as caught:
            choose_slot(_slots((6169, False),), 12298)
        self.assertIn("6169 tokens against 12298", str(caught.exception))

    def test_a_small_drift_is_tolerated(self):
        choose_slot(_slots((12295, False),), 12298)

    def test_all_slots_busy_is_refused(self):
        with self.assertRaises(PrewarmError) as caught:
            choose_slot(_slots((10, True), (20, True)), 12298)
        self.assertIn("every slot is busy", str(caught.exception))

    def test_a_nonpositive_expectation_is_refused(self):
        # Zero would match an empty slot and publish an artifact of nothing.
        for bad in (0, -1):
            with self.assertRaises(PrewarmError):
                choose_slot(_slots((0, False),), bad)


if __name__ == "__main__":
    unittest.main()


class _Artifact:
    def __init__(self, model, tokens, digest="d"):
        self.manifest = {"runtime_model": model, "prompt_token_count": tokens}
        self.digest = digest
        self.basename = digest + ".state"


class BuildOnWhatTheModelAlreadyHas(unittest.TestCase):
    """Warming from nothing recomputes text the model has already read.

    Restoring this model's own previous attachment first means the replay prefills only what
    is new. Measured on a growing prefix: 820 of 892 tokens reused, 108ms -> 19ms.
    """

    def test_the_largest_prior_attachment_is_chosen(self):
        from kv_rosetta.prewarm import best_prior
        arts = [_Artifact("m", 500, "a"), _Artifact("m", 12000, "b"), _Artifact("m", 900, "c")]
        self.assertEqual(best_prior(arts, "m").digest, "b")

    def test_another_models_attachment_is_never_a_head_start(self):
        # llama.cpp would not catch this: it checks the token prefix, not the weights.
        from kv_rosetta.prewarm import best_prior
        self.assertIsNone(best_prior([_Artifact("other", 40000)], "m"))

    def test_no_prior_attachment_returns_none_rather_than_guessing(self):
        from kv_rosetta.prewarm import best_prior
        self.assertIsNone(best_prior([], "m"))

    def test_an_artifact_without_a_token_count_does_not_win_by_accident(self):
        from kv_rosetta.prewarm import best_prior
        blank = _Artifact("m", 0, "blank")
        blank.manifest.pop("prompt_token_count")
        self.assertEqual(best_prior([blank, _Artifact("m", 10, "real")], "m").digest, "real")
