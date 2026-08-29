"""A protocol that is absent, false, malformed or unknown must never enable hybrid import.

Support is enabled by a complete and exact runtime statement, and by nothing else - not an
architecture name, a filename, a strings(1) match, an artifact size, or the mere presence of
the SCKP magic. These run offline against a stubbed /props so every malformed shape can be
exercised without a GPU.
"""

import unittest

from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter


class _StubAdapter(LlamaCppHTTPAdapter):
    """An adapter whose runtime answers exactly what a case needs."""

    def __init__(self, props: dict):
        super().__init__("http://127.0.0.1:1", "/nonexistent")
        self._props = props

    def props(self, refresh: bool = False) -> dict:
        return self._props


def _props(**overrides) -> dict:
    base = {
        "slot_checkpoint_persistence": True,
        "slot_checkpoint_format": "sckp/1",
        "sequence_state_version": 3,
        "supports_target_checkpoint_state": True,
        "supports_draft_checkpoint_state": False,
        "supports_speculative_checkpoint_state": False,
        "sckp_serializes_target_state": True,
        "sckp_serializes_draft_state": True,
        "sckp_serializes_speculative_state": True,
    }
    base.update(overrides)
    return base


class ProtocolGatingTests(unittest.TestCase):
    def _complete(self, props):
        adapter = _StubAdapter(props)
        return adapter._protocol_is_complete(adapter.checkpoint_protocol())

    def test_a_complete_protocol_is_accepted(self):
        ok, reason = self._complete(_props())
        self.assertTrue(ok, reason)

    def test_absent_protocol_is_refused(self):
        ok, reason = self._complete({})
        self.assertFalse(ok)
        self.assertIn("no checkpoint-persistence protocol", reason)

    def test_persistence_false_is_refused(self):
        ok, _ = self._complete(_props(slot_checkpoint_persistence=False))
        self.assertFalse(ok)

    def test_unknown_format_is_refused(self):
        ok, reason = self._complete(_props(slot_checkpoint_format="sckp/9"))
        self.assertFalse(ok)
        self.assertIn("unrecognised checkpoint format", reason)

    def test_missing_format_is_refused(self):
        ok, _ = self._complete(_props(slot_checkpoint_format=""))
        self.assertFalse(ok)

    def test_unsupported_sequence_version_is_refused(self):
        ok, reason = self._complete(_props(sequence_state_version=99))
        self.assertFalse(ok)
        self.assertIn("unsupported sequence-state version", reason)

    def test_malformed_sequence_version_is_refused(self):
        for bad in ("3", None, 3.5, [3]):
            with self.subTest(value=bad):
                ok, _ = self._complete(_props(sequence_state_version=bad))
                self.assertFalse(ok)

    def test_serialization_alone_does_not_enable_support(self):
        """The exact confusion this protocol split exists to prevent: the format carries
        the blob, but nothing has shown it restores."""
        ok, reason = self._complete(_props(
            supports_target_checkpoint_state=False,
            sckp_serializes_target_state=True))
        self.assertFalse(ok)
        self.assertIn("serialization alone is not a capability", reason)

    def test_unproven_draft_and_speculative_are_reported_as_false(self):
        protocol = _StubAdapter(_props()).checkpoint_protocol()
        self.assertTrue(protocol["target"])
        self.assertFalse(protocol["draft"], "untested draft support advertised as proven")
        self.assertFalse(protocol["speculative"])
        # ...while the format's serialization is still reported, separately.
        self.assertTrue(protocol["serializes"]["draft"])
        self.assertTrue(protocol["serializes"]["speculative"])

    def test_capabilities_withhold_when_the_protocol_is_incomplete(self):
        """End to end: an incomplete protocol must leave the capability unadvertised."""
        for props in ({}, _props(slot_checkpoint_format="sckp/9"),
                      _props(supports_target_checkpoint_state=False)):
            with self.subTest(props=sorted(props)[:2]):
                adapter = _StubAdapter(props)
                ok, _ = adapter._protocol_is_complete(adapter.checkpoint_protocol())
                self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
