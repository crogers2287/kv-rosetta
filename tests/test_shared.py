"""The shared drive: one content document, per-model cache attachments.

The load-bearing property is negative. A drive that hands one model another model's tensors
produces fluent, wrong output - the exact failure §20, §29 and §30 established cannot be
repaired - so most of these tests are about what it refuses to return.
"""

import json
import tempfile
import unittest
from pathlib import Path

from kv_rosetta.identity import CacheABIIdentity, IdentityError, ModelIdentity
from kv_rosetta.shared import Content, Entry, SharedDrive, SharedError, attachment_key


def _model(weights="a" * 64):
    return ModelIdentity(architecture="qwen2", weights_sha256=weights,
                         tokenizer_sha256="b" * 64, chat_template_sha256="c" * 64)


def _abi(**over):
    base = dict(runtime="llama.cpp", runtime_revision="ca3d5a3", state_format="ggsq/3",
                k_dtype="f16", v_dtype="f16", rope_kind="normal", rope_base=1e6)
    base.update(over)
    return CacheABIIdentity(**base)


def _content(*, memory="remembered: the user prefers metric units. "):
    return Content(tokenizer_id="qwen2-tok", entries=(
        Entry("system", "system", "You are a careful assistant. ", (1, 2, 3, 4)),
        Entry("tools", "tools", '{"name":"search"} ', (5, 6, 7)),
        Entry("recall", "memory", memory, (8, 9, 10, 11, 12)),
    ))


class ContentIsUniversal(unittest.TestCase):
    def setUp(self):
        self.drive = SharedDrive(Path(tempfile.mkdtemp()) / "drive")

    def test_any_model_can_read_the_content_without_an_attachment(self):
        # The whole point of the shared half: a model nobody has warmed still gets the text.
        digest = self.drive.publish(_content())
        self.assertIsNone(self.drive.cache_for(digest, _model(), _abi()))
        self.assertEqual(self.drive.content(digest).text.count("careful assistant"), 1)

    def test_token_ids_concatenate_in_order(self):
        content = _content()
        self.assertEqual(content.token_ids, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

    def test_reordering_entries_changes_the_digest(self):
        # The same text in a different order is a different cache, so it must be a
        # different identity.
        a = _content()
        b = Content(tokenizer_id=a.tokenizer_id, entries=tuple(reversed(a.entries)))
        self.assertNotEqual(a.digest(), b.digest())

    def test_the_tokenizer_is_part_of_the_identity(self):
        a = _content()
        b = Content(tokenizer_id="other-tok", entries=a.entries)
        self.assertNotEqual(a.digest(), b.digest())

    def test_a_changed_memory_entry_is_a_different_drive_entry(self):
        # Memory is content, so appending to it produces new content rather than quietly
        # invalidating an attachment that still looks valid.
        self.assertNotEqual(_content().digest(),
                            _content(memory="remembered: two facts now. ").digest())

    def test_regions_record_the_chain_each_was_prefilled_behind(self):
        # compose.plan needs this to tell ordinary prefix reuse from a region placed behind
        # a context it never saw.
        regions = _content().regions()
        self.assertEqual(regions[0].prefilled_after, "root")
        self.assertNotEqual(regions[1].prefilled_after, "root")
        self.assertEqual(regions[2].token_start, regions[1].token_end)

    def test_publishing_is_idempotent(self):
        first = self.drive.publish(_content())
        self.assertEqual(first, self.drive.publish(_content()))

    def test_empty_content_is_refused(self):
        with self.assertRaises(SharedError):
            self.drive.publish(Content(tokenizer_id="t", entries=()))

    def test_tampered_content_is_refused_rather_than_returned(self):
        # Every attachment is keyed to this digest. If the text changed underneath, those
        # attachments describe something else and handing the new text back would pair them.
        digest = self.drive.publish(_content())
        path = self.drive.root / f"{digest}.content.json"
        data = json.loads(path.read_text())
        data["entries"][0]["text"] = "You are a reckless assistant. "
        path.write_text(json.dumps(data))
        with self.assertRaises(SharedError) as caught:
            self.drive.content(digest)
        self.assertIn("modified since it was published", str(caught.exception))

    def test_unknown_content_is_refused(self):
        with self.assertRaises(SharedError):
            self.drive.content("d" * 64)

    def test_a_malformed_digest_never_reaches_a_path(self):
        with self.assertRaises(IdentityError):
            self.drive.content("../../etc/passwd")


class MalformedContentIsRefused(unittest.TestCase):
    """A drive document is a file on disk that another process may have written."""

    def test_a_non_object_document_is_refused(self):
        with self.assertRaises(SharedError) as caught:
            Content.from_dict(["tokenizer", "entries"])
        self.assertIn("JSON object", str(caught.exception))

    def test_a_blank_tokenizer_is_refused(self):
        # "" would let content under two different tokenizers digest identically, and an
        # attachment warmed under one would then be offered for the other.
        for blank in (None, "", "   ", 7):
            with self.assertRaises(SharedError) as caught:
                Content.from_dict({"tokenizer_id": blank, "entries": [
                    {"name": "s", "role": "system", "text": "x", "token_ids": [1]}]})
            self.assertIn("tokenizer_id", str(caught.exception))

    def test_missing_or_empty_entries_are_refused(self):
        for entries in (None, [], {}, "system"):
            with self.assertRaises(SharedError) as caught:
                Content.from_dict({"tokenizer_id": "t", "entries": entries})
            self.assertIn("at least one entry", str(caught.exception))

    def test_a_non_object_entry_is_refused(self):
        with self.assertRaises(SharedError) as caught:
            Entry.from_dict(["system", "text"])
        self.assertIn("must be a JSON object", str(caught.exception))

    def test_an_entry_missing_a_field_is_refused(self):
        with self.assertRaises(SharedError) as caught:
            Entry.from_dict({"name": "s", "role": "system", "text": "x"})
        self.assertIn("not readable", str(caught.exception))

    def test_non_integer_token_ids_are_refused(self):
        with self.assertRaises(SharedError):
            Entry.from_dict({"name": "s", "role": "system", "text": "x",
                             "token_ids": ["one", "two"]})


class AttachmentsAreModelSpecific(unittest.TestCase):
    def setUp(self):
        self.drive = SharedDrive(Path(tempfile.mkdtemp()) / "drive")
        self.digest = self.drive.publish(_content())
        self.state = Path(tempfile.mkdtemp()) / "s.state"
        self.state.write_bytes(b"qsgg" + b"\x00" * 64)

    def test_a_model_gets_back_its_own_attachment(self):
        self.drive.attach(self.digest, _model(), _abi(), self.state)
        self.assertIsNotNone(self.drive.cache_for(self.digest, _model(), _abi()))

    def test_another_model_is_not_given_it(self):
        # The failure this project exists to prevent: fluent output conditioned on tensors
        # a different model wrote.
        self.drive.attach(self.digest, _model(), _abi(), self.state)
        self.assertIsNone(self.drive.cache_for(self.digest, _model("f" * 64), _abi()))

    def test_a_different_cache_abi_is_not_given_it(self):
        # Same weights, different KV quantisation: byte-incompatible, and the model digest
        # alone would not catch it.
        self.drive.attach(self.digest, _model(), _abi(), self.state)
        self.assertIsNone(
            self.drive.cache_for(self.digest, _model(), _abi(k_dtype="q8_0")))

    def test_a_different_runtime_revision_is_not_given_it(self):
        self.drive.attach(self.digest, _model(), _abi(), self.state)
        self.assertIsNone(
            self.drive.cache_for(self.digest, _model(), _abi(runtime_revision="other")))

    def test_the_same_content_holds_attachments_for_many_models(self):
        # The shared drive: several models, one copy of the text.
        for weights in ("a" * 64, "b" * 64, "c" * 64):
            self.drive.attach(self.digest, _model(weights), _abi(), self.state)
        self.assertEqual(len(self.drive.attachments(self.digest)), 3)
        for weights in ("a" * 64, "b" * 64, "c" * 64):
            self.assertIsNotNone(self.drive.cache_for(self.digest, _model(weights), _abi()))

    def test_attaching_to_unpublished_content_is_refused(self):
        # A dangling attachment would be returned later as though its content existed.
        with self.assertRaises(SharedError):
            self.drive.attach("e" * 64, _model(), _abi(), self.state)

    def test_a_missing_state_file_is_refused(self):
        with self.assertRaises(SharedError):
            self.drive.attach(self.digest, _model(), _abi(), self.state.with_name("gone"))

    def test_a_state_of_the_wrong_length_is_refused(self):
        # 12 tokens of content; a state warmed on something else must not be deposited.
        with self.assertRaises(SharedError) as caught:
            self.drive.attach(self.digest, _model(), _abi(), self.state, token_count=99)
        self.assertIn("not warmed on this content", str(caught.exception))

    def test_a_state_of_the_right_length_is_accepted(self):
        self.drive.attach(self.digest, _model(), _abi(), self.state, token_count=12)

    def test_the_attachment_key_covers_model_and_abi(self):
        self.assertNotEqual(attachment_key(_model(), _abi()),
                            attachment_key(_model("9" * 64), _abi()))
        self.assertNotEqual(attachment_key(_model(), _abi()),
                            attachment_key(_model(), _abi(v_dtype="q8_0")))

    def test_staging_puts_the_attachment_where_the_runtime_looks(self):
        # llama.cpp resolves a restore filename only inside its own --slot-save-path, so a
        # correct path in the drive is useless to it. Without this the caller gets an
        # opaque HTTP 400 instead of a miss.
        self.drive.attach(self.digest, _model(), _abi(), self.state)
        slots = Path(tempfile.mkdtemp())
        name = self.drive.stage(self.digest, _model(), _abi(), slots)
        self.assertIsNotNone(name)
        self.assertTrue((slots / name).is_file())
        self.assertEqual((slots / name).read_bytes(), self.state.read_bytes())

    def test_staging_a_missing_attachment_returns_none(self):
        slots = Path(tempfile.mkdtemp())
        self.assertIsNone(self.drive.stage(self.digest, _model(), _abi(), slots))
        self.assertEqual(list(slots.iterdir()), [])

    def test_staging_never_stages_another_models_attachment(self):
        self.drive.attach(self.digest, _model(), _abi(), self.state)
        slots = Path(tempfile.mkdtemp())
        self.assertIsNone(self.drive.stage(self.digest, _model("f" * 64), _abi(), slots))
        self.assertEqual(list(slots.iterdir()), [])

    def test_staging_twice_is_idempotent(self):
        self.drive.attach(self.digest, _model(), _abi(), self.state)
        slots = Path(tempfile.mkdtemp())
        first = self.drive.stage(self.digest, _model(), _abi(), slots)
        self.assertEqual(first, self.drive.stage(self.digest, _model(), _abi(), slots))
        self.assertEqual(len(list(slots.iterdir())), 1)

    def test_staging_into_a_non_directory_is_refused(self):
        self.drive.attach(self.digest, _model(), _abi(), self.state)
        target = Path(tempfile.mkdtemp()) / "file"
        target.write_bytes(b"")
        with self.assertRaises(SharedError):
            self.drive.stage(self.digest, _model(), _abi(), target)

    def test_a_drive_pointed_at_a_file_is_refused(self):
        path = Path(tempfile.mkdtemp()) / "not-a-dir"
        path.write_bytes(b"")
        with self.assertRaises(SharedError):
            SharedDrive(path, create=False)


if __name__ == "__main__":
    unittest.main()


class PrefixAddressedAttachments(unittest.TestCase):
    """Growing a memory must not re-prefill the system prompt that did not change.

    Safe only because llama.cpp checks the token prefix itself, so a wrong guess costs a
    re-prefill rather than producing wrong output. That check covers tokens, not weights,
    which is why a different model's attachment is still refused outright below.
    """

    def setUp(self):
        self.drive = SharedDrive(Path(tempfile.mkdtemp()) / "drive")
        self.state = Path(tempfile.mkdtemp()) / "s.state"
        self.state.write_bytes(b"qsgg" + b"\x00" * 64)

    def _content(self, memory_ids):
        return Content(tokenizer_id="t", entries=(
            Entry("system", "system", "sys ", (1, 2, 3, 4)),
            Entry("recall", "memory", "mem ", tuple(memory_ids))))

    def _warm(self, content):
        digest = self.drive.publish(content)
        self.drive.attach(digest, _model(), _abi(), self.state)
        return digest

    def test_a_grown_memory_still_reuses_the_unchanged_head(self):
        self._warm(self._content((5, 6)))
        grown = self._content((5, 6, 7, 8))
        match = self.drive.best_attachment(grown, _model(), _abi())
        self.assertIsNotNone(match)
        self.assertEqual(match.shared_tokens, 6)      # 4 system + 2 memory
        self.assertEqual(match.target_tokens, 8)
        self.assertFalse(match.exact)
        self.assertAlmostEqual(match.reusable_fraction, 0.75)

    def test_an_exact_match_is_marked_exact(self):
        content = self._content((5, 6))
        self._warm(content)
        match = self.drive.best_attachment(content, _model(), _abi())
        self.assertTrue(match.exact)
        self.assertEqual(match.shared_tokens, match.target_tokens)

    def test_the_longest_prefix_wins(self):
        # Several generations of a growing memory; the newest that still fits must be used.
        self._warm(self._content((5,)))
        self._warm(self._content((5, 6, 7)))
        self._warm(self._content((5, 6)))
        match = self.drive.best_attachment(self._content((5, 6, 7, 8)), _model(), _abi())
        self.assertEqual(match.shared_tokens, 7)      # 4 system + 3 memory

    def test_a_rewritten_memory_falls_back_to_the_shared_head(self):
        # Not appended but replaced: only the system region is common, and that is still
        # worth more than nothing.
        self._warm(self._content((5, 6, 7)))
        match = self.drive.best_attachment(self._content((9, 9, 9)), _model(), _abi())
        self.assertEqual(match.shared_tokens, 4)

    def test_content_sharing_nothing_is_not_offered(self):
        self._warm(self._content((5, 6)))
        alien = Content(tokenizer_id="t", entries=(
            Entry("other", "system", "x ", (99, 98)),))
        self.assertIsNone(self.drive.best_attachment(alien, _model(), _abi()))

    def test_a_minimum_can_be_required(self):
        self._warm(self._content((5, 6, 7)))
        grown = self._content((9, 9, 9))
        self.assertIsNotNone(self.drive.best_attachment(grown, _model(), _abi()))
        self.assertIsNone(self.drive.best_attachment(grown, _model(), _abi(), minimum=5))

    def test_another_models_attachment_is_never_offered_as_a_prefix(self):
        # The whole safety argument for prefix matching is that the runtime checks tokens.
        # It does not check weights, so this case stays refused.
        self._warm(self._content((5, 6)))
        self.assertIsNone(
            self.drive.best_attachment(self._content((5, 6, 7)), _model("f" * 64), _abi()))

    def test_another_cache_abi_is_never_offered_as_a_prefix(self):
        self._warm(self._content((5, 6)))
        self.assertIsNone(self.drive.best_attachment(
            self._content((5, 6, 7)), _model(), _abi(k_dtype="q8_0")))

    def test_tampered_content_is_skipped_rather_than_offered(self):
        digest = self._warm(self._content((5, 6)))
        path = self.drive.root / f"{digest}.content.json"
        data = json.loads(path.read_text())
        data["entries"][0]["text"] = "tampered "
        path.write_text(json.dumps(data))
        self.assertIsNone(self.drive.best_attachment(
            self._content((5, 6, 7)), _model(), _abi()))
