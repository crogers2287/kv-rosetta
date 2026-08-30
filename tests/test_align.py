"""Byte-span alignment across two tokenizations of one text.

Two models with different tokenizers share no token positions at all, so a cache cannot move
between them position by position. What they share is the bytes. These tests pin the
arithmetic and, more importantly, the refusals: a misalignment here is a quiet one-token
shift, not a crash, and a shifted cache produces fluent wrong output.
"""

import unittest
from pathlib import Path

import numpy as np

from kv_rosetta.mappers.align import (
    AlignError,
    Span,
    align,
    byte_spans,
    pool,
    text_of,
)


class ByteSpanTest(unittest.TestCase):
    def test_pieces_become_contiguous_intervals(self):
        spans = byte_spans([b"the ", b"natural", b"ist"])
        self.assertEqual(spans, (Span(0, 0, 4), Span(1, 4, 11), Span(2, 11, 14)))
        self.assertEqual([s.nbytes for s in spans], [4, 7, 3])

    def test_multibyte_characters_are_counted_in_bytes(self):
        """A character is not a byte; counting characters would shift every later span."""
        spans = byte_spans(["café ".encode(), b"au", b" lait"])
        self.assertEqual(spans[0].nbytes, 6)
        self.assertEqual(spans[1].start, 6)

    def test_an_empty_piece_is_refused(self):
        """A zero-width span would match anything or nothing depending on comparison order."""
        with self.assertRaises(AlignError) as caught:
            byte_spans([b"a", b"", b"b"])
        self.assertIn("no bytes", str(caught.exception))

    def test_no_pieces_at_all_is_refused(self):
        with self.assertRaises(AlignError):
            byte_spans([])

    def test_str_pieces_are_refused_rather_than_encoded(self):
        """Encoding here would pick an encoding on the caller's behalf and could disagree
        with how the text was actually decoded."""
        with self.assertRaises(AlignError) as caught:
            byte_spans(["the "])
        self.assertIn("not bytes", str(caught.exception))


class AlignmentTest(unittest.TestCase):
    def test_an_identical_tokenization_aligns_one_to_one(self):
        pieces = [b"the ", b"cat"]
        found = align(pieces, pieces)
        np.testing.assert_array_equal(found.weights, np.eye(2))

    def test_overlap_is_weighted_by_bytes(self):
        """'the' takes all of 'the '; ' naturalist' spans 1, 7 and 3 bytes of three tokens."""
        found = align([b"the ", b"natural", b"ist"], [b"the", b" naturalist"])
        np.testing.assert_allclose(found.weights[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(found.weights[1], [1 / 11, 7 / 11, 3 / 11])

    def test_every_row_sums_to_one_so_pooling_is_a_mean(self):
        found = align([b"ab", b"cd", b"ef"], [b"abc", b"def"])
        np.testing.assert_allclose(found.weights.sum(axis=1), [1.0, 1.0])

    def test_one_source_token_read_by_several_targets(self):
        found = align([b"abcdef"], [b"ab", b"cd", b"ef"])
        self.assertEqual(found.weights.shape, (3, 1))
        np.testing.assert_allclose(found.weights, [[1.0], [1.0], [1.0]])

    def test_several_source_tokens_pooled_into_one_target(self):
        found = align([b"ab", b"cd", b"ef"], [b"abcdef"])
        np.testing.assert_allclose(found.weights, [[1 / 3, 1 / 3, 1 / 3]])

    def test_the_covered_source_tokens_are_reported(self):
        found = align([b"the ", b"natural", b"ist"], [b"the", b" naturalist"])
        self.assertEqual(found.sources_for(0), (0,))
        self.assertEqual(found.sources_for(1), (0, 1, 2))

    def test_tokenizations_of_different_text_are_refused(self):
        """A leading space or a unicode normalisation difference lands here, and the
        alignment it would otherwise produce is a silent one-token shift."""
        with self.assertRaises(AlignError) as caught:
            align([b"the cat"], [b" the cat"])
        self.assertIn("not two views of one text", str(caught.exception))

    def test_same_length_but_different_bytes_is_still_refused(self):
        """Equal lengths would pass a size check; the bytes have to be compared."""
        with self.assertRaises(AlignError):
            align([b"the cat"], [b"the dog"])

    def test_the_byte_count_is_recorded(self):
        found = align([b"ab", b"cd"], [b"abcd"])
        self.assertEqual((found.nbytes, found.source_tokens, found.target_tokens), (4, 2, 1))


class PoolTest(unittest.TestCase):
    def cache(self, tokens, layers=2, heads=2, dim=3):
        return np.arange(layers * 2 * tokens * heads * dim, dtype=np.float32).reshape(
            layers, 2, tokens, heads, dim)

    def test_an_identity_alignment_returns_the_cache_unchanged(self):
        cache = self.cache(3)
        pieces = [b"ab", b"cd", b"ef"]
        np.testing.assert_allclose(pool(cache, align(pieces, pieces)), cache)

    def test_pooling_a_constant_cache_preserves_the_constant(self):
        """Rows summing to one is what guarantees this; a sum would inflate the vectors."""
        cache = np.full((1, 2, 3, 2, 3), 7.0, dtype=np.float32)
        pooled = pool(cache, align([b"ab", b"cd", b"ef"], [b"abcdef"]))
        np.testing.assert_allclose(pooled, np.full((1, 2, 1, 2, 3), 7.0))

    def test_pooling_averages_by_the_declared_weights(self):
        cache = np.zeros((1, 2, 3, 1, 1), dtype=np.float32)
        cache[0, 0, :, 0, 0] = [1.0, 2.0, 3.0]
        pooled = pool(cache, align([b"ab", b"cd", b"ef"], [b"abcdef"]))
        self.assertAlmostEqual(float(pooled[0, 0, 0, 0, 0]), 2.0)

    def test_only_the_token_axis_changes(self):
        pooled = pool(self.cache(3), align([b"ab", b"cd", b"ef"], [b"abc", b"def"]))
        self.assertEqual(pooled.shape, (2, 2, 2, 2, 3))

    def test_a_post_rope_cache_is_refused(self):
        """Averaging keys rotated by different position angles encodes no position at all."""
        with self.assertRaises(AlignError) as caught:
            pool(self.cache(2), align([b"ab", b"cd"], [b"abcd"]), rope_state="applied")
        self.assertIn("Strip the rotation first", str(caught.exception))

    def test_a_cache_of_the_wrong_shape_is_refused(self):
        alignment = align([b"ab", b"cd"], [b"abcd"])
        for bad in (np.zeros((2, 2, 2, 2)), np.zeros((2, 3, 2, 2, 3))):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(AlignError):
                    pool(bad, alignment)

    def test_a_cache_with_the_wrong_token_count_is_refused(self):
        """An alignment built for another prompt would otherwise pool the wrong vectors."""
        with self.assertRaises(AlignError) as caught:
            pool(self.cache(5), align([b"ab", b"cd"], [b"abcd"]))
        self.assertIn("5 tokens but the alignment was built for 2", str(caught.exception))

    def test_text_of_joins_the_pieces(self):
        self.assertEqual(text_of([b"ab", b"cd"]), b"abcd")


if __name__ == "__main__":
    unittest.main()


class CoverageInvariantTest(unittest.TestCase):
    """Every target token always covers at least one source token.

    align() has no branch for "overlaps nothing" because it cannot happen: the source spans
    tile the byte range contiguously, both sides decode to identical bytes, and no piece is
    empty. A guard for it could never fire, and an untestable guard is decoration - so the
    invariant is asserted here across a range of tokenizations instead.
    """

    CASES = (
        ([b"abcdef"], [b"a", b"b", b"c", b"d", b"e", b"f"]),
        ([b"a", b"b", b"c", b"d", b"e", b"f"], [b"abcdef"]),
        ([b"the ", b"natural", b"ist"], [b"t", b"he natu", b"ralist"]),
        ([b"\xc3\xa9"], [b"\xc3", b"\xa9"]),
        ([b"x"], [b"x"]),
    )

    def test_no_target_row_is_ever_empty(self):
        for source, target in self.CASES:
            with self.subTest(source=source, target=target):
                found = align(source, target)
                self.assertTrue(all(found.sources_for(row)
                                    for row in range(found.target_tokens)))
                np.testing.assert_allclose(found.weights.sum(axis=1),
                                           np.ones(found.target_tokens))

    def test_a_split_multibyte_character_still_aligns(self):
        """One tokenizer's byte-fallback can cut a character in half; the bytes still line
        up even though the characters do not."""
        found = align([b"\xc3\xa9"], [b"\xc3", b"\xa9"])
        np.testing.assert_allclose(found.weights, [[1.0], [1.0]])


QWEN25 = Path("/mnt/storage/pre1940_finetune/base_qwen25_3b")
ORNITH = Path("/mnt/storage/ornith-a1-src")
try:
    from transformers import AutoTokenizer
    HAVE_TOKENIZERS = QWEN25.is_dir() and ORNITH.is_dir()
except ImportError:                                # pragma: no cover
    HAVE_TOKENIZERS = False


@unittest.skipUnless(HAVE_TOKENIZERS, "the two local tokenizers are not on this host")
class RealTokenizerTest(unittest.TestCase):
    """Two tokenizers with different vocabularies, on real text.

    Everything above is constructed. This is qwen2.5 (151,643 tokens) against ornith-a1
    (248,044), which is the case the whole module exists for: the same sentence tokenizes to
    17 pieces in one and 16 in the other, and no token index means the same thing in both.
    """

    TEXT = ("The naturalist recorded the following observations in detail, including a "
            "café au lait.")

    @classmethod
    def setUpClass(cls):
        import warnings
        warnings.filterwarnings("ignore")
        cls.a = AutoTokenizer.from_pretrained(str(QWEN25))
        cls.b = AutoTokenizer.from_pretrained(str(ORNITH))

    def pieces(self, tokenizer):
        ids = tokenizer.encode(self.TEXT, add_special_tokens=False)
        return [tokenizer.decode([i]).encode("utf-8") for i in ids]

    def test_per_token_decoding_reconstructs_the_text_exactly(self):
        """If it did not, align would refuse - and this is how a caller checks first."""
        for tokenizer in (self.a, self.b):
            with self.subTest(tokenizer=type(tokenizer).__name__):
                self.assertEqual(b"".join(self.pieces(tokenizer)), self.TEXT.encode())

    def test_the_two_tokenizations_differ_in_length(self):
        """Otherwise this proves nothing that the identity case does not."""
        self.assertNotEqual(len(self.pieces(self.a)), len(self.pieces(self.b)))

    def test_they_align_over_the_full_byte_range(self):
        found = align(self.pieces(self.a), self.pieces(self.b))
        self.assertEqual(found.nbytes, len(self.TEXT.encode()))
        np.testing.assert_allclose(found.weights.sum(axis=1),
                                   np.ones(found.target_tokens))

    def test_a_token_split_by_one_tokenizer_is_pooled_by_byte_count(self):
        """qwen2.5 writes ' lait' as ' la' + 'it'; ornith-a1 keeps it whole. Three bytes and
        two of five, so 0.6 and 0.4."""
        source, target = self.pieces(self.a), self.pieces(self.b)
        found = align(source, target)
        whole = target.index(b" lait")
        covered = found.sources_for(whole)
        self.assertEqual([source[i] for i in covered], [b" la", b"it"])
        np.testing.assert_allclose([found.weights[whole][i] for i in covered], [0.6, 0.4])

    def test_pooling_a_real_alignment_keeps_the_token_axis_honest(self):
        source, target = self.pieces(self.a), self.pieces(self.b)
        found = align(source, target)
        cache = np.random.default_rng(0).normal(
            size=(2, 2, len(source), 2, 4)).astype(np.float32)
        pooled = pool(cache, found)
        self.assertEqual(pooled.shape, (2, 2, len(target), 2, 4))
        self.assertTrue(np.isfinite(pooled).all())
