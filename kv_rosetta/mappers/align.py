"""Aligning two tokenizations of the same text, by bytes rather than by token index.

Two models that share a tokenizer share token positions, and a cache maps across position by
position. Two models that do not share one have nothing in common at the token level: "the
naturalist" may be three tokens in one vocabulary and five in another, and token 2 of the
first has no relationship whatever to token 2 of the second.

What they do share is the bytes. Decode each tokenization to a byte offset per token, and the
two sequences become intervals over one common line. A target token covering three source
tokens takes a weighted mean of their vectors; a source token covering three target tokens is
read by each of them.

Two things this refuses rather than paper over:

* **The two tokenizations must decode to identical bytes.** If they do not, they are not two
  views of one text, and every alignment computed from them is fiction. A tokenizer that adds
  a leading space, or normalises unicode differently, produces exactly this - and the
  resulting misalignment is a quiet one-token shift, not a crash.
* **Keys must have had RoPE stripped.** Pooling post-RoPE keys averages vectors that have been
  rotated by *different* position angles, so the result encodes neither position faithfully.
  The rotation has to come off before the mean and go back on after.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class AlignError(ValueError):
    """Raised when an alignment cannot be justified. Never a best effort."""


@dataclass(frozen=True)
class Span:
    """One token's half-open byte interval in the decoded text."""

    index: int
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def byte_spans(pieces: list[bytes] | tuple[bytes, ...]) -> tuple[Span, ...]:
    """Token pieces to contiguous byte intervals.

    `pieces` is what each token contributes to the decoded text, in order. It is taken from
    the caller rather than produced here because every tokenizer library spells incremental
    decoding differently, and guessing wrong would shift every span silently.

    A token contributing no bytes is refused. Some special tokens genuinely decode to nothing,
    and such a token has no interval to align on: giving it a zero-width span would let it
    match anything or nothing depending on comparison order. The caller drops them.
    """
    if not pieces:
        raise AlignError("no token pieces to align")
    spans, offset = [], 0
    for index, piece in enumerate(pieces):
        if not isinstance(piece, (bytes, bytearray)):
            raise AlignError(f"piece {index} is {type(piece).__name__}, not bytes; decoding "
                             f"to str would make the offsets characters, not bytes")
        if not piece:
            raise AlignError(f"token {index} contributes no bytes, so it has no interval to "
                             f"align on; drop empty pieces before aligning")
        spans.append(Span(index=index, start=offset, end=offset + len(piece)))
        offset += len(piece)
    return tuple(spans)


def text_of(pieces) -> bytes:
    return b"".join(bytes(piece) for piece in pieces)


@dataclass(frozen=True)
class Alignment:
    """Per target token, the source tokens it covers and how much of each.

    `weights` has one row per target token and one column per source token. Each row sums to
    one, so pooling is a weighted mean and never changes the scale of the vectors.
    """

    weights: np.ndarray
    source_tokens: int
    target_tokens: int
    nbytes: int

    def sources_for(self, target_index: int) -> tuple[int, ...]:
        return tuple(int(i) for i in np.nonzero(self.weights[target_index])[0])


def align(source_pieces, target_pieces) -> Alignment:
    """Weight every target token by the bytes it shares with each source token.

    Overlap is measured in bytes, so a target token covering two thirds of one source token
    and one third of another weights them 2:1. This is the only defensible reading when the
    boundaries do not line up, and they usually do not.
    """
    source_text, target_text = text_of(source_pieces), text_of(target_pieces)
    if source_text != target_text:
        raise AlignError(
            f"the two tokenizations decode to different bytes ({len(source_text)} against "
            f"{len(target_text)}); they are not two views of one text, and an alignment "
            f"between them would be fiction")
    source, target = byte_spans(source_pieces), byte_spans(target_pieces)

    weights = np.zeros((len(target), len(source)), dtype=np.float64)
    for row, span in enumerate(target):
        for column, other in enumerate(source):
            overlap = min(span.end, other.end) - max(span.start, other.start)
            if overlap > 0:
                weights[row, column] = overlap
        # No "overlaps nothing" branch here. The source spans tile the whole byte range
        # contiguously, both sides decode to the same bytes, and no piece is empty - so every
        # target span necessarily meets at least one source span. A guard for it could never
        # fire, and an untestable guard is decoration. The invariant is asserted by test
        # instead, over a range of tokenizations.
        weights[row] /= weights[row].sum()
    return Alignment(weights=weights, source_tokens=len(source), target_tokens=len(target),
                     nbytes=len(source_text))


def pool(source_kv: np.ndarray, alignment: Alignment, *,
         rope_state: str = "not_applied") -> np.ndarray:
    """Source vectors gathered onto the target tokenization.

    `source_kv` is canonical `(layer, kv, token, head, dim)`. Only the token axis moves; layer,
    head and dimension are the linear mapper's problem, not this one's.
    """
    if rope_state != "not_applied":
        raise AlignError(
            f"rope_state is {rope_state!r}; pooling post-RoPE keys averages vectors rotated "
            f"by different position angles, and the mean encodes no position faithfully. "
            f"Strip the rotation first and reapply it at the target positions")
    array = np.asarray(source_kv)
    if array.ndim != 5 or array.shape[1] != 2:
        raise AlignError(f"expected canonical (layer, kv, token, head, dim) with kv=2, got "
                         f"{array.shape}")
    if array.shape[2] != alignment.source_tokens:
        raise AlignError(f"cache holds {array.shape[2]} tokens but the alignment was built "
                         f"for {alignment.source_tokens}")
    # (target, source) x (layer, kv, source, head, dim) over the token axis.
    return np.einsum("ts,lkshd->lkthd", alignment.weights.astype(array.dtype), array)
