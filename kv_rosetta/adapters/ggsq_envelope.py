"""Explicit envelope parsing for llama.cpp GGSQ sequence-state buffers.

WHY this module exists
----------------------
``ggsq.py`` already parses the *body* of a sequence state and is verified against
it.  The body is not self-describing: it carries no offset telling where it
begins.  Two different sources produce these buffers and their framing differs,
so the body starts at a different absolute offset in each:

  * SOURCE A (``buffer``) comes from ``llama_state_seq_get_data()`` and is an
    in-process byte string.  It is optionally prefixed by an implementation
    defined ``io_magic`` and an ``int32`` ``source_seq_id`` before the body.
  * SOURCE B (``file``) is written by ``llama_state_seq_save_file()`` and is
    prefixed by the ``b"GGSQ"`` magic, a ``uint32`` version, a ``uint32``
    ``n_token_count`` and that many ``int32`` token ids before the body.

``ggsq.py`` must never be handed an unknown byte buffer directly: guessing the
offset silently misparses the body.  This module inspects the framing, decides
which source produced the buffer, and reports the *absolute* offset where the
body begins.  ``body()`` then slices exactly the body bytes, which can be handed
to ``ggsq.py`` with certainty.

Everything is little-endian.  A truncated or malformed envelope raises
:class:`EnvelopeError` (a :class:`ValueError`) rather than returning a
plausible-looking but wrong envelope.  Every error names the offending field and
the byte offset it was found at.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "EnvelopeError",
    "GGSQ_MAGIC",
    "SUPPORTED_VERSIONS",
    "Source",
    "Envelope",
    "parse_file_envelope",
    "parse_buffer_envelope",
    "detect",
    "parse",
    "body",
]


class EnvelopeError(ValueError):
    """Raised when a GGSQ sequence-state blob cannot be parsed as an envelope.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers keep
    working, while callers can catch this more specific type to distinguish
    envelope problems from unrelated value errors.
    """


# LLAMA_FILE_MAGIC_GGSQ is 0x67677371 written as a little-endian uint32, so the bytes on
# disk read "qsgg", not "GGSQ". Assuming the human-readable spelling parses a real state
# file as an in-process buffer and yields a plausible but wrong envelope.
GGSQ_MAGIC = (0x67677371).to_bytes(4, "little")   # b"qsgg"
GGSN_MAGIC = (0x6767736E).to_bytes(4, "little")   # b"nsgg", whole-session files
_KNOWN_MAGICS = (GGSQ_MAGIC, GGSN_MAGIC, b"GGSQ", b"GGSN")
# Observed: a binary built 2026-08-19 writes version 2 while the checked-out header
# declares LLAMA_STATE_SEQ_VERSION 3. The running binary is the authority, not the header,
# so accepted versions are a set and an unknown one is refused rather than assumed.
SUPPORTED_VERSIONS = frozenset({2, 3})


class Source(str, Enum):
    """Which framing produced the buffer."""

    FILE = "file"
    BUFFER = "buffer"
    #: Not recognisable as either. Refused rather than guessed: treating unknown bytes as a
    #: buffer is how a real state file was parsed into a plausible but wrong envelope.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Envelope:
    """The framing of a sequence-state buffer and where its body begins."""

    source: Source
    version: int  # 0 when the source carries no version
    token_ids: tuple[int, ...]  # empty for BUFFER
    seq_id: int  # -1 when unknown
    body_offset: int  # absolute offset where the state body begins
    body_length: int  # len(blob) - body_offset


def _require(blob: bytes, offset: int, length: int, field: str) -> bytes:
    """Return ``blob[offset:offset+length]`` or raise if not fully present."""
    end = offset + length
    if offset < 0 or end > len(blob):
        raise EnvelopeError(
            f"truncated envelope: need {length} byte(s) for field {field!r} at "
            f"offset {offset}, but only {len(blob) - offset} byte(s) remain"
        )
    return blob[offset:end]


def parse_file_envelope(blob: bytes) -> Envelope:
    """Parse the envelope of a ``llama_state_seq_save_file()`` buffer.

    Raises :class:`EnvelopeError` if the magic, version, or token array is
    missing, malformed, or would read past the end of ``blob``.  Never guesses.
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise EnvelopeError(
            f"expected bytes for a file envelope, got {type(blob).__name__}"
        )
    blob = bytes(blob)

    magic = _require(blob, 0, 4, "magic")
    if magic != GGSQ_MAGIC:
        raise EnvelopeError(
            f"bad magic at offset 0: found {magic!r}, expected {GGSQ_MAGIC!r}"
        )

    version = struct.unpack_from("<I", _require(blob, 4, 4, "version"), 0)[0]
    if version not in SUPPORTED_VERSIONS:
        raise EnvelopeError(
            f"unsupported version at offset 4: {version}; supported versions are "
            f"{sorted(SUPPORTED_VERSIONS)}"
        )

    n_token_count = struct.unpack_from("<I", _require(blob, 8, 4, "n_token_count"), 0)[0]

    ids_start = 12
    ids_end = ids_start + n_token_count * 4
    if ids_end > len(blob):
        raise EnvelopeError(
            f"token array at offset {ids_start}: declared {n_token_count} token(s) "
            f"({n_token_count * 4} byte(s)) but only {len(blob) - ids_start} "
            f"byte(s) remain"
        )

    token_ids: tuple[int, ...] = (
        struct.unpack_from(f"<{n_token_count}i", blob, ids_start)
        if n_token_count
        else ()
    )

    body_offset = ids_end
    return Envelope(
        source=Source.FILE,
        version=version,
        token_ids=token_ids,
        seq_id=-1,
        body_offset=body_offset,
        body_length=len(blob) - body_offset,
    )


def _reject_known_magic(blob: bytes) -> None:
    head = blob[:4]
    if head in _KNOWN_MAGICS:
        raise EnvelopeError(
            f"buffer starts with the state-file magic {head!r}; this is a sequence-state "
            f"FILE, not an in-process buffer - parse it with parse_file_envelope")


def parse_buffer_envelope(
    blob: bytes,
    seq_id: int = -1,
    io_magic: bytes | None = None,
) -> Envelope:
    """Parse the envelope of an in-process ``llama_state_seq_get_data()`` buffer.

    When ``io_magic`` is given it must match the leading bytes or
    :class:`EnvelopeError` is raised.  The ``int32`` ``source_seq_id`` following
    the optional magic is read from the buffer; ``seq_id`` overrides it when it
    is not ``-1``.  ``body_offset`` follows the ``source_seq_id``.
    """
    if not io_magic:
        raise EnvelopeError(
            "parse_buffer_envelope requires the runtime's internal IO magic; a buffer is not "
            "self-describing and must not be inferred from its first bytes")
    _reject_known_magic(blob)
    if not isinstance(blob, (bytes, bytearray)):
        raise EnvelopeError(
            f"expected bytes for a buffer envelope, got {type(blob).__name__}"
        )
    blob = bytes(blob)

    offset = 0
    if io_magic is not None:
        if len(io_magic) == 0:
            raise EnvelopeError(
                "io_magic at offset 0 must be a non-empty byte string"
            )
        head = _require(blob, 0, len(io_magic), "io_magic")
        if head != io_magic:
            raise EnvelopeError(
                f"bad io_magic at offset 0: found {head!r}, expected {io_magic!r}"
            )
        offset += len(io_magic)

    source_seq_id = struct.unpack_from(
        "<i", _require(blob, offset, 4, "source_seq_id"), 0
    )[0]

    reported_seq_id = seq_id if seq_id != -1 else source_seq_id
    body_offset = offset + 4
    return Envelope(
        source=Source.BUFFER,
        version=0,
        token_ids=(),
        seq_id=reported_seq_id,
        body_offset=body_offset,
        body_length=len(blob) - body_offset,
    )


def detect(blob: bytes) -> Source:
    """Classify a blob by evidence only.

    Anything that is not recognisably a sequence-state FILE is UNKNOWN, never BUFFER.
    Treating unrecognised bytes as an in-process buffer is how a real state file came to be
    parsed into a plausible envelope with zero tokens and a body offset of 4.
    """
    if blob[:4] == GGSQ_MAGIC:
        return Source.FILE
    return Source.UNKNOWN


def parse(blob: bytes, source: Source | None = None, **kwargs: object) -> Envelope:
    """Parse an envelope. An unknown blob is refused, not guessed.

    A buffer has no self-describing magic, so it can only be parsed when the caller states
    that is what it is and supplies the runtime's internal IO magic.
    """
    if source is None:
        source = detect(blob)
        if source is Source.UNKNOWN:
            raise EnvelopeError(
                f"cannot classify blob: first bytes {blob[:4]!r} are not {GGSQ_MAGIC!r}; "
                f"pass source=Source.BUFFER with io_magic to parse an in-process buffer")
    if source is Source.FILE:
        return parse_file_envelope(blob)
    if source is Source.BUFFER:
        return parse_buffer_envelope(blob, **kwargs)  # type: ignore[arg-type]
    raise EnvelopeError(f"cannot parse source {source}")


def body(blob: bytes, envelope: Envelope) -> bytes:
    """Return exactly the body bytes described by ``envelope``.

    Raises :class:`EnvelopeError` if the declared body range is not fully inside
    ``blob``.
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise EnvelopeError(f"expected bytes, got {type(blob).__name__}")
    blob = bytes(blob)

    start = envelope.body_offset
    end = start + envelope.body_length
    if start < 0 or end > len(blob):
        raise EnvelopeError(
            f"body range [{start}, {end}) is not fully inside blob of length "
            f"{len(blob)}"
        )
    return blob[start:end]


# ---------------------------------------------------------------------------
# Prompt payload
# ---------------------------------------------------------------------------

LLAMA_TOKEN_NULL = -1
SERVER_TOKENS_STATE_VERSION = 1


def decode_prompt_tokens(packed: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Recover real token IDs from the envelope's prompt array.

    llama-server does not write a plain token list. It writes
    ``server_tokens::serialize()`` reinterpreted as ``llama_token *``
    (tools/server/server-context.cpp), whose layout is::

        LLAMA_TOKEN_NULL (-1)            format marker
        SERVER_TOKENS_STATE_VERSION      currently 1
        tokens        as [count][elements...]
        media_keys    as [count][elements...]
        media chunks

    Treating that array as token IDs yields -1 as the first "token", which the server
    rejects. Older versions wrote a plain list, which ``deserialize`` still detects by the
    absence of the marker, so both shapes are handled.

    Returns () when the prompt cannot be decoded to a pure text token list - including
    when media chunks are present, because a text-only reuse of a multimodal prompt would
    silently drop the media.
    """
    packed = list(packed)
    if not packed:
        return ()
    if packed[0] != LLAMA_TOKEN_NULL:
        return tuple(packed)                       # plain list, older format
    if len(packed) < 3:
        raise EnvelopeError("prompt state truncated before the token count")
    version = packed[1]
    if version != SERVER_TOKENS_STATE_VERSION:
        raise EnvelopeError(
            f"unsupported server tokens state version {version}; "
            f"expected {SERVER_TOKENS_STATE_VERSION}")
    count = packed[2]
    if count < 0 or 3 + count > len(packed):
        raise EnvelopeError(
            f"prompt declares {count} token(s) but only {len(packed) - 3} word(s) remain")
    tokens = tuple(packed[3:3 + count])
    rest = packed[3 + count:]
    media = rest[0] if rest else 0
    if media:
        return ()                                   # multimodal: not a pure text prompt
    return tokens
