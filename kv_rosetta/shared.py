"""One artifact many models can use: shared content, per-model cache attachments.

Three measurements decided this shape. Cache tensors do not survive a change of weights -
not across geometries (§20), not through a fitted converter (§29), not partially (§30) -
because the difference is the target's own weight drift and the source cache holds no
information about it. So a single set of tensors every model can read does not exist and
this module does not pretend otherwise.

What *is* shareable is everything else, and it is the part that matters for the stated goal.
The prompt regions - system text, tool schemas, memory entries - are model-neutral. A drive
holds one canonical copy of those plus a cache attachment per (model, cache ABI) that has
warmed them. A model arriving with no attachment still gets the content, prefills it, and
deposits its own attachment for next time, so the drive fills in as it is used rather than
needing every model warmed up front.

The load-bearing refusal is that an attachment is only ever returned to the exact model and
cache ABI it was written by. A shared drive that hands one model another's tensors is the
failure this whole project exists to prevent, and it is a refusal here rather than a
convention.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import compose
from .identity import CacheABIIdentity, ModelIdentity, digest_of, require_digest

CONTENT_SUFFIX = ".content.json"
ATTACHMENT_SUFFIX = ".state"
META_SUFFIX = ".meta.json"
DRIVE_MODE = 0o700


class SharedError(ValueError):
    """A refusal. Never downgraded, and never satisfied with an approximate match."""


@dataclass(frozen=True)
class Entry:
    """One region of shareable content, with the text a model needs to prefill it."""

    name: str
    role: str
    text: str
    token_ids: tuple[int, ...]

    def region(self, start: int, prefilled_after: str = compose.AT_FRONT) -> compose.Region:
        return compose.Region(
            name=self.name, role=self.role, token_start=start,
            token_end=start + len(self.token_ids),
            tokens_sha256=compose.token_digest(self.token_ids),
            prefilled_after=prefilled_after)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "role": self.role, "text": self.text,
                "token_ids": list(self.token_ids)}

    @classmethod
    def from_dict(cls, data: Any) -> "Entry":
        if not isinstance(data, dict):
            raise SharedError("a content entry must be a JSON object")
        try:
            return cls(name=str(data["name"]), role=str(data["role"]),
                       text=str(data["text"]), token_ids=tuple(int(v) for v in data["token_ids"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise SharedError(f"content entry is not readable: {exc}") from exc


@dataclass(frozen=True)
class Content:
    """The model-neutral half of a drive: what to prefill, in order."""

    tokenizer_id: str
    entries: tuple[Entry, ...] = field(default_factory=tuple)

    @property
    def token_ids(self) -> tuple[int, ...]:
        out: list[int] = []
        for entry in self.entries:
            out.extend(entry.token_ids)
        return tuple(out)

    @property
    def text(self) -> str:
        return "".join(entry.text for entry in self.entries)

    def digest(self) -> str:
        """Identity of the content, order and text included.

        The text is covered, not only the token ids. An earlier version digested ids,
        names and roles alone, which meant editing an entry's text left the identity
        unchanged - so a drive would hand a model text that its attachments had never been
        warmed on, and the tamper check below could not fire. The text is what gets
        prefilled, so the text is part of the identity.

        The tokenizer is in it too: the same text under a different tokenizer is a
        different token sequence, and an attachment warmed under one is meaningless under
        the other even when every other field matches.
        """
        return digest_of(
            "SharedContent", self.tokenizer_id, compose.token_digest(self.token_ids),
            *(f"{e.name}\x1f{e.role}\x1f"
              f"{hashlib.sha256(e.text.encode()).hexdigest()}" for e in self.entries))

    def regions(self) -> tuple[compose.Region, ...]:
        """Regions in order, each recording the chain it was prefilled behind."""
        out, start, chain = [], 0, compose.AT_FRONT
        for entry in self.entries:
            region = entry.region(start, chain)
            out.append(region)
            chain = compose.chain_digest(chain, region)
            start = region.token_end
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {"tokenizer_id": self.tokenizer_id,
                "entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, data: Any) -> "Content":
        if not isinstance(data, dict):
            raise SharedError("content must be a JSON object")
        tokenizer = data.get("tokenizer_id")
        if not isinstance(tokenizer, str) or not tokenizer.strip():
            raise SharedError("content.tokenizer_id must be a non-empty string")
        raw = data.get("entries")
        if not isinstance(raw, list) or not raw:
            raise SharedError("content needs at least one entry")
        return cls(tokenizer_id=tokenizer, entries=tuple(Entry.from_dict(e) for e in raw))


def attachment_key(model: ModelIdentity, abi: CacheABIIdentity) -> str:
    """Which model, and which cache layout, an attachment belongs to.

    Both halves are required. Two runtimes serving identical weights with different KV
    quantisation produce byte-incompatible states, so the model digest alone would let one
    be handed to the other.
    """
    return digest_of("SharedAttachment", model.digest(), abi.digest())


def _common_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


@dataclass(frozen=True)
class AttachmentInfo:
    """What is known about one stored attachment, including whether it can pay.

    An attachment is storable and restorable for any architecture, and for a hybrid one it
    will still reuse nothing: the model resumes from a context checkpoint that an unpatched
    slot save does not persist. Measured on this drive - a 27B `qwen35` and a `qwen35moe`
    MoE both restored their own attachments cleanly and re-prefilled all 676 tokens, while
    a dense `qwen2` went from 676 prefilled to 1.

    Recorded rather than refused, because a patched runtime can use these and the bytes are
    not wrong. What would be wrong is a drive that reports a hit and lets the caller assume
    it saved something.
    """

    key: str
    path: Path
    architecture: str = ""
    expects_reuse: bool | None = None
    reason: str = ""

    @property
    def pays(self) -> bool:
        """True only when reuse is known to be supported. Unknown is not a promise."""
        return self.expects_reuse is True


@dataclass(frozen=True)
class PrefixMatch:
    """An attachment that covers a prefix of the content being asked for."""

    path: Path
    content_digest: str
    shared_tokens: int
    attachment_tokens: int
    target_tokens: int
    exact: bool

    @property
    def reusable_fraction(self) -> float:
        return self.shared_tokens / self.target_tokens if self.target_tokens else 0.0


class SharedDrive:
    """A directory holding one content document and any number of cache attachments."""

    def __init__(self, root: Path | str, *, create: bool = True) -> None:
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            os.chmod(self.root, DRIVE_MODE)
        if not self.root.is_dir():
            raise SharedError(f"drive {self.root} is not a directory")

    # -- content -----------------------------------------------------------------------

    def _content_path(self, digest: str) -> Path:
        return self.root / f"{require_digest(digest, 'content_digest')}{CONTENT_SUFFIX}"

    def publish(self, content: Content) -> str:
        """Write the shareable content and return its digest. Idempotent."""
        if not content.entries:
            raise SharedError("refusing to publish content with no entries")
        digest = content.digest()
        path = self._content_path(digest)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(content.to_dict(), indent=1))
        os.replace(tmp, path)
        return digest

    def content(self, digest: str) -> Content:
        """The content any model can read, whether or not it has an attachment."""
        path = self._content_path(digest)
        if not path.is_file():
            raise SharedError(f"no content published under {digest[:16]}")
        loaded = Content.from_dict(json.loads(path.read_text()))
        if loaded.digest() != digest:
            raise SharedError(
                f"content under {digest[:16]} digests to {loaded.digest()[:16]}; it has "
                f"been modified since it was published and every attachment keyed to it "
                f"is now describing different text")
        return loaded

    # -- attachments -------------------------------------------------------------------

    def _attachment_path(self, content_digest: str, key: str) -> Path:
        return self.root / (f"{require_digest(content_digest, 'content_digest')}."
                            f"{require_digest(key, 'attachment_key')}{ATTACHMENT_SUFFIX}")

    def _meta_path(self, content_digest: str, key: str) -> Path:
        return self.root / (f"{require_digest(content_digest, 'content_digest')}."
                            f"{require_digest(key, 'attachment_key')}{META_SUFFIX}")

    def describe(self, content_digest: str) -> tuple[AttachmentInfo, ...]:
        """Every attachment for this content, and whether each is expected to pay."""
        out = []
        for key in self.attachments(content_digest):
            meta_path = self._meta_path(content_digest, key)
            meta = {}
            if meta_path.is_file():
                try:
                    loaded = json.loads(meta_path.read_text())
                    if isinstance(loaded, dict):
                        meta = loaded
                except (OSError, ValueError):
                    meta = {}      # unreadable metadata is unknown, never assumed good
            expects = meta.get("expects_reuse")
            out.append(AttachmentInfo(
                key=key, path=self._attachment_path(content_digest, key),
                architecture=str(meta.get("architecture", "")),
                expects_reuse=expects if isinstance(expects, bool) else None,
                reason=str(meta.get("reason", ""))))
        return tuple(out)

    def attach(self, content_digest: str, model: ModelIdentity, abi: CacheABIIdentity,
               state: Path | str, *, token_count: int | None = None,
               architecture: str = "") -> Path:
        """Deposit this model's cache for this content.

        `token_count` is checked against the content when supplied, because a state file
        holding a different number of cells was not warmed on this text, and a drive that
        accepts it will hand it back later as though it had been.
        """
        content = self.content(content_digest)
        source = Path(state)
        if not source.is_file():
            raise SharedError(f"state file does not exist: {source}")
        if token_count is not None and token_count != len(content.token_ids):
            raise SharedError(
                f"state holds {token_count} tokens against the content's "
                f"{len(content.token_ids)}; it was not warmed on this content")
        key = attachment_key(model, abi)
        dest = self._attachment_path(content_digest, key)
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)

        # Record whether this architecture can reuse a restored prefix at all. Known before
        # the cache is ever built, so a caller never has to discover it by measuring a
        # cache_n of zero.
        arch = architecture or model.architecture
        expects, reason = None, ""
        if arch:
            from .gguf import supports_prefix_reuse
            expects, reason = supports_prefix_reuse(arch)
        meta = self._meta_path(content_digest, key)
        meta_tmp = meta.with_name(meta.name + ".tmp")
        meta_tmp.write_text(json.dumps({"architecture": arch, "expects_reuse": expects,
                                        "reason": reason, "tokens": token_count}, indent=1))
        os.replace(meta_tmp, meta)
        return dest

    def cache_for(self, content_digest: str, model: ModelIdentity,
                  abi: CacheABIIdentity) -> Path | None:
        """This model's attachment, or None. Never another model's.

        Addressed by the digest of the exact (model, cache ABI) pair, so a near miss is a
        miss. There is deliberately no fallback to "the closest available attachment": the
        measurements say a foreign cache degrades output, and degraded output that looks
        fluent is the failure mode this project was built to refuse.
        """
        path = self._attachment_path(content_digest, attachment_key(model, abi))
        return path if path.is_file() else None

    def stage(self, content_digest: str, model: ModelIdentity, abi: CacheABIIdentity,
              slot_dir: Path | str) -> str | None:
        """Place this model's attachment where its runtime will look, return the basename.

        A drive and a runtime's slot directory are different places, and llama.cpp resolves
        a restore filename only inside its own --slot-save-path. Without this step a caller
        holds a correct path the server cannot use, which surfaces as an opaque HTTP 400
        rather than as a miss.

        Hard-linked when the drive and the slot directory share a filesystem, because these
        files reach gigabytes and copying one per request is the cost this project exists to
        avoid. Falls back to a copy across filesystems.
        """
        source = self.cache_for(content_digest, model, abi)
        if source is None:
            return None
        target_dir = Path(slot_dir)
        if not target_dir.is_dir():
            raise SharedError(f"slot directory {target_dir} is not a directory")
        dest = target_dir / source.name
        if dest.exists():
            return dest.name
        tmp = dest.with_name(dest.name + ".tmp")
        try:
            os.link(source, tmp)
        except OSError:
            shutil.copyfile(source, tmp)
        os.replace(tmp, dest)
        return dest.name

    def published(self) -> tuple[str, ...]:
        """Every content digest on the drive."""
        return tuple(sorted(p.name[:-len(CONTENT_SUFFIX)]
                            for p in self.root.glob(f"*{CONTENT_SUFFIX}")))

    def best_attachment(self, content: Content, model: ModelIdentity,
                        abi: CacheABIIdentity, *, minimum: int = 1) -> "PrefixMatch | None":
        """The attachment sharing the longest token prefix with `content`, for this model.

        Exact-digest lookup is the wrong instrument when a memory entry grows. Editing one
        region changes the whole content digest, so `cache_for` misses and the unchanged
        system and tools regions are prefilled again. Measured on this host: appending to a
        memory and restoring the *previous* attachment reused 820 of 892 tokens and cut the
        prefill from 108ms to 18ms. Refusing that is not caution, it is waste.

        It is safe for one specific reason, and the reason does not generalise. llama.cpp
        compares the restored cache's tokens against the incoming prompt itself and reuses
        only the common prefix, so a wrong guess costs a re-prefill rather than producing
        wrong output. That check covers tokens, not weights - which is exactly why a
        foreign *model's* attachment is still refused outright here. Same model, different
        text: the runtime protects us. Different model, same text: nothing does.
        """
        target = content.token_ids
        best: PrefixMatch | None = None
        for digest in self.published():
            path = self._attachment_path(digest, attachment_key(model, abi))
            if not path.is_file():
                continue
            try:
                candidate = self.content(digest)
            except SharedError:
                continue        # tampered or unreadable content is never offered
            shared = _common_prefix(candidate.token_ids, target)
            if shared < minimum:
                continue
            if best is None or shared > best.shared_tokens:
                best = PrefixMatch(path=path, content_digest=digest, shared_tokens=shared,
                                   attachment_tokens=len(candidate.token_ids),
                                   target_tokens=len(target),
                                   exact=candidate.token_ids == target)
        return best

    def attachments(self, content_digest: str) -> tuple[str, ...]:
        """Which attachment keys are warmed for this content, newest last."""
        require_digest(content_digest, "content_digest")
        prefix = f"{content_digest}."
        keys = [p.name[len(prefix):-len(ATTACHMENT_SUFFIX)]
                for p in sorted(self.root.glob(f"{prefix}*{ATTACHMENT_SUFFIX}"))]
        return tuple(keys)
