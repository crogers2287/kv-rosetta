"""EXPERIMENTAL: a vLLM KV connector that speaks the canonical KVX representation.

vLLM has no HTTP restore endpoint - I enumerated all 25 on a running server and there is no
seam for handing it a cache. Its seam is KVConnectorBase_V1, an in-process interface that
receives the paged KV tensors directly. So a vLLM adapter cannot talk to vLLM over the wire;
it has to live inside the worker. That is what this is.

vLLM's paged buffer is (num_pages, page_size, ...) and is addressed by a slot mapping, where
slot = block_id * block_size + offset. llama.cpp's is contiguous rows per layer. Neither is
canonical, and neither is asked to be: both convert to (token, head, dim), which is the only
thing the two runtimes can agree on.

The bridge below holds all of that and imports nothing from vLLM, so it can be tested without
a GPU or a vLLM install. The connector subclass is a thin shell over it, because a class that
can only be constructed inside a vLLM worker is a class that never gets tested.

Fail-closed here means returning zero matched tokens: vLLM then prefills exactly as it would
without a connector. A connector that guesses wrong does not degrade the cache, it corrupts
the response, so every uncertainty resolves to "prefill it".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TORCH_MISSING = "torch is required inside a vLLM worker; the bridge is testable without it"


class ConnectorError(ValueError):
    """Raised for a programming error. Identity mismatches are not errors - they are zeros."""


@dataclass(frozen=True)
class ShardIdentity:
    """Which slice of a tensor-parallel model produced a cache.

    A rank's cache holds only its own attention heads. Treating one shard as a whole cache
    would silently restore a fraction of the state and leave the rest stale, so the shard is
    part of identity rather than a note attached to it.
    """

    tp_rank: int = 0
    tp_size: int = 1

    def validate(self) -> list[str]:
        problems = []
        if self.tp_size < 1:
            problems.append(f"tp_size {self.tp_size} is not positive")
        if not 0 <= self.tp_rank < max(self.tp_size, 1):
            problems.append(f"tp_rank {self.tp_rank} outside 0..{self.tp_size - 1}")
        return problems

    @property
    def whole(self) -> bool:
        return self.tp_size == 1


@dataclass
class BridgeStats:
    extracted_layers: int = 0
    injected_layers: int = 0
    refusals: list[str] = field(default_factory=list)


class CanonicalBridge:
    """Moves tensors between vLLM's paged buffer and canonical (token, head, dim).

    Deliberately free of vLLM imports so it can be exercised offline.
    """

    def __init__(self, shard: ShardIdentity | None = None) -> None:
        self.shard = shard or ShardIdentity()
        problems = self.shard.validate()
        if problems:
            raise ConnectorError("; ".join(problems))
        self.stats = BridgeStats()

    @staticmethod
    def _flat(layer):
        """vLLM stores (num_pages, page_size, ...); slots address the first two flattened."""
        if layer.ndim < 2:
            raise ConnectorError(f"paged layer has {layer.ndim} dimensions, expected at "
                                 f"least 2 (num_pages, page_size, ...)")
        pages, page_size = layer.shape[0], layer.shape[1]
        return layer.reshape(pages * page_size, -1), pages * page_size

    def extract(self, layer, slot_mapping, *, n_head: int, head_dim: int):
        """Canonical (token, head, dim) for the slots this request occupies."""
        flat, slots = self._flat(layer)
        if len(slot_mapping) == 0:
            raise ConnectorError("empty slot mapping")
        highest = int(max(slot_mapping))
        if highest >= slots:
            raise ConnectorError(f"slot {highest} is outside the {slots}-slot buffer")
        if any(int(s) < 0 for s in slot_mapping):
            raise ConnectorError("negative slot in the mapping")
        gathered = flat[list(int(s) for s in slot_mapping), ...]
        width = gathered.shape[-1]
        if width != n_head * head_dim:
            raise ConnectorError(f"layer width {width} does not match the declared geometry "
                                 f"{n_head}x{head_dim}={n_head * head_dim}")
        self.stats.extracted_layers += 1
        return gathered.reshape(len(slot_mapping), n_head, head_dim)

    def inject(self, layer, slot_mapping, canonical) -> None:
        """Write canonical tensors back into the paged buffer, in place."""
        flat, slots = self._flat(layer)
        if canonical.shape[0] != len(slot_mapping):
            raise ConnectorError(f"{canonical.shape[0]} tokens for {len(slot_mapping)} "
                                 f"slots; refusing a partial write that would leave the "
                                 f"remainder stale")
        if len(slot_mapping) == 0:
            raise ConnectorError("empty slot mapping")
        highest = int(max(slot_mapping))
        if highest >= slots:
            raise ConnectorError(f"slot {highest} is outside the {slots}-slot buffer")
        # numpy would take a negative index as an offset from the end and write there
        # without complaint, so an out-of-range slot on this side corrupts a live token's
        # cache rather than failing. extract refuses the same input; so must this.
        if any(int(s) < 0 for s in slot_mapping):
            raise ConnectorError("negative slot in the mapping")
        width = flat.shape[-1]
        flattened = canonical.reshape(canonical.shape[0], -1)
        if flattened.shape[-1] != width:
            raise ConnectorError(f"canonical width {flattened.shape[-1]} does not match the "
                                 f"buffer's {width}")
        flat[list(int(s) for s in slot_mapping), ...] = flattened
        self.stats.injected_layers += 1

    # -- the fail-closed decision ------------------------------------------------------

    def matched_tokens(self, *, artifact_shard: ShardIdentity, artifact_model: str,
                       live_model: str, artifact_tokens: list[int],
                       request_tokens: list[int]) -> int:
        """How many tokens may be served from an artifact. Zero on any doubt.

        Returning zero is not an error path - it is vLLM prefilling exactly as it would
        without a connector. A wrong answer here does not slow a response down, it makes it
        wrong, so every uncertainty resolves to zero.
        """
        def refuse(reason: str) -> int:
            self.stats.refusals.append(reason)
            return 0

        if not artifact_model or artifact_model != live_model:
            return refuse(f"model identity {artifact_model!r} does not match {live_model!r}")
        if artifact_shard != self.shard:
            return refuse(f"artifact is rank {artifact_shard.tp_rank}/{artifact_shard.tp_size}, "
                          f"this worker is {self.shard.tp_rank}/{self.shard.tp_size}; a "
                          f"shard is not a whole cache")
        if not artifact_tokens or not request_tokens:
            return refuse("no tokens to compare")
        shared = 0
        for a, b in zip(artifact_tokens, request_tokens):
            if a != b:
                break
            shared += 1
        if shared < len(artifact_tokens):
            return refuse(f"artifact diverges from the request at token {shared}; a prefix "
                          f"cache is only valid for the exact prefix it holds")
        return shared


@dataclass(frozen=True)
class Selection:
    """What a lookup decided, including why it decided nothing."""

    digest: str = ""
    matched: int = 0
    refusals: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.digest) and self.matched > 0


def artifact_shard(manifest: dict) -> ShardIdentity:
    """The shard an artifact was written from. Absent fields mean whole, not unknown-ok.

    A manifest with no shard fields is treated as tp_size 1 because that is what a
    single-worker export is. It is not a licence to skip the comparison: a rank-0-of-2
    artifact reaching a tp_size-1 worker still fails, which is the case that matters.

    Present-but-nonsense is not coerced. An earlier version wrote `or 1`, which turned a
    manifest claiming tp_size 0 into a claim of a whole cache - inventing the one fact the
    comparison exists to check. The caller validates what comes back.
    """
    rank = manifest.get("tp_rank", 0)
    size = manifest.get("tp_size", 1)
    return ShardIdentity(tp_rank=int(0 if rank is None else rank),
                         tp_size=int(1 if size is None else size))


def select_artifact(objects, *, live_model: str, request_tokens: list[int],
                    bridge: "CanonicalBridge") -> Selection:
    """The longest admitted prefix that may serve this request, or nothing.

    Every rejection is recorded rather than swallowed, because "no artifact matched" and
    "an artifact matched but was refused" are different operational facts and the second one
    is the one worth reading a log for.

    An artifact without recorded prompt tokens is refused, not trusted. Reuse is a claim
    about which tokens are already in the cache, and there is no way to check that claim
    against a request without the tokens. This is the same rule the sidecar applies.
    """
    best = Selection()
    refusals: list[str] = []
    for obj in objects:
        manifest = obj.manifest or {}
        tokens = manifest.get("prompt_token_ids")
        if not tokens:
            refusals.append(f"{obj.digest[:12]}: no prompt tokens recorded, so reuse "
                            f"cannot be verified")
            continue
        try:
            shard = artifact_shard(manifest)
            problems = shard.validate()
        except (TypeError, ValueError) as exc:
            problems = [f"unreadable shard fields: {exc}"]
        if problems:
            refusals.append(f"{obj.digest[:12]}: {'; '.join(problems)}; a malformed shard "
                            f"is not a whole cache")
            continue
        before = len(bridge.stats.refusals)
        matched = bridge.matched_tokens(
            artifact_shard=shard,
            artifact_model=str(manifest.get("runtime_model") or ""),
            live_model=live_model,
            artifact_tokens=[int(t) for t in tokens],
            request_tokens=request_tokens)
        for reason in bridge.stats.refusals[before:]:
            refusals.append(f"{obj.digest[:12]}: {reason}")
        # Ties break on the digest so a store holding two equal prefixes always picks the
        # same one; an unstable choice would make a bad artifact look intermittent.
        if matched > best.matched or (matched == best.matched and matched > 0
                                      and obj.digest < best.digest):
            best = Selection(digest=obj.digest, matched=matched)
    return Selection(digest=best.digest, matched=best.matched, refusals=tuple(refusals))


def build_connector(vllm_config: Any, role: Any):
    """Construct the real connector. Imports vLLM only when actually used inside a worker."""
    def store_for(config):
        """The admitted store named in the connector's extra config, or None.

        Absent configuration yields None and the connector behaves exactly as vLLM does
        without one. A configured store that cannot be opened is an error rather than a
        silent None: an operator who named a store meant to use it.
        """
        extra = getattr(getattr(config, "kv_transfer_config", None),
                        "kv_connector_extra_config", None) or {}
        root = extra.get("kv_rosetta_store")
        if not root:
            return None
        from kv_rosetta.admitted_store import AdmittedStore
        return AdmittedStore(root, create=False)

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    except ImportError as exc:                     # pragma: no cover - needs a vLLM install
        raise ConnectorError(f"vLLM is not importable here: {exc}") from exc

    class KVRosettaConnector(KVConnectorBase_V1):  # pragma: no cover - needs a live worker
        """Thin shell: every decision lives in CanonicalBridge, which is tested offline."""

        def __init__(self, config, connector_role):
            super().__init__(config, connector_role)
            parallel = getattr(getattr(config, "parallel_config", None), "__dict__", {})
            self.bridge = CanonicalBridge(ShardIdentity(
                tp_rank=int(parallel.get("rank", 0) or 0),
                tp_size=int(parallel.get("tensor_parallel_size", 1) or 1)))
            self.store = store_for(config)
            self.live_model = str(getattr(getattr(config, "model_config", None),
                                          "model", "") or "")
            self.selection = Selection()

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            """Ask the store what it can serve. No store configured means prefill."""
            if self.store is None:
                return 0, False
            tokens = [int(t) for t in getattr(request, "prompt_token_ids", []) or []]
            chosen = select_artifact(self.store.list_objects(),
                                     live_model=self.live_model,
                                     request_tokens=tokens, bridge=self.bridge)
            self.selection = chosen
            # Already-computed tokens are not new ones, and a load that would not add
            # anything is not worth a transfer.
            gain = max(0, chosen.matched - int(num_computed_tokens or 0))
            return (gain, False) if gain else (0, False)

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            return None

        def build_connector_meta(self, scheduler_output):
            from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
            return KVConnectorMetadata()

        def start_load_kv(self, forward_context, **kwargs):
            return None

        def wait_for_layer_load(self, layer_name):
            return None

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
            return None

        def wait_for_save(self):
            return None

    return KVRosettaConnector(vllm_config, role)
