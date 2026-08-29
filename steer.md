# KV Rosetta research steer: quarantine premature runtime bridges and restore the real hybrid gate

Status basis: default-branch head 874bea186908475eb8bc5ab550044fbc7316a99b.

This steer supersedes 0987a34. Four commits added a ggml-verified q8_0/q4_0 encoder, a transfer policy, active sidecar restoration, and a vLLM connector-shaped module. The quantizer oracle is useful evidence. The latter three are scaffolding, not a proven portable transfer path, and two moved ahead of the ordered gates. Freeze new surfaces and return to the earliest causal evidence gap: a complete, fail-closed parse of the retained production 8K hybrid artifact.

## Mission and non-negotiable boundary

Deliver one canonical KVX prefix artifact that can be transformed across cache dtypes, inference runtimes, and CUDA/ROCm/Vulkan backends, with quality-gated cross-model transformation later.

The opaque llama.cpp admitted-store path remains the strict exact-tuple foundation for the qwen35 family. Preserve refusal on unpatched, incomplete, mismatched, unsupported, or active draft/speculative runtimes and artifacts. Never weaken CacheABIIdentity, model/GGUF identity, compound GGSQ/3+SCKP/1 identity, checkpoint metadata, or admitted-store guards to simulate portability.

Primary opaque evidence remains:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep patches 0001 and 0002 pinned and identified. Do not post upstream.

## Evidence that is now established

### Production-shape opaque reuse

The exact tested Qwen3.8-27B/qwen35 8K gate passed 3/3:

- 8,188/8,192 tokens reused;
- median 1.848 s restored versus 4.876 s cold, 62.1% faster;
- output and nonempty probability-vector parity;
- persistent ext4/NVMe natural-cache conditions recorded;
- zero request-path payload copying;
- unpatched refusal with zero state-endpoint calls.

This proves exact-tuple process-restart economics, not host-restart portability or canonical conversion.

### Numeric codec unit evidence

Commit 233499a compares local q8_0 and q4_0 encoders byte-for-byte against exported pinned ggml reference functions over designed edge cases, and verifies local decoders against ggml-produced bytes. This materially strengthens the numeric oracle.

It does not prove that a converted KV cache can be encoded into either target runtime, restored, or used with behavioral parity. Keep `SUPPORTED_DTYPE_PAIRS` descriptive of experimentally proposed pairs until the live gate passes; offline codec equality alone must not advertise a usable transfer.

### Real parsing evidence remains non-hybrid

The only retained real state-file decode is a 49-token, non-hybrid, sequence-state-v2 llama.cpp artifact. It validates attention structure for that file. It does not validate recurrent state, SCKP, state-v3, qwen35 geometry, or the retained 1,007,783,892-byte 8K compound artifact.

No later commit closes the inherited reader/schema trust boundaries or retains the required real 8K structural record.

## Corrections to premature surfaces

### Sidecar: active restore is not yet proven no-wake

Commit 60921fd wires `/v1/ensure` to `/upstream/<model>` after a separate `/running` check. The tests use fakes and monkeypatched restore reports. No retained live llama-swap record proves a successful real restore while every route leaves an unloaded sacrificial model unloaded.

A status check followed by a separate upstream request is a time-of-check/time-of-use race: the named model can unload or be replaced between the two calls, and the upstream route can then wake a model. A model name is also not a same-instance runtime identity.

Until the live gate below passes, the sidecar must fail closed before any active upstream restore unless an explicit research-only opt-in is set. Production/default behavior remains native prefill. Do not treat the permanently zero `models_woken` counter or URL-construction-count tests as evidence.

Red test first:

1. Report the target ready.
2. Unload or replace it before the restore request.
3. Assert no target-model wake, no state endpoint call to the replacement, and native-prefill fallback.
4. Bind readiness and restoration to one exact runtime instance/generation carrying full live CacheABIIdentity. If llama-swap exposes no atomic identity/lease seam, retain that failure and keep active restore disabled rather than approximating it with two requests.
5. Run a live sacrificial unloaded-model matrix across every sidecar route. Retain swap state before/after and actual target endpoint traces.

### vLLM: inert connector plus a contradicted layout assumption

Commit 874bea adds a useful offline `CanonicalBridge`, but the real `KVConnectorBase_V1` subclass always reports zero matched tokens and all scheduler/worker load/save methods are no-ops. It is intentionally inert and must be described as such.

More importantly, `CanonicalBridge._flat` assumes the first two dimensions are page and page-size. The current official vLLM example describes the standardized per-layer view as `[B, H, N, C]`; non-MLA slot lookup indexes block and the `N` offset while preserving `H`. Only MLA (`H == 1`) is flattened. The current synthetic tests therefore do not establish correct non-MLA extraction/injection. Rank derivation from `parallel_config.__dict__["rank"]` is also an unproven guess, not live shard identity.

Keep the connector's matched-token result at zero and all load paths inert. Do not register or deploy it.

Red test first, against the exact installed vLLM 0.27.1 source/runtime:

1. Capture a real standardized per-layer K/V view, attention metadata, block size, slot mapping, dtype, model digest, TP/PP rank, backend, and device provenance without admitting reuse.
2. Build separate non-MLA and MLA fixtures from that capture.
3. Require extraction and injection to match the version-pinned official indexing contract; a test shaped so flattening `B*H` instead of indexing `B,N` must fail on current head.
4. Obtain TP/PP identity from live distributed worker state and prove every shard is present and correctly attributed.
5. Keep `get_num_new_matched_tokens == 0` until actual scheduler metadata, allocation, layer load, completion parity, and fallback behavior pass end to end.

This validation is subordinate to the canonical source-artifact gate below. It may produce a retained layout record now, but it must not consume implementation time beyond the smallest red fixture/fail-closed correction.

## P0 — close inherited reader and schema trust boundaries

Write mutation-sensitive red tests, then the smallest causal fixes:

1. Refuse a physically truncated final K, V, R, or S payload even when declared section bounds remain self-consistent.
2. Bound attention and recurrent `cell_count` before looping/allocation by remaining bytes and exact source geometry.
3. Bind cell-extension presence and positive width to the pinned architecture descriptor; refuse `has_cell_ext=True, cell_ext_size=0`.
4. Match transposed q8_0 to the pinned writer's 34-byte block contract, or refuse it until writer-produced bytes resolve the semantics.
5. Require exact attention/recurrent layer counts and a unique, ascending, in-range recurrent map bound to the exact GGUF/model.
6. Require at least one positive-coverage qwen35 checkpoint; keep draft/speculative payloads refused.
7. Account for every byte as envelope, attention, recurrent state, or SCKP without scanning for magic.
8. Replace coercive schema decoding: exact JSON types, finite positive RoPE theta, exact lowercase SHA-256 digests, descriptor-consistent geometry, unique segment roles, and deliberate versioned unknown-field policy.

Keep metadata reads and peak memory bounded independently of artifact size. Preserve sparse multi-GiB read-ceiling tests.

## P1 — largest justified break-first experiment: exact 8K hybrid structural parse

Parse the admitted GGSQ/3+SCKP/1 object with digest `2af6ca68737a1888bd65c67cf4d36746123cf18e51824e50f94c888f6be80c72`, bound to the exact Qwen3.8-27B GGUF/model and pinned patched writer. Locate structure only; do not materialize the whole payload.

Retain a machine-readable record containing:

- clean repository/parser commit;
- artifact and model/GGUF digests plus unchanged file facts;
- resolved architecture descriptor;
- physical length and exact envelope, attention, recurrent, and SCKP bounds;
- cells, layers, dtypes, strides, transposition, recurrent map, checkpoint extents, and spans;
- proof every span is in-bounds, non-overlapping, writer-ordered, and collectively accounts for every byte;
- checkpoint count, positions, and coverage equal to admitted metadata;
- bytes read and peak memory;
- wrong-geometry/map/tuple, physical-truncation, and one-byte-boundary refusal controls.

On the first mismatch, retain it, identify the earliest divergent writer field, apply only the causal fix, and rerun the identical gate. Do not infer fields from model names or byte patterns.

## P2 — bounded numeric samples and canonical physical representation

After P1:

1. Use the pinned ggml oracle to validate deterministic f16/q8_0/q4_0 samples.
2. Decode bounded attention, recurrent, and checkpoint samples from the real 8K object and bind every shape to exact geometry.
3. Choose and version canonical on-disk dtype/chunking with measured size, conversion cost, and quality consequences. Float32 may be an intermediate, not an accidental persistent default.
4. Require bounded-memory conversion with no whole-artifact Python copies.
5. Emit the first canonical 8K artifact only after these gates pass.

## P3 — same-model f16-to-q8_0 behavioral conversion

Add the narrowest target encoder/import seam for the exact same model. Opaque f16-to-q8_0 remains refused; only the canonical route may transform it.

Fix thresholds before the live run and compare with target-native q8_0 reuse for exact prefix/model identity, reused tokens, output, nonempty probability/logit divergence, and full runtime/dtype/backend/device provenance. Every unsupported or failed case must prefill natively.

Only a pass unlocks the first live vLLM target connector and then the CUDA/ROCm/Vulkan matrix. The generic transfer orchestrator remains offline policy plumbing until a real encoder, target restore, and behavioral gate exercise it end to end.

## P4 — runtime integration

1. Complete the live vLLM connector against version-pinned standardized layouts and full shard identity; prove target-native parity before reporting any matched token.
2. Activate the demand-driven sidecar only with exact same-instance readiness plus the live no-wake matrix.
3. Expose only the proven transfer seam; no scheduling/proactive warming.

## Deferred

- 32K opaque benchmarking until a source-derived size estimator is validated;
- host-restart/cold-boot claims;
- 131K;
- learned cross-model mapping/token alignment;
- MTP/draft/speculative state;
- broad service/authentication/distributed scheduling work;
- upstream comments/submissions.

## Required execution order

1. Quarantine the premature sidecar and keep vLLM inert; retain the two red safety/layout fixtures.
2. Close inherited reader and strict-schema defects.
3. Retain the exact structural parse of the real 8K hybrid artifact.
4. Prove bounded hybrid numeric decoding and choose the canonical physical representation.
5. Prove same-model f16-to-q8_0 behavioral conversion.
6. Complete a live vLLM target path, then the backend matrix.
7. Activate the demand-driven sidecar only after its same-instance/no-wake gate.
