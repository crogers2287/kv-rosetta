# KV Rosetta research steer: prove real-artifact canonical decode before wiring the sidecar

Status basis: default-branch head bf2202663bc5191e65ec1712d56b042e769c2e3e.

This steer supersedes 92546ba. The 8K opaque path remains proven and closed. Canonical parsing has expanded to recurrent state and SCKP, and a demand-driven sidecar skeleton has landed. Neither change yet proves portable reuse: the parser has only synthetic structural fixtures and inherited unresolved fail-open boundaries, while the sidecar's `ensure` method always falls back and performs no restore.

## Mission

Deliver one canonical KVX prefix artifact that can be transformed across cache dtypes, inference runtimes, and CUDA/ROCm/Vulkan backends, with quality-gated cross-model transformation later.

The opaque llama.cpp admitted-store path remains a strict exact-tuple foundation for qwen35-family hybrids. Preserve refusal on unpatched, incomplete, mismatched, or unsupported runtimes and artifacts. Do not weaken its CacheABIIdentity, compound tuple, model/GGUF identity, checkpoint metadata, or admitted-store file/ownership guards to simulate portability.

Primary opaque-runtime evidence remains:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep patches 0001 and 0002 pinned and identified. Do not post upstream.

## Retained evidence

### Production-shape opaque gate

The exact tested Qwen3.8-27B/qwen35 8K gate passed 3/3:

- 8,188/8,192 tokens reused;
- median 1.848 s restored versus 4.876 s cold, 62.1% faster;
- output and nonempty probability-vector parity;
- persistent ext4/NVMe, natural page-cache conditions recorded;
- zero request-path payload copying;
- unpatched refusal with zero state-endpoint calls.

This proves exact-tuple process-restart economics, not host restart or portability.

### Canonical parser status

Commit bf22026 adds source-layer-map-driven recurrent R/S spans and recursively parses each SCKP target payload rather than forwarding opaque recurrent bytes. It correctly refuses draft/speculative payloads and requires exact declared target consumption.

This is meaningful structural progress, but all fixtures are constructed by the same repository from the documented field order. No retained record demonstrates the parser against the admitted 1,007,783,892-byte 8K artifact.

The prior P0 was not completed before parser expansion. Current source still:

- lets `Reader.skip()` authorize bytes beyond physical EOF when the caller's declared end is larger;
- loops and allocates from unbounded attention and recurrent `cell_count`;
- permits `has_cell_ext=True` with `cell_ext_size=0`;
- interprets transposed q8_0 `v_size_el` as 1 although the pinned writer contract records `ggml_type_size(q8_0)=34`;
- accepts recurrent layer maps that are not in source-writer order;
- validates attention/recurrent layer counts only as 1..512, not against exact bound source geometry;
- accepts an empty checkpoint appendix and a checkpoint with `n_tokens=0`, although the canonical hybrid schema requires at least one positive-coverage checkpoint;
- retains coercive schema decoding, including `bool("false") == True`, length-only digests, non-finite/non-positive RoPE theta, and negative recurrent widths.

Do not add more parser surface until these defects are red-tested and closed.

## RA-003 — answered

The categorical claim that 32K cannot fit the current NVMe is withdrawn. Do not admit 32K from a two-point affine fit. Later replace the conservative guard with a componentized estimator derived from the pinned writer and exact source geometry, then validate it against retained 2K and 8K sizes.

32K remains deferred by product priority and an unvalidated estimator, not by a claim of known storage insufficiency.

## RA-004 — answered

Scheduled or proactive warming of arbitrary models is excluded from the sidecar contract. The sidecar must be demand-driven and may act only for a model already reported ready because waking an unused model recreates the eviction harm regardless of whether the work is prefill or restore.

The staged order is:

1. retain the inert demand-driven API skeleton and its offline tests;
2. complete and behaviorally prove the canonical conversion seam;
3. then connect one proven exact-tuple or canonical restore action;
4. only afterward use the service in a harness.

Do not expand the HTTP surface while `ensure` always falls back. Commit 7998bda is scaffolding, not a functional kvwarm replacement.

Before any live restore wiring, close the identity/race gap:

- a model name returned by llama-swap `/running` is not CacheABIIdentity;
- obtain build, model digest, cache dtypes, architecture, checkpoint protocol, and active-state facts only through a path behaviorally proven not to wake an unloaded model;
- bind those facts to the same ready runtime instance used for restore, using a generation/PID/instance token or an equivalent before-and-after check;
- if readiness or instance continuity cannot be proven atomically enough, refuse and prefill natively;
- never restore based only on a caller-supplied model string and slot number.

The current `models_woken` counter is initialized to zero and has no path that increments it, so observing zero is not evidence. Keep the offline URL-interception tests, but add a live llama-swap integration gate using a sacrificial unloaded model: capture swap/runtime request logs, exercise every public sidecar route, and prove the model remains unloaded and no waking endpoint was called. Run this only when an active restore is about to be connected; it does not preempt canonical decode.

## P0 — close inherited parser and schema trust boundaries

Write mutation-sensitive red tests first.

### Reader and section parsers

Require:

1. A final K, V, R, or S payload physically truncated while the declared section end remains unchanged must refuse without materializing the payload. Verify file extent independently of declared bounds before returning spans.
2. Attention and recurrent `cell_count` must be bounded before looping or allocating, both by conservative bytes remaining and an explicit expected maximum from bound source geometry/context.
3. Cell-extension presence and exact positive width must come from the pinned architecture descriptor. Ambiguous or zero width when present must refuse.
4. Transposed q8_0 must follow the pinned writer's 34-byte `ggml_type_size` contract; a declared value of 1 must refuse. If independent confirmation is unavailable, refuse quantized transposed V explicitly.
5. Expected attention and recurrent `n_layer` must equal the bound source geometry.
6. The recurrent layer map must be unique, ascending in writer order, in range, and tied to the exact GGUF/model digest. A merely size-compatible caller map is not evidence.
7. For qwen35 hybrid artifacts, SCKP count and checkpoint `n_tokens` must be positive; draft/speculative payloads remain refused.
8. A top-level parser must account for every byte as envelope, attention, recurrent state, or SCKP. No scanning for magic inside payloads and no unexplained trailing bytes.

Keep reads and peak memory bounded independently of payload size. Retain sparse multi-GiB tests that assert a small maximum metadata-byte count.

### Canonical schema

Require exact JSON types rather than `str`, `int`, `float`, or `bool` coercion:

- booleans are JSON booleans;
- integer fields are integers and booleans are rejected as integers;
- RoPE theta is finite and positive;
- digests are exactly 64 lowercase hexadecimal characters;
- recurrent dimensions are nonnegative, with qwen35-required fields positive and descriptor-consistent;
- segment references resolve to unique segments with the required roles when integrated with the container;
- unknown fields refuse unless deliberately versioned.

Acceptance: each guard fails red on current head, survives mutation checking, and the full offline suite passes. Do not alter opaque import/export behavior.

## P1 — largest justified break-first gate: parse the retained real 8K artifact

After P0, parse the exact admitted GGSQ/3+SCKP/1 object with digest `2af6ca68737a1888bd65c67cf4d36746123cf18e51824e50f94c888f6be80c72`, using the bound GGUF/model digest and pinned writer source.

Locate structure only; do not materialize tensor payloads.

Retain a machine-readable record with:

- parser and repository commit;
- artifact digest and unchanged before/after file facts;
- exact source GGUF/model identity and resolved geometry;
- physical file length and exact envelope, attention, recurrent, and SCKP boundaries;
- cell/layer counts, types, strides, transposition, recurrent layer map, checkpoint extents, and every span;
- proof spans are in-bounds, non-overlapping, writer-ordered, and collectively account for all bytes;
- parsed checkpoint facts equal to the admitted manifest;
- total actual bytes read and peak memory;
- refusal controls for wrong geometry, wrong recurrent map, wrong tuple/version, physical truncation, and one-byte boundary shifts.

Never mutate the admitted store object. Use private or sparse copies for corruptions.

Decision rule:

- First mismatch: retain it, identify the earliest divergent writer field, make only the causal fix, and rerun the identical gate.
- Exact structural agreement: proceed to numeric canonical materialization.
- Any field not uniquely recoverable from the exact source GGUF plus pinned descriptor fires the upstream-seam falsifier. Record that exact field; do not infer from model names or byte patterns.

## P2 — prove canonical numbers on an independent oracle

Materialize only a deterministic small fixture into canonical little-endian attention and recurrent tensors. Normalize transposed V during decode.

The expected offsets, shapes, and numeric values must come from the pinned writer, a separate reference implementation, or HF where appropriate—not the decoder or an encoder paired with it. A self-consistent round trip is not evidence.

Require exact and quantized dtype cases plus corruption, truncation, wrong geometry/map, unsupported type, and trailing-byte controls. After the small oracle passes, decode a bounded sample from the retained 8K artifact; do not allocate the complete 8K canonical cache merely to test the decoder.

## P3 — prove same-model f16-to-q8_0 conversion

Add the narrowest target encoder/import seam for the exact same model. The opaque path must refuse the dtype mismatch; only the canonical route may transform it.

Fix quality thresholds before the run and compare with target-native q8_0 reuse:

- successful target prefix reuse;
- output and nonempty probability/logit divergence;
- exact prompt and target model identity;
- source/target runtime, dtype, backend, and device provenance recorded outside canonical cache identity;
- native target prefill fallback on every unsupported or failed case.

A pass unlocks one same-model CUDA/ROCm/Vulkan matrix. It does not unlock cross-model transformation.

## P4 — connect the minimal demand-driven sidecar

After P3, connect only the proven seam to `/v1/ensure`.

Require:

- the model is already ready;
- exact runtime identity and same-instance continuity;
- no scheduled target enumeration or warming mode;
- live proof that every route leaves an unloaded sacrificial model unloaded;
- reused-token count, transfer mode, gate verdict, and fallback reason;
- native prefill on refusal or failure.

Do not add broad daemon, authentication, multi-tenant, or distributed scheduling surfaces.

## Deferred

- 32K opaque benchmarking;
- host-restart/cold-boot claims;
- 131K;
- learned cross-model projection and token alignment;
- production vLLM/HF adapters;
- MTP/draft/speculative state;
- upstream comments or submissions.

HF may be used earlier only as an independent numeric oracle.

## Required execution order

1. Close inherited reader and strict-schema defects.
2. Retain an exact structural parse of the real 8K artifact.
3. Prove canonical numeric decoding with an independent oracle.
4. Prove same-model f16-to-q8_0 conversion.
5. Wire that proven seam into the demand-driven sidecar and run the no-wake live gate.
