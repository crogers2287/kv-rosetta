# KV Rosetta research steer: move from non-hybrid decode proof to the real 8K hybrid gate

Status basis: default-branch head cd61dec5a08be44e65d633c3df32d3949e7085bc.

This steer supersedes e41977e. The decoder now materializes canonical-shaped tensors and has parsed one real llama.cpp state file. That is genuine progress, but the real file was a 49-token non-hybrid sequence-state-v2 artifact. It does not validate recurrent state, SCKP, the patched hybrid compound tuple, or the retained production-shape 8K object. The next gate remains the exact 8K hybrid artifact, after closing inherited trust boundaries that recent commits still did not address.

## Mission

Deliver one canonical KVX prefix artifact that can be transformed across cache dtypes, inference runtimes, and CUDA/ROCm/Vulkan backends, with quality-gated cross-model transformation later.

The opaque llama.cpp admitted-store path remains a strict exact-tuple foundation for qwen35-family hybrids. Preserve refusal on unpatched, incomplete, mismatched, or unsupported runtimes and artifacts. Do not weaken CacheABIIdentity, compound tuple, model/GGUF identity, checkpoint metadata, or admitted-store guards to simulate portability.

Primary opaque-runtime evidence:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep patches 0001 and 0002 pinned and identified. Do not post upstream.

## Evidence state

### Production-shape opaque reuse is proven

The exact tested Qwen3.8-27B/qwen35 8K gate passed 3/3:

- 8,188/8,192 tokens reused;
- median 1.848 s restored versus 4.876 s cold, 62.1% faster;
- output and nonempty probability-vector parity;
- persistent ext4/NVMe natural-cache conditions recorded;
- zero request-path payload copying;
- unpatched refusal with zero state-endpoint calls.

This proves exact-tuple process-restart economics, not portability or host restart.

### Real non-hybrid attention parsing is now measured

Commit cd61dec parsed a real 1,808,028-byte llama.cpp state file produced from a small non-hybrid model:

- 49 cells;
- 36 layers and 72 K/V spans matching GGUF geometry;
- a 512-byte f16 row stride matching 2 KV heads × 128 dimensions;
- exact consumption of the non-hybrid file;
- one materialized span shaped (49, 2, 128), finite and nonzero.

This independently falsifies many attention-header/stride mistakes and shows the attention layout is compatible with one unpatched sequence-state-v2 artifact.

It does not prove:

- patched sequence-state-v3 plus SCKP/1;
- attention-to-recurrent boundary recovery;
- recurrent layer attribution;
- checkpoint target recursion;
- the qwen35 cell/geometry descriptor;
- the 1,007,783,892-byte 8K artifact;
- numeric equality with runtime-native tensors or completions.

The live test passes `has_cell_ext=False` directly and uses physical file size as the attention-section end. For hybrids it explicitly skips the exact-consumption assertion. Do not describe this as the real hybrid gate.

### Materialization and dequantization are useful but preliminary

Commit 53ec55b materializes a span as float32 in token/head/dimension order, normalizes transposed V, and implements f32, f16, bf16, q8_0, and q4_0 decoding. Its constructed tests are useful independent byte-layout checks.

Audit limits:

- The parser's transposed-q8_0 header contract remains inconsistent with the pinned writer, so that path cannot reach trustworthy materialization.
- The real-file numeric assertion is only finite/nonzero. It does not compare decoded values to an independent runtime/native oracle.
- `materialise` reads an entire span into memory. That is acceptable for the current small oracle and bounded per-span sampling, but it is not yet a production streaming-conversion design.
- Float32 is a compute/interchange array in this implementation, not yet an approved on-disk canonical dtype. Do not silently double large-prefix storage. Fix the canonical physical dtype and size/economic consequences before writing an 8K container.

## RA-003 — answered

Withdraw the categorical 32K storage refusal, but do not admit 32K from a two-point affine fit. Later derive a componentized estimator from exact writer geometry and validate it against retained 2K and 8K sizes. 32K remains deferred by priority and an unvalidated estimator.

## RA-004 — answered

Scheduled/proactive warming is excluded. The sidecar is demand-driven and may act only on a model already reported ready.

The existing localhost API remains inert scaffolding: `ensure` always falls back. Do not expand or activate it before a restore seam is behaviorally proven. A model name from `/running` is not CacheABIIdentity; active wiring later requires exact build/model/cache/checkpoint identity tied to the same ready runtime instance plus a live sacrificial-model no-wake gate. The permanently zero `models_woken` counter is not evidence by itself.

## P0 — stop feature expansion and close inherited trust boundaries

Write mutation-sensitive red tests first, then the smallest causal fixes.

### Reader and parser

Require:

1. A final K, V, R, or S payload physically truncated while the declared section end stays unchanged must refuse without materializing it. Validate physical extent independently of declared bounds before returning spans.
2. Attention and recurrent `cell_count` must be bounded before looping or allocating, both by conservative remaining bytes and an explicit maximum from bound source geometry/context.
3. Cell-extension presence and exact positive width must come from the pinned architecture descriptor. `has_cell_ext=True, cell_ext_size=0` must refuse.
4. Transposed q8_0 must match the pinned writer's `ggml_type_size(q8_0)=34` contract. If actual quantized transposed-V semantics cannot be confirmed with writer-produced bytes, refuse that combination explicitly rather than guessing.
5. Attention and recurrent `n_layer` must equal exact bound source geometry, not merely fall in 1..512.
6. The recurrent layer map must be unique, ascending in writer order, in range, and bound to the exact source GGUF/model identity.
7. A qwen35 hybrid SCKP appendix must contain at least one checkpoint with positive `n_tokens`; draft/speculative payloads remain refused.
8. A top-level compound parser must account for every byte as envelope, attention, recurrent state, or SCKP. No magic scanning and no unexplained trailing bytes.

Keep reads and peak memory bounded independently of artifact size. Retain sparse multi-GiB tests with an asserted metadata-read ceiling.

### Canonical schema

Remove coercive untrusted JSON decoding:

- booleans must be JSON booleans;
- integer fields must be integers and reject booleans;
- RoPE theta must be finite and positive;
- digests must be exactly 64 lowercase hexadecimal characters;
- recurrent dimensions must be nonnegative, with qwen35-required fields positive and descriptor-consistent;
- segment references must resolve uniquely with correct roles at container integration;
- unknown fields must refuse unless deliberately versioned.

Acceptance: every guard fails red on current head, survives mutation checking, and the full offline suite passes. Do not alter opaque import/export behavior.

## P1 — largest justified break-first experiment: exact real 8K hybrid structural parse

Parse the admitted GGSQ/3+SCKP/1 object with digest `2af6ca68737a1888bd65c67cf4d36746123cf18e51824e50f94c888f6be80c72`, using the exact bound Qwen3.8-27B GGUF/model identity and pinned patched writer.

Locate structure only. Do not materialize the full payload.

Retain a machine-readable record with:

- repository/parser commit and clean-tree state;
- artifact digest and unchanged before/after file facts;
- exact GGUF/model digest and fully resolved architecture descriptor;
- physical file length plus exact envelope, attention, recurrent, and SCKP bounds;
- cell/layer counts, types, strides, transposition, recurrent map, checkpoint extents, and all spans;
- proof all spans are in-bounds, non-overlapping, writer-ordered, and collectively account for every byte;
- parsed checkpoint count/coverage/positions equal the admitted manifest;
- actual bytes read and peak memory;
- refusal controls for wrong geometry/map, wrong tuple/version, physical truncation, and one-byte boundary shifts.

Never mutate the admitted object. Use private or sparse corruptions.

Decision rule:

- On the first mismatch, retain it, identify the earliest divergent writer field, apply only the causal fix, and rerun the identical gate.
- Exact agreement unlocks bounded hybrid numeric sampling.
- Any field not uniquely recoverable from exact source GGUF plus pinned descriptor fires the smallest-upstream-seam falsifier. Never infer from model name or byte patterns.

## P2 — independent numeric gate and canonical physical representation

After P1:

1. Compare deterministic f16, q8_0, and q4_0 decoded values with an oracle that calls or is generated by pinned ggml reference code. Manual packing remains useful unit coverage but is not the only numeric authority.
2. Decode bounded attention and recurrent samples from the real 8K hybrid object. Confirm shapes, finiteness, and exact source geometry.
3. Where runtime/native cache values cannot be exported independently, require a behavioral reconstruction test before calling the numbers proven.
4. Decide and version the on-disk canonical physical dtype and chunking. Float32 may be used for computation, but do not default the persistent 8K artifact to float32 without measured size, conversion cost, and quality rationale.
5. Require chunked or bounded-memory conversion; no whole-artifact Python byte copies.

Only after this gate may a canonical 8K KVX artifact be emitted.

## P3 — same-model f16-to-q8_0 target conversion

Add the narrowest target encoder/import seam for the exact same model. The opaque path must refuse the dtype mismatch; only the canonical route may transform it.

Fix thresholds before the live run and compare against target-native q8_0 reuse:

- successful prefix reuse;
- output and nonempty probability/logit divergence;
- exact prompt and target identity;
- source/target runtime, dtype, backend, and device provenance;
- native target prefill fallback on every unsupported or failed case.

A pass unlocks one same-model CUDA/ROCm/Vulkan matrix, not cross-model conversion.

## P4 — activate the minimal demand-driven sidecar

After P3, connect only the proven seam to `/v1/ensure`.

Require ready-model status, exact same-instance runtime identity, no scheduling mode, a live no-wake sacrificial-model test, reused-token and transfer-mode reporting, and native-prefill fallback.

Do not add broad service, authentication, multi-tenant, or distributed scheduling scope.

## Deferred

- 32K opaque benchmarking;
- host-restart/cold-boot claims;
- 131K;
- learned cross-model mapping/token alignment;
- production vLLM/HF adapters;
- MTP/draft/speculative state;
- upstream comments/submissions.

HF may be used earlier only as an independent numeric oracle.

## Required execution order

1. Close the inherited reader and strict-schema defects.
2. Retain the exact structural parse of the real 8K hybrid artifact.
3. Prove bounded hybrid numeric decoding and choose the canonical physical representation.
4. Prove same-model f16-to-q8_0 conversion.
5. Activate the demand-driven sidecar through that proven seam.
