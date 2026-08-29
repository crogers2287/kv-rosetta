# KV Rosetta research steer: harden canonical ingestion against the real 8K artifact

Status basis: default-branch head d58777db689f6796cf34d76df22d276726dc646c.

This steer supersedes c867cda. The production-shape opaque gate has passed. The immediate critical path is no longer opaque-store economics; it is making the canonical decoder trustworthy enough to inspect the retained 8K artifact without guessing, allocating from attacker-controlled counts, or accepting bytes it never verified.

## Mission

Deliver one canonical KVX prefix artifact that a sidecar can transform for multiple cache dtypes, inference runtimes, and CUDA/ROCm/Vulkan backends, with quality-gated cross-model transformation later.

The opaque llama.cpp admitted-store path remains a strict source/runtime-specific foundation for qwen35-family hybrid models. Do not relax CacheABIIdentity, compound-tuple allowlists, model and GGUF identity, checkpoint metadata, store ownership/file guards, or unpatched-runtime refusals to simulate portability.

Primary opaque-runtime track:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep checkpoint patches 0001 and 0002 pinned and identified. Do not post upstream.

## Evidence that changes the priority

### 8K production-shape gate: passed

The retained three-repetition record on the exact tested Qwen3.8-27B/qwen35 model reports:

- 8,188/8,192 tokens reused in every repetition, with the four-token tail derived from checkpoint metadata;
- median restored total 1.848 s versus 4.876 s cold, a 62.1% reduction;
- three of three paired wins;
- output and nonempty probability-vector parity;
- complete fail-closed unpatched control with zero state-endpoint calls;
- zero request-path payload copying and unchanged admitted-object facts;
- persistent ext4 on nvme0n1 under explicitly recorded natural page-cache conditions;
- a 1,007,783,892-byte artifact and approximately two-restore admission break-even.

This is process-restart evidence on one host and exact model/runtime tuple. It is not host-restart, cross-dtype, cross-runtime, or cross-backend evidence. It nevertheless closes the required 8K economic gate and justifies moving the main track to canonical conversion.

### Canonical schema and first reader slice landed

`kvx-hybrid/1` now has explicit source geometry, recurrent layer mapping, checkpoint records, canonical attention layout, and draft/speculative refusal. The first GGSQ reader locates attention payload spans without materializing them.

These are useful foundations, not yet a trustworthy decode. Source review finds trust-boundary defects that must be made red before the reader touches the retained 8K object:

1. `Reader.skip()` checks only the caller-declared section end. It does not verify physical file length. A final payload truncated on disk can be accepted if the declared end remains unchanged because no later header read forces EOF.
2. `cell_count` is a raw uint32 used to drive a loop and grow `positions` before any count bound. A tiny malicious section can demand billions of iterations.
3. `has_cell_ext=True` accepts the default `cell_ext_size=0`. Because the file carries no cell-extension flag or width, that is an ambiguous architecture descriptor and must refuse.
4. The source-derived writer inventory says transposed V stores `v_size_el = ggml_type_size(v_type)`. The reader currently compares against bytes-per-block divided by block-size. For q8_0 that accepts 1 instead of the writer's 34. Existing transposed tests do not exercise a quantized type.
5. The module-level statement that unexplained trailing body bytes are refused is not implemented by this slice. For a hybrid body, attention is intentionally followed by recurrent state, so exact composition belongs in the top-level parser, not in an attention-only helper.

The schema loader also coerces untrusted JSON values with `bool()`, `str()`, `int()`, and `float()`. In particular, `"false"` becomes true. Digests are checked only for length, RoPE theta is not checked for finiteness or positivity, and recurrent widths can be negative. These are fail-open canonical identity defects.

## RA-003 — answered

Withdraw the categorical statement that a 32K artifact cannot fit the current NVMe. Its premise was a linear bytes-per-token projection contradicted by the measured 2K and 8K artifacts.

Do not replace that guard with a two-point affine fit. Two points exactly determine a line and do not validate the model. The production predictor should be componentized from the pinned writer and exact source geometry:

- attention K/V bytes derived from cell count, per-layer row geometry, and cache dtypes;
- recurrent R/S bytes derived separately from layer map and recurrent widths;
- checkpoint appendix bytes derived from checkpoint count and canonical/native recurrent payload geometry;
- container metadata and atomic-admission peak factors added explicitly;
- the safety margin applied after those components.

Test the estimator against the retained 2K and 8K sizes before allowing it to admit a larger generation. Keep the existing conservative refusal until that replacement is proven.

Do not run a 32K opaque benchmark now. The 8K result already proves production-shaped economics, while canonical portability is the product-critical path. A future 32K run is deferred by priority and by an unvalidated size guard, not by a claim that available storage is known to be insufficient.

## P0 — red-test and close canonical trust-boundary defects

Make the smallest causal fixes before expanding the decoder.

### Bounded reader

Add independent tests that fail on current head for all of these cases:

1. The section's declared end is unchanged while bytes are removed from the final K or V payload. The reader must refuse before returning spans. Validate physical extent at reader construction or before a skip can authorize an unread region; do not force payload materialization.
2. A huge `cell_count` in a tiny declared section must refuse before looping or allocating. Bound it both by the bytes remaining under a conservative minimum cell record and by an explicit expected maximum from the resolved source geometry/runtime context.
3. `has_cell_ext=True, cell_ext_size=0` and all unsupported cell-extension sizes must refuse. The exact size must come from the pinned architecture descriptor; do not infer it from payload alignment.
4. A transposed q8_0 header using the writer's 34-byte `ggml_type_size` contract must be interpreted correctly. A header using 1 must refuse. If quantized transposed-V semantics cannot be independently confirmed from the pinned writer and a real or source-authored fixture, explicitly refuse that combination rather than guess.
5. The top-level parser, once added, must account for every byte as envelope, attention, recurrent state, or SCKP appendix. The attention helper should report its consumed end and must not pretend recurrent trailing bytes are an error.

Keep reads bounded and measure actual metadata bytes read. A sparse multi-GiB test must not become a multi-GiB read.

### Schema decoding

Make malformed JSON values red, then:

- require booleans to be JSON booleans; never use truthiness for `has_cell_ext`, draft, or speculative flags;
- require integer fields to be integers without string/float coercion and reject booleans as integers;
- require RoPE theta to be finite and positive;
- require digests to be exactly 64 lowercase hexadecimal characters;
- require recurrent dimensions to be nonnegative, and require the qwen35 descriptor's expected recurrent fields to be positive and internally consistent;
- validate named segment references against the actual container segment table at integration time, including role and uniqueness;
- reject unknown fields or version them deliberately; do not silently preserve an identity the decoder did not understand.

Do not weaken the opaque path or connect this schema to production import in P0.

Acceptance: every new guard has a mutation-sensitive red test, the offline suite passes, and no fixture obtains its expected offsets or values from the parser under test.

## P1 — break-first structural parse of the retained 8K artifact

After P0, run the largest currently justified falsifiable experiment: parse the exact admitted 8K GGSQ/3+SCKP/1 object identified by digest `2af6ca68737a1888bd65c67cf4d36746123cf18e51824e50f94c888f6be80c72`, using the bound source GGUF digest and pinned patched-writer source.

This gate locates structure only; it does not yet allocate or convert tensor payloads.

Require a retained machine-readable record containing:

- artifact digest and before/after file facts;
- exact source GGUF/model digests and resolved architecture geometry;
- physical file length, declared sequence body, attention end, recurrent section bounds, and SCKP appendix bounds;
- cell count, layer count, K/V types, strides, transposition status, and every payload span;
- proof that spans are in-bounds, non-overlapping, in writer order, and collectively leave no unexplained bytes;
- agreement between parsed checkpoint count/coverage/positions and the admitted manifest;
- total bytes actually read and peak memory, both bounded independently of payload size;
- refusal controls for wrong source geometry, wrong tuple/version, a physically truncated copy, and a one-byte boundary shift.

Never mutate the admitted store object. Use private copies or sparse fixtures for corruption controls.

Decision rule:

- Any mismatch: retain the record, stop, identify the earliest divergent field against the pinned writer, and fix only that cause.
- Exact structural agreement: proceed to recurrent/SCKP decoding.
- A missing field that cannot be uniquely recovered from the exact source GGUF plus pinned architecture descriptor fires the upstream-seam falsifier. Record the exact field and request the smallest versioned export metadata addition; do not infer from model name or byte patterns.

## P2 — decode recurrent state and SCKP, then prove numbers

Extend the parser in writer order:

1. Parse the recurrent R/S section with the explicit source-layer map.
2. Parse each 16-byte SCKP record and recursively decode its PARTIAL_ONLY recurrent target state.
3. Materialize only a deterministic small fixture into canonical little-endian `layer,kv,token,head,dim` attention tensors and canonical recurrent tensors.
4. Normalize transposed V during decode; backend-native layout is not canonical identity.
5. Compare exact offsets, shapes, and decoded numeric values with an independent oracle derived from the pinned writer or a separate reference implementation. A decoder/encoder self-round-trip is not evidence.

Require truncation, overlap, impossible count/shape/stride, unsupported dtype, bad layer map, wrong source identity, nonempty draft/speculative state, and trailing-byte mutation tests.

Only after the small numeric oracle passes may a bounded sample of the retained 8K artifact be decoded and compared. Do not materialize the full 8K cache merely to prove the parser.

## P3 — same-model cross-dtype conversion

After canonical decode is trustworthy, add one narrow target encoder/import seam for f16 source to q8_0 target of the exact same model.

The opaque route must continue refusing the dtype mismatch. Only the canonical route may transform it.

Fix quality thresholds before the live run. Gate against target-native q8_0 reuse on:

- exact prompt and target model identity;
- successful target prefix reuse;
- output and nonempty probability/logit divergence versus target-native reuse;
- clean fallback to native target prefill on every unsupported or failed case;
- explicit source/target runtime, dtype, backend, and device provenance outside canonical cache identity.

A pass unlocks one same-model CUDA/ROCm/Vulkan transfer matrix. It does not unlock cross-model projection.

## P4 — minimal sidecar

After one canonical dtype conversion passes, expose only:

- canonical prefix resolution;
- transfer/import into a named target runtime;
- reused-token count, transfer mode, gate verdict, and fallback reason;
- automatic native target prefill on failure.

Do not build broad daemon, authentication, multi-tenant, or distributed scheduling surfaces before the proven transfer seam is reachable.

## Deferred

- 32K opaque benchmarking until canonical parsing reaches its real-artifact gate and the componentized estimator is validated;
- host-restart/cold-boot persistence claims;
- 131K;
- learned cross-model projection and token alignment;
- production vLLM/HF adapters;
- MTP/draft/speculative state;
- upstream comments or submissions.

HF may be used earlier only as an independent numeric oracle.

## Required execution order

1. Red-test and close the bounded-reader and strict-schema defects in P0.
2. Structurally parse the retained real 8K artifact and retain the gate record.
3. Decode recurrent/SCKP structure and prove numeric decoding on an independent small oracle.
4. Add one narrow target encoder and prove same-model f16-to-q8_0 conversion.
5. Exercise the proven seam through the minimal sidecar.
