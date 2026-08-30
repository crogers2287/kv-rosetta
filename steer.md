# KV Rosetta research steer: contain failed-restore state before expanding native portability

Status basis: default-branch head f3fa574ab3d9dac2ab46f87498d3ebd684090968.

This steer supersedes 619d496. The newest work proves that one native llama.cpp state file can cross NVIDIA/CUDA and AMD/ROCm in both directions at 8K and 32K for one exact non-hybrid qwen2 tuple. At 32K it reused 31,999/32,000 tokens with matching generated text and token IDs. Retain that as strong same-model native-portability evidence.

The same batch exposed a more urgent fail-closed boundary. In a three-model raw-restore matrix, all six wrong-model restores returned HTTP 400, but llama.cpp logged that it restored the foreign checkpoint appendix before rejecting the main state body. The harness did not issue a completion afterward, so it did not prove that the target slot remained pristine. A rejected restore is not safe merely because the request returned an error.

Before the production qwen35 backend matrix, run the smallest retained poisoning test and close any state-mutation path. Also make gate-protocol provenance and legacy requirements fail closed. Then resume the exact production qwen35 8K HIP↔Vulkan matrix. Do not substitute the non-hybrid CUDA↔ROCm result for that hybrid gate.

## Mission and non-negotiable boundaries

Deliver portable reuse of large agentic prefixes:

1. use an exact native llama.cpp fast path wherever a serialization tuple is behaviorally proven portable;
2. use canonical KVX conversion across cache dtypes, inference runtimes, and backend pairs the native format cannot cross;
3. consider cross-model transformation only after same-model native and canonical paths pass quality gates.

For qwen35, GGSQ/3+SCKP/1 remains a strict opaque foundation. Preserve refusal on unpatched, incomplete, mismatched, unsupported, active-draft/speculative, or incompletely attested runtimes and artifacts. Do not weaken model/GGUF identity, CacheABIIdentity, checkpoint metadata, admitted-store guarantees, or exact runtime/protocol identity.

Primary hybrid evidence remains:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep patches 0001 and 0002 pinned and identified. Do not post upstream.

## P-1 — prove that failed restoration cannot poison a live slot

Treat this as the immediate safety gate because it can invalidate fallback behavior.

Use one process-owned target runtime and a known foreign artifact whose raw restore reproduces the logged checkpoint-before-rejection sequence:

1. Establish a pristine target baseline: cold/native completion, token IDs, nonempty probability vectors, cache reuse, and save/checkpoint metadata.
2. Attempt the foreign raw restore and retain the exact server log, response, endpoint-call count, checkpoint metadata, and process identity.
3. Without clearing or restarting the slot, immediately run the identical target completion and compare it with the pristine baseline.
4. Repeat through the public sidecar with both a requirements-bearing wrong-model artifact and a legacy artifact missing requirements. Both must refuse before any state endpoint call.
5. If the raw failure changes any slot state, completion, probability, reuse, or checkpoint metadata, retain that failure and add the smallest recovery boundary: never expose the raw mismatch path, clear/restart or restore pristine state before fallback, and rerun the identical test.

Pass requires zero public-path state endpoint calls for every mismatched or incomplete artifact and a bit-for-bit/threshold-equivalent pristine target after refusal. HTTP 400 alone is not acceptance evidence.

Close the adjacent provenance gap in the same red-test-first patch set:

- GateBinding currently defaults a missing scoring protocol to teacher_forced. Missing is not the same as explicitly recorded. Deserialization or construction without a protocol must refuse.
- Teacher-forced scoring is a diagnostic quality measure, not a substitute for the separately frozen free-generation/behavioral admission gate.
- Legacy manifests missing runtime requirements must refuse active restoration or be explicitly re-admitted through full validation.

## RA-005 answer — split the shippable product from cross-model research

RA-005 is answered as follows:

1. **Yes, ship the proven half when its production gates pass.** The honest product claim is durable exact-prefix reuse for the same model, with explicitly allowlisted native backend/runtime tuples and strict fallback. It is not “one cache for any model.”
2. **Cross-backend and cross-dtype portability remain product work.** Native HIP↔Vulkan can be allowlisted only where behaviorally proven. CUDA, different runtimes, and different cache dtypes require canonical conversion or an independently proven native tuple.
3. **Do not fund a larger nonlinear cross-model mapper yet.** The qwen38→Tiel failure changed model weights, capacity, dense/MoE structure, attention-layer/head counts, weight quantization, and possibly KV-cache quantization at once. Matching tokenizer and several dimensions made it favorable but did not isolate the cause.
4. **Do not attempt recurrent cross-model mapping yet.** Supplying Tiel's own recurrent state made the attention experiment a useful upper-bound diagnostic, not a complete transferable hybrid cache. Recurrent transformation remains research-only until attention crosses controlled isolation gates.

The project headline and capability reporting must distinguish:

- **proven now:** same-model persistence and selected same-runtime/backend transfer;
- **next product work:** same-model conversion across cache dtypes, CUDA/ROCm/Vulkan, and live runtime connectors;
- **research only:** different-model transformation, learned projectors, tokenizer remapping, and recurrent-state mapping.

## What CUDA↔ROCm now proves—and does not

Retain the new native same-model evidence:

- one exact non-hybrid qwen2 GGUF moved CUDA→ROCm and ROCm→CUDA;
- 8K restored 8,191/8,192 tokens in both directions;
- 32K restored 31,999/32,000 tokens in both directions;
- generated text and token IDs matched, and top-1 agreement was 1.0 for the retained positions;
- writer and reader used the same llama.cpp revision.

This is strong evidence that backend vendor alone need not force canonical conversion when model, runtime, state format, cache dtype, and context tuple are otherwise identical. It is not yet an admission-quality shipping allowlist:

1. The retained records do not bind the model-weights digest, explicit K/V cache dtype value, prompt digest, complete launch commands, process identities, and device identity strongly enough.
2. Only eight generated positions were compared.
3. The 32K reverse leg evicted production models through llama-swap and left them unloaded until their next request. Keep that operational effect in the record; benchmark authorization is not proof of the no-wake sidecar contract.
4. The model is non-hybrid qwen2. It contains neither qwen35 recurrent state nor SCKP checkpoint persistence.

Do not repeat 32K merely for scale. After P-1 and the production hybrid P0, retain one admission-quality 8K CUDA↔ROCm repetition with complete identity and no hidden fleet side effects before allowlisting this tuple.

## What the latest Tiel live work proves—and does not

Retain the live Tiel-Coder evidence:

- patched runtime: 508/512 reuse through the public pipeline;
- stock runtime: refused before restore because state version/checkpoint persistence were absent;
- requirements and model identity are now bound from the admitted bytes and caller-derived weights identity;
- 8K RAM-backed f16/f16 run: 8,188/8,192 reused, reported 410.9 ms restored versus 6,029.9 ms cold.

Treat the 1.8×/4.3×/14.7× economics as directional, not admission-quality:

1. bench/tiel-coder-restore-latency.json retains only three medians, not the three raw repetitions.
2. It omits exact model, prompt, artifact, binary, patch, library, launch, process, and device digests/identity.
3. It records reuse but not output/token/probability parity or same-backend native-reuse parity.
4. The phase totals do not reconcile: 2K leaves about 8.7 ms unattributed and 8K about 20.7 ms.
5. Storage was /dev/shm; NVMe cost is arithmetic, not measured.
6. KV was f16/f16 on an idle W6800; actual kvwarm uses quantized KV, longer prompts, two slots, and live contention.
7. The producing runner was not committed with the record.

Also close these fail-closed gaps before shipping:

- artifacts missing the new requirements field currently skip the capability check; active restore must refuse incomplete/legacy manifests or explicitly re-admit them through full validation;
- the live run found that the admitted store must equal the server-visible slot-save path, but this is not yet an enforced invariant;
- the sidecar still needs atomic binding to one already-ready runtime generation; a separate capability/readiness check followed by restore can race an unload/replacement and must not wake a model;
- test commands must preserve the real exit status (pipefail or no status-hiding pipe); a green-looking piped failure already landed once.

## What the checkpoint work now proves

For the pinned Qwen3.8-27B/qwen35 geometry and patched llama.cpp lineage:

- the base 256-token state is exactly 173,679,168 bytes;
- the recurrent section is exactly 156,894,356 bytes and is fixed across 256→257 tokens;
- the attention/header marginal cost is exactly 65,564 bytes/token;
- the SCKP appendix has a 12-byte header;
- each checkpoint has a 16-byte record, three interleaved uint64 buffer lengths, an eight-byte unidentified target preamble, and one recurrent-state payload when draft/speculative buffers are empty;
- the derived total equals the independently written 256-token/one-checkpoint artifact: 330,573,584 bytes;
- it also equals the earlier 2,048-token/two-checkpoint artifact: 604,958,676 bytes.

The same component arithmetic reconstructs the retained 8,192-token/two-checkpoint size exactly:

- base: 693,995,072 bytes;
- two-checkpoint appendix: 313,788,820 bytes;
- total: 1,007,783,892 bytes.

This is sufficient to use the derived size formula for admission only when the exact writer revision, patch/protocol tuple, model geometry, cache dtype, checkpoint count, and empty draft/speculative state are attested. Any other revision or state class falls back conservatively or refuses. The eight-byte target preamble is still semantically unidentified, so do not describe the appendix as canonically decoded or portable merely because its byte count closes.

RA-003 is answered: source-derived component sizing replaces the flat per-token estimate for the proven tuple. A qwen35 32K object is no longer blocked by the old 9.7 GiB estimate, but the 32K live run remains deferred behind P0 and the structural trust boundary. Tmpfs still proves mechanism, not persistence.

## Checkpoint/parser trust boundaries still open

Mutation coverage is useful, but 42/42 exercised guards is not proof that all necessary guards exist. The current parser still has known fail-open or unproven boundaries:

1. `Reader.skip()` checks only declared logical bounds; it does not prove physical EOF for skipped final payloads.
2. Attention and recurrent cell counts are not bounded before loops/allocation.
3. A declared cell extension can still resolve ambiguously to zero instead of an exact positive, revision-bound descriptor.
4. Transposed q8_0 V still uses an element-size rule inconsistent with the writer's 34-byte block contract; refuse until proven from the pinned writer.
5. Layer counts and recurrent maps are not bound exactly to source GGUF geometry and ordering.
6. `SCKP count == 0` is accepted by the low-level parser, and checkpoint extents are not yet bound there to live save metadata.
7. The eight-byte checkpoint target preamble is size-known but meaning-unknown; exact end closure reduces risk but does not establish semantics.
8. Schema JSON must remain strict on types, finite positive geometry, exact digest forms, unique roles, and unknown-field version policy.
9. Architecture-based sizing switches must be bound to the pinned llama.cpp revision/protocol, not treated as timeless architecture facts.
10. A digest-identified real patched 8K artifact still lacks a retained complete live parse proving physical EOF, exhaustive spans, metadata equality, and bounded peak memory.

These block canonical admission. They do not block the raw behavioral transfer experiment below.

## Cross-tokenizer alignment is a quarantined hypothesis

Keep `mappers/align.py` inert and out of capability reporting, transfer planning, and sidecar/runtime paths.

What is proven:

- two supplied token-piece sequences can be represented as byte intervals;
- identical reconstructed bytes are required;
- overlap weights sum to one;
- a short qwen2.5/ornith-a1 example produces the expected 0.6/0.4 boundary weights when both local tokenizers are present.

What is not proven:

- that byte-overlap averaging produces valid K or V states across models;
- that removing RoPE makes hidden states from different causal tokenizations linearly interchangeable;
- any behavioral cache restoration, reuse, logits, or output quality;
- recurrent-state, layer/head/dimension, tokenizer-special-token, or model transformation semantics;
- scalable operation on agentic prefixes.

The implementation is currently quadratic and dense: `align()` loops over every target/source token pair and stores float64 weights. At equal token counts, that is about 512 MiB for 8K and 7.6 GiB for 32K before cache tensors or `einsum` temporaries. This contradicts the large-prefix goal. If this work is reopened after same-model portability succeeds, first replace it with a sparse two-pointer interval representation with O(source+target) construction and O(overlaps) storage, and label byte pooling as a candidate feature—not an admitted cache transformation. Only held-out behavioral divergence can validate it.

Do not call this “the last unbuilt piece of T3.” Cross-model transformation still lacks a fitted/derived state mapper, architecture correspondence, recurrent-state treatment, live target encoding, and behavioral admission evidence.

## Cross-model evidence is real, failed, confounded, and quarantined

The new raw three-model matrix adds a narrower fact: same-model controls restored 508/512, and all six direct wrong-model restores returned HTTP 400. It does not prove safe refusal because the runtime restored a foreign checkpoint before rejecting the remaining file and post-failure slot state was never tested. Do not expose direct cross-model raw restore through a product path.

The qwen38-27b → Tiel-Coder experiment is useful negative evidence:

- exact shared tokenizer, matching head width and recurrent dimensions;
- whole-prompt holdout instead of leaked adjacent-token splits;
- Tiel-native recurrent state and an exact identity control;
- translated attention restored 764/768 tokens but failed target-native behavior;
- teacher forcing later measured 0.733 and 0.903 agreement on two prompts, separating the translation from accepted blends;
- margin-threshold sweeping corrected the claim that skipping near-ties materially improves the gate. Keep teacher forcing; do not base policy on the chosen margin.

It is not a causal verdict on cross-model translation. The run changed model weights, parameter capacity, qwen35 versus qwen35moe structure, 16/4 versus 10/2 attention layers/heads, and apparently Q4_K_XL versus Q5 weight quantization. The retained records do not prove matched KV-cache dtypes. The result says only that this simple linear mapper did not overcome the complete bundle of differences.

The implementation also remains incomplete or fail open:

1. LinearMapper.require_applicable() is not called with exact source/target model digests.
2. Architecture, layer counts, full target-layer coverage, head geometry, and complete RoPE semantics are not identity-bound.
3. Missing target layers can be omitted or zero-filled.
4. Only attention is mapped; recurrent state is target-native.
5. Cross-tokenizer alignment is dense/quadratic and was bypassed.
6. Benchmark files do not retain the executable runner and full provenance.

Keep compose, translate, alignment, learned mapping, vLLM activation, and all cross-model capability reporting inert.

When T3 reopens after the same-model gates, use this isolation ladder and change one variable per rung:

1. exact same model/GGUF/cache dtype across backend only—complete the hybrid P0;
2. exact same model and weights across KV-cache dtype via canonical conversion;
3. same base model, size, and architecture across weight quantizations with KV dtype held fixed;
4. same base architecture and size across finetunes;
5. only then change parameter size, dense/MoE structure, heads/layers, tokenizer, or recurrent geometry.

Every rung needs exact model/weight/runtime/artifact/prompt identities, teacher-forced per-position logits, per-layer/per-kind cache error, identity/noise/native controls, and a separately frozen free-generation admission gate. A failure identifies the earliest causal boundary; it does not authorize skipping to a richer mapper.

## P0 — resume after P-1: exact qwen35 8K HIP ↔ Vulkan

First apply only the smallest red-test-first runner corrections needed for an admission-quality record. Freeze the verdict before measurement:

- exact model/GGUF and prompt-token digests;
- full llama.cpp source identity, patch set, binary and loaded-library digests;
- complete CacheABIIdentity, including actual K/V dtypes and context parameters;
- retained launch commands, environment, device identity, backend startup evidence, and offloaded-layer count;
- proof of writer stop before reader start, with distinct process identities;
- save/restore metadata equality: state version, SCKP version, checkpoint count, positions, coverage, and active-state classes;
- exact expected uncovered tail, not merely positive reuse;
- same-backend self-restore and cold-prefill controls for each reader;
- same-source unpatched hybrid refusal before any state endpoint call;
- identical generated token ids/text against the reader-native reference;
- nonempty probability vectors at every declared generated position under thresholds fixed before the run;
- unchanged artifact digest and file facts before and after transfer;
- missing provenance, partial vectors, silent backend fallback, or post-hoc thresholds cause refusal.

Then run the exact production-tested Qwen3.8-27B/qwen35 8,192-token prefix:

1. HIP save → full process stop → Vulkan restore → completion.
2. Vulkan save → full process stop → HIP restore → completion.
3. HIP and Vulkan same-backend self-restore and cold controls.
4. Same-source unpatched controls on both reader paths.

Hold model, prompt, cache dtype, context, generation settings, runtime revision, patch set, and checkpoint policy fixed. Change only backend.

Decision:

- **Pass:** allowlist this exact llama.cpp HIP/Vulkan serialization tuple as a native fast path. Backend may be omitted from refusal identity only for this explicitly proven compatible pair; model, dtype, runtime, protocol, patch, checkpoint, and active-state identity remain strict.
- **Fail:** retain the earliest divergent field or behavioral position, make the smallest causal fix, and rerun the identical matrix.
- **Missing attestation:** refuse the run. Backend names supplied by the caller or inferred from binary paths are not evidence.

Do not perform more unrelated work while the live matrix is pending. If hardware availability blocks it, retain that concrete blocker and work only on the runner prerequisites—not T3 or service features.

## P0.5 — smallest shippable same-model sidecar gate

Only after P0 passes or retains its first causal failure, run one product-shaped same-model gate on the models the owner actually serves:

- demand-driven request for an already-ready Tiel or qwen38 instance; no scheduled/proactive warming;
- atomic identity and readiness for the exact runtime generation used for restore;
- sacrificial unloaded model remains unloaded across every sidecar route;
- current requirements-bearing artifact only; missing/incomplete requirements refuse;
- server-visible persistent store with enforced path/device/ownership/digest invariants;
- actual production KV dtype and context settings;
- 8K minimum, three raw retained repetitions under normal slot/concurrent load;
- cold, native-reuse, patched restore, stock/unpatched refusal, wrong-model/dtype/runtime, legacy-manifest, post-refusal slot-pristine, and unload-race controls;
- output/token/probability parity and completely reconciled phase timings.

Pass allows the opaque same-model route to replace kvwarm for that exact tuple. Failure falls back to native prefill and retains the earliest causal defect. This is a narrow shippable feature, not canonical or cross-model portability.

## P1 — real patched 8K structural account

After P0, close the ten parser/schema boundaries above with red tests. Then parse the exact digest-identified 8K GGSQ/3+SCKP/1 artifact written by the live patched runtime:

- prove physical EOF and exact total consumption;
- prove spans ordered, non-overlapping, exhaustive, and bounded-memory;
- bind attention/recurrent layer maps, row geometry, dtypes, and cell extensions to source GGUF plus pinned writer revision;
- identify or explicitly preserve/refuse the checkpoint preamble rather than assigning invented semantics;
- require positive checkpoint coverage equal to live save metadata;
- retain truncation, oversized-count, wrong-map/geometry/tuple, zero-checkpoint, unknown-preamble, and one-byte-boundary refusal controls.

Behavioral P0 and structural P1 are separate claims. Neither substitutes for the other.

## P2 — same-model canonical dtype conversion

Only after P1:

1. validate bounded attention, recurrent, and checkpoint numeric samples against pinned ggml reference code;
2. choose and version a canonical physical dtype/chunking from measured fidelity, storage, and conversion costs;
3. build the narrowest same-model f16→q8_0 target encoder/import seam;
4. freeze quality thresholds before the live run;
5. compare converted reuse against target-native q8_0 reuse with exact identity, output, probability, bounded-memory, and native-prefill fallback gates.

The q8_0/q4_0 codecs are unit evidence only until a converted cache passes behavioral restoration.

## P3 — broader portability and service

Only after P2:

1. extend the native/canonical distinction to CUDA and a second device;
2. prove a live second inference-runtime connector;
3. make tokenizer alignment sparse and test it only as one input to a full cross-model behavioral experiment;
4. activate vLLM only after its real cache layout, MLA/non-MLA indexing, scheduler metadata, and TP/PP shard identity pass end to end;
5. activate the demand-driven sidecar only after readiness and restoration are atomic to one exact runtime generation and all routes pass a sacrificial no-wake matrix.

Keep vLLM inert and sidecar active restoration quarantined. No scheduled or proactive warming. RA-004 remains answered on that basis.

## Required execution order

1. Freeze compose/translate/alignment, nonlinear mapping, and unrelated service expansion.
2. Run P-1: reproduce the checkpoint-before-rejection path and prove the target slot remains pristine after failure.
3. Red-test and close missing GateBinding protocol, legacy requirements, and public-path zero-state-endpoint refusal.
4. Harden only the P0 runner provenance, raw records, exit-status handling, and predeclared verdict.
5. Run the exact production qwen35 8K HIP↔Vulkan patched/unpatched matrix.
6. Run P0.5: one demand-driven, persistent, production-KV, no-wake same-model sidecar gate.
7. Close parser/schema trust boundaries and parse the real patched 8K artifact completely.
8. Prove bounded hybrid numeric decoding and choose canonical storage.
9. Prove same-model f16→q8_0 behavioral conversion.
10. Repeat one admission-quality 8K CUDA↔ROCm tuple, then extend to a live second runtime.
11. Revisit cross-model work only through the isolation ladder above.
12. Attempt nonlinear/tokenizer/recurrent mapping only after a controlled rung identifies that need.

Deferred: qwen35 32K execution, host-restart/cold-boot claims, 131K, learned cross-model mapping, MTP/draft/speculative support, broad service/authentication/distributed scheduling work, and upstream submissions/comments.
