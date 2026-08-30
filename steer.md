# KV Rosetta research steer: close pinned checkpoint sizing and enforce the production transfer gate

Status basis: default-branch head 2a326bbb76190cf979cf3d6c07f3f55da3278af9.

This steer supersedes da0e3ea. Two materially different changes landed:

1. The checkpoint appendix was structurally accounted well enough to close byte sizing for the exact pinned qwen35 llama.cpp/patch tuple.
2. A cross-tokenizer byte-overlap mapper was added before the ordered production qwen35 HIP↔Vulkan gate ran.

Retain the checkpoint evidence. Quarantine the alignment prototype. The exact production qwen35 8K HIP↔Vulkan patched/unpatched matrix has now been skipped twice and remains the single next break-first experiment. No new sizing, tokenizer, cross-model, sidecar, vLLM, CUDA, or canonical-format feature work should start until that matrix produces a retained pass or first-causal-failure record.

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

## P0 — the only next experiment: exact qwen35 8K HIP ↔ Vulkan

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

1. Stop new work outside P0.
2. Harden only the cross-backend runner provenance and predeclared verdict.
3. Run the exact qwen35 8K HIP↔Vulkan patched/unpatched matrix.
4. Close parser/schema trust boundaries and parse the real patched 8K artifact completely.
5. Prove bounded hybrid numeric decoding and choose canonical storage.
6. Prove same-model f16→q8_0 behavioral conversion.
7. Extend to CUDA and a live second runtime.
8. Revisit sparse cross-tokenizer/cross-model mapping only after same-model gates pass.
9. Activate vLLM and the demand-driven sidecar only after their live gates.

Deferred: qwen35 32K execution, host-restart/cold-boot claims, 131K, learned cross-model mapping, MTP/draft/speculative support, broad service/authentication/distributed scheduling work, and upstream submissions/comments.
