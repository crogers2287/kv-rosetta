# KV Rosetta steering directive

## Mission

Build a reliable, model-aware system for reusing and translating KV state across inference runtimes and hardware backends.

The non-negotiable rule is fail closed: an artifact that is corrupt, ambiguous, incompatible, insufficiently tested, or below its quality gate must never reach a runtime import call. The final fallback is always normal target-model prefill.

KV Rosetta must distinguish four separate identities:

1. Prompt identity: canonical messages, tool schemas, and exact token IDs.
2. Model semantic identity: weights, tokenizer, adapters, and architecture.
3. Runtime cache ABI identity: runtime revision, state format, KV configuration, position configuration, and context flags.
4. Artifact identity: representation, segment table, mapper, payload, and integrity metadata.

Hardware is provenance unless a runtime-native opaque format makes it part of the cache ABI.

## Current truth

Commit fb3ecf7 establishes useful scaffolding:

- KVX container and manifest
- artifact store
- admission metrics
- capability tiers
- GGSQ body parser
- cfrproxy prefix-manifest reader
- standard RoPE helper

It does not yet prove runtime extraction, runtime injection, same-model parity, CUDA-to-HIP transfer, or cross-model transfer. Do not describe those paths as implemented until retained integration tests prove them.

## Working method

Use the break-first method:

1. Test the smallest real path until it fails.
2. Capture the exact artifact, configuration, logs, timings, and first divergent token.
3. Identify the violated contract.
4. Fix the narrowest responsible layer.
5. Add a permanent regression test.
6. Scale context only after the smaller case is clean.

Use a context ladder rather than beginning at 131K:

- 256 tokens for parser and identity failures
- 2K for functional round trips
- 8K for conversion and storage behavior
- 32K for memory and throughput behavior
- 131K for the actual target

Never spend a long benchmark run proving a bug that a 256-token fixture could expose.

## P0: revise the contract before building adapters

### 1. Artifact keys and coexistence

The cfrproxy fingerprint identifies a reusable prompt prefix. It is not a unique KV artifact key.

The store must allow all of these to coexist for the same prompt:

- CUDA opaque state
- HIP opaque state
- canonical raw state
- target-native translated state
- multiple KV dtypes
- multiple runtime revisions
- multiple mapper versions

Define an artifact key from a canonical structure containing at least:

- prompt fingerprint
- exact token-ID digest
- model semantic ID
- cache ABI ID
- encoding and format version
- representation or segment-table digest
- mapper ID when applicable

Do not truncate the identity for database uniqueness. A shortened digest may be used only as a display label after collision handling.

Validate all externally supplied hashes as lowercase fixed-length hexadecimal before using them in paths. A prefix manifest must never be able to escape the store root.

Acceptance criteria:

- four representations for one prompt can be inserted and retrieved independently
- replacing a model behind the same path or alias produces a different semantic identity
- moving identical weights to another path does not change semantic identity
- a malformed fingerprint cannot create a path outside the store
- failed file deletion does not silently delete the only database record

### 2. Replace the single homogeneous tensor with a segment table

The fixed layer,kv,token,head,dim tensor is useful for standard MHA/GQA, but it is not a sufficient model-agnostic boundary.

KVX must be able to describe named segments with fields such as:

- semantic role
- layer or layer range
- dtype
- shape
- logical layout
- byte order
- offset and byte count
- payload digest
- quantization metadata
- position semantics

One file may still contain a contiguous payload. The header must describe multiple non-overlapping segments within it.

This is required for:

- different K and V dtypes
- quantized values and scales
- DeepSeek MLA compressed state
- hybrid attention/recurrent state
- cross-attention state
- per-layer shape or cache-kind differences
- future runtime-specific auxiliary state

A runtime may export an expanded standard K/V representation for an unusual model, but it must report that expansion and its cost. Do not call the format model-agnostic when the current adapter only supports standard decoder-only attention.

Preserve read compatibility with KVX 0.1 and 0.2. Write the revised format under a new schema/container version.

### 3. Integrity and semantic validation

Integrity must cover the canonical header and every payload segment, not only the payload bytes. Model identity, token identity, offsets, shapes, and dtypes must not be mutable without detection.

Before exposing an artifact to an adapter, enforce:

- supported container and schema versions
- bounded header length
- canonical JSON rules
- valid encoding
- valid and nonnegative offsets and lengths
- alignment requirements
- segments begin after the header
- segments do not overlap
- declared sizes equal dtype times shape when applicable
- raw tensor dtype matches the manifest
- little-endian serialization
- complete file length and trailing-byte policy
- header and payload integrity
- manifest semantic validation

The public verifier must never raise on arbitrary bytes. It returns a structured failure reason for every malformed input. Add fuzz and property tests.

Do not import first and verify afterward.

### 4. Exact model and cache identity

Do not use model path as weight identity.

Model semantic identity must include stable content-derived identities for:

- weights or GGUF tensor content
- tokenizer vocabulary, merges, normalization, and special tokens
- chat template where prompt tokenization depends on it
- active LoRA or other adapters
- architecture and cache semantics

Cache ABI identity must include all state-affecting runtime values, including:

- runtime name, revision, and state-format version
- K and V dtype
- context type, including MTP
- RoPE type, base, scaling, YaRN parameters, and partial rotary dimensions
- SWA and hybrid-cache settings
- unified or per-sequence KV behavior
- opaque format and endianness
- any model-specific cache layout flags

Opaque import requires an exact cache ABI match. An override flag must not permit unsafe opaque import. Mapped import may cross semantic identities only through an explicitly compatible mapper and quality gate.

### 5. Make the admission gate fail closed

The configured holdout count is a minimum requirement, not merely a maximum slice. A one-token comparison must not satisfy a 64-token gate.

Validate all thresholds:

- holdout count is positive
- agreement is finite and between zero and one
- KL limits are finite and nonnegative
- optional logit limits are finite and nonnegative
- no NaN or infinity can disable a comparison

A gate result must be bound to:

- source model semantic ID
- target model semantic ID
- source artifact digest
- target representation digest
- mapper implementation and weights digest
- calibration corpus and holdout token digest
- threshold policy version
- measured metrics

Treat gate evaluation as compilation/certification work whenever possible. Do not put a full native target prefill in the hot restore path and then claim the prefill was avoided.

### 6. Parse llama.cpp envelopes explicitly

Do not pass an unknown byte buffer directly to the current GGSQ body parser.

Support distinct, explicit sources:

1. In-process sequence-state buffer:
   - internal IO magic
   - source sequence ID
   - state body

2. Sequence-state file:
   - GGSQ magic
   - sequence-state version
   - token count and token IDs
   - state body

Verify the envelope and version before parsing the body. Do not guess the starting offset.

Commit small binary fixtures captured from the exact llama.cpp revision used for CUDA and HIP. Fixtures must cover:

- ordinary values
- transposed values
- empty streams
- multiple sequences
- truncated data
- unsupported cell extensions
- unknown GGML types
- trailing bytes

BF16 is a canonical representation and should be handled explicitly. Quantized K/V should remain opaque only until a tested dequantization path and gate exist.

### 7. Make adapter capabilities directional and specific

A single maximum tier is too coarse. An adapter may export canonical state while importing only opaque state, or support one dtype and not another.

Capabilities must report independently:

- export representations
- import representations
- supported dtypes
- supported opaque formats
- model/cache semantic families
- device and staging modes
- zero-copy constraints
- runtime revision

A capability probe is evidence from the live runtime. It is not a hard-coded promise.

## P1: implementation hardening

### Streaming and bounded memory

Do not materialize an entire large cache through both a contiguous NumPy copy and a bytes copy.

Implement:

- chunked or memoryview-based writes
- incremental hashing
- preallocated or memory-mapped output
- per-segment streaming
- bounded host staging
- peak-RSS and staging-byte telemetry

At every benchmark size record:

- native prefill time
- export time
- restore time
- conversion time
- verification time
- bytes read and written
- host peak RSS
- source and target VRAM
- effective transfer bandwidth
- first-token and next-token parity

A cache path is beneficial only when total restore, verification, and conversion cost is lower than normal prefill for the intended reuse pattern.

### RoPE dispatch

The current helper implements one standard half-split rotation. It must not silently process every model.

Dispatch by explicit position semantics and fail closed for unsupported variants, including:

- normal versus NeoX pairing
- partial rotary dimensions
- linear and dynamic scaling
- YaRN
- mRoPE and multimodal positions
- model-specific long-context schemes

Round-trip tensor similarity is insufficient. Every supported position adapter requires downstream target-logit tests.

### BF16 and conversion behavior

Use a defined BF16 rounding policy, preferably round-to-nearest-even, rather than silent truncation. Validate the physical NumPy representation associated with the declared source dtype.

Every lossy dtype conversion must be named in the import report and covered by the quality gate.

### Watcher and store robustness

The cfrproxy watcher must validate required fields, recompute or verify fingerprints, reject invalid types, and deduplicate repeated manifests.

Do not rely only on filesystem mtime for change detection. Track a stable snapshot containing path, size, mtime, and content identity so equal timestamps and atomic replacement cannot hide changes.

Storage budgets must account for actual artifact bytes and estimated runtime KV memory, not only prompt-token counts.

## Required retained tests

The repository needs committed tests for:

- container read, write, verification, and corruption
- segment bounds, overlap, dtype, and size validation
- arbitrary malformed input never escaping the verifier
- store coexistence, collision handling, and path containment
- concurrent store writers
- GGSQ API and file envelopes
- gate minimum-holdout and invalid-threshold rejection
- identity changes and non-changes
- adapter conformance with successful round trips
- exact token-ID matching
- same-runtime export and import
- same-model CUDA-to-HIP canonical transfer
- explicit refusal of incompatible opaque state
- cfrproxy manifest ingestion and deduplication

The current conformance suite is insufficient if an adapter that always returns a failed ImportReport can pass. A conforming import-capable adapter must prove a successful round trip for every capability it advertises.

Add CI for supported Python versions, unit tests, static checks, and malformed-input tests. Independent one-time checks belong in the permanent suite.

## Execution order

1. Write the revised identity, segment-table, and integrity specification.
2. Implement the new validator and backward-compatible reader.
3. Migrate the store to composite artifact identity.
4. Harden and bind the quality gate.
5. Implement explicit llama.cpp sequence-state envelopes.
6. Prove same model, same runtime, same backend opaque save/restore.
7. Prove same model, same runtime revision, CUDA-to-HIP canonical transfer with matching KV configuration.
8. Add Transformers as a reference/calibration adapter.
9. Add the vLLM connector after the canonical contract survives the llama.cpp tests.
10. Implement standard-family mapping.
11. Implement MLA or other heterogeneous cache semantics explicitly.
12. Attempt learned cross-model projection only after every prior fallback and gate is operational.

Do not fan out production adapters in parallel before steps 1 through 5 stabilize the shared contract.

## Immediate next action

Stop at the portable boundary and implement the revised KVX specification plus regression tests. The first runtime milestone after that is a 256-token llama.cpp same-backend round trip with exact token IDs and token-for-token logit parity.

Only after that passes should the test scale through 2K, 8K, 32K, and 131K, followed by a same-revision CUDA-to-HIP canonical transfer.
