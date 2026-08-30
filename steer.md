# KV Rosetta research steer: promote the production hybrid cross-backend gate

Status basis: default-branch head 0d205a09ff86ae84aea926d6f422b3813c952ccb.

This steer supersedes fd3aeff. The latest retained HIP/Vulkan series is material discovery evidence: an unpatched non-hybrid llama.cpp state file from one backend was accepted by the other at 128, 8,192, and 32,000 tokens in both directions. At 32K, 31,999/32,000 tokens were reused and generated text/token ids matched. This shows that native llama.cpp sequence state can be backend-independent when runtime revision, model, state format, and cache dtype are held fixed.

That narrows the canonical product problem. A canonical conversion layer remains necessary across cache dtypes, inference runtimes, and later models. It may not be necessary merely to cross HIP and Vulkan inside one compatible llama.cpp tuple. The next break-first experiment is therefore the exact production qwen35 8K compound state across HIP and Vulkan, after making the runner admission-quality.

## Mission and boundaries

Deliver portable reuse of large agentic prefixes:

1. exploit a strict native fast path wherever an exact llama.cpp tuple is genuinely portable;
2. use canonical KVX transformation at dtype/runtime/backend boundaries the native format cannot cross;
3. add quality-gated cross-model transformation only after same-model paths are proven.

The qwen35 opaque foundation remains fail closed. Preserve refusal on unpatched, incomplete, mismatched, unsupported, or active draft/speculative runtimes and artifacts. Never weaken model/GGUF identity, CacheABIIdentity, GGSQ/3+SCKP/1 identity, checkpoint metadata, or admitted-store guards.

Primary hybrid evidence remains:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep patches 0001 and 0002 pinned and identified. Do not post upstream.

## What the new records prove

For one qwen2 Q4_K_M model on one W6800, using llama.cpp revision ca3d5a3 through HIP and Vulkan builds:

- 127/128, 8,191/8,192, and 31,999/32,000 tokens were reused in both directions;
- each direction used a source-backend native state file unchanged by the reader;
- generated text and token ids matched;
- same-backend self-restore controls reused the expected prefix;
- reader cold controls reused zero tokens;
- at 8K and 32K, foreign-cache divergence was of the same order as the two backends' ordinary cold-run divergence;
- the non-hybrid size law `908 + 36,880 * tokens` predicted the held-out 32K artifact exactly.

This is strong evidence that the state serialization itself is not tied to HIP versus Vulkan for that tuple.

It does not prove:

- qwen35 recurrent state or SCKP portability;
- CUDA, a second device, or a second model;
- cache-dtype conversion;
- canonical KVX encoding;
- cross-runtime or cross-model transfer;
- persistent-storage or host-restart behavior.

Each length is one retained run, and probability comparison covers only eight generated positions.

## Audit corrections required before the next live run

The current cross-backend runner is research instrumentation, not an admission gate:

1. It records a model path and architecture but no model/GGUF digest or before/after file facts.
2. K/V cache dtypes, context parameters, launch arguments, environment, actual device, offloaded-layer count, and backend/runtime attestation are absent from the record. Backend names are caller-supplied labels.
3. It compares only abbreviated source revisions. Retain the fullest available source identity plus binary/library digests.
4. It records booleans and deltas but does not fail the run against thresholds fixed before measurement. Any positive `cache_n` makes `both_directions_reuse` true.
5. It does not require `cache_n == prompt_tokens - documented_tail`, save/restore token counts, complete probability positions, output parity against both the reader's cold run and writer reference, or artifact immutability as a final verdict.
6. It does not retain process ids/generations or explicit writer-stop-before-reader-start evidence.
7. The records do not establish that the configured backend actually executed rather than silently falling back.

Write red tests for these omissions and apply only the smallest runner fixes. Do not broaden product code.

Predeclare the hybrid gate thresholds in the runner or a checked-in gate specification before the live result exists:

- exact model/GGUF digest and exact prompt-token digest;
- same full llama.cpp source revision and patched protocol tuple on both legs;
- complete live CacheABIIdentity, including f16/f16 or the actual matching K/V dtypes;
- verified HIP and Vulkan device/backend execution with retained command lines and startup/device evidence;
- writer process fully stopped before reader start, with different process identities;
- save and restore metadata equality, including checkpoint count, positions, coverage, state version, SCKP version, and active-state classes;
- expected constant uncovered tail only; no generic `cache_n > 0` pass;
- zero reuse on cold controls and unpatched hybrid refusal with zero state-endpoint calls;
- identical generated token ids/text against the reader-native reference;
- nonempty probability vectors for every declared generated position;
- frozen divergence thresholds defined relative to the reader's same-backend self-restore and cross-backend cold floor;
- artifact digest and file facts unchanged before and after each transfer;
- any missing provenance or partial vector is a refusal, not a warning.

## P0 — largest justified break-first experiment: exact qwen35 8K HIP ↔ Vulkan

Use the exact production-tested Qwen3.8-27B/qwen35 model and the existing 8,192-token prefix. Hold model digest, prompt, K/V dtype, context parameters, llama.cpp revision, patch set, and checkpoint policy fixed. Change only HIP versus Vulkan.

Run both directions through full process replacement:

1. HIP save → stop → Vulkan restore → completion.
2. Vulkan save → stop → HIP restore → completion.
3. Same-backend native self-restore and cold-prefill controls for each reader.
4. Same-source unpatched hybrid controls that must refuse before state endpoint use.
5. Retain one failure record at the earliest violated predeclared gate; do not massage thresholds or continue to CUDA.

Why this is now first: it directly tests whether the new non-hybrid backend-portability result extends across the exact recurrent/SCKP boundary that blocked the production model. It is larger and more product-relevant than another synthetic parser test, while all prerequisites have already been demonstrated separately on this host.

Decision:

- **Pass:** classify exact-tuple native state as a proven llama.cpp HIP/Vulkan fast path for this model. Backend alone must no longer be part of the opaque refusal identity when every serialization-affecting field is identical and the compatible backend pair is explicitly allowlisted by retained evidence. This does not relax dtype, runtime, model, or patch identity.
- **Fail:** retain the first divergent save/restore/checkpoint field. Apply the smallest causal parser/runtime/identity fix and rerun the identical 8K matrix.
- **Missing attestation:** refuse the run; do not infer backend execution from a binary pathname.

Do not run 32K hybrid, CUDA, or a second model before this gate resolves.

## P1 — real 8K hybrid structural account

Whether P0 passes or fails, retain a bounded structural record of the exact admitted GGSQ/3+SCKP/1 object. P0 can establish behavioral portability; P1 establishes that the project understands and can safely transform the bytes.

First close the inherited trust boundaries:

- physical EOF independent of declared bounds;
- bounded cell counts before loops/allocations;
- descriptor-bound cell extensions and exact layer/recurrent maps;
- correct q8_0 transposed-V block semantics or explicit refusal;
- positive checkpoint coverage and continued draft/speculative refusal;
- exact byte accounting without magic scanning;
- strict schema JSON types, finite geometry, exact digests, unique segment roles, and versioned unknown-field policy.

Then retain:

- exact envelope, attention, recurrent, and SCKP bounds;
- every span in-bounds, ordered, non-overlapping, and collectively exhaustive;
- checkpoint positions/coverage equal to live save metadata;
- bounded bytes read and peak memory;
- wrong-geometry/map/tuple, truncation, and one-byte-boundary refusal controls.

## P2 — canonical dtype/runtime seam

The ggml-verified f16/q8_0/q4_0 codecs are useful unit evidence. They are not behavioral conversion proof.

After P1:

1. Decode bounded attention, recurrent, and checkpoint samples from the real 8K object against the pinned ggml oracle.
2. Choose and version the canonical physical dtype/chunking with measured storage and conversion costs; float32 is an intermediate unless explicitly justified.
3. Build the narrowest same-model f16→q8_0 target encoder/import seam.
4. Fix quality thresholds before the run and compare converted reuse against target-native q8_0 reuse.
5. Require exact prompt/model identity, reuse, output parity, nonempty probability divergence, bounded memory, and native-prefill fallback.

Only a pass unlocks cross-runtime work and the broader CUDA/ROCm/Vulkan matrix. Native same-dtype backend portability and canonical cross-dtype portability are separate capabilities and must remain separately labeled.

## P3 — runtime integrations

The sidecar and vLLM work remain quarantined:

- Sidecar active restore stays off by default until readiness and restoration are bound to one exact runtime instance/generation and a live sacrificial no-wake matrix passes.
- vLLM returns zero matched tokens and performs no loads until its actual `[B,H,N,C]` layout, MLA/non-MLA indexing, scheduler metadata, and TP/PP shard identity are captured from the pinned live runtime and proven end to end.
- No scheduled/proactive warming.

## RA-003 — answered

Use a componentized affine estimator derived from writer geometry, not a single worst-case bytes/token rate. The non-hybrid held-out 32K point validates that method for qwen2, not its fitted constants for qwen35.

For qwen35, the 2K/8K affine law remains a hypothesis until it predicts a held-out hybrid point. A tmpfs run is acceptable as a research-only size/mechanism falsifier after P0, but it is not evidence for persistent storage or host restart. The current NVMe admission decision changes only after a held-out hybrid measurement and a fresh recorded space budget.

Do not prioritize hybrid 32K ahead of P0/P1/P2.

## RA-004 — answer unchanged

Scheduled/proactive warming is excluded outright. The service is demand driven and may touch only an already-requested, exact ready runtime instance. A separate `/running` check followed by `/upstream/<model>` is not atomic and remains a wake race; active restore stays quarantined until the live no-wake gate passes.

## Deferred

- hybrid 32K until the held-out size gate is justified;
- host-restart/cold-boot claims;
- 131K;
- learned cross-model mapping/token alignment;
- MTP/draft/speculative state;
- broad service/authentication/distributed scheduling work;
- upstream comments/submissions.

## Required execution order

1. Harden only the cross-backend runner's provenance and predeclared verdict.
2. Run the exact qwen35 8K HIP↔Vulkan patched/unpatched matrix.
3. Retain the exact structural account of the real 8K hybrid artifact.
4. Prove bounded hybrid numeric decoding and choose canonical physical storage.
5. Prove same-model f16→q8_0 behavioral conversion.
6. Extend to CUDA and then a live second runtime.
7. Activate vLLM and the demand-driven sidecar only after their own live gates.
