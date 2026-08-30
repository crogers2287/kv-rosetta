# KV Rosetta research steer: retain the real-layout correction and resume the production transfer gate

Status basis: default-branch head 6ef470d7287caf6fc864bb53db0369e74e3dce0a.

This steer supersedes ae04ce7. The sizing detour produced one material causal correction: the recurrent section written by `llama_memory_recurrent::state_write` has no `n_stream` field. The decoder, layout inventory, and synthetic fixtures had agreed with one another but not with llama.cpp. A real qwen35 artifact exposed the error. Preserve that fix and its failure history.

Do not continue expanding sizing work now. The exact production qwen35 8K HIP↔Vulkan gate ordered in the prior steer has not run and remains the largest justified falsifiable experiment. Checkpoint-bearing size prediction correctly refuses rather than extrapolating from one residual; keep that refusal and return execution to transfer portability.

## Mission and boundaries

Deliver portable reuse of large agentic prefixes:

1. exploit a strict native fast path wherever an exact llama.cpp tuple is proven portable;
2. use canonical KVX transformation across cache dtypes, inference runtimes, and backend pairs the native format cannot cross;
3. add quality-gated cross-model transformation only after same-model paths are proven.

For qwen35, GGSQ/3+SCKP/1 remains a strict opaque foundation. Preserve refusal on unpatched, incomplete, mismatched, unsupported, active-draft/speculative, or incompletely attested runtimes and artifacts. Do not weaken model/GGUF identity, CacheABIIdentity, checkpoint metadata, admitted-store guarantees, or exact runtime/protocol identity.

Primary hybrid evidence remains:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep patches 0001 and 0002 pinned and identified. Do not post upstream.

## New retained evidence

The corrected recurrent layout closes a real unpatched Qwen3.8-27B/qwen35 256-token state file exactly:

- total: 173,679,168 bytes;
- header: 12 bytes plus 4 × 260 token-id bytes;
- attention: 16,783,760 bytes, 16 layers, 256 cells, f16 rows of 2,048;
- recurrent: 156,894,356 bytes, one cell, 48 present layers of 64 declared;
- leftover: zero bytes.

The 256→257 delta is exactly 65,564 bytes and the recurrent section is byte-identical. Recurrent state is therefore a fixed per-model/configuration term and accounts for 90.3% of the 256-token artifact. GGUF metadata yields the recurrent row sizes exactly: 122,880 bytes for convolution state and 3,145,728 bytes for SSM state. The NextN/MTP layer is not present in the KV cache, so the attention count is 16 rather than the 17 implied by a naive recurrence rule.

This is valuable evidence because eight tests failed when the parser was corrected: the fixtures had encoded a body no writer produces. Future parser claims must include a retained real-writer artifact, digest, exact-consumption result, and independent metadata source. Synthetic self-consistency is insufficient.

The new hybrid size code derives the unpatched/no-checkpoint object exactly. For the patched 2,048-token object it predicts 291,169,840 bytes before checkpoints versus 604,958,676 observed. The residual equals two recurrent sections plus 124 bytes, but that is one observation, not a decoded checkpoint law. Refusing nonzero or unspecified checkpoint counts and falling back to a clearly labeled conservative estimate is correct fail-closed behavior.

## Audit corrections still open

The parser is not yet an artifact trust boundary:

1. `Reader.skip()` advances a logical offset but does not prove physical EOF for skipped final payloads.
2. Attacker-controlled cell counts are not bounded before loops and allocation.
3. A declared cell extension can still resolve ambiguously to zero rather than an exact positive descriptor-bound size.
4. Transposed q8_0 V still uses an element-size interpretation inconsistent with the writer's 34-byte block contract; refuse it until the pinned writer layout is proven.
5. Layer counts and recurrent maps are range-checked but not bound exactly to source GGUF geometry and ordering.
6. Empty/zero-coverage checkpoint sections can be accepted by the low-level parser.
7. Schema JSON decoding must use strict types, finite positive geometry, exact digest forms, unique roles, and a versioned unknown-field policy.
8. Architecture-based sizing semantics are hardcoded. Bind them to the pinned llama.cpp revision/protocol rather than treating architecture names as timeless truth.
9. The retained real-artifact evidence must open and parse a digest-identified file; arithmetic constants copied into a test are evidence summaries, not a live parse gate.

These defects block canonical transformation and admission, but they do not block the raw behavioral cross-backend experiment below. Do not let parser expansion displace that gate again.

## P0 — exact qwen35 8K HIP ↔ Vulkan

First harden only the cross-backend runner omissions already identified, with red tests. Predeclare the verdict before collecting the live result:

- exact model/GGUF digest and prompt-token digest;
- full llama.cpp source identity, patch set, binary and loaded-library digests;
- complete CacheABIIdentity, including actual K/V dtypes and context parameters;
- retained launch commands, environment, device identity, backend startup evidence, and offloaded-layer count;
- proof that the writer stopped before the reader started, with distinct process identities;
- save/restore metadata equality, including state version, SCKP version, checkpoint count, positions, coverage, and active-state classes;
- exact expected uncovered tail, not merely `cache_n > 0`;
- same-backend self-restore and cold-prefill controls for each reader;
- unpatched hybrid refusal before any state endpoint call;
- identical output token ids/text against the reader-native reference;
- nonempty probability vectors at every declared generated position under frozen divergence thresholds;
- unchanged artifact digest and file facts before and after transfer;
- missing provenance, partial vectors, silent backend fallback, or a threshold not fixed in advance is a refusal.

Then run, on the exact production-tested Qwen3.8-27B/qwen35 8,192-token prefix:

1. HIP save → full stop → Vulkan restore → completion.
2. Vulkan save → full stop → HIP restore → completion.
3. Same-backend self-restore and cold controls for HIP and Vulkan.
4. Same-source unpatched hybrid controls.

Hold model, prompt, K/V dtype, context, runtime revision, patch set, checkpoint policy, and generation settings fixed. Change only backend.

Decision:

- **Pass:** allowlist this exact llama.cpp HIP/Vulkan serialization tuple as a native fast path. Backend alone may be omitted from refusal identity only for the explicitly proven compatible pair; dtype, runtime, model, protocol, patch, and active-state identities remain strict.
- **Fail:** retain the first divergent field or behavioral position, make the smallest causal fix, and rerun the identical matrix.
- **Missing attestation:** refuse the run. Do not infer backend execution from a filename or caller-supplied label.

Do not run qwen35 32K, CUDA, a second model, or more sizing experiments before P0 resolves.

## P1 — real patched 8K structural account

After P0, close the parser/schema defects above red-test first. Then structurally parse the exact digest-identified 8K GGSQ/3+SCKP/1 artifact written by the live patched runtime:

- prove physical EOF, exact total consumption, ordered/non-overlapping/exhaustive spans, and bounded peak memory;
- bind attention/recurrent layer maps, row geometry, cache dtypes, and cell extensions to the source GGUF and pinned writer revision;
- decode checkpoint framing rather than inferring it from residual size;
- require positive checkpoint coverage matching the live save metadata;
- retain truncation, oversized-count, wrong-map/geometry/tuple, zero-checkpoint, and one-byte-boundary refusal controls.

Only this decoded result may replace the conservative checkpoint size fallback. If it yields a source-derived component formula, validate it on a held-out hybrid length before changing persistent-storage admission.

## P2 — canonical dtype/runtime seam

After P1:

1. Validate bounded attention, recurrent, and checkpoint numeric samples against pinned ggml reference code.
2. Choose and version a canonical physical dtype and chunking from measured fidelity, storage, and conversion costs; float32 remains an intermediate unless justified.
3. Build the narrowest same-model f16→q8_0 target encoder/import seam.
4. Freeze quality thresholds before the live run.
5. Compare converted reuse against target-native q8_0 reuse with exact identity, output, probability, bounded-memory, and native-prefill fallback gates.

The ggml-verified q8_0/q4_0 codecs remain unit evidence only. No converted cache is behaviorally admitted until this gate passes.

## P3 — broader portability and service

Only after P2:

1. extend the proven native/canonical distinction to CUDA and a second device;
2. prove a live second inference-runtime connector;
3. activate vLLM only after its actual `[B,H,N,C]`/MLA layout, scheduler metadata, and TP/PP shard identity pass end to end;
4. activate the demand-driven sidecar only after readiness and restoration are atomic to one exact runtime generation and every route passes a live sacrificial no-wake matrix.

Keep vLLM inert and sidecar active restoration quarantined meanwhile. No scheduled or proactive warming.

## RA-003 — answered with source-derived evidence

The affine mechanism is now demonstrated for the exact qwen35 configuration, not merely fitted:

- fixed unpatched recurrent term: 156,894,356 bytes;
- exact attention marginal term: 65,564 bytes/token;
- exact closure at 256 and 257 tokens.

The earlier ~449 MiB fitted “fixed term” combined base recurrent state with patched checkpoint copies/framing. The observed patched residual of two recurrent sections plus 124 bytes is suggestive but not an admission formula. Therefore:

- use source-derived component accounting for no-checkpoint/unpatched artifacts;
- keep nonzero-checkpoint prediction conservative and explicitly labeled;
- do not generate hybrid 32K on an inferred checkpoint law;
- decode the patched SCKP appendix and validate a held-out hybrid prediction before changing the persistent-space guard.

Tmpfs remains valid for research-only mechanism tests, not persistence claims. RA-003 can be marked answered.

## RA-004 — answer unchanged

Scheduled/proactive warming is excluded. The sidecar is demand driven and may touch only an already-requested, exact ready runtime instance. A separate readiness check followed by a model-routed request is a wake/replacement race; active restore stays quarantined until readiness and restore are bound atomically and the live no-wake gate passes. RA-004 can be marked answered.

## Required execution order

1. Retain the recurrent-layout failure and exact real-artifact evidence.
2. Harden only cross-backend runner provenance and its predeclared verdict.
3. Run the exact qwen35 8K HIP↔Vulkan patched/unpatched matrix.
4. Red-test parser/schema trust boundaries and parse the real patched 8K artifact completely.
5. Prove bounded hybrid numeric decoding and choose canonical physical storage.
6. Prove same-model f16→q8_0 behavioral conversion.
7. Extend to CUDA and a live second runtime.
8. Activate vLLM and the demand-driven sidecar only after their live gates.

Deferred: qwen35 32K, host-restart/cold-boot claims, 131K, learned cross-model mapping/token alignment, MTP/draft/speculative support, broad service/authentication/distributed scheduling work, and upstream submissions/comments.
