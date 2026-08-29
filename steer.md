# KV Rosetta research steer: production 27B checkpoint-contract gate

Status basis: default-branch head 858731619bee54ab3cffcc4ea5cf00ca545d84e5.

This steer supersedes 3cd8143. The retained restart harness and runtime protocol now exist. The next milestone is no longer “prove that some hybrid model can restart.” It is to prove, on the exact production Qwen3.5/Qwen3.6 27B, that the declared checkpoint contract survives a real restart, differs from an unpatched runtime, and is strong enough for the adapter to consume without guessing.

## Mission

Persist the exact production Hermes/system/tool prefix across slot eviction and a complete llama-server restart, while failing closed on an unpatched runtime, incomplete SCKP state, dishonest metadata, or unproven MTP/speculative configuration.

A short verified tail re-prefill is acceptable. Exact-prefix means exact token, model, runtime, cache-ABI, and checkpoint identity; it does not mean cache_n must equal the full prompt length.

Primary upstream evidence remains:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Do not post upstream or replace the pinned PR patch during this track.

## Evidence now retained

The repository now has:

- a process-owning positive restart harness;
- proof that the first PID exits, the port closes, the second PID differs, and the fresh process has cache_n=0 before restore;
- patched hybrid restore on OpenMythos-Q6_K with a four-token tail and output/token parity;
- an explicit patched/unpatched test classifier;
- a second, digest-guarded local llama.cpp patch;
- machine-readable /props fields for sckp/1 and sequence-state version;
- save/restore checkpoint count, byte count, coverage, and position metadata;
- an in-process comparison where declared checkpoint coverage 252 equals observed cache_n 252;
- removal of the old artifact-size threshold.

Keep these results. Do not reopen exact-boundary checkpoint work: the tail remained four tokens at 256, 1,024, and 4,096 tokens across large checkpoint-policy changes.

## Audit corrections

Do not overstate what the new commits prove.

1. tests/test_hybrid_restart_harness.py proves only the patched positive path. It calls _require_patched and skips any unpatched runtime. The unpatched negative control is retained separately and is in-process, not the same two-process experiment.

2. The restart harness checks n_restored > 0 but does not assert the new checkpoint metadata. The retained equality between checkpoint_n_tokens and cache_n is currently in-process. It has not yet been bound across the two owned process lifetimes.

3. The successful harness model is still OpenMythos-Q6_K, not the production 27B.

4. /props currently reports target, draft, and speculative checkpoint support as true unconditionally. Target-state behavior has evidence on the tested non-MTP path. Draft/speculative behavior does not. The source struct containing data_dft and data_spec, and PR #26004 serializing those fields, are source evidence—not behavioral proof.

5. The KV Rosetta adapter still withholds hybrid opaque capability and still labels slot artifacts as plain ggsq/<sequence-version>. No production hybrid artifact is enabled yet. That is correct until the contract below passes.

6. SCKP magic scanning is acceptable as a test-only classifier for a build carrying upstream patch 0001 without protocol patch 0002. It must not become the adapter capability signal or a successful import criterion.

## Steering decision

Run one paired, process-owned, 256-token checkpoint-contract matrix on the exact production 27B before adding larger benchmarks.

The matrix must use the same GGUF, exact token IDs, launch flags, KV dtypes, checkpoint policy, slot number, and sampling settings for both binaries:

- unpatched pinned llama-server;
- patched pinned llama-server with patches 0001 and 0002.

Parameterize the existing harness rather than creating another manual script. Use a unique temporary slot directory and dynamically allocated port per leg.

This is the next smallest falsifiable experiment because it can answer all of the immediate production questions with two short prompts and four process lifetimes. Do not run 2K or larger contexts until it passes.

## P0: make protocol claims honest before consuming them

Change the draft/speculative capability semantics before the adapter reads them.

Accept either design:

- report separate “serialized” fields for data_tgt, data_dft, and data_spec, while behavioral support remains false or unknown until tested; or
- keep the current support field names but report draft=false and speculative=false until a retained MTP/speculative restart test passes.

Target support may be true only for the exact configuration covered by the positive test. Do not treat a compile-time struct member as a runtime capability.

Retain a unit/live test proving an absent, false, malformed, or unknown protocol never enables hybrid import.

## P1: production 27B paired restart matrix

For each leg, record a machine-readable result containing:

- model path and full model digest;
- tokenizer/prompt token digest and token count;
- general.architecture and quantization;
- binary digest, build_info, upstream base, and patch digests;
- sequence-state and checkpoint-format versions;
- K/V dtypes, n_ctx, checkpoint policy, GPU split/offload, MTP/draft/spec flags;
- first and second PIDs;
- slot artifact digest and exact bytes;
- save and restore responses;
- cache_n, prompt_n, generated token IDs, top probabilities, and timings.

The patched leg must prove:

1. /props advertises slot_checkpoint_persistence=true, sckp/1, and the exact sequence version.
2. Save reports at least one checkpoint, positive checkpoint_bytes, valid positions, and positive checkpoint_n_tokens.
3. The first process exits and its port closes.
4. The second process has a different PID and cache_n=0 before restore.
5. The restored file digest equals the file digest created by the first process.
6. Restore metadata equals save metadata for checkpoint count, bytes, coverage, and positions.
7. The exact prompt then reports cache_n equal to checkpoint_n_tokens.
8. prompt_n equals token_count minus cache_n and the uncovered tail is within the declared experimental ceiling of eight.
9. Persisted output tokens and probability vectors match native in-memory checkpoint reuse. Compare to cold output only when native reuse itself matches cold.
10. If MTP/draft/speculative execution is enabled, its behavioral support must be separately proven; otherwise its capability remains withheld.

The unpatched leg must prove:

1. It advertises no checkpoint-persistence protocol.
2. The same save/stop/start/restore lifecycle may restore sequence cells, but reports no usable checkpoint coverage.
3. The post-restore exact prompt has cache_n=0 and prompt_n equal to the full token count.
4. The adapter continues to advertise no hybrid opaque import/export.
5. The test fails if the unpatched leg is accidentally pointed at a patched binary; do not silently skip the required negative half of the paired record.

Commit one JSON benchmark/evidence record only after both legs complete. A skipped leg is not a passing matrix.

## P2: consume the protocol fail closed

Only after the production matrix passes, teach LlamaCppHTTPAdapter the hybrid compound format.

Required behavior:

- Hybrid/recurrent opaque capability is enabled only when the live protocol is complete and exact: persistence true, format sckp/1, supported sequence version, and configuration-appropriate target/draft/spec evidence.
- Label the artifact ggsq/<sequence-version>+sckp/1, never plain ggsq/N.
- Bind exact prompt token digest/count, whole payload digest, checkpoint count/bytes/coverage/positions, runtime checkpoint policy, K/V dtypes, and any target/draft/spec identities into the manifest and cache ABI.
- Export refuses if save metadata says zero checkpoints or is internally inconsistent.
- Import requires outer KVX integrity before staging, an exact compound format and cache-ABI match, and restore metadata matching the artifact.
- The verification completion must satisfy cache_n == declared checkpoint_n_tokens and prompt_n == exact uncovered tail within the tested bound.
- Any missing appendix, truncated appendix, unknown version, zero restored checkpoints, metadata mismatch, or observed-reuse mismatch returns a failed ImportReport and falls back to native prefill.
- HTTP 200 and n_restored alone are never success.
- Keep ordinary-attention ggsq/N behavior and its one-token tail unchanged.

Add corruption cases for a truncated SCKP appendix and a sequence-only hybrid slot. llama.cpp may accept both as sequence restores; KV Rosetta must reject both as hybrid checkpoint imports.

## P3: economics after correctness

After the production 256-token paired matrix and adapter round trip pass, run 2K, 8K, and 32K with at least three clean repetitions per rung.

Measure native cold prefill, native in-memory reuse, save, verification, restore, tail prefill, total user-visible restore, sequence bytes, checkpoint bytes, total bytes, cache_n/prompt_n, RSS, VRAM, and output parity. Test tmpfs and NVMe separately.

The decision criterion is:

total verified restore + tail prefill < native full prefill

Do not infer 27B economics from the earlier 3B q4 ladder. Do not run 131K until 32K produces a credible size, latency, and memory projection.

## MTP/speculative gate

PR #26004 persisting data_tgt, data_dft, and data_spec is necessary but insufficient.

Before claiming MTP/speculative support:

- run the process-owned production restart with the real feature enabled;
- compare persisted restore with native in-memory reuse;
- demonstrate a negative build or fixture with required draft/spec state omitted;
- require refusal or a measured behavioral failure;
- bind draft/spec model identities and settings into CacheABIIdentity.

Until then, report those behavioral capabilities as false or unknown.

## Required execution order

1. Correct draft/speculative protocol semantics so untested support is not advertised as true.
2. Parameterize the existing harness into the mandatory patched/unpatched production-27B matrix.
3. Run and retain the 256-token paired record.
4. If either leg fails, stop and explain the smallest violated invariant; do not compensate with a larger prompt or looser tail bound.
5. If it passes, enable the adapter only through ggsq/<version>+sckp/1 with metadata-bound fail-closed import.
6. Add missing/truncated/unknown SCKP corruption tests.
7. Prove or withhold MTP/speculative behavior.
8. Run the 2K, 8K, and 32K economic ladder.
9. Keep exact-boundary checkpoint work closed unless the production measurements falsify the bounded-tail result.
10. Defer 131K, cross-backend, canonical extraction, vLLM, Transformers, and upstream submission.

## Definition of the next milestone

The next milestone is complete only when one retained production-27B record contains both patched and unpatched two-process legs; the patched restore metadata survives the restart and equals observed reuse; the unpatched leg reuses zero; unproven draft/spec support is not advertised; and no adapter capability has been enabled by architecture, filename, strings output, artifact size, or SCKP magic alone.

## Reporting discipline

Classify every statement as one of:

- proven by retained automated test;
- measured once on Fred;
- independently measured upstream;
- confirmed by source review;
- inferred;
- untested;
- failed.

Current truthful status:

- hybrid checkpoint persistence is proven by a retained two-process test on OpenMythos-Q6_K;
- runtime checkpoint metadata is proven in-process on that model;
- the exact production 27B paired restart contract is untested;
- draft/speculative behavioral restoration is untested;
- the adapter correctly continues to withhold hybrid opaque transfer;
- production economics remain unmeasured.
