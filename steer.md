# KV Rosetta research steer: productionize hybrid checkpoint persistence

Status basis: default-branch head 4a302d9133de86ade65a898fc33505f115f51ab6.

This steer supersedes bdaa391. The hybrid checkpoint seam is no longer hypothetical. The next work is to turn a successful one-time experiment into a reproducible, detectable, fail-closed production path on the actual Qwen3.5/Qwen3.6 27B model.

## Mission

Persist the exact stable Hermes/system/tool prefix for the production 27B hybrid model across slot eviction and a complete llama-server restart, then prove that total restore latency beats native prefill at useful context lengths.

Exact-prefix means the artifact is bound to the exact token sequence, model, runtime, and cache ABI. It does not require every token to be reused when llama.cpp deliberately rolls back to the nearest valid recurrent checkpoint. A small, declared, verified tail re-prefill is acceptable when it preserves native behavior and the economics remain favorable.

## What changed

The repository now contains:

- a source map for the hybrid checkpoint failure;
- an unpatched-runtime negative control;
- upstream PR #26004 preserved as a pinned patch;
- a guarded build script;
- a patched llama-server built on Fred;
- an in-process hybrid save, erase, restore, and reuse result;
- a one-time full-process restart result on a qwen35 hybrid model;
- a bounded-tail import invariant that accepts exact-prompt reuse only when 1–8 tokens are uncovered and prompt_n equals that exact tail.

Measured on OpenMythos-Q6_K:

- unpatched slot file: 173,678,144 bytes;
- patched slot file: 330,573,584 bytes;
- restored checkpoint: about 149.626 MiB;
- post-restore reuse: cache_n=252, prompt_n=4 for a 256-token prompt;
- output and token IDs matched the recorded cold run;
- the same result survived a manually controlled full server restart;
- the uncovered tail remained exactly four tokens at 256, 1,024, and 4,096 tokens under two substantially different checkpoint policies;
- ordinary-attention live tests still report the established one-token tail.

This proves the architecture is restorable when checkpoint state travels with the slot file.

It does not yet prove the production 27B, MTP/speculative state, repeatable restart automation, or favorable economics on Fred.

## Primary upstream evidence

Use these as evidence, not as substitutes for local tests:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

PR #26004 remains open. Its current design appends a versioned SCKP payload to the same slot file and preserves data_tgt, data_dft, and data_spec.

Independent upstream testing reported:

- Qwen3.5-35B-A3B hybrid;
- 58,202-token cold prompt;
- full server restart between measurements;
- unpatched restore: cache_n=0, prompt_n=58,202, 181.9 seconds;
- patched restore: cache_n=57,686, prompt_n=516, 4.7 seconds;
- successful ten-turn continuation after restore.

That is a roughly 39x first-request improvement despite reprocessing 516 tokens. It is evidence that checkpoint granularity is not automatically a blocker.

A separate CUDA-to-Metal result is interesting but not yet a cross-backend proof because its initial prompt continued straight ahead rather than diverging at a checkpoint. Keep cross-backend work deferred.

## Critical evidence correction

tests/test_hybrid_checkpoint_restart.py does not start, stop, or restart llama-server. Its module documentation explicitly says restart orchestration is the caller's responsibility.

Therefore:

- the full-restart result is measured once on Fred;
- the test retains in-process properties;
- the repository does not yet contain an automated retained restart proof.

Do not describe the restart as proven by the retained test until the harness owns and verifies both process lifetimes.

This is the first item to fix because every later benchmark depends on knowing the old process and its in-memory checkpoints are gone.

## Steering decision

Do not spend another cycle forcing cache_n from 252 to 255.

Commit 4a302d9 tested 256, 1,024, and 4,096-token prompts while changing checkpoint density by roughly 16x. The uncovered tail remained four tokens. It is a stable runtime resume rule for this tested hybrid path, not a checkpoint-density problem. The adapter now verifies a bounded tail and requires prompt_n to equal the exact uncovered count.

Forcing an endpoint checkpoint is removed from the plan unless the production model falsifies this result.

Prioritize:

1. a real restart harness;
2. machine-readable capability advertisement;
3. the actual production 27B;
4. measured context-ladder economics;
5. exact-boundary checkpoint creation only if measured tail reprocessing is material.

Preserve the unpatched negative control throughout.

## P0-A: make the restart proof genuinely retained

Replace the manual restart convention with a harness that owns llama-server.

Inputs:

- patched llama-server path;
- unpatched llama-server path;
- exact GGUF model path;
- slot directory;
- launch flags;
- test port or dynamically allocated port.

The harness must:

1. create a unique temporary slot directory;
2. launch the first server process;
3. record its PID, binary digest, build information, model identity, and launch flags;
4. wait for readiness;
5. cold-prefill the exact token sequence;
6. save the slot artifact;
7. record artifact digest and size;
8. stop the process;
9. wait for process exit and confirm the port is closed;
10. launch a second process with a different PID and an empty in-memory state;
11. prove the same request before restore has cache_n=0;
12. restore the persisted artifact;
13. issue the exact request to the exact slot;
14. record cache_n, prompt_n, output tokens, probabilities, and timings;
15. stop and reap the second process;
16. clean temporary files.

The test must fail if:

- the two PIDs are equal;
- the first process remains alive;
- the port never closes;
- cache reuse exists before restore on the second process;
- the restored artifact is not the exact file created by the first process;
- the expected hybrid checkpoint coverage is not reused.

Run the same harness against the unpatched binary and require the negative result: sequence cells restore, but cache_n remains zero.

Do not retire the negative control. Parameterize the matrix:

- unpatched hybrid: must fail closed;
- patched hybrid: must restore checkpoint coverage;
- ordinary-attention model: must continue to use the existing GGSQ path.

## P0-B: stop inferring the runtime feature

The build script's strings search proves that SCKP code compiled into one shared library. It is not a runtime protocol and must not enable the adapter.

Keep upstream PR #26004 as the unchanged first patch. Add a second local patch for KV Rosetta integration so upstream provenance remains auditable.

Expose machine-readable fields from the live server, preferably through /props:

- slot_checkpoint_persistence: true;
- slot_checkpoint_format: sckp/1;
- sequence_state_version;
- supports_target_checkpoint_state;
- supports_draft_checkpoint_state;
- supports_speculative_checkpoint_state.

Save and restore responses should additionally report:

- n_checkpoints_saved or restored;
- checkpoint_bytes;
- newest checkpoint n_tokens;
- newest checkpoint pos_min and pos_max;
- total on-disk bytes.

Do not enable hybrid capabilities from:

- architecture name alone;
- llama.cpp commit alone;
- binary filename;
- strings output;
- artifact size greater than a model-specific threshold.

The current static refusal remains correct until this protocol exists and the live test passes.

## P0-C: validate and bind the bounded-tail contract

The default branch now replaces the standard-attention-only invariant cache_n=L-1 and prompt_n=1 with a bounded-tail rule. Preserve that correction, but do not treat one class-level bound as universal proof for every architecture.

For a checkpoint-aware artifact, bind the following into the manifest:

- exact prompt token count and digest;
- checkpoint format;
- saved checkpoint n_tokens;
- pos_min and pos_max;
- declared uncovered tail;
- runtime checkpoint policy;
- context and speculative configuration.

Import succeeds only when the post-restore request agrees with the persisted checkpoint coverage.

The current default max_uncovered_tail of 8 is acceptable as a fail-closed experimental ceiling for Qwen3.5. Before general adapter capability is advertised, bind the supported bound to the tested runtime/architecture contract rather than silently applying it to every future hybrid family.

A positive cache_n alone remains insufficient.

Required checks:

- cache_n is at least the declared reusable checkpoint coverage adjusted by the runtime's documented one-token evaluation rule;
- prompt_n equals the exact uncovered tail plus required evaluation tokens;
- no cache beyond the artifact's declared coverage is credited;
- output matches the runtime's native in-memory checkpoint reuse behavior.

If the metadata and observed reuse disagree, fail closed to native prefill.

Do not force an exact endpoint checkpoint unless this measured contract leaves a tail large enough to damage the economics.

## P0-D: label the artifact truthfully

PR #26004 produces one file, not a sidecar pair. The previous steer’s two-segment requirement is obsolete.

Keep the slot file as one opaque KVX payload whose whole-byte digest covers both the GGSQ state and appended SCKP checkpoint data.

Label it explicitly:

- ggsq/<sequence-version>+sckp/1

Do not label it as plain ggsq/2 or ggsq/3.

CacheABIIdentity must include:

- exact llama.cpp build;
- sequence-state version;
- SCKP version;
- target, draft, and speculative model identities;
- K and V dtypes;
- context flags;
- checkpoint policy;
- hardware/backend fields that affect opaque compatibility.

The outer KVX verifier must reject truncation or modification before runtime import. A truncated SCKP appendix currently degrades to a sequence-only restore; that behavior is backward compatible for llama.cpp but unacceptable as a successful hybrid KV Rosetta import.

## P0-E: remove brittle evidence tests

The current positive test infers checkpoint presence from a file larger than 200 MiB. That threshold is specific to one model and can pass on a large sequence state without proving SCKP exists.

Replace it with one of:

- explicit save-response checkpoint metadata from P0-B;
- a bounded parser for the SCKP appendix using a server-reported offset;
- an exact known-format fixture.

Do not scan arbitrary binary bytes for SCKP magic and call that proof.

The current parity test compares restored output directly to cold prefill. For quantized KV, the project already proved llama.cpp's own warm reuse can diverge from cold processing.

Use three controls:

1. cold versus cold to prove deterministic sampling;
2. native in-memory checkpoint reuse;
3. persisted checkpoint restore.

Persisted restore must match native checkpoint reuse. Require cold equality only when native checkpoint reuse itself equals cold.

## P1: prove the actual production model at 256 tokens

The successful local run used OpenMythos-Q6_K, not the production 27B named by the project.

Run the automated restart matrix on the exact production Qwen3.5/Qwen3.6 27B GGUF with its actual Fred launch configuration.

Record:

- model and tokenizer content digests;
- general.architecture;
- model quantization;
- K and V cache dtypes;
- GPU split and offload flags;
- MTP, draft, and speculative configuration;
- server and patch digests;
- sequence and checkpoint format versions.

Start at 256 tokens.

Acceptance:

- patched runtime restores nonzero declared checkpoint coverage after a true restart;
- unpatched runtime restores sequence cells but reuses zero;
- persisted behavior matches native in-memory checkpoint reuse;
- MTP/speculative state is either explicitly proven or capability is withheld for that configuration.

Do not proceed to larger contexts if the exact production model fails at 256.

## P2: measure production economics before exact-boundary work

After the production 256-token proof, run:

- 2K;
- 8K;
- 32K.

Run 131K only after the 32K result gives a credible size and time projection.

At least three clean repetitions per rung. Report medians and ranges for:

- native cold prefill;
- native in-memory checkpoint reuse;
- slot save;
- artifact verification;
- slot restore;
- post-restore tail prefill;
- total user-visible restore;
- sequence-state bytes;
- checkpoint bytes;
- total artifact bytes;
- cache_n and prompt_n;
- effective throughput;
- host RSS and VRAM;
- output parity against native reuse.

Test tmpfs and NVMe separately.

Use actual production KV quantization. Do not transfer the 3B q4 economics to the 27B by arithmetic.

The relevant decision is not whether prompt_n equals one. It is whether:

total restore + verified tail prefill < native full prefill

If the checkpoint tail is a few tokens or a few hundred tokens against tens of thousands, leave checkpoint creation policy alone.

## P3: test MTP and speculative state deliberately

PR #26004 serializes data_tgt, data_dft, and data_spec, but field presence is not behavioral proof.

For the production MTP configuration:

- prove the saved artifact contains the required state blobs through explicit metadata;
- restore after a full restart;
- compare generated token IDs and probability vectors with native in-memory checkpoint reuse;
- run a negative build or fixture omitting data_dft or data_spec and require refusal or demonstrated failure;
- bind the draft/speculative identities into the cache ABI.

No MTP claim may rest solely on the upstream struct containing data_spec.

## P4: harden corruption and compatibility behavior

Retained cases:

- no SCKP appendix on a hybrid artifact;
- unknown SCKP version;
- truncated appendix;
- absurd checkpoint count;
- oversized individual blob;
- excessive aggregate blob size;
- wrong model identity;
- wrong prompt token digest;
- wrong runtime or sequence-state version;
- wrong slot;
- target state present but required draft/speculative state missing.

The upstream loader intentionally ignores an unusable appendix and may still return a successful sequence restore. KV Rosetta must interpret that as hybrid checkpoint failure and fall back to native prefill.

Do not rely on the server's HTTP status alone.

## P5: keep exact-boundary work closed

The four-token tail stayed constant across prompt length and checkpoint-policy changes. Do not implement forced endpoint checkpoints.

Reopen this question only if the actual production model:

- exceeds the bounded-tail contract;
- shows tail growth with context;
- loses meaningful latency to tail reprocessing;
- or diverges from native in-memory checkpoint reuse.

A production result inside the verified bound closes this line of research.

## Work still deferred

Until the production 27B passes the automated restart and 32K economic gates:

- canonical extraction;
- CUDA-to-HIP claims;
- vLLM adapter;
- Transformers adapter;
- cross-model mapping;
- DeepSeek compatibility;
- 131K brute-force runs;
- upstream comments or PR submissions.

The upstream CUDA-to-Metal observation is a research lead, not authorization to claim opaque portability.

## Required execution order

1. Replace the manual restart convention with a process-owning retained harness.
2. Keep an explicit patched/unpatched test matrix.
3. Add a second local llama.cpp patch for capability and checkpoint metadata.
4. Teach the adapter the compound GGSQ+SCKP format and coverage contract.
5. Replace artifact-size heuristics with explicit format evidence.
6. Prove the exact production 27B at 256 tokens after a full restart.
7. Prove or withhold MTP/speculative support.
8. Run 2K, 8K, and 32K production ladders.
9. Keep exact-boundary checkpoint work closed unless production measurements falsify the bounded-tail result.
10. Attempt 131K only if artifact size, restore latency, and memory remain viable.
11. Resume canonical and cross-backend work only after those gates.

## Definition of the next milestone

The production hybrid milestone is complete only when:

- one automated test owns both server processes;
- the second process has a different PID and no preexisting cache;
- the unpatched binary fails closed;
- the patched binary advertises sckp/1 through the runtime API;
- the artifact is labelled ggsq/<version>+sckp/1;
- exact prompt, model, runtime, cache, and checkpoint identities are bound;
- the actual production 27B reuses its declared checkpoint coverage;
- persisted behavior matches native in-memory checkpoint reuse;
- corrupt or incomplete checkpoint state cannot report successful hybrid import;
- 2K, 8K, and 32K total restore economics are measured with repetitions.

## Reporting discipline

Separate every claim into:

- proven by retained automated test;
- measured once on Fred;
- independently measured upstream;
- confirmed by source review;
- inferred;
- untested;
- failed.

Current status must be stated precisely:

- hybrid checkpoint persistence is feasible;
- one manual full-restart run succeeded locally;
- the retained test does not yet automate that restart;
- the bounded four-token hybrid tail is now measured across 256–4,096 tokens and enforced by the adapter;
- the exact production 27B and its economics remain unproven.
