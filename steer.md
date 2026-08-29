# KV Rosetta research steer: seal compound import, prove the adapter path, then run 2K

Status basis: default-branch head 7c8e32bd1a981b268157b7581c8d1df38119a73e.

This steer supersedes cd2afb0. The corrected 256-token runtime matrix, one configuration-aware support predicate, complete restore-metadata comparison, mandatory hybrid reuse verification, and bounded appendix parsing have landed. Do not repeat those discoveries. Close the remaining compound-header bypass, then prove the actual KVX export/import path across owned process lifetimes before widening to 2K.

## Mission

Deliver persistent exact-prefix restoration for the deployed 27B qwen35-family hybrid model across a complete llama-server restart, with an auditable compound artifact and no path that advertises, exports, restores, or reports success on an unpatched runtime, incomplete SCKP state, inconsistent manifest, unsupported protocol tuple, or unproven active state class.

Primary upstream evidence:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Keep llama.cpp patch 0001 pinned. Do not post upstream during this track.

## Evidence now retained

### Corrected process-owned 256-token matrix

`bench/production-27b-matrix.json` was rerun from committed runner `56600b98b0ac3e9584c8b04942310b46ef1ae560` on the exact file:

- `Qwen3.8-27B-UD-Q4_K_XL.gguf`;
- architecture `qwen35`;
- model content digest `9bf3b07e1fb6531e91d970384cc3bdbc34b26dcfed993a743ecb3a9773aa8886`;
- current deployed llama.cpp fleet SKU on the measured host;
- 256 exact prompt tokens;
- upstream llama.cpp source revision `ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc` for both builds.

Do not transfer the model claim to a Qwen3.5-27B or Qwen3.6-27B digest. Those exact variants remain untested.

Measured once on Fred:

| Result | Patched | Same-source unpatched |
|---|---:|---:|
| Native in-process reuse | 252/256 | 252/256 |
| Fresh process before restore | 0/256 | 0/256 |
| Fresh process after restore | 252/256 | 0/256 |
| Tail reprocessed | 4 | 256 |
| Artifact bytes | 487,926,936 | 174,138,116 |
| Checkpoint appendix bytes | 313,788,820 | 0 |

The retained vectors are now non-vacuous: eight generated positions, five `top_logprobs` alternatives per position, generated token present in each vector, native and restored vectors exactly equal within the declared `1e-6` tolerance. Token IDs and content also match.

The runner refuses a dirty kv-rosetta worktree and records its own digest, repository HEAD, launch argv, build flags, binary digests, source-tree HEADs, and patched-tree diff digest. The patched llama.cpp tree is intentionally modified by the retained patch set; do not describe that source tree as clean.

### One fail-closed support decision

`hybrid_support()` now governs capability, export, and import. On the target hybrid path it requires:

- complete `sckp/1` advertisement;
- exact tested tuple `ggsq/3+sckp/1`;
- reported active state classes;
- target-only active state;
- proven state classes only.

Active state and the protocol tuple are bound into `CacheABIIdentity`. Import refuses unsupported configurations before any restore POST, compares all five checkpoint metadata fields with the manifest, forces reuse verification for a recognized compound artifact, and returns `ok=false` plus erasure when pristine restoration fails.

Keep all 14 tests in `tests/test_hybrid_support_predicate.py`.

### Appendix validation is bounded

Production validation now seeks directly to `n_written - checkpoint_bytes`, reads only framing, skips payloads arithmetically, bounds individual and aggregate lengths, and requires exact EOF termination. The general classifier scans in bounded 4 MiB chunks.

Retained evidence:

- 4 GiB sparse artifact: no more than 256 bytes read at the known boundary;
- 400 MiB measured artifact: 0.0 MiB peak RSS growth after the change, versus 400 MiB before;
- oversized and past-EOF lengths fail without allocation.

Keep these properties and tests.

## P0: close the remaining compound-header bypass

Source review at `7c8e32b` found one remaining fail-open mismatch in `import_()`:

- compatibility is checked from `header["blob"]["opaque_format"]`;
- compound behavior is selected independently from whether `header["coverage"]["format"]` contains `+`;
- `container.verify()` verifies header and payload integrity but does not require those two format fields to agree.

Therefore a correctly hashed artifact whose blob says `ggsq/3+sckp/1` but whose coverage format is absent or plain `ggsq/3` reaches the compound-capable runtime with `is_compound=False`. That skips nonzero coverage, restore-metadata equality, and forced reuse verification; a caller passing `verify_reuse=False` can receive success based only on sequence-state `n_restored`.

Fix red-test first:

1. Derive the compound decision from the authoritative blob format, not from optional coverage text.
2. Require `coverage.format == blob.opaque_format` for opaque artifacts.
3. If the blob format is compound, require a coverage object and complete nonzero checkpoint fields before staging or restore.
4. Refuse missing, plain, malformed, or mismatched coverage format before any restore POST.
5. Keep `verify_reuse=False` ineffective on every artifact whose blob format is compound.
6. Validate numeric coverage types fail closed; do not let malformed strings raise out of the report boundary.
7. Bind the exact compound format used for the decision into the artifact key and cache ABI as already intended.

Required retained tests:

- compound blob plus missing coverage format refuses before restore;
- compound blob plus plain coverage format refuses before restore;
- plain blob plus compound coverage refuses before restore;
- malformed numeric coverage refuses cleanly before restore;
- each case remains refused with `verify_reuse=False`;
- the genuine exported artifact still imports.

Do not solve this by trusting a plus sign alone. Parse the exact supported tuple and compare exact strings.

## P1: finish the 256-token end-to-end adapter gate

The retained production matrix exercises raw llama.cpp save/restore endpoints. It is not evidence that the KVX container, integrity verification, staging, adapter import, mandatory probe, pristine re-restore, and unpatched refusal work together across a true restart.

Run the next smallest process-owning experiment on the same exact Qwen3.8-27B digest and 256-token prompt:

### Patched leg

1. Start patched server A and record PID, props, binary/source identity, launch argv, and exact token IDs.
2. Use the public adapter path to export one `ggsq/3+sckp/1` KVX artifact.
3. Record outer-container digest, embedded payload digest, artifact key, complete coverage, sequence bytes, checkpoint bytes, and total bytes.
4. Stop A; prove PID death and port closure.
5. Start fresh patched server B; prove zero reuse before import.
6. Import through the public adapter with its mandatory reuse verification.
7. Require observed `cache_n=252`, `prompt_n=4`, exact restore-metadata equality, and successful pristine re-restore.
8. Issue the retained deterministic completion and require token, content, and nonempty probability-vector parity with native in-memory reuse.
9. Prove the staged copy is removed.

### Unpatched leg

Against a same-source unpatched fresh process:

- capabilities are empty for this hybrid model;
- explicit export refuses before a save POST;
- importing the patched compound artifact refuses before staging or restore;
- `verify_reuse=False` does not change the result;
- no response is accepted from `n_restored` alone.

Record endpoint-call evidence so “before POST” is mechanically checkable.

### Timing

Measure end-to-end, not just runtime restore:

- KVX integrity verification;
- payload staging/extraction;
- runtime restore;
- mandatory reuse probe;
- pristine re-restore;
- tail processing;
- total user-visible adapter import latency.

Keep raw runtime restore timing separately. Save/export time is not restore latency, but artifact creation time and bytes still belong in the record.

The 256 gate is correctness evidence. The existing raw numbers—about 0.554 s cold versus about 0.358 s restore plus 0.203 s tail before outer verification, staging, probe, and pristine re-restore—do not establish an economic win.

## P2: run the 2K economic rung

Only after P0 and P1 are green, parameterize the committed runner and run 2,048 exact prompt tokens three clean repetitions on the same model, hardware, dtypes, and launch configuration.

For every repetition retain:

- native cold prefill;
- native in-memory reuse;
- outer KVX verification;
- staging;
- runtime restore;
- mandatory reuse probe;
- pristine re-restore;
- tail processing;
- total user-visible adapter import;
- sequence bytes, checkpoint bytes, container overhead, and total bytes;
- process RSS and VRAM peaks;
- token/content parity;
- nonempty probability-vector parity against native reuse;
- exact cache and prompt token accounting.

Run tmpfs first to isolate compute and serialization. Run NVMe separately to measure the deployable path. State the actual K/V cache dtypes explicitly; Q4 weight quantization does not imply K/V dtype.

Decision rule:

`total verified adapter import + tail processing < native full prefill`

Proceed to 8K only if all three 2K repetitions pass correctness and show either break-even or a credible measured path toward it. Otherwise stop and attribute the dominant phase. Proceed to 32K only if 8K remains correct, favorable, and memory-bounded.

## P3: exact model transfer and active-state expansion remain deferred

The result is currently for Qwen3.8-27B with architecture `qwen35`. If the production target changes to a Qwen3.5-27B or Qwen3.6-27B file, rerun the sealed 256 gate on that exact digest before claiming support.

Draft/MTP/speculative serialization is not behavioral support. Keep those active classes withheld until each is actually enabled in a process-owned restart experiment, compared with native reuse, subjected to required-blob corruption/removal, and bound to model and launch identity.

## Required execution order

1. Add red tests for blob/coverage compound-format disagreement.
2. Make one exact format decision govern import semantics and fail malformed coverage closed.
3. Run all offline tests.
4. Commit the implementation before producing evidence.
5. Run the patched/unpatched process-owned 256-token public-adapter matrix.
6. Retain the end-to-end record and phase timings.
7. Run 2K three times.
8. Run 8K and 32K only when the previous rung passes.
9. Keep exact-boundary work, 131K, cross-backend, canonical conversion, vLLM, Transformers, and upstream submission deferred.

## Reporting discipline

Classify every claim as one of:

- proven by retained automated test;
- measured once on Fred;
- independently measured upstream;
- confirmed by source review;
- inferred;
- untested;
- failed.

Current truthful status:

- checkpoint persistence is behaviorally proven once on the exact tested Qwen3.8-27B qwen35 model;
- patched raw restore reuses 252/256 tokens and same-source unpatched restore reuses 0/256;
- token, content, and nonempty probability-vector parity against native reuse are proven by the retained record;
- target-only capability/export/import configuration gating is substantially fail closed;
- bounded appendix parsing is proven by retained tests and a one-host RSS measurement;
- a compound blob/coverage disagreement can still bypass compound import semantics;
- the public KVX adapter path has not yet been proven across a true process restart;
- 2K, 8K, and 32K verified adapter economics remain unmeasured.
