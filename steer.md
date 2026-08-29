# KV Rosetta research steer: bind cache dtype, then measure the zero-copy 2K lower bound

Status basis: default-branch head 2efffccfe86e0341409c31ff3598358773343851.

This steer supersedes 0315e90. The corrected economic timer, reconciled phase accounting, one clean 2K break-test, and one-pass verified staging have landed. Correctness remains green and one redundant payload read is gone, but the public adapter is still 2.52× slower than cold because copying and hashing the 577 MiB payload dominates. Do not run 8K or NVMe yet. First close the missing K/V-cache-dtype identity boundary, then measure a direct-raw 2K lower bound before choosing a zero-copy artifact design.

## Mission

Deliver persistent exact-prefix restoration for the deployed 27B qwen35-family hybrid model across a complete llama-server restart, with fail-closed artifact identity and request-path economics that beat native prefill.

Primary upstream evidence remains open as of 2026-08-29:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep checkpoint-persistence patch 0001 pinned. Do not post upstream during this track.

## Evidence now retained

### Economic measurement is now truthful

At `4814702`:

- the completion after public-adapter import is timed directly as `adapter_tail_completion_wall_s`;
- the raw-endpoint tail remains diagnostic only;
- the verdict refuses to substitute when the adapter timer is missing;
- preflight is a named import phase;
- phase sum, reported seconds, and unclassified remainder are retained and checked.

The corrected 2K run reconciles to less than 0.005 s unclassified. Keep `tests/test_matrix_runner.py`.

### Two-pass 2K tmpfs baseline

`bench/ladder-2k-tmpfs.json`, produced from committed runner `4814702`, records one corrected break-first repetition:

- exact prompt tokens: 2,048;
- patched: `cache_n=2044`, `prompt_n=4`;
- same-source unpatched: `cache_n=0`, `prompt_n=2048`;
- token, content, and nonempty probability-vector parity against native reuse;
- public-adapter patched import succeeds;
- unpatched export and cross-import refuse without a restore POST;
- total KVX artifact: 604,960,596 bytes;
- payload: 604,958,676 bytes;
- sequence state: 291,169,856 bytes;
- checkpoint state: 313,788,820 bytes;
- checkpoint coverage: 2 checkpoints, 2,044 tokens;
- server peak RSS: about 17.1 GiB;
- two NVIDIA devices: about 9.0 and 9.4 GiB used after load.

Measured:

| Phase | Seconds |
|---|---:|
| Native cold request | 1.772 |
| Container payload verification | 1.986 |
| Staging copy | 2.404 |
| Preflight | 0.542 |
| Runtime restore | 0.515 |
| Reuse probe | 0.207 |
| Pristine re-restore | 0.477 |
| Actual post-adapter completion | 0.430 |
| Adapter import plus tail | 6.664 |
| Ratio to cold | 3.76× |

The runtime restore itself is about 3.4× faster than prefill. KV Rosetta's two full payload passes caused the loss.

### One-pass verified staging is correct and materially faster

At `9ce3ceb`, opaque import now:

1. verifies structure and the authenticated header without reading the payload;
2. extracts the payload to a temporary staged file;
3. hashes every byte in the same pass that writes it;
4. compares the payload digest before publishing the staged file;
5. issues no restore POST on any mismatch;
6. removes temporary and staged files on failure.

Segmented artifacts do not take this path because extraction does not validate their per-segment digests; they retain the full verification path.

Retained negative tests cover corrupted payload, corrupted header, truncation, exception containment, cleanup, before-restore refusal, genuine import, segmented fallback, and a read-count ceiling below 1.6× payload bytes. Keep `tests/test_one_pass_staging.py`.

`bench/ladder-2k-tmpfs-onepass.json`, from committed implementation `9ce3ceb`, records:

| Phase | Seconds |
|---|---:|
| Native cold request | 1.734 |
| Separate payload verification | 0.000 |
| Verified staging | 2.222 |
| Preflight | 0.525 |
| Runtime restore | 0.438 |
| Reuse probe | 0.232 |
| Pristine re-restore | 0.443 |
| Actual post-adapter completion | 0.425 |
| Adapter import plus tail | 4.371 |
| Ratio to cold | 2.52× |

Correctness and unpatched refusal are unchanged. Phase accounting reconciles to 0.004 s. The separate verification pass was removed, not merely accelerated.

The remaining dominant phase is verified staging: 2.222 s, about 57% of import, to copy and hash 577 MiB on tmpfs. The result crosses the previous stop threshold, so repetitions, NVMe, 8K, and 32K are not next.

## P0: bind the actual K/V cache dtype before changing storage identity

Both retained 2K records truthfully report:

- `kv_dtype_k=""`;
- `kv_dtype_v=""`.

The live patched `/props` does not expose the cache K/V types. It exposes weight quantization information, which is not a substitute. `CacheABIIdentity.k_dtype` and `.v_dtype` are therefore empty, so two otherwise identical launches that differ only in K/V cache type currently produce the same cache ABI digest.

That is an artifact-identity gap. Do not introduce a direct raw artifact path while it remains.

Extend the local capability-advertisement patch, not checkpoint-persistence patch 0001, to report the actual target context K and V cache types used by the running server. Requirements:

- values come from the live context/runtime configuration, not model filename, weight quantization, defaults guessed by the adapter, or launch-argument text;
- advertisement is machine-readable and exact;
- target, draft, and speculative contexts are distinguished if more than target exists;
- target-only hybrid support requires nonempty advertised target K/V types;
- the exact types are bound into `CacheABIIdentity`;
- missing or unrecognized types withhold hybrid capabilities and refuse export/import before save, staging, or restore;
- a runtime with a different K or V type refuses the artifact on ABI identity before touching state;
- same-type patched restart remains accepted;
- same-source unpatched hybrid remains withheld.

Required retained tests:

1. differing only in K dtype changes cache ABI and refuses cross-import;
2. differing only in V dtype changes cache ABI and refuses cross-import;
3. missing dtype advertisement yields empty hybrid capabilities;
4. missing dtype refuses explicit export and import before state endpoints;
5. weight quantization cannot populate cache dtype fields;
6. exact advertised types survive into the benchmark record;
7. target-only live 256 restart remains green after rebuilding the patched binary.

Because this changes capability patch 0002 and the patched binary identity, record the new patch digest, binary digest, source-tree diff digest, and build configuration. Do not transfer old live measurements to the rebuilt binary without a 256 smoke gate.

## P1: remove the avoidable state-version save probe

The one-pass record attributes about 0.525 s to preflight. Endpoint evidence shows `state_version()` saves a slot to discover the sequence version even though the complete patched checkpoint protocol advertises `sequence_state_version=3`.

After P0 makes runtime identity complete:

- use the advertised sequence version for import compatibility only when the protocol is complete, the tuple is allowlisted, active state classes are proven, and K/V cache types are present;
- keep byte-derived sequence-version validation on export and require emitted bytes to match the advertisement;
- retain a fail-closed fallback for runtimes without a complete protocol;
- an unpatched hybrid must refuse from support evidence, not perform a save probe to discover a format it is not allowed to use.

Measure the removed endpoint call and preflight time. Do not merely relabel it.

Required tests:

- complete patched hybrid import makes no version-probe save;
- advertised/emitted version disagreement refuses export;
- incomplete or malformed protocol never enables the fast path;
- unpatched hybrid import still issues no save or restore;
- cache ABI remains exact.

## P2: measure the direct-raw 2K lower bound before choosing a format

Do not build a production zero-copy store yet. First answer whether eliminating staging can actually cross break-even.

Run a research-only, process-owned lower-bound experiment on the exact existing 2K model, prompt, binary pair, and tmpfs environment. Keep the public adapter unchanged and clearly label this path non-production and pre-admitted.

### Artifact admission outside the timed restore window

- start with the exact raw `ggsq/3+sckp/1` state produced by the patched prefix export;
- validate its sequence framing, SCKP appendix, full payload digest, model identity, prompt identity, protocol tuple, active state classes, and K/V cache types;
- record admission time separately;
- store it under its content digest;
- record device, inode, size, mtime_ns, ctime_ns, and digest;
- ensure the file is unchanged before and after the timed restore;
- never count an unverified or changed file as admitted.

This is a lower bound for a trusted, pre-admitted local cache. It is not permission to skip integrity on arbitrary files.

### Timed direct path

Across a real server restart:

1. prove the new process has zero reuse;
2. use the already-admitted raw state directly, with no KVX payload extraction or byte copy;
3. issue the runtime restore;
4. compare every restore checkpoint field with admitted metadata;
5. run mandatory reuse verification;
6. restore the pristine prefix;
7. time the actual post-import completion;
8. require `cache_n=2044`, `prompt_n=4`, and full token/content/logprob parity;
9. prove the raw file and any temporary link are unchanged/removed;
10. run the unpatched control and require refusal before file linking or restore.

Record two direct variants if needed:

- current exact verification semantics with the preflight save retained;
- complete advertised identity with the version-probe save removed after P1.

Decision rule remains:

`timed verified/admitted restore path + actual tail < native cold request`

The one-pass measurements imply, but do not prove, a lower bound near:

`runtime restore + reuse probe + pristine restore + tail ≈ 1.54 s`

against a 1.73 s cold request. The margin is small enough that it must be measured, not projected.

### Branch on the lower-bound result

- Lower bound is not cheaper: stop 2K storage redesign. Preserve the result and determine whether probe/pristine semantics or longer context is the only remaining path.
- Lower bound is within 10% of cold: run three repetitions before choosing a format.
- Lower bound is clearly cheaper: then design the smallest admitted raw-artifact contract.

Only after a measured win choose among:

- content-addressed raw state plus authenticated sidecar;
- same-filesystem hardlink/reflink into the slot directory;
- restore-from-container offset support;
- filesystem integrity such as fs-verity.

Any chosen design must maintain atomic admission, digest identity, TOCTOU resistance, cleanup, and unpatched refusal. A filename, read-only bit, or prior hash alone is not sufficient proof that the bytes restored are the bytes admitted.

## P3: keep scaling and scope deferred

Do not run NVMe, 8K, 32K, or 131K until the direct 2K lower bound is measured and the identity gap is closed.

The proven model is exactly `Qwen3.8-27B-UD-Q4_K_XL.gguf`, architecture `qwen35`. Exact Qwen3.5-27B and Qwen3.6-27B digests remain untested.

Draft/MTP/speculative state remains withheld until behaviorally tested while active.

## Required execution order

1. Add live K/V cache dtype advertisement and fail-closed ABI tests.
2. Rebuild the patched runtime and run the sealed 256 smoke gate.
3. Remove the patched import's avoidable state-version save probe with exact fallback tests.
4. Commit code before evidence generation.
5. Run one pre-admitted direct-raw 2K lower-bound repetition on tmpfs.
6. Follow the measured branch; do not automatically design a new format or run 8K.
7. Keep upstream submission, exact-boundary work, canonical conversion, cross-backend work, vLLM, and Transformers deferred.

## Reporting discipline

Classify claims as:

- proven by retained automated test;
- measured once on Fred;
- independently measured upstream;
- confirmed by source review;
- inferred;
- untested;
- failed.

Current truthful status:

- persistent hybrid restoration is correct on the exact tested Qwen3.8-27B qwen35 model;
- the public KVX adapter remains fail closed on the tested unpatched runtime;
- 2K correctness holds with `2044/2048` reuse and a four-token tail;
- one-pass verified staging safely removes one complete 577 MiB payload read;
- 2K improves from 3.76× to 2.52× cold but remains economically unsuccessful;
- verified staging is now the dominant phase;
- actual K/V cache dtypes are missing from runtime advertisement and cache ABI identity;
- a direct pre-admitted raw restore is inferred to be near break-even but is unmeasured;
- NVMe, 8K, 32K, exact Qwen3.5/Qwen3.6 digests, and active draft/speculative state remain untested.
