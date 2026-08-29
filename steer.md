# KV Rosetta research steer: correct the economic timer, then break-test 2K

Status basis: default-branch head 67efd9813f4ce068779eddf77d07d3bf474dd76a.

This steer supersedes 233a897. The compound-header bypass is sealed and the public KVX adapter now passes a process-owned patched/unpatched restart gate on the deployed Qwen3.8-27B qwen35 model. Do not repeat the 256-token correctness work. Correct two measurement-attribution defects in the newly parameterized runner, then run the next smallest 2K tmpfs experiment once. Use that result to decide whether repetitions or I/O work come next.

## Mission

Deliver persistent exact-prefix restoration for the deployed 27B qwen35-family hybrid model across a complete llama-server restart, with an auditable compound artifact, fail-closed behavior on unpatched or inconsistent runtimes, and measured request-path economics rather than raw restore timing alone.

Primary upstream evidence:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Keep llama.cpp patch 0001 pinned. Do not post upstream during this track.

## What is now complete

### The compound import bypass is sealed

At `4a5ba0f`, import now treats `blob.opaque_format` as authoritative, recognizes only exact tested compound tuples, and requires `coverage.format` to agree exactly. A compound blob needs complete integer coverage before staging or restore. Malformed strings, NaN, bools, lists, dictionaries, absent coverage, and plain/compound disagreement return `ok=false` rather than escaping as exceptions.

The retained tests rebuild and re-hash altered artifacts, so they pass container integrity and exercise semantic consistency rather than being rejected for unrelated corruption. `verify_reuse=False` cannot bypass any recognized compound path.

Keep `tests/test_compound_format_agreement.py`.

### Live export corrected a self-consistent parser error

The prior SCKP parser and all its fixtures assumed a 12-byte checkpoint record containing three int32 values. The patched writer emits 16 bytes: `n_tokens` is int64 and `pos_min`/`pos_max` are int32.

A real adapter export failed closed and exposed the error. At `5144bec` the parser was corrected to the writer's `<qii` layout, real observed framing was pinned, range checks were added, and the old 12-byte form was made a negative test.

This was a false negative, not a false positive: genuine hybrid exports were refused; malformed artifacts were not accepted.

Keep the live-framing fixture and the explicit old-layout rejection.

### Prefix artifacts now stop at the prefix

The first adapter experiment exported after eight generated tokens. The artifact held 263 sequence cells but its last checkpoint covered 252, leaving an 11-token tail. The earlier “four-token tail” result applies when replaying the 256-token prefix against a checkpoint at 252; it is not a universal constant after arbitrary generation.

At `a9e617e`, the public-adapter path now erases, performs a prompt-only prefill, and exports before generation. Do not widen the tail ceiling to accommodate post-generation state. A reusable prompt cache should contain the exact prompt boundary being named.

### The process-owned public-adapter 256 gate passes

The retained `bench/production-27b-matrix.json` was produced by committed runner `fd4e9d01fe0ad6f99ad4cacd7d01bebf2b44c326` on:

- `Qwen3.8-27B-UD-Q4_K_XL.gguf`;
- architecture `qwen35`;
- model digest `9bf3b07e1fb6531e91d970384cc3bdbc34b26dcfed993a743ecb3a9773aa8886`;
- exact 256-token prompt;
- same llama.cpp base `ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc` for patched and unpatched builds.

Patched public-adapter path:

- advertises OPAQUE and `ggsq/3+sckp/1`;
- exports an exact-prefix KVX artifact;
- stops server A and starts a distinct fresh server B;
- proves zero reuse before import;
- verifies the outer container;
- stages and restores the payload;
- verifies `cache_n=252`, `prompt_n=4`, and declared coverage 252;
- restores the pristine prefix after the verification probe;
- matches native reuse in generated tokens, content, and eight nonempty five-entry logprob vectors;
- removes the staged copy.

The retained adapter artifact is:

- sequence bytes: 173,679,168;
- checkpoint bytes: 156,894,416;
- payload bytes: 330,573,584;
- container overhead: 1,920;
- total bytes: 330,575,504.

Unpatched same-source path:

- advertises no hybrid capability;
- refuses explicit export with no POST inside the export window;
- refuses the patched compound artifact with `verify_reuse=True` and `False`;
- stages nothing and issues no restore POST;
- fails first on cache ABI identity, which is an earlier fail-closed boundary than the support predicate.

Capability discovery itself currently performs a state-version save probe. Keep its calls separate from the export/import refusal windows so “before POST” claims remain precisely scoped.

### 256-token economics identify the bottleneck

Measured once on Fred:

- native cold request: 0.588 s;
- full public-adapter import: 3.693 s;
- runtime restore within import: 0.250 s;
- container verification: 1.054 s;
- staging: 1.295 s;
- reuse probe: 0.212 s;
- pristine re-restore: 0.275 s.

The runtime restore is faster than the cold request. The public adapter still loses badly because it makes multiple passes over a 315 MiB artifact. At this size, container verification plus staging alone costs 2.349 s.

This proves correctness and identifies an I/O problem. It does not prove where the break-even context lies.

## P0: correct the runner before producing the 2K record

Commit `67efd981` parameterizes prompt tokens, repetitions, storage notes, K/V dtypes, RSS, VRAM, and a decision verdict. Two source-review corrections are required before trusting that verdict.

### 1. Time the actual post-adapter completion

The current verdict computes:

`adapter.import_seconds_end_to_end + patched.tail_completion_wall_s`

But `tail_completion_wall_s` times the completion after the separate raw-endpoint restore. The completion issued after the public adapter import is checked for parity but is not timed.

Those states should behave alike, but an economic record must measure the path it names rather than substitute a neighboring measurement.

Fix red-test first:

- time the deterministic completion immediately after successful adapter import;
- store it as a distinct `adapter_tail_completion_wall_s`;
- compute the verdict from `import_seconds_end_to_end + adapter_tail_completion_wall_s`;
- retain the raw-endpoint tail timer separately for diagnostic comparison;
- require both paths to have identical token/content/logprob results, but never use one path's timer as the other's.

### 2. Reconcile phase attribution with end-to-end time

The retained phases sum to about 3.085 s while `ImportReport.seconds` is 3.609 s, leaving roughly 0.524 s unclassified. Endpoint calls show a save during import because `state_version()` probes the emitted sequence format before restore.

The total end-to-end number already includes this cost, so the economic conclusion is conservative. The phase attribution is incomplete.

Before 2K:

- add an explicit preflight/state-version-probe phase, or an `other_preflight` phase that contains it;
- store phase-sum and unclassified seconds;
- require phase sum plus a small scheduling tolerance to reconcile with `ImportReport.seconds`;
- retain endpoint calls beside timings;
- do not hide the save probe inside runtime restore or staging.

Do not optimize the probe yet unless measurement shows its exact cost. If removing it later, use the runtime's advertised sequence version only when that advertisement is complete and exact, retain byte-derived format validation on export, and preserve fail-closed fallback behavior for unpatched runtimes.

Required retained runner tests:

1. verdict uses the timed post-adapter completion, not raw warm completion;
2. changing only the raw tail timer cannot change the adapter verdict;
3. every successful import second is assigned to a named phase within tolerance;
4. failed pre-staging imports still report no staging or restore phase;
5. the actual 256 record remains readable by the parameterized runner schema.

## P1: run one 2K tmpfs break-first experiment

After P0 is committed, run one clean repetition at exactly 2,048 prompt tokens on the same exact Qwen3.8-27B digest and same build pair.

Use tmpfs first. The purpose is to isolate compute, hashing, memory copying, and runtime restoration from deployable-device latency.

Retain:

- exact committed runner SHA and clean kv-rosetta worktree;
- source-tree and binary identities;
- exact tokenizer output count of 2,048;
- patched and unpatched owned process pairs;
- public-adapter artifact and complete compound coverage;
- patched reuse equal to declared checkpoint coverage;
- exact bounded tail and prompt accounting;
- token, content, and nonempty logprob parity against native reuse;
- unpatched export and cross-import refusals with endpoint-call evidence;
- actual K/V cache dtypes;
- sequence, checkpoint, container, and total bytes;
- first/second server peak RSS, runner peak RSS, and per-device VRAM;
- raw restore and raw tail;
- every adapter import phase;
- actual post-adapter tail completion;
- fully reconciled end-to-end verdict.

Decision rule:

`verified adapter import + actual post-adapter tail completion < native cold request`

This is one falsifiable break-first run, not a projection.

### Branch after the first 2K result

- Correctness fails: stop. Preserve the record and isolate the first violated invariant.
- Correctness passes and total is within 25% of cold: run two more tmpfs repetitions, then NVMe.
- Correctness passes and restore is clearly cheaper: run two more tmpfs repetitions, then NVMe.
- Correctness passes but adapter total is more than 1.25× cold: do not spend time on repetitions or 8K yet. Attribute bytes and phases, then attack the largest linear I/O pass.

If I/O is again dominant, the next smallest optimization hypothesis is a verified extraction primitive that validates header integrity first, then hashes the payload while writing the staged file in one pass, compares the digest before any runtime restore POST, and deletes the staged file on mismatch. This can remove the separate full-payload verification read without weakening fail-closed behavior.

Keep direct-from-container restore, raw artifact plus sidecar manifests, reflinks, and format redesign deferred until the one-pass verified staging result is measured.

## P2: scale only after the 2K branch decision

Run 8K only after:

- the 2K public-adapter path is correct;
- three repetitions are justified by the branch above;
- the measured economics show a credible path;
- RSS and VRAM remain bounded.

Run 32K only if 8K remains correct and favorable. Keep 131K deferred.

Test tmpfs and NVMe as distinct records. Never relabel one as the other with `--storage-note`; record the slots path and filesystem/mount identity so storage provenance is observable rather than asserted.

## P3: exact-model and state-class scope remains narrow

The behavior is proven on Qwen3.8-27B, architecture `qwen35`. Exact Qwen3.5-27B and Qwen3.6-27B model digests remain untested. Rerun the sealed 256 gate on the exact target digest before transferring the claim.

Draft/MTP/speculative serialization is not behavioral support. Keep those active classes withheld until each is actually active in a process-owned restart test, matches native reuse, survives required-blob corruption/removal tests, and is bound to model and launch identity.

## Required execution order

1. Add runner tests for actual post-adapter tail timing and phase reconciliation.
2. Correct the verdict and phase attribution.
3. Run all offline tests.
4. Commit the runner before evidence generation.
5. Run one 2K tmpfs patched/unpatched public-adapter repetition.
6. Follow the measured 2K branch; do not automatically run 8K.
7. Keep exact-boundary work, 131K, cross-backend, canonical conversion, vLLM, Transformers, and upstream submission deferred.

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

- the exact tested Qwen3.8-27B qwen35 model restores persistent hybrid state across a full restart;
- the public KVX adapter path is proven once at 256 tokens;
- same-source unpatched capability, export, and cross-import fail closed;
- token, content, probability-vector, coverage, process-replacement, cleanup, and artifact-integrity checks pass;
- the remaining compound-header bypass is closed;
- the 256-token adapter path is about 6.3× slower than cold because artifact I/O dominates, even though runtime restore itself is faster;
- the parameterized runner does not yet time the actual post-adapter tail in its verdict and leaves preflight time unattributed;
- no 2K economic record has landed;
- exact Qwen3.5/Qwen3.6 digests and active draft/speculative state remain untested.
