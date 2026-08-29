# KV Rosetta research steer: turn the measured 2K lower-bound win into a fail-closed admitted store

Status basis: default-branch head 520509b583e64da8610c135a9da9a096bda0aef4.

This steer supersedes f081b53. The K/V-cache identity gap is closed, the patched import no longer performs the avoidable state-version save probe, the rebuilt 256 gate is sealed, and a retained direct-raw 2K run is 12.4% faster than cold prefill. The next smallest falsifiable experiment is not 8K: implement the narrowest local admitted-state contract that can reproduce that win without treating an arbitrary filename or prior hash as trusted.

## Mission

Deliver persistent exact-prefix restoration for the deployed 27B qwen35-family hybrid model across a complete llama-server restart, with exact runtime/artifact identity, fail-closed behavior on unpatched or incomplete runtimes, and request-path economics that beat native prefill.

Primary upstream evidence remains open as of 2026-08-29:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep checkpoint-persistence patch 0001 pinned. Do not post upstream during this track.

## Evidence now retained

### Runtime identity and fail-closed boundaries

Commits 1638e14 through 4613be8 establish and retain:

- live target K and V cache types from the runtime configuration;
- f16/f16 in the tested deployment, distinct from Q4_K model-weight quantization;
- target K/V types bound into CacheABIIdentity;
- missing or differing cache types withholding capability and refusing before state endpoints;
- active draft/speculative state still withheld;
- complete patched import using advertised sequence version 3 without a discovery save;
- unpatched hybrid import refusing before save or restore;
- reconciled phase accounting that fails the runner on an unexplained interval.

The rebuilt 256 record remains correct: patched cache_n=252 and prompt_n=4; same-source unpatched cache_n=0 and prompt_n=256.

### First measured request-path economic win

Retained record bench/direct-raw-2k-tmpfs.json, produced by committed runner cd3722b and committed at 520509b, records one research-only pre-admitted run on Qwen3.8-27B-UD-Q4_K_XL.gguf, architecture qwen35:

| Measurement | Result |
|---|---:|
| Native cold request | 1.731 s |
| Direct raw timed import | 1.081 s |
| Actual tail completion | 0.434 s |
| Direct import plus tail | 1.516 s |
| Ratio to cold | 0.876 |
| Request-path saving | 0.216 s / 12.4% |
| Reuse | 2,044 / 2,048 tokens |
| Uncovered tail | 4 tokens |
| Raw state size | 604,958,676 bytes |
| Untimed admission | 3.267 s |

Correctness checks include a real PID replacement, zero reuse before restore, field-for-field checkpoint metadata equality, cache_n equal to declared coverage, token/content/nonempty probability-vector parity, unchanged file facts and digest after restore, no temporary leftovers, and unpatched refusal from support evidence. Timed phases reconcile within 0.00003 s.

Classify this accurately: measured once on Fred, not yet a production path. Admission is intentionally excluded from request latency. At the observed 0.216 s per-request saving, its 3.267 s admission cost amortizes only after about 16 successful restores; retain that lifecycle fact in future records.

The patched llama.cpp source tree is recorded as modified with a source diff digest. Continue recording base head, patch-file digests, source diff digest, binary digests, and build flags; do not describe it as a clean upstream build.

## P0: smallest admitted-state contract

Build a local-only prototype around the raw ggsq/3+sckp/1 state. Do not change the portable KVX format and do not weaken its verification semantics.

The contract must distinguish two phases:

1. Admission, off the request path: validate every raw byte and all semantic identity, then atomically publish a state object into an adapter-owned private store on the same filesystem as the llama-server slot directory.
2. Restore, on the request path: resolve only an already-admitted object and restore it in place without payload extraction, copying, or a full payload read.

Use a full SHA-256 content address, not the current 128-bit filename abbreviation. Bind a small sidecar/manifest to that full digest and include at least:

- complete raw-state digest and size;
- sequence version and exact ggsq/3+sckp/1 tuple;
- checkpoint appendix offset, size, coverage, count, and positions;
- model content/weights digest;
- prompt token digest and token count;
- target K/V cache types;
- active state classes;
- adapter/runtime ABI identity;
- creation patch/binary identity sufficient to reproduce the state.

Admission must validate sequence framing, exact prompt tokens, the appendix at the declared offset, full payload digest, complete checkpoint metadata, allowlisted compound tuple, proven active state classes, and exact cache ABI. Hash while writing if a write is required; fsync the file and containing directory; publish data and manifest atomically or fail closed and clean up.

State the trust boundary explicitly. An admitted object may avoid rehashing only while it remains in an adapter-owned private directory and its pinned identity/metadata remain unchanged. A basename, read-only mode bit, prior hash, content-looking filename, or manifest alone is not admission.

Because llama-server restores by basename rather than file descriptor, directly address the name-to-object race. Before any live benchmark, retain red tests for symlink/path traversal, replacement after admission, changed inode/device/size/mtime/ctime/link count, non-private store permissions, incomplete atomic publish, manifest substitution, truncation, and payload corruption. If the prototype cannot make the filename lookup resistant within its stated local threat model, stop and evaluate fs-verity or an upstream restore-from-open-fd/verified-offset primitive; do not silently narrow the threat model.

## P1: public prototype gate

Expose the admitted-store operation through a clearly experimental adapter API. It must reuse the same support predicate and CacheABIIdentity as export/import.

Required red-test-first behavior:

- unpatched or incomplete hybrid runtimes refuse before opening/linking state or issuing save/restore;
- missing/different model, prompt, K/V dtype, protocol tuple, active state class, or ABI refuses before restore;
- an arbitrary raw file cannot enter the restore path without successful admission;
- corrupt/truncated bytes and altered metadata refuse during admission;
- a changed or replaced admitted object refuses before restore;
- restore-response metadata must equal admitted metadata field for field;
- mandatory reuse verification requires cache_n equal declared coverage and a 1–8 token uncovered tail;
- verification failure erases/restores safely and leaves no temporary objects;
- active draft/MTP/speculative configurations remain withheld;
- the request-time path performs no payload copy and no full payload read, enforced with retained byte/read-count instrumentation.

Do not call this zero-copy unless instrumentation proves no payload bytes are copied by KV Rosetta. Runtime-internal reads are expected and must remain separately attributed.

## P2: one falsifiable live 2K experiment

After code and offline tests are committed, run three process-owned repetitions on the same exact model, prompt, patched/unpatched binary pair, and tmpfs environment. Reuse one admitted object so admission is measured once and every restore crosses a fresh server process.

For every repetition require:

- new PID and zero reuse before restore;
- patched cache_n=2044 and prompt_n=4;
- complete restore-metadata equality;
- token, content, and nonempty probability-vector parity against native reuse;
- pristine-prefix restoration after the verification probe;
- unpatched refusal before file access/linking and before restore;
- unchanged admitted-object identity across the timed window;
- zero staging/copy bytes and bounded metadata-only request-path reads;
- phase sum reconciliation.

Retain raw per-repetition cold, restore, probe, pristine restore, tail, and total times. Report median and range; do not report only the best run.

Decision rule:

- Pass: median admitted-store import plus actual tail is below median native cold, at least 2 of 3 paired repetitions win, and all correctness/security gates pass.
- Marginal or noisy: if median is within 10% of cold or fewer than 2 paired wins occur, repeat only after explaining the variance; do not scale.
- Fail: if the median loses to cold, stop storage-format work and profile runtime restore, verification probe, and pristine re-restore separately.
- Any safety failure: fail the experiment regardless of speed.

Also retain admission time and the measured break-even restore count: ceiling(admission seconds / median per-restore saving). A request-path win is not automatically a lifecycle win.

## P3: only then test storage realism

If P2 passes, the next rung is the same admitted-store contract at 2K on the intended persistent filesystem/NVMe. Do not jump to 8K merely because tmpfs passes. Measure cold, admission, request-path total, and amortization under the exact filesystem and mount options.

Only after the 2K persistent-filesystem gate passes may 8K be considered. Keep 32K, 131K, canonical conversion, cross-backend work, vLLM, Transformers, and upstream submission deferred.

## Scope and reporting discipline

The proven model is exactly Qwen3.8-27B-UD-Q4_K_XL.gguf, architecture qwen35. Exact Qwen3.5-27B and Qwen3.6-27B model digests remain untested. Do not generalize the evidence beyond the tested digest and runtime ABI.

Classify claims as:

- proven by retained automated test;
- measured once on Fred;
- independently measured upstream;
- confirmed by source review;
- inferred;
- untested;
- failed.

Current truthful status:

- persistent hybrid restoration is correct on the exact tested 27B qwen35 model;
- patched reuse is 252/256 and 2044/2048, with a stable four-token tail;
- target K/V cache identity is now explicit and fail closed;
- the public KVX adapter remains correct but slower than cold because verified staging copies the payload;
- a research-only pre-admitted direct raw path is measured once at 12.4% faster than cold;
- that result justifies the smallest admitted-store prototype, not a general zero-copy claim;
- admission costs 3.267 s and is inferred to amortize after about 16 restores at the one-run saving;
- NVMe, production admitted-store behavior, 8K, exact Qwen3.5/Qwen3.6 digests, and active draft/speculative state remain untested.

## Required execution order

1. Write the admitted-store trust contract and red tests for atomicity, identity, TOCTOU, and unpatched refusal.
2. Implement only the local experimental admission and direct-restore path; leave portable KVX unchanged.
3. Commit code before generating evidence.
4. Run the three-repetition 2K tmpfs gate and retain the full record.
5. Follow the decision rule; do not automatically run NVMe or 8K.
