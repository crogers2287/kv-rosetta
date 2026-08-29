# KV Rosetta research steer: remove the measured support-check tax without weakening refusal

Status basis: default-branch head 767bc5d1fb431808423854fc51ea08a2fc54c7b0.

This steer supersedes 9a883d9. The experimental admitted store and adapter path landed with strong retained safety tests, but the truthful three-repetition 2K gate failed 0/3. Profiling localizes the entire regression to hybrid_support reading the whole GGUF metadata block on every restore. The next experiment is therefore a narrow parser optimization plus one missing ownership guard, followed by the same paired 2K gate. Do not cache away, pre-warm away, or move the fail-closed support decision outside the measured path.

## Mission

Deliver persistent exact-prefix restoration for the deployed 27B qwen35-family hybrid model across a complete llama-server restart, with exact artifact/runtime identity, fail-closed behavior on unpatched or incomplete runtimes, and request-path economics that beat native prefill.

Primary upstream track:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep checkpoint-persistence patch 0001 pinned. Do not post upstream during this track.

## Evidence now retained

### The admitted-state contract is implemented

Commits cd78b22 and 464c5ce add an experimental local-only store and adapter path while leaving portable KVX unchanged.

Admission:

- hashes while copying every raw byte;
- uses the full SHA-256 as the object address;
- validates the tested ggsq/3+sckp/1 tuple, exact prompt tokens, byte counts, appendix at the declared offset, complete checkpoint coverage, model identity, K/V cache types, active state classes, and cache ABI;
- publishes through fsync and atomic replace into a mode-0700 store;
- pins device, inode, size, mtime_ns, ctime_ns, and link count.

Restore:

- reuses the same hybrid support predicate and CacheABIIdentity as KVX;
- refuses arbitrary, missing, replaced, symlinked, traversing, truncated, hard-linked, or manifest-mismatched objects;
- reads 965 metadata bytes and zero state-payload bytes in KV Rosetta;
- compares restore metadata field for field;
- requires cache_n equal admitted coverage and a 1–8-token uncovered tail;
- restores the pristine prefix after the mandatory reuse probe;
- erases/refuses on failure;
- withholds unpatched hybrid support before touching the store or issuing any endpoint call.

The code truthfully documents the remaining basename race: llama-server opens the object after KV Rosetta checks it. The stated local threat model excludes another writer acting as the same UID or root. Do not broaden this into a general untrusted-file claim without fs-verity or an upstream restore-from-open-fd primitive.

### The real admitted-store economic gate failed

Retained bench/admitted-store-2k-tmpfs.json records three paired repetitions:

| Measurement | Result |
|---|---:|
| Median admitted total | 2.167 s |
| Admitted range | 2.029–2.277 s |
| Median native cold | 1.726 s |
| Cold range | 1.724–1.730 s |
| Paired wins | 0 / 3 |
| Median ratio to cold | 1.255 |
| Reuse | 2,044 / 2,048 every run |
| Request-path state payload read | 0 bytes |
| Unpatched endpoint calls | 0 |

All correctness and refusal gates passed. This is an economic failure, not a correctness failure.

### Profiling identifies one exact cause

Retained bench/admitted-store-2k-profiled.json records another three repetitions:

| Median/typical phase | Seconds |
|---|---:|
| resolve_support | about 0.652 |
| resolve_store | below 0.001 |
| resolve_abi | below 0.001 |
| resolve_identity | about 0.001 |
| runtime restore | about 0.429 |
| reuse probe | about 0.205 |
| pristine re-restore | about 0.441 |
| actual tail | about 0.426 |

Median total is 2.182 s against 1.730 s cold, again 0/3 wins. Restore, probe, pristine restore, and tail match the earlier lower-bound run within noise. The whole loss is resolve_support.

Source review confirms why: prefix_reuse_support calls gguf.architecture(model_path). gguf.read_metadata filters what it returns but still decodes every metadata value, including large tokenizer arrays, before returning general.architecture. This costs roughly 0.65 s per restore.

Audit correction: the earlier 1.516 s direct-raw “win” excluded hybrid_support from its timed window. It proved the restore mechanics can win, but it was not a like-for-like request-path result. Do not report it as the admitted-store performance.

## P0: exact-key early-exit architecture lookup

Preserve the fail-closed support predicate on every restore. Optimize only the evidence retrieval.

Add a dedicated exact-key GGUF lookup used by architecture():

1. parse the GGUF header normally;
2. walk keys in file order;
3. decode general.architecture when encountered;
4. return immediately after that exact value is decoded;
5. do not decode later tokenizer arrays or tensor metadata;
6. return unknown/refuse if the key is absent, malformed, duplicated ambiguously, truncated, or not a string.

Do not:

- memoize hybrid_support across server generations;
- trust a filename or model family string;
- skip hybrid_support for an admitted object;
- pre-run the support check outside the economic timer;
- use model weight quantization as architecture or cache-dtype evidence;
- special-case qwen35 as automatically supported.

The classification after lookup remains unchanged: hybrid qwen35 still requires the complete, allowlisted checkpoint protocol, target-only active state, and exact K/V cache types. An unpatched qwen35 runtime must still refuse before store access and endpoint calls.

Required red tests:

- an architecture key placed before a deliberately huge or invalid later metadata value returns without reading the later value;
- read-count instrumentation proves architecture() consumes only the prefix through the exact key;
- missing, wrong-type, duplicate-conflicting, and truncated architecture metadata fail closed;
- ordinary, recurrent, and hybrid classifications are unchanged;
- a patched qwen35 complete protocol remains supported;
- unpatched qwen35 remains withheld with zero state endpoint calls;
- repeated calls do not rely on stale cached architecture evidence.

If the exact key appears too late in this specific GGUF to remove the measured cost, stop and use the already-advertised live model_arch from /props only after retaining equivalence tests against the GGUF value and refusing missing/disagreeing values. Do not silently switch evidence sources.

## P0 safety correction: enforce the documented store owner

AdmittedStore says the store must be owned by the current user, but _require_private currently checks only mode 0700. Add an explicit st_uid == os.geteuid() requirement and a red test that a mode-0700 directory with a different reported owner is refused.

Do not expand this into unrelated store redesign. Keep the documented same-UID/root residual risk explicit.

Also retain the existing guards for mode, confinement, O_NOFOLLOW, pinned facts, link count, manifest binding, atomic publication, and failed-admission cleanup.

## P1: rerun the identical paired 2K gate

Commit the parser and owner fixes before generating evidence. Then run exactly three paired repetitions using:

- the same Qwen3.8-27B-UD-Q4_K_XL.gguf digest;
- the same 2,048-token prompt digest;
- the same patched/unpatched binaries and patch digests unless a rebuild is genuinely required;
- the same tmpfs store;
- one admission reused across fresh server processes;
- the same cold-versus-admitted timing boundaries.

For every repetition require:

- a new PID and zero reuse before restore;
- cache_n=2044 and prompt_n=4;
- complete restore-metadata equality;
- token, content, and nonempty probability-vector parity;
- pristine-prefix restoration;
- unchanged admitted-object facts;
- zero KV Rosetta payload bytes and bounded metadata reads;
- phase reconciliation;
- unpatched refusal before store access and with zero endpoint calls.

Retain resolve_support as its own phase. Record its read count and time. Report all raw repetitions, median, and range.

Decision rule:

- Pass: resolve_support median is below 0.010 s, admitted total median beats cold median, at least 2 of 3 paired runs win, and every correctness/safety gate passes.
- Parser fixed but economics still lose: stop and profile the remaining restore/probe/pristine/tail phases; do not run NVMe or 8K.
- Support lookup still expensive: retain the failed record and test live /props architecture equivalence as the next smallest experiment.
- Any refusal regression or safety failure: fail regardless of speed and restore the previous fail-closed behavior.

If P1 wins, report admission cost and the measured break-even number of restores. A request-path win is not automatically a lifecycle win.

## P2: only after a real pass

After the admitted-store path passes the paired tmpfs gate with the support predicate included, the next rung is the same 2K experiment on the intended persistent filesystem/NVMe. Do not run 8K first.

Before calling the local store production-ready, separately close or explicitly accept:

- the same-UID/root basename race, preferably with fs-verity or an upstream open-fd primitive;
- exact model variants beyond the tested digest;
- active draft/MTP/speculative state;
- reproducible checkpoint-patch/implementation identity in artifact records.

Keep 8K, 32K, 131K, canonical conversion, cross-backend work, vLLM, Transformers, portable-KVX redesign, and upstream submission deferred.

## Truthful current status

- Persistent hybrid restoration is correct on the exact tested Qwen3.8-27B qwen35 model.
- Patched reuse is 252/256 and 2044/2048 with a stable four-token tail.
- Target K/V cache identity is explicit as f16/f16 and unpatched behavior remains fail closed.
- The experimental admitted store passes its retained correctness and adversarial tests.
- The real admitted-store 2K economic gate fails 0/3 at about 1.26x cold.
- Profiling attributes essentially all avoidable loss to decoding the entire GGUF metadata block during hybrid_support.
- The lower-bound run omitted that support cost and is mechanics evidence, not a full request-path win.
- The store does not yet enforce its documented current-user ownership condition.
- NVMe, a true admitted-path economic win, 8K, exact Qwen3.5/Qwen3.6 digests, and active draft/speculative state remain untested.

## Required execution order

1. Add exact-key early-exit architecture parsing and isolated fail-closed/read-count tests.
2. Add the current-user store-owner guard and red test.
3. Run the full offline suite and commit code.
4. Run the identical three-repetition 2K tmpfs gate with resolve_support timed.
5. Follow the decision rule; do not automatically run NVMe or 8K.
