# KV Rosetta steer: verified restore before translation

Status basis: default-branch head 9c3763b7bad1733ce50b4d08a8186ff8b385273f.

This steer supersedes the earlier integration steer. Preserve the architectural work already landed, but change the immediate sequence to reflect the first real llama.cpp runtime proof and the failure exposed by the context ladder.

## Mission

Eliminate repeated large-prefix prefill by restoring a previously compiled cache only when the system can prove all of the following:

1. the prompt token IDs are exact;
2. the model and tokenizer contents are exact;
3. the runtime cache ABI and state-file version are exact;
4. the artifact is structurally intact and bound to those identities;
5. the intended runtime slot actually reused the expected prefix;
6. the restored path is faster or otherwise operationally preferable to native prefill.

Every uncertainty fails closed to native prefill. A server-reported restore count is diagnostic data, not proof of a usable cache.

## Evidence now established

The repository has crossed an important boundary:

- A real llama.cpp HTTP save, erase, restore, and reuse cycle succeeded for a 256-token prompt on Qwen2.5-3B Q4_K_M.
- Generated text and token IDs matched across all tested positions.
- The maximum reported top-5 probability delta was zero.
- The erase-and-rerun control proved the cache had actually been removed before restoration.
- A real GGSQ fixture pinned the file magic and sequence-state layout to captured bytes.
- The 2K, 8K, and 32K ladder exposed a fail-open adapter defect and retained the correction.

Do not weaken or replace this proof.

The same ladder also established limits:

- For the tested 3B model with f16 KV, full restore was slower than native prefill at 2K, 8K, and 32K.
- The 32K artifact was about 1.1 GB and took about 7.99 seconds to restore versus about 3.41 seconds to prefill.
- The 27B run did not restore a reusable cache. cache_n remained zero, so its timing says nothing about restore economics.
- The projected q4_0 advantage is an estimate. It is not a result until the actual production KV type and model are measured.
- The reported 206 passing tests have no GitHub CI status attached to the head.

## Steering decision

Stay on the same-runtime llama.cpp path until it is both correct and economically measured on the production-class 27B configuration.

Do not add another backend adapter. Do not begin CUDA-to-HIP translation. Do not claim cross-backend, cross-model, canonical, or 131K support.

The immediate work is to remove the remaining ways an artifact can be mislabeled, restored into the wrong place, partially reused, or accepted under a false identity.

## P0-A: make the opaque format truthful

The adapter currently labels every opaque state as ggsq/3 even though the captured runtime writes sequence-state version 2. Remove the hardcoded format claim.

On export:

1. save the slot;
2. parse the actual file envelope;
3. require the real qsgg file magic;
4. read the emitted sequence-state version;
5. label the artifact with that exact envelope and version;
6. bind that version into CacheABIIdentity and ArtifactKey.

On import:

- require an exact opaque format and sequence-state version match;
- require the exact runtime revision and build identity used to write the artifact;
- refuse a version supported by the parser when the live runtime cannot load it;
- never translate version 2 to version 3 by relabeling it.

File and in-process buffer envelopes are separate formats. Detection must return unknown for an unknown blob. parse_buffer_envelope must require a known internal IO magic supplied by the exact runtime contract. It must not treat arbitrary first bytes as a plausible sequence ID.

Acceptance tests:

- the committed real fixture identifies as the exact emitted GGSQ version;
- synthetic version-2 and version-3 files remain distinct;
- an artifact mislabeled with the other version is refused before the restore endpoint is called;
- random bytes, qsgg-like truncations, and unknown buffer magic are refused;
- no non-qsgg blob is automatically classified as an in-process buffer.

## P0-B: prove reuse in the exact slot

Every save, erase, restore, probe, and completion call must specify the intended slot ID. The current post-restore probe is not sufficient for a multi-slot server if it can be scheduled onto a different slot.

Recover the exact prompt token IDs and derive the expected reusable-prefix count. Import succeeds only when the intended slot reports the complete expected reuse contract. cache_n greater than zero is insufficient.

For the current llama.cpp behavior, record and assert the relationship among:

- exact artifact token count;
- cache_n on the verification request;
- prompt_n on the verification request;
- any BOS, final-token, or sampler-specific adjustment.

Do not hide a mismatch behind a loose tolerance. Encode any legitimate one-token adjustment explicitly and test it.

The verification request must not leave an extra generated token appended to the restored prefix. Use a non-mutating inspection path if the runtime provides one. Otherwise probe, then restore the original artifact again before returning success.

Retained tests must include:

- restore to slot A and verify full reuse on slot A;
- prove slot B does not satisfy slot A's verification;
- partial-prefix reuse is refused;
- positive cache_n below the expected prefix is refused;
- n_restored positive with cache_n zero is refused;
- verification leaves the slot at the exact imported prefix;
- greedy token IDs and probability vectors remain identical after the verified import.

## P0-C: connect identity, KVX, ArtifactStore, and the daemon

The llama.cpp adapter must use the identity system already present in the repository. Its current filename, chat-template, BOS, and EOS hash is not a model-content identity.

ModelIdentity must derive from:

- model weight contents or a previously validated content digest;
- tokenizer contents;
- chat-template contents;
- architecture;
- ordered adapter contents and scales when present.

Renaming identical weights must preserve identity. Replacing a file behind the same name must change identity. Cache expensive digests using strong file metadata and shard inventories; do not hash a 100-plus-GB model on every request.

PromptIdentity must bind the tokenizer identity, exact serialized token IDs, token count, and token-ID digest. The adapter must not emit an empty tokenizer ID or empty token digest.

The exported manifest must not use empty strings and zeros as stand-ins for unknown model, prompt, KV, producer, or runtime fields. Either supply the value required by that representation or model its absence explicitly without making a canonical-state claim.

Complete the end-to-end chain:

1. construct PromptIdentity, ModelIdentity, and CacheABIIdentity from live evidence;
2. construct the representation identity from the parsed opaque envelope;
3. compute ArtifactKey;
4. embed the component identities and final key in KVX;
5. recompute and verify them from the file;
6. publish only through ArtifactStore;
7. retrieve through ArtifactStore for import;
8. migrate daemon production paths away from the legacy prompt-only Store.

No adapter may receive unverified artifact bytes. No production lookup may key only on the CFRProxy prefix fingerprint.

Acceptance tests:

- same filename with changed weights produces a different model identity;
- renamed identical weights retain identity;
- changed tokenizer, template, adapter order, adapter scale, runtime revision, KV dtype, context mode, or state version changes the appropriate identity;
- a valid artifact stored under any wrong identity is refused;
- every representation of one prompt can coexist;
- the daemon retrieves the exact composite artifact;
- repository search finds no production prompt-only artifact lookup.

## P0-D: eliminate whole-artifact memory copies

The current adapter reads the full slot file into Python bytes and creates additional full copies while wrapping it in KVX. That is already wasteful at 1.1 GB and is unacceptable at 131K.

Implement bounded-memory file-to-container and container-to-file paths:

- incremental payload and segment hashing;
- chunked reads and writes;
- file preallocation where useful;
- unique temporary paths;
- atomic publication;
- bounded staging;
- cleanup of saved, restore, and failed temporary files;
- peak-RSS telemetry.

Use a streaming or file-range API between the adapter, KVX, and ArtifactStore. Do not call read_bytes on a cache artifact. Do not join the entire payload into one Python bytes object.

The hardware benchmark must test both:

- tmpfs or RAM-backed slot storage, which isolates serialization and runtime import cost;
- NVMe-backed storage, which measures the deployable persistent path.

Report both. Do not mix them into one timing.

## P0-E: make conformance claims non-vacuous in every dimension

The conformance suite improved, but testing only the intersection of advertised import and export representations does not prove every advertised import capability.

For each advertised capability, independently exercise:

- every import representation;
- every export representation;
- every opaque format version;
- every dtype;
- every staging mode that changes behavior;
- every supported slot mode.

An adapter advertising canonical export plus canonical and opaque import must prove both imports. One successful canonical round trip does not validate the opaque claim.

The content-derived identity test must detect basename-derived hashes. Merely rejecting values that resemble absolute paths is insufficient. Use controlled rename and content-mutation fixtures.

## P0-F: add CI before treating test counts as evidence

Add GitHub Actions for:

- supported Python versions;
- unit and integration tests;
- malformed-container corpus;
- deterministic identity vectors;
- static or type checks;
- package installation and CLI smoke tests.

Keep the live llama.cpp test gated from ordinary hosted runners, but record Fred hardware runs as machine-readable artifacts with:

- repository commit;
- llama.cpp commit and build information;
- model and tokenizer identities;
- cache ABI;
- slot ID and slot count;
- context length and exact token digest;
- KV dtype and bytes per token;
- artifact digest and size;
- native prefill time;
- export, verify, read, restore, and post-restore verification times;
- total user-visible restore latency;
- host RAM and VRAM peaks;
- cache_n, prompt_n, parity result, and first divergent token.

## P1: isolate and fix the 27B failure at 256 tokens

Do not resume the 27B ladder at 2K or higher. Reproduce cache_n zero at 256 tokens with the exact production model, actual kvarn4 or q4_0 KV configuration, and exact runtime build.

Capture:

- save and restore API responses;
- server logs;
- parsed envelope version;
- exact slot ID;
- token count and token digest;
- runtime and cache-ABI identities;
- artifact size and digest;
- all state-affecting launch flags.

Classify the failure before changing code. Likely categories include runtime/state-version mismatch, incompatible KV type, GPU-offload layout, slot selection, unified-KV behavior, context configuration, or artifact corruption. Change one variable at a time and retain the smallest failing artifact or header slice that reproduces the violated contract.

The 27B path is not working until save, erase, cold-control, restore, exact-slot full-prefix reuse, and output parity all pass.

## P2: measure actual q4 economics

After the 27B 256-token proof passes, run this ladder:

- 256
- 2K
- 8K
- 32K

Run 131K only after 32K passes and projected memory use fits the bounded-I/O design.

For each rung, use the actual production model and KV quantization. Run at least three clean repetitions and report medians plus the range. Record tmpfs and NVMe separately.

Use this accounting:

total restore latency =
artifact read + integrity and identity verification + runtime restore +
reuse verification + any required second restore that returns the slot to a pristine prefix

Compare total restore latency with native prefill latency. Server-only restore time is not the user-visible result.

Report:

- milliseconds saved or lost;
- speedup ratio;
- artifact bytes per token;
- effective read throughput;
- effective restore throughput;
- break-even context length;
- peak RAM and VRAM;
- whether parity held.

Do not promote the current q4_0 projection as measured performance. Replace it with actual results.

## Gates for later work

Canonical extraction and cross-backend translation remain gated until all of these are true:

- the exact opaque version is bound into identity;
- the correct slot reuses the complete expected prefix;
- verification leaves a pristine prefix;
- model and tokenizer identities are content-derived;
- KVX and ArtifactStore recompute and enforce the same ArtifactKey;
- the daemon uses the composite store;
- cache I/O is bounded-memory;
- CI is green;
- the real 27B q4 path passes through 32K with parity;
- total restore economics are measured on tmpfs and NVMe.

After those gates:

1. scale the proven same-runtime path to 131K;
2. implement a canonical llama.cpp extractor with explicit layouts and dtypes;
3. validate canonical round trips within llama.cpp;
4. add Transformers only as a reference/calibration path;
5. attempt same-revision CUDA-to-HIP transfer;
6. add vLLM only after the canonical contract survives retained tests;
7. add standard-family mapping;
8. add explicit MLA semantics before any DeepSeek compatibility claim.

## Required execution order

1. Remove hardcoded ggsq/3 and make envelope detection fail closed.
2. Bind every request to an exact slot and require full-prefix reuse.
3. Ensure verification leaves the imported prefix unchanged.
4. Wire content-derived identities into the adapter manifest.
5. Bind KVX to ArtifactKey and migrate production storage to ArtifactStore.
6. Replace whole-artifact Python copies with bounded streaming.
7. Expand conformance and add GitHub CI.
8. Fix the 27B path at 256 tokens.
9. Measure the actual 27B q4 ladder on tmpfs and NVMe.
10. Scale to 131K only if the 32K evidence supports it.
11. Begin canonical and cross-backend work only after every gate above is satisfied.

## Reporting discipline

Every status report must separate:

- proven by retained automated test;
- measured once on Fred;
- inferred from code or arithmetic;
- untested;
- failed.

Include exact commits and configurations. Do not report restore success from n_restored alone. Do not report economic success from estimated q4 artifact size. Preserve failures and first-divergence evidence as fixtures or machine-readable benchmark records.
