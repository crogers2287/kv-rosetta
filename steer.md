# KV Rosetta re-steer

Status basis: default-branch head e433052, after commits 29d55d0, 6e7c1fc, and e433052.

## Mission

Eliminate repeated large-prefix prefill by safely restoring or translating previously compiled model state.

The product is not the file format. The product is an end-to-end path that:

1. identifies the exact prompt, model, cache ABI, and artifact;
2. proves the stored bytes match those identities;
3. imports them into a compatible live runtime;
4. verifies token behavior;
5. falls back to native prefill on every uncertainty.

No artifact reaches a runtime import call before structural verification, identity verification, compatibility checks, and any required translation certificate pass.

## What is now complete

The following architectural corrections are real progress and should be preserved:

- Gate holdout is a minimum and invalid thresholds fail closed.
- Header semantics are covered by integrity checking.
- Store paths require full lowercase SHA-256 values and remain under the store root.
- BF16 conversion uses round-to-nearest-even.
- KVX v2 has a multi-segment payload description.
- Prompt, model, cache-ABI, and artifact identities are separate types.
- A composite-key ArtifactStore permits several representations of one prompt.
- Older KVX containers remain readable.

Do not redo these from scratch. Harden and connect them.

## Current truth

The new parts are still parallel scaffolding rather than an operating system:

- ArtifactStore is referenced only by its implementation and tests.
- The legacy Store remains active in the codebase.
- ArtifactStore accepts an arbitrary file and a separately supplied ArtifactKey.
- The KVX header does not provide an enforced end-to-end binding to ArtifactKey.
- Identity dataclasses allow empty or malformed semantic fields.
- GateBinding is optional and can be entirely empty.
- read() can expose an artifact before verify() runs.
- write_raw() can serialize bytes whose physical dtype disagrees with the manifest.
- segmented writing joins the full cache into one Python bytes object.
- no real llama.cpp state has been exported and restored.
- no GitHub CI status is attached to the head commit.

The self-reported 165 passing tests show useful local coverage. They do not prove a live cache path.

## Operating method

Use break-first development:

1. Reproduce one concrete failure at the smallest possible context.
2. Save the exact configuration, fixture, logs, and first divergent token.
3. Fix the narrowest violated contract.
4. retain the failure as a regression test.
5. scale only after the small case is clean.

Context ladder:

- 256 tokens: parsing, identity, import, and first-logit proof
- 2K: functional round trip
- 8K: conversion and transfer behavior
- 32K: memory and throughput behavior
- 131K: target workload

Do not begin a 131K run to discover a parser or identity failure.

## P0-A: make identity, container, and store one chain

The next commit must connect the pieces end to end.

### KVX identity header

KVX v2 must carry enough canonical identity information to recompute:

- PromptIdentity digest
- ModelIdentity digest
- CacheABIIdentity digest
- representation digest
- mapper ID
- final ArtifactKey digest

Store both the component digests and the canonical fields required to audit them. Reject disagreement between a supplied digest and recomputed fields.

The artifact payload digest, representation-schema digest, and artifact-key digest are different concepts. Keep them separate:

- payload digest identifies bytes;
- representation digest identifies segment roles, dtypes, shapes, layouts, and quantization semantics;
- artifact key identifies prompt plus model plus cache ABI plus representation plus mapper.

Do not call a digest a representation digest if it changes solely because payload values change.

### Verified storage

ArtifactStore.put must:

1. require a real KVX file;
2. call a non-raising verifier;
3. parse it through a verified-open path;
4. recompute all identity components from the header;
5. recompute ArtifactKey;
6. compare it with the requested key;
7. reject mismatches before copying;
8. copy through a unique temporary file;
9. atomically publish the file and database record.

The current test fixture containing only KVX1 plus zero bytes is not an acceptable stored artifact.

Add tests proving rejection of:

- random bytes;
- truncated KVX;
- valid KVX stored under the wrong prompt;
- valid KVX stored under the wrong model;
- valid KVX stored under the wrong cache ABI;
- valid KVX stored under the wrong mapper;
- header identity changed without updating integrity;
- two concurrent writers for the same key;
- two concurrent writers for different keys.

### Store migration

There must be one production store API.

Migrate required pruning, touching, lookup, byte accounting, and LRU behavior to ArtifactStore. Update daemon code and every non-test caller. Then remove or explicitly quarantine the legacy Store so new code cannot silently use prompt-only identity.

Do not leave index.sqlite and artifacts.sqlite as competing sources of truth.

Acceptance:

- repository search shows production code using ArtifactStore;
- no production path stores by cfrproxy prompt fingerprint alone;
- all representations for one prefix coexist;
- pruning operates on actual artifact bytes;
- failed file operations cannot leave an untracked artifact or a dead database row.

## P0-B: enforce content-derived identities

The current identity classes describe the right fields but do not validate them.

Add fail-closed construction or validation.

### PromptIdentity

Require:

- non-empty tokenizer identity;
- exact token-ID SHA-256;
- positive token count;
- valid corpus fingerprint when supplied.

A prompt artifact is not exact unless its actual token IDs are known.

### ModelIdentity

Require:

- non-empty architecture;
- valid weights content digest;
- valid tokenizer content digest;
- valid chat-template digest where applicable;
- ordered adapter identity including each adapter digest and scale.

Free-form notes must not participate in semantic identity. Adapter ordering must not be sorted away if runtime composition can depend on order.

Do not hash 100-plus GB on every startup. Compute a content digest once, cache it with strong file metadata, and invalidate it when size, timestamps, shard list, or content evidence changes. A path alone never becomes identity.

### CacheABIIdentity

Validate types and require all state-affecting fields for the selected runtime, including:

- runtime revision;
- exact state format and envelope version;
- K and V physical dtype;
- context kind;
- position scheme;
- partial rotary dimensions;
- SWA or hybrid-memory behavior;
- value transposition or equivalent layout flags;
- host versus on-device opaque state;
- byte order;
- runtime-specific cache flags.

Canonical JSON hashing must reject NaN and infinity. Equivalent mappings must hash identically regardless of input dictionary order.

Acceptance:

- empty identities cannot produce a storable ArtifactKey;
- moved identical weights retain identity;
- changed weights behind the same path change identity;
- a cache-affecting runtime flag changes CacheABIIdentity;
- human notes do not change ModelIdentity;
- adapter scale or order changes identity when semantically relevant.

## P0-C: close remaining container gaps

Add a verified-open API and make it the only path adapters receive.

The verifier must:

- impose a maximum header size;
- reject negative, non-integer, boolean, overflowed, and nonsensical numeric fields;
- catch every malformed-input exception and return a reason;
- validate payload offset begins after the aligned header;
- validate exact file-length or explicit trailing-byte policy;
- validate segment bounds and non-overlap;
- validate segment hashes;
- validate representation digest;
- validate header identity and ArtifactKey;
- validate raw dtype, shape, byte order, and byte count together.

write_raw must reject a NumPy array whose physical dtype does not match the declared KV dtype. It must serialize explicitly little-endian data.

Add coherent-hostile tests: valid JSON, valid header hash, and deliberately invalid numeric or semantic fields. Random-byte fuzzing alone cannot reach these deeper branches.

## P0-D: distinguish a measurement from a certificate

The current GateBinding defaults allow an admitted result with no meaningful binding.

Keep metric calculation usable for experiments, but a reusable translation certificate must require:

- valid source model digest;
- valid target model digest;
- valid source artifact digest;
- valid target representation or artifact digest;
- non-empty mapper implementation and weights digest;
- calibration and holdout-token digest;
- a digest of the complete threshold policy;
- policy version;
- measured metrics.

Changing any threshold must invalidate the certificate even if a human forgets to increment a version string.

An empty GateBinding may produce an unbound measurement. It must never produce a certifiable artifact.

## P0-E: implement explicit llama.cpp envelopes

Do this only after P0-A through P0-D are retained in tests.

Support distinct formats rather than guessing offsets:

### In-process sequence-state buffer

Parse and verify:

- internal IO magic;
- source sequence ID;
- state body.

### Sequence-state file

Parse and verify:

- GGSQ magic;
- sequence-state version;
- token count;
- exact token IDs;
- state body.

Capture small fixtures from the exact llama.cpp revision used on Fred. Include ordinary values, transposed values, empty streams, multiple sequences, unsupported cell extensions, unknown GGML types, truncation, and trailing bytes.

The parser must report exactly how many bytes were consumed and reject unexplained remainder.

## P0-F: make capabilities directional and prove them

Replace the single maximum tier with explicit live capabilities:

- export encodings and formats;
- import encodings and formats;
- supported dtypes;
- supported cache semantic families;
- supported opaque envelope versions;
- staging modes;
- zero-copy restrictions;
- device and runtime revision.

A capability is a claim the conformance suite must prove.

An import-capable adapter that always returns a failed ImportReport must not pass conformance. Every advertised import/export pair requires at least one successful round trip.

## P1: bounded-memory I/O

The current segmented writer constructs a full list of chunks and joins them into one large bytes object. That cannot survive large contexts efficiently.

Implement:

- file preallocation;
- incremental header and payload hashing;
- memoryview or chunked writes;
- per-segment streaming;
- unique temporary files;
- bounded host staging;
- fsync policy before atomic publication;
- peak-RSS telemetry.

Do not require two additional full-cache copies in host RAM.

## Required CI

Add GitHub Actions before relying on the test count.

Required jobs:

- supported Python-version matrix;
- unit and integration tests;
- static or type checks;
- malformed-container corpus;
- deterministic identity vectors;
- package installation and CLI smoke test.

Hardware tests may remain a separate Fred workflow, but its results must be recorded with:

- exact commit;
- model identity;
- runtime revisions;
- cache ABI;
- context size;
- artifact digest;
- native prefill time;
- export, verify, conversion, and restore time;
- peak host RAM and VRAM;
- first divergent token or parity result.

## Execution order from this head

1. Bind identities into KVX and enforce them in ArtifactStore.
2. Migrate production callers from legacy Store.
3. Add strict identity validation and deterministic vectors.
4. Add verified-open and close raw dtype/numeric validation gaps.
5. Split gate measurements from bound certificates.
6. Add CI.
7. Implement GGSQ API and file envelopes with real fixtures.
8. Implement directional capabilities and meaningful conformance.
9. Prove a 256-token same-model, same-runtime, same-backend llama.cpp round trip.
10. Scale through 2K, 8K, and 32K.
11. Prove same-revision CUDA-to-HIP canonical transfer with matching KV configuration.
12. Scale to 131K.
13. Add Transformers as reference/calibration.
14. Add vLLM only after the canonical contract survives llama.cpp.
15. Begin standard-family mapping.
16. Add explicit MLA semantics before claiming DeepSeek compatibility.

Do not fan out runtime adapters yet. The immediate work is integration and enforcement, not more parallel abstractions.

## Definition of the next milestone

The integration milestone is complete only when:

- one verified KVX artifact embeds complete identities;
- its ArtifactKey is recomputed from the file;
- ArtifactStore refuses the same file under every wrong identity variation;
- the daemon retrieves it through ArtifactStore;
- no production code uses prompt fingerprint as artifact identity;
- malformed coherent inputs never raise;
- tests run under GitHub CI;
- the repository contains no claim that a live cache transfer has occurred.

After that, the next milestone is one real 256-token llama.cpp same-backend save and restore with exact token IDs and token-for-token logit parity.
