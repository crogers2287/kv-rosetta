# KV Rosetta research steer: validate 8K, then canonicalize hybrid state

Status basis: default-branch head 21eedeeac7f7739a715c6a0d2ab8436847a84612.

This steer supersedes 7ded4e1. The project remains realigned on portable agentic-prefix reuse. New source evidence changes the immediate execution details: the 8K harness still contains two 2K-only acceptance/record constants, and the source-derived ggsq/3+sckp/1 inventory proves that canonical extraction needs the bound source GGUF plus new canonical representations for recurrent layers and checkpoints.

## Mission

Deliver one canonical KVX prefix artifact that a sidecar can transform for multiple cache dtypes, inference runtimes, and CUDA/ROCm/Vulkan backends, with quality-gated cross-model transformation later.

The opaque llama.cpp admitted-store path remains a strict source/runtime-specific fast path. Do not relax its CacheABIIdentity, compound-tuple allowlist, model identity, checkpoint metadata, owner/file guards, or unpatched-runtime refusals to simulate portability.

Primary hybrid-runtime track:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep checkpoint patches 0001 and 0002 pinned and identified. Do not post upstream.

## Evidence landed since the realignment

### Persistent 2K foundation

The retained ext4/NVMe natural-cache gate passed 3/3:

- median restored total 1.486 s versus 1.731 s cold;
- 2,044/2,048 reuse;
- output and nonempty probability-vector parity;
- complete checkpoint metadata equality;
- unpatched refusal before store access with zero state-endpoint calls.

This is process-restart evidence under natural Linux page-cache state, not host-restart evidence.

### The 8K attempts have not measured the hypothesis

The first 8K attempts returned HTTP 400 because llama-server still received hardcoded `-c 8192`; an 8,192-token prompt needs room for generated tokens. Commit 21eedee now passes the derived context size to the actual server argv.

That failure is harness configuration, not an economic or correctness result. Do not retain it as an 8K benchmark.

Source review at head 21eedee finds two additional stale 2K constants in `scripts/admitted_store_gate.py`:

- successful restore is still required to equal `cache_n=2044, prompt_n=4`;
- the record `kind` is still hardcoded as `admitted-store-2k-gate`.

An 8K run is not admissible until these are parameterized and behaviorally tested.

### Canonical layout inventory

`docs/ggsq-layout.md` is useful primary-source-derived evidence from the pinned llama.cpp writer:

- hybrid slot state is attention KV followed immediately by recurrent state, without a section delimiter;
- SCKP target payloads are PARTIAL_ONLY recurrent-state serializations;
- K/V payload dtype and row stride are recorded;
- head split, layer identity, RoPE status, cell-extension presence, and the recurrent-layer map are not self-described by the state file;
- those missing values must come from the exact bound source GGUF/architecture descriptor;
- the current KVX schema cannot represent the recurrent layer map or canonical hybrid checkpoints.

This partially fires the prior falsifier but does not require an upstream export primitive yet. Canonicalization may use the exact source GGUF during ingestion. The resulting canonical KVX artifact must carry the resolved geometry and state metadata so target-side reuse does not require access to the source GGUF.

Do not carry `data_tgt` forward as an allegedly portable opaque checkpoint. It is another native recurrent-state serialization and must be parsed into canonical recurrent state. Draft/speculative blobs remain unsupported and must fail closed when nonempty.

## P0 — close the two remaining 8K harness constants

Make the smallest red-test-first correction before another live run:

1. Derive expected covered tokens from retained save/checkpoint metadata. For the current stable-tail policy, require:
   - `prompt_n == prompt_tokens - checkpoint_n_tokens`;
   - `cache_n == checkpoint_n_tokens`;
   - `cache_n + prompt_n == prompt_tokens`.
   Do not hardcode either 2,044 or 8,188.
2. Make the record kind length-neutral, such as `admitted-store-gate`, while retaining the exact `prompt_tokens` field.
3. Add a behavioral unit test that exercises argv construction without starting a live server. Source-string assertions are not sufficient because the previous nonfunctional fix passed them.
4. Add an 8,192-token synthetic report test proving the acceptance logic expects 8,188/4 and rejects 2,044/4.
5. Keep the clean-worktree requirement. Commit the runner and tests before generating evidence.

Do not alter adapter behavior or persistence code in this step.

## P1 — run one truthful 8K break-first repetition

Run 8,192 prompt tokens on the exact tested Qwen3.8-27B/qwen35 model, persistent ext4/NVMe, natural page-cache state, with `n_ctx >= 9,216`.

The preflight space estimate is acceptable: using the retained 295,390 bytes/token rate, the 8K run needs approximately 5.41 GiB including the current double-copy peak and 20% margin. Recompute against actual free space immediately before generation and refuse if it no longer fits.

Require:

- new server PID and zero reuse before restore;
- `cache_n == checkpoint_n_tokens`, `prompt_n == prompt_tokens - checkpoint_n_tokens`, and total coverage of 8,192;
- complete save/restore checkpoint metadata equality;
- token, content, and nonempty probability-vector parity against native reuse;
- pristine-prefix restoration after verification;
- unchanged admitted-object facts and zero request-path payload copying;
- support detection inside the timed request path;
- reconciled phase timings;
- unpatched refusal before store access and with zero state-endpoint calls.

Decision rule:

- Any correctness, identity, parity, or refusal failure: stop and diagnose.
- Restored total beats paired cold prefill: retain the record and run two more paired repetitions. Pass requires at least 2/3 paired wins and a winning median.
- Restored total loses: retain the result and attribute the delta once. Do not resume opaque-store optimization unless the same cause blocks canonical transfer.

Record actual bytes/token, artifact size, admission time, request-path saving, and break-even restores.

## P2 — define the minimum canonical hybrid schema before writing a decoder

The layout inventory invalidates a decoder-first implementation because the current container cannot faithfully hold everything the decoder would produce.

Add the smallest versioned schema extension that can represent:

1. normalized attention K and V tensors in canonical little-endian `layer,kv,token,head,dim` order;
2. recurrent R/S tensors plus an explicit source-layer map;
3. checkpoint records carrying `n_tokens`, `pos_min`, `pos_max`, and canonical target recurrent state;
4. the resolved source geometry and architecture facts obtained from the exact source GGUF;
5. explicit RoPE-state semantics;
6. absence of draft/speculative state.

Normalize transposed V during decode; do not add a backend-native transposed layout to the canonical identity merely because the source used it.

Fail closed on:

- missing or mismatched source GGUF identity;
- unknown architecture descriptor;
- ambiguous layer map or head geometry;
- nonempty draft/speculative checkpoint blobs;
- unsupported ggml types;
- checkpoint state that cannot be parsed into the same canonical recurrent schema.

Acceptance is offline round-trip validation of the new container structures and malformed-input refusal. It is not yet a portability claim.

## P3 — smallest fail-closed GGSQ decoder

Implement only enough of `adapters/llamacpp_ggsq.py` to decode a deterministic small fixture using:

- the ggsq/3+sckp/1 bytes;
- the exact bound source GGUF/architecture descriptor;
- the versioned canonical hybrid schema from P2.

Requirements:

- streaming or bounded-memory parsing;
- exact section bounds and no scanning for magic inside payloads;
- rejection of truncation, overlap, impossible counts/shapes/strides, wrong tuple/version, unsupported types, missing source metadata, and trailing unexplained bytes;
- canonical normalization of transposed V;
- recursive parsing of SCKP target recurrent state;
- an independent numeric oracle not produced by the decoder under test;
- mutation tests for each trust boundary;
- no changes to opaque import/export capability gates.

Falsifier: if source GGUF plus the pinned architecture descriptor still cannot uniquely recover head geometry, layer mapping, cell metadata size, RoPE semantics, or recurrent shapes, retain the exact missing field and request the smallest versioned upstream metadata/export seam. Do not guess.

## P4 — prove same-model dtype portability

After canonical decode, add the narrow target encoder/import seam needed to compare one f16 source against one q8_0 target of the same model.

The opaque route must refuse this mismatch. Only the canonical route may transform it.

Gate against target-native q8_0 reuse with thresholds fixed before the run:

- exact prompt and target model identity;
- successful target prefix reuse;
- output and probability/logit divergence compared with target-native reuse;
- native target prefill fallback on every unsupported or failed case.

Only a behavioral pass unlocks same-model CUDA/ROCm/Vulkan transfer. Hardware provenance must not become canonical artifact identity.

## P5 — minimal sidecar

After one canonical dtype conversion passes, expose only:

- canonical prefix resolution;
- transfer/import into a named target runtime;
- reused-token count, transfer mode, gate verdict, and fallback reason;
- automatic native target prefill on failure.

Do not build broad daemon, authentication, multi-tenant, or distributed scheduling surfaces before this transfer seam exists.

## Deferred

- 32K on the current NVMe; estimated peak plus margin does not fit;
- 131K;
- learned cross-model projection and token alignment;
- production vLLM/HF adapters;
- MTP/draft/speculative state;
- upstream comments or submissions.

HF may be used earlier only as an independent numeric oracle.

## Required execution order

1. Parameterize 8K coverage acceptance and record identity; add behavioral tests.
2. Run the one-repetition 8K gate and follow its decision rule.
3. Define and test the minimum canonical hybrid schema.
4. Implement the bounded GGSQ+SCKP decoder with an independent oracle.
5. Add one narrow target encoder and prove same-model dtype conversion.
6. Expose only the proven seam through the sidecar.
