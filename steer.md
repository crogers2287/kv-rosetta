# KV Rosetta research steer: realign on portable agentic-prefix reuse

Status basis: default-branch head b6d8728b6c244850a3a4bbbedc2e56738b1d6cb9.

This steer answers re-align.md entries RA-001 and RA-002 and supersedes b76bc8c. The persistence track established an important prerequisite, but it is no longer the product roadmap. The project target is a sidecar that lets agentic harnesses reuse large, stable prefixes through a portable KVX representation across runtimes, cache dtypes, hardware backends, and—behind an explicit quality gate—model architectures.

## Answers to re-align.md

### RA-001 — answered: cross-backend portability and the sidecar are in scope

Yes. Cross-backend portability is the core project goal, and daemon/server.py (or the smallest equivalent service boundary) is in scope.

The existing opaque llama.cpp route is a source/target-specific fast path, not the portable format. Its strict CacheABIIdentity, ggsq/3+sckp/1 tuple, checkpoint appendix, and local admitted-store invariants are correct for opaque restoration and must not be relaxed. An f16 opaque artifact must continue to refuse a q8_0 target; an unpatched qwen35 runtime must continue to refuse before any state endpoint or store access.

Portability comes from a separate canonical conversion path:

source-native state -> canonical KVX tensors/state -> target-native state -> behavioral gate -> reuse or native-prefill fallback

The missing conversion and orchestration layer is now the main research track. Same-model cross-backend/cache-dtype transfer comes before cross-model mapping. Cross-model claims require the held-out divergence/task-quality gate already described by the repository; tensor similarity alone is never admission evidence.

The sidecar should be implemented only far enough to exercise a proven transfer seam. Do not build a broad daemon API around unimplemented conversions.

### RA-002 — answered: measure 8K now; do not attempt 32K on the current NVMe

Yes. The next live benchmark is 8K because it is the first retained measurement in the stated 8K–32K agentic-prefix range.

The persistent natural-cache 2K gate already passed 3/3 on ext4/NVMe at median 1.486 s versus 1.731 s cold, with 2,044/2,048 reuse and all refusal/parity gates intact. The file-scoped eviction mechanism and residency check have also landed. Do not spend another steering cycle optimizing 2K cold-cache behavior. If a cold-cache run is already in progress, retain and classify that single result, but it does not gate 8K.

Do not generate a 32K artifact on the current NVMe. The estimated approximately 9.7 GB object needs approximately 19 GB transient under the current admission copy while only 16.6 GB is free. A 32K persistent rung requires either:

- a different persistent target with at least the measured transient requirement plus a safety margin; or
- removal of the admission double-space requirement without weakening atomic full-digest admission.

Tmpfs is not a substitute for the 32K persistent claim.

## Mission

Deliver portable reuse of heavy system prompts, skills, MCP schemas, and tool definitions for agentic workloads:

1. persistent exact-prefix restoration within a supported runtime;
2. same-model canonical transfer across cache dtypes and hardware backends;
3. quality-gated cross-model/architecture transformation;
4. a small sidecar interface that orchestrates transfer, verification, and native-prefill fallback.

The llama.cpp checkpoint-persistence work in issue #25913 and PR #26004 remains required for qwen35 hybrid source/target state. Keep patch 0001 pinned and its identity recorded. Do not post upstream from this track.

## What the persistence work proved

Retain these results as the T1 foundation:

- genuine process replacement and persistent qwen35 checkpoint restoration;
- 252/256 and 2,044/2,048 reuse with a stable four-token tail;
- output and nonempty probability-vector parity against native reuse;
- complete checkpoint metadata equality;
- patched/unpatched paired controls, with the unpatched runtime refusing before endpoint calls;
- explicit f16/f16 CacheABIIdentity;
- a complete admitted path that beats cold prefill 3/3 on tmpfs and 3/3 on persistent NVMe under natural page-cache state;
- atomic content admission, bounded metadata reads, file-fact pinning, owner/mode checks, and adversarial cleanup/refusal coverage.

These results do not establish cross-backend portability, canonical extraction, cross-model transfer, host-restart economics, or exact Qwen3.5/Qwen3.6 model-digest coverage.

## P0 — one 8K production-shape economic rung

Run the existing process-owned paired harness at 8,192 prompt tokens on the exact tested Qwen3.8-27B/qwen35 model and persistent NVMe, using natural page-cache state. Before generation, record free space, predicted object size, and predicted peak transient usage; refuse if the measured safety margin is inadequate.

Start with one break-first repetition. Require:

- a new server PID and zero reuse before restore;
- cache_n = 8,188 and prompt_n = 4, unless retained token evidence falsifies the stable-tail hypothesis;
- complete save/restore metadata equality;
- token, content, and nonempty probability-vector parity against native reuse;
- pristine-prefix restoration after the probe;
- unchanged admitted-object identity and zero request-path payload copying;
- hybrid_support inside the timed path;
- unpatched refusal before store access and with zero state-endpoint calls;
- reconciled phase timings and truthful natural-cache labeling.

Decision rule:

- Correctness/refusal failure: stop and diagnose; speed cannot pass the rung.
- Restored total beats paired cold prefill: retain the record and run two more paired repetitions. Pass requires at least 2/3 wins and a winning median.
- Restored total loses: retain the result, attribute the delta once, and do not optimize the opaque store further unless the cause would also block canonical transfer.

Report artifact bytes per token, admission cost, per-restore saving, and break-even restore count. Interpret break-even against repeated harness requests, not a one-shot workload.

## P1 — smallest falsifiable canonical-conversion experiment

After the 8K record, stop extending opaque persistence and build the smallest read-only canonical extraction proof.

First produce a source-backed layout inventory for ggsq/3 and sckp/1 that identifies, with byte ranges and invariants:

- token and sequence metadata;
- per-layer K and V payloads, ggml type IDs, row strides, shapes, and ordering;
- recurrent/hybrid state and checkpoint records;
- which fields are model/runtime identity versus portable tensor/state data;
- which data cannot yet be represented by the current KVX container.

Then implement only enough of adapters/llamacpp_ggsq.py to decode one admitted artifact into canonical KVX segments. Do not add a general encoder, daemon, vLLM adapter, or learned mapper in this step.

The first fixture must be deterministic, small enough for offline CI, and mutation-tested. Acceptance requires:

- exact bounds checking and streaming/bounded-memory reads;
- rejection of truncation, overlap, impossible shapes/strides, unsupported ggml types, wrong tuple/version, and missing hybrid checkpoint data;
- canonical little-endian layer,kv,token,head,dim ordering;
- explicit source dtype and dequantization policy;
- a retained numeric oracle derived independently from the decoder under test;
- no weakening of the opaque import/export gates.

Falsifier: if ggsq/3 lacks enough typed shape/layout information to reconstruct canonical tensors without model-specific inference, retain the exact missing information and pivot to the smallest upstream export primitive that supplies it. Do not guess offsets or bless a self-consistent round trip.

## P2 — prove dtype portability before hardware portability

Once canonical extraction passes, prove one same-model cache-dtype conversion on a single available host: f16 source to q8_0 target, or the closest actually deployed pair.

This is not expected to be token-exact. Gate it against the target runtime's own native q8_0 reuse:

- same prompt tokens and target model digest;
- target-native output/logit or probability-vector reference;
- explicit divergence thresholds fixed before the run;
- successful target reuse at the intended prefix;
- native target prefill fallback on any conversion, identity, or quality failure.

The opaque route must still refuse the dtype mismatch. Only the canonical route may convert it.

A pass unlocks a same-model CUDA/ROCm/Vulkan transfer using the same source artifact and gate. A fail retains the numeric evidence and identifies whether layout, quantization, RoPE treatment, or hybrid recurrent state is the blocker.

## P3 — minimal sidecar seam

After one canonical dtype conversion is behaviorally accepted, add the smallest service surface needed by an agentic harness:

- resolve a canonical prefix identity;
- request import/transfer into a named target runtime;
- return reused-token count, transfer mode, quality-gate verdict, and fallback reason;
- fall back to native target prefill on every unsupported, ambiguous, stale, corrupt, or failed case.

Reuse watcher.py, budget.py, resolve.py, gate.py, and metrics.py where their contracts fit. Do not create a second identity or admission system. Authentication, multi-tenant policy, distributed scheduling, and a broad management API are later work.

## Deferred until the above gates pass

- 32K on the current NVMe;
- 131K;
- learned cross-model projection and token alignment;
- vLLM and Transformers production adapters;
- MTP/speculative-state enablement;
- general untrusted same-UID storage claims;
- upstream submission or external comments.

HF may be added earlier only as a narrow independent numeric oracle for P1/P2, not as a production adapter.

## Required execution order

1. Retain any already-running single state-cold result, but do not iterate on it.
2. Run the break-first persistent-NVMe 8K experiment; expand to three repetitions only on a correct economic win.
3. Produce the ggsq/3+sckp/1 canonical-layout inventory.
4. Land the smallest fail-closed canonical decoder and independent offline oracle.
5. Prove one same-model dtype conversion against target-native behavior.
6. Only then expose that proven transfer seam through a minimal sidecar.
