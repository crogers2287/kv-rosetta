# KV Rosetta research steer: prove the admitted path on persistent NVMe

Status basis: default-branch head 815930439046780605d66174038a6ea4bc8c9ece.

This steer supersedes 3cacbba. The exact-key GGUF lookup and store-owner guard landed, and the real admitted-store path now beats cold prefill in three of three paired 2K repetitions with hybrid_support still inside the timed request path. The next smallest falsifiable experiment is the same 2K gate on Fred's intended persistent NVMe. Do not jump to 8K, and do not describe an NVMe file served from warm Linux page cache as a cold-host-restart result.

## Mission

Deliver persistent exact-prefix restoration for the deployed 27B qwen35-family hybrid model across a complete llama-server restart, with exact artifact/runtime identity, fail-closed behavior on unpatched or incomplete runtimes, and request-path economics that beat native prefill.

Primary upstream track:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Keep checkpoint-persistence patch 0001 pinned. Do not post upstream during this track.

## Evidence now retained

### The support check is fast without being removed

Commit 799f2ad replaces full GGUF metadata materialization with an mmap cursor that can read one exact string key without allocating later tokenizer arrays.

The safety/performance split is explicit:

- architecture() returns at general.architecture for the request path;
- architecture_exhaustive() scans the complete metadata header during off-request admission and refuses conflicting duplicates;
- malformed, missing, wrong-type, truncated, and ambiguous admission headers fail closed;
- classification remains unchanged for ordinary, recurrent, and hybrid architectures;
- no architecture value is cached across calls or server generations;
- hybrid_support still runs for every restore;
- the admitted store now requires both mode 0700 and st_uid equal to the effective UID.

Fifteen focused tests landed and 472 offline tests pass.

### The complete tmpfs request path passes

Retained bench/admitted-store-2k-fastkey.json records the same model, prompt, binaries, admitted object, timing boundaries, and refusal gates as the prior failed run:

| Measurement | Result |
|---|---:|
| Median admitted total | 1.551 s |
| Admitted range | 1.509–1.696 s |
| Median native cold | 1.734 s |
| Cold range | 1.731–1.737 s |
| Paired wins | 3 / 3 |
| Median ratio to cold | 0.894 |
| Median saving | 0.183 s |
| Median resolve_support | about 0.0022 s |
| Admission | 2.834 s |
| Break-even | 16 restores |
| Reuse | 2,044 / 2,048 every run |
| Request-path state payload read by KV Rosetta | 0 bytes |
| Unpatched endpoint calls | 0 |

Every run has cache_n=2044 and prompt_n=4, complete restore-metadata equality, token/content/nonempty probability-vector parity, pristine-prefix restoration, unchanged admitted-object facts, and reconciled phases.

Classify this correctly: the experimental local path wins for a process restart when the admitted state resides on tmpfs. It does not yet prove persistent-filesystem economics or host-restart behavior. The first paired win was only 2.2%, while the other two were about 10.4% and 13.1%; retain raw ranges and do not reduce this to the median alone.

## P0: make the persistent-filesystem record self-identifying

The current runner records only the filesystem type and a free-form note. Before an NVMe record is accepted, add small evidence-only fields and fail if the target is memory-backed.

Record at minimum:

- resolved slot/store path;
- stat device ID;
- findmnt source, target, filesystem type, and mount options;
- backing block-device name and rotational flag when resolvable;
- whether the path is on the same mount as the model;
- available bytes before admission;
- admitted object size and file facts;
- explicit page-cache policy for the run.

Refuse the persistent rung when findmnt reports tmpfs, ramfs, overlay backed by transient storage, or an unresolved target. Do not infer NVMe from a pathname such as /mnt/storage. Evidence must identify the actual mounted source.

Do not record device serial numbers or other unnecessary host identifiers.

Keep the existing exact model, prompt, patch, source-diff, binary, build, K/V dtype, compound-tuple, and admitted-object digests. The patched llama.cpp tree remains a base head plus recorded diff, not a clean upstream binary.

## P1: natural-cache 2K NVMe gate

Use the exact committed admitted-store runner and the intended persistent NVMe slot directory. The store must be the llama-server slot directory, owned by the effective UID at mode 0700. Admit one object once, then run three paired repetitions across fresh llama-server processes.

Do not force page eviction for this first rung. Label it precisely as:

persistent NVMe file, natural page-cache state after admission and process restarts

This matches the immediate llama-server restart use case: the process dies, the file persists, and the OS page cache may survive. It is not evidence for a machine reboot or a long-idle cache eviction.

For every repetition retain and require:

- a new PID and zero reuse before restore;
- cache_n=2044 and prompt_n=4;
- complete restore-metadata equality;
- token, content, and nonempty probability-vector parity;
- pristine-prefix restoration;
- unchanged admitted-object device/inode/size/mtime/ctime/link count;
- zero KV Rosetta state-payload bytes and bounded metadata reads;
- hybrid_support timed, not cached or pre-warmed;
- phase reconciliation;
- unpatched refusal before store access with zero endpoint calls.

Decision rule:

- Pass: median admitted total beats median cold, at least 2 of 3 paired repetitions win, and all correctness/safety gates pass.
- Marginal: median wins but fewer than 2 paired runs win, or the median margin is under 5%; retain the record and run three more repetitions before proceeding.
- Economic fail: stop NVMe scaling and attribute the delta versus tmpfs, especially runtime_restore and pristine_restore. Do not run 8K.
- Any correctness or refusal failure: fail regardless of speed.

Record admission cost and the measured break-even restore count. A request-path win does not establish lifecycle savings for prefixes restored fewer times than break-even.

## P2: state-file cold-cache sensitivity, only after P1 passes

If the natural-cache persistent rung passes, test one deliberately state-cold repetition before considering 8K.

Evict only the admitted state file from page cache using a file-scoped mechanism such as POSIX_FADV_DONTNEED after its bytes are durable. Do not use the system-wide drop_caches control: evicting model weights and unrelated files changes the cold baseline and makes the comparison uninterpretable.

The sensitivity record must:

- state the exact eviction mechanism and its return status;
- measure or otherwise verify state-file residency before restore when practical;
- leave model/runtime warm-up policy identical to P1;
- retain the full correctness, refusal, identity, phase, and parity gates;
- label the result state-file cold-cache sensitivity, not host reboot.

Branch:

- Cold-state still beats cold prefill: repeat to three paired runs, then persistent 2K economics are credible under both natural and state-cold page-cache conditions.
- Cold-state loses: preserve both results. The proven win is immediate process restart with a naturally warm page cache; host-restart or long-idle persistence remains economically unproven. Profile storage read/restore before 8K.
- Eviction cannot be verified: do not publish a cold-cache claim; retain only the natural-cache result.

## P3: production-boundary work after persistent economics

Only after P1 passes—and after P2 is honestly classified—choose the next smallest production-hardening item.

Still open:

- llama-server opens the state by basename after KV Rosetta checks it, leaving a same-UID/root race; a general untrusted-file claim needs fs-verity or an upstream restore-from-open-fd/verified-offset primitive;
- the manifest records runtime build_info but not a reproducible local checkpoint-patch implementation identity;
- the exact tested model is Qwen3.8-27B-UD-Q4_K_XL.gguf, architecture qwen35, not exact Qwen3.5-27B or Qwen3.6-27B digests;
- active draft/MTP/speculative state remains withheld and behaviorally untested.

Do not conflate these with the P1 performance gate. Preserve the experimental label and fail-closed behavior.

## Scaling rule

Do not run 8K merely because tmpfs passed.

An 8K rung is justified only after:

1. the persistent NVMe natural-cache 2K gate passes;
2. the state-cold sensitivity is retained and truthfully classified;
3. no safety/refusal regression is present;
4. the expected 8K artifact size and free-space requirement are recorded before generation.

Keep 32K, 131K, canonical conversion, cross-backend work, vLLM, Transformers, portable-KVX redesign, and upstream submission deferred.

## Truthful current status

- Persistent hybrid restoration is correct on the exact tested Qwen3.8-27B qwen35 model.
- Patched reuse is 252/256 and 2044/2048 with a stable four-token tail.
- Target K/V cache identity is explicit as f16/f16.
- The admitted store enforces full content addressing, pinned file facts, current-user ownership, mode 0700, and extensive fail-closed guards.
- Unpatched qwen35 refuses before store access and makes zero state endpoint calls.
- The complete admitted request path on tmpfs beats cold prefill 3/3, with a median 10.6% request-latency saving.
- Admission costs 2.834 s and currently amortizes after 16 restores.
- Persistent NVMe economics, state-cold behavior, host restart, 8K, exact Qwen3.5/Qwen3.6 digests, and active draft/speculative state remain untested.

## Required execution order

1. Add self-identifying persistent-filesystem evidence and a memory-backed-target refusal to the runner.
2. Commit the runner change before evidence generation.
3. Run the three-repetition natural-cache 2K gate on the intended persistent NVMe.
4. Follow the P1 decision rule.
5. If P1 passes, run one file-scoped state-cold sensitivity repetition.
6. Do not automatically run 8K.
