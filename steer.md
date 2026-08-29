# KV Rosetta research steer: validate the 27B proof, seal import, then measure economics

Status basis: default-branch head 652471486061f0a9f004a29570d0135a7999a5db.

This steer supersedes 53c390a. A paired patched/unpatched process-owned 27B result now exists and the export-side fail-closed gaps were substantially repaired. The next work is not to repeat the checkpoint-persistence discovery. It is to correct two evidence defects in the retained record, apply the same gate to import and capability reporting, remove a new full-file memory cost, and then start the production context ladder at 2K.

## Mission

Deliver persistent exact-prefix restoration for the deployed 27B qwen35-family hybrid model across a complete llama-server restart, with one auditable artifact contract and no path that succeeds on an unpatched runtime, incomplete SCKP state, unsupported protocol tuple, unproven active state class, or unverifiable restore.

Primary upstream evidence:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Keep upstream patch 0001 pinned and unchanged. Do not post upstream during this track.

## Major result now retained

bench/production-27b-matrix.json records a paired experiment on:

- Qwen3.8-27B-UD-Q4_K_XL.gguf;
- general.architecture qwen35;
- exact model digest 9bf3b07e…;
- 256 exact prompt tokens;
- the same upstream source revision ca3d5a3e1 for both binaries;
- two owned process lifetimes per leg.

Measured:

Patched:

- first process PID 2281304, second PID 2283813;
- save/restore both report 2 checkpoints;
- checkpoint bytes 313,788,820;
- total slot bytes 487,926,936;
- declared checkpoint coverage 252 tokens;
- post-restart cache_n=252, prompt_n=4;
- generated token IDs and content match native in-memory reuse.

Unpatched control:

- first process PID 2286310, second PID 2289098;
- no checkpoint protocol advertised;
- sequence restore still reports n_restored=263;
- artifact bytes 174,138,116;
- post-restart cache_n=0, prompt_n=256.

This is strong behavioral evidence that checkpoint persistence, not ordinary sequence restoration, is the load-bearing variable on the tested 27B qwen35 architecture.

The runner now enforces PID replacement, port closure, fresh-process zero reuse, artifact digest continuity, save/restore metadata equality, bounded tail, and the patched/unpatched contrast.

## Export-side work now complete enough to retain

The current branch also adds:

- explicit OPAQUE export refusal on an unpatched hybrid runtime before any save POST;
- an exact tested compound tuple allowlist, currently ggsq/3+sckp/1;
- target-only active-state gating;
- exact appendix-offset validation from n_written - checkpoint_bytes;
- EOF and byte-count agreement;
- structured SCKP parsing with corruption cases;
- preservation of checkpoint position zero.

Keep these properties and their red tests.

## Audit corrections before claiming the 256 gate complete

### 1. Probability-vector parity is currently vacuous

The retained record stores eight empty objects under warm_after_restore.top_probs.

scripts/production_matrix.py looks for top_probs or probs directly inside each completion_probabilities entry. The llama.cpp response contract places the alternatives under top_logprobs by default. If post_sampling_probs=true is requested, the nested field becomes top_probs.

Therefore:

- probs(native) and probs(warm) currently both evaluate to eight empty objects;
- the equality check passes without comparing a single probability;
- the commit’s claim that probability vectors match is not yet proven.

Fix one of these exact ways:

- keep the default response and parse each entry’s top_logprobs with id and logprob; or
- send post_sampling_probs=true and parse each entry’s top_probs with id and prob.

Require:

- one vector per generated token;
- each vector nonempty;
- the generated token itself present;
- the expected n_probs count unless the API explicitly returns fewer;
- native and persisted vectors equal within a declared numeric tolerance.

Store both native and persisted vectors in the JSON. Empty vectors must fail the run.

### 2. Runner provenance in the record is inconsistent

The record says repo_commit=a0540a7, while the acceptance-enforcing runner and final record landed in 6524714.

That happened because the modified runner was executed before it was committed. The record is evidence from an uncommitted implementation even though acceptance_checked=true.

Repair by:

1. running from a clean committed tree;
2. refusing a dirty worktree before the experiment;
3. recording git HEAD;
4. recording the SHA-256 of scripts/production_matrix.py;
5. recording the full launch argv for both legs;
6. recording source-tree HEADs and build flags for both llama.cpp builds;
7. rerunning the 256 matrix;
8. committing the new record after the runner is already committed.

The record’s repo_commit may name the prior commit containing the runner. That is valid when runner_sha256 proves the exact file used. It must not name a commit that lacks the acceptance logic.

### 3. Model naming must remain exact

The retained model is named Qwen3.8-27B, while the earlier project wording named Qwen3.5/Qwen3.6 27B. It exercises the qwen35 hybrid architecture and appears to be the current deployed 27B target, but do not blur model names.

Report:

- exact model filename and content digest;
- architecture separately;
- whether this is the currently deployed production SKU.

If the production service still uses a different Qwen3.5/Qwen3.6 27B file, run the same corrected 256 matrix on that exact digest before transferring the claim. Do not discard the Qwen3.8 evidence; classify it as architecture-level evidence until the deployment identity matches.

## P0: make one predicate govern capability, export, and import

The export boundary is now gated, but capabilities() and import_() do not yet enforce the same full predicate.

Current gaps:

- capabilities can advertise OPAQUE for an active draft/speculative configuration and only append a warning;
- capabilities can advertise a compound tuple that export would refuse;
- active_checkpoint_state_classes is not included in CacheABIIdentity;
- import does not re-check active state classes or the tested compound tuple;
- import does not compare the restore response’s checkpoint metadata with the manifest;
- verify_reuse=False can bypass the behavioral proof;
- _restore_pristine() swallows a failed re-restore and the caller can receive ok=true with a verification token left in the slot.

Create one configuration-aware decision function used by all three surfaces.

For a hybrid/recurrent artifact it must require:

- complete protocol;
- exact tested compound tuple;
- active state classes present and all behaviorally proven;
- exact model and prompt identity;
- active state classes and protocol tuple bound into CacheABIIdentity;
- complete nonzero manifest coverage;
- exact restore-response equality for checkpoint count, bytes, n_tokens, pos_min, and pos_max;
- observed cache_n equal to declared checkpoint_n_tokens;
- prompt_n equal to the exact bounded tail.

For hybrid compound import, reuse verification is mandatory. Remove the public bypass or ignore verify_reuse=False for this path.

If restoring the pristine prefix after the probe fails:

- erase the slot if possible;
- return ok=false;
- never report a usable imported cache.

Required retained tests:

1. active draft/speculative state yields empty capabilities;
2. the same configuration refuses explicit import before a restore POST;
3. ggsq/2+sckp/1 is absent from capabilities and refused on import;
4. each restore-metadata field mismatch independently returns ok=false;
5. zero/missing compound coverage is refused before restore;
6. verify_reuse=False cannot bless a hybrid compound import;
7. failed pristine re-restore cannot return ok=true.

## P1: restore streaming behavior before large contexts

checkpoint_appendix_at() and parse_checkpoint_appendix() currently call Path.read_bytes().

The production 256-token patched slot is already 487,926,936 bytes. Reading the whole file into Python to validate a small amount of framing adds roughly 488 MB of transient memory and will scale badly at 2K, 8K, and 32K.

Replace full-file reads with seek-based bounded parsing:

- seek directly to the declared appendix offset for production export;
- read fixed headers only;
- skip blobs by checked offsets or stream them in bounded chunks when a digest is needed;
- compare the final offset with file size;
- never allocate a declared blob length;
- retain the 16 GiB per-buffer and aggregate bounds.

The general corruption helper may scan in chunks for tests, but the production adapter knows the exact offset and must use it.

Add a test using a sparse multi-gigabyte fixture or a read-counting wrapper proving validation does not read the sequence body or allocate proportional to artifact size.

## P2: rerun the corrected 256 record

Do not widen the prompt yet.

Acceptance:

- clean committed runner provenance;
- exact model/deployment identity;
- both patched and unpatched process-owned legs;
- nonempty native and persisted probability vectors;
- metadata and cache-coverage checks enforced;
- active classes ["target"];
- adapter capabilities, export, and import agree on the same support decision;
- end-to-end KVX export/import round trip succeeds patched and refuses unpatched.

Record raw timing honestly:

- patched cold wall: about 0.557 s;
- raw runtime restore: about 0.424 s;
- restored tail completion: about 0.207 s.

At 256 tokens, raw restore plus tail processing is roughly 0.631 s, slightly slower than the measured cold request. This is a correctness gate, not an economic win.

Do not include save time in user-visible restore latency, but do include KVX integrity verification, staging, runtime restore, and verification-probe cost in the end-to-end adapter number.

## P3: production economic ladder

After the corrected 256 gate, run 2K first with three clean repetitions.

Compare:

- native cold prefill;
- native in-memory checkpoint reuse;
- KVX outer verification;
- staging;
- runtime restore;
- mandatory reuse verification;
- tail prefill;
- total user-visible adapter restore;
- sequence bytes;
- checkpoint bytes;
- total bytes;
- peak process RSS and VRAM;
- output tokens and probability parity against native reuse.

Only proceed:

- 2K passes correctness and shows a credible path toward break-even -> run 8K;
- 8K remains favorable and memory is bounded -> run 32K;
- otherwise stop at the first failing rung and attribute the bottleneck.

Run tmpfs and NVMe separately. Use the actual production KV dtypes and record them explicitly. The model’s Q4 weight quantization does not state the K/V cache dtype.

Decision criterion:

total verified adapter restore + tail processing < native full prefill

The new 256 record shows the fixed checkpoint payload is about 314 MB. Storage and verification cost may dominate small prefixes, so the break-even context is now the first economic question.

## P4: MTP/speculative only after target economics

Serialization is still not behavioral support.

Before enabling another active state class:

- run the process-owned restart with it actually active;
- compare persisted behavior to native in-memory reuse;
- remove or corrupt its required blob and require refusal or demonstrated failure;
- bind its model identity and launch settings into CacheABIIdentity.

Until then, target-only is the only supported hybrid configuration.

## Required execution order

1. Fix probability extraction and make empty vectors fail.
2. Commit the runner, then rerun the 256 record from a clean committed tree with runner provenance.
3. Centralize the configuration-aware support predicate across capabilities, export, and import.
4. Bind and compare full restore metadata; make hybrid reuse verification mandatory.
5. Make failed pristine restoration fail closed.
6. Replace full-file appendix reads with seek-based bounded validation.
7. Run the end-to-end patched/unpatched KVX round trip on the production 27B.
8. Run 2K three times.
9. Run 8K and 32K only when the prior rung passes.
10. Keep exact-boundary, 131K, cross-backend, canonical, vLLM, Transformers, and upstream submission deferred.

## Definition of the next milestone

The 256 production gate is complete only when:

- the record is produced by an exact committed runner;
- probability-vector parity is non-vacuous;
- capability, export, and import share one fail-closed decision;
- restore metadata is bound to the artifact;
- validation is streaming;
- patched end-to-end KVX reuse succeeds after a true restart;
- the same-source unpatched runtime cannot advertise, export, or successfully import a hybrid artifact.

## Reporting discipline

Classify every claim as:

- proven by retained automated test;
- measured once on Fred;
- independently measured upstream;
- confirmed by source review;
- inferred;
- untested;
- failed.

Current truthful status:

- paired checkpoint persistence is behaviorally proven on the tested Qwen3.8-27B qwen35 model;
- patched reuse is 252/256 and unpatched reuse is 0/256;
- token/content parity against native reuse is proven;
- probability-vector parity is not yet proven because the stored vectors are empty;
- export is substantially fail closed;
- import and capability reporting do not yet enforce the complete active-state/metadata contract;
- 256-token raw restore is slightly slower than cold;
- production 2K, 8K, and 32K economics remain unmeasured.
