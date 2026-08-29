# KV Rosetta research steer: close the hybrid fail-closed seam, then run production 27B

Status basis: default-branch head fff2b23f3ff04f5a8b5721aab7bcb7634c95791a.

This steer supersedes c40f71d. The runtime protocol and compound GGSQ+SCKP adapter path now exist, but the adapter was enabled before the required production-27B paired restart gate. The next work is a small fail-closed hardening experiment, followed immediately by the exact production Qwen3.5/Qwen3.6 27B matrix.

## Mission

Persist the exact production Hermes/system/tool prefix across slot eviction and a complete llama-server restart, while refusing unpatched runtimes, sequence-only hybrid artifacts, incomplete checkpoint metadata, unsupported protocol tuples, and unproven active draft/speculative configurations.

Primary upstream evidence remains:

- https://github.com/ggml-org/llama.cpp/issues/25913
- https://github.com/ggml-org/llama.cpp/pull/26004

Both remain open as of 2026-08-29. The upstream evidence continues to support checkpoint persistence and bounded tail re-prefill; it does not substitute for the production 27B run.

## Evidence added since the prior steer

The default branch now contains:

- compound artifact labels of the form ggsq/<sequence-version>+sckp/1;
- manifest checkpoint coverage fields;
- an import-time equality check between declared checkpoint_n_tokens and observed cache_n;
- outer-container truncation refusal;
- protocol separation between what SCKP serializes and what behavior has been proven;
- target support true, draft/speculative support false;
- offline malformed-protocol tests;
- a fix that prevents setUpClass network I/O from turning wrong-runtime skips into errors.

The recorded live positive result remains OpenMythos-Q6_K with target-only behavior. No production-27B paired record or new benchmark record landed.

Keep exact-boundary checkpoint work closed. The measured four-token tail remains the correct working contract.

## Audit findings that change priority

### 1. Explicit unpatched OPAQUE export is not fail closed

LlamaCppHTTPAdapter.export() checks only:

- requested representation is OPAQUE;
- slot_save_path exists.

It does not require capabilities(), a complete checkpoint protocol, or reusable hybrid state.

The retained unpatched test calls export with Representation.CANONICAL. That is rejected before runtime capability matters, so it does not prove that an explicit OPAQUE export is refused.

Current consequence: a caller that bypasses capability discovery can request OPAQUE from an unpatched hybrid runtime and receive a plain sequence-state artifact that cannot provide hybrid reuse.

This is the immediate red test and fix.

### 2. Compound labeling still relies on an unbounded magic scan

Export labels an artifact compound when has_checkpoint_appendix() finds the four bytes SCKP anywhere in the opaque slot file.

That is acceptable for a test classifier. It is not sufficient format evidence for a production artifact: the same bytes can occur inside the GGSQ body, and the scan proves neither the appendix boundary nor that the reported checkpoint payload occupies the file tail.

The save response already exposes total n_written and checkpoint_bytes. Extend the runtime protocol to expose checkpoint_offset or sequence_bytes, or derive the offset as:

checkpoint_offset = n_written - checkpoint_bytes

Then require SCKP exactly at that offset and require the payload bounds to agree with the whole file. Do not label a production artifact from an arbitrary magic occurrence.

### 3. Export records metadata but does not validate it

For a checkpoint-bearing hybrid export, require all of these before producing KVX:

- complete supported protocol;
- n_checkpoints_saved >= 1;
- checkpoint_bytes > 0;
- checkpoint_n_tokens > 0;
- valid pos_min and pos_max;
- exact appendix offset and magic;
- compound format matching the live protocol.

The current int(value or -1) conversion also turns a legitimate position 0 into -1. Preserve zero explicitly.

### 4. Import does not bind restore metadata to the manifest

Import currently compares observed cache_n with manifest checkpoint_n_tokens only when the declared value is nonzero.

It does not require a nonzero declaration for a compound artifact and does not compare the restore response’s checkpoint count, bytes, n_tokens, pos_min, or pos_max with the manifest.

For ggsq/N+sckp/1, zero or missing coverage must be refusal, not an optional contract. The restore response must exactly match the saved manifest metadata before the verification completion is credited.

### 5. Protocol support must be an exact tuple

The adapter currently accepts sequence versions 2 and 3 with sckp/1 because ordinary GGSQ versions 2 and 3 have been exercised somewhere in the project.

The retained checkpoint-persistence proof is for the patched sequence-version-3 build. Plain ggsq/2 may remain supported for ordinary attention. Hybrid checkpoint capability must use an explicit tested tuple allowlist, currently:

- ggsq/3+sckp/1 with proven target state.

Do not infer that sckp/1 on sequence version 2 works from plain ggsq/2 support.

### 6. Active draft/speculative configuration is not gated

The protocol correctly reports draft/speculative behavioral support as false, but capabilities still enables hybrid opaque transfer whenever target support is true and merely adds a note.

A note is not a gate. The live protocol must state which checkpoint state classes the current launch actually requires. If draft or speculative state is active and its behavioral support is false, export and import capabilities must be empty.

Until the server exposes active required state, treat a configuration that cannot be proven target-only as unsupported.

### 7. The production paired restart experiment is still outstanding

The process-owning harness still proves the patched positive path only. The unpatched control remains a separate in-process test and may skip when connected to the wrong binary.

There is still no single machine-readable production-27B record containing both required two-process legs.

## Steering decision

Do not run the expensive context ladder and do not widen capabilities further.

First execute one small offline/live fail-closed slice that makes the current export hole red, fixes it, and binds compound metadata. Then run the paired production-27B 256-token matrix from the previous steer.

This is the smallest falsifiable sequence because the first failures require no large model and prevent an expensive 27B run from blessing an adapter contract that can still emit useless hybrid artifacts.

## P0: red-test the fail-closed boundary

Add retained tests before changing behavior.

Required cases:

1. Unpatched hybrid props plus a requested OPAQUE export must raise AdapterError before any slot-save POST occurs.
2. Patched props with persistence=false, unknown format, unsupported compound tuple, target=false, or incomplete checkpoint metadata must also refuse export before publishing an artifact.
3. A sequence-state fixture containing incidental SCKP bytes inside its GGSQ body but no appendix at the declared offset must remain plain or be refused; it must never be labelled compound.
4. A compound artifact with zero or missing checkpoint coverage must be refused before restore.
5. Restore metadata differing from the manifest in any one of count, bytes, n_tokens, pos_min, or pos_max must produce ok=false.
6. ggsq/2+sckp/1 must be refused until a retained checkpoint test proves that exact tuple.
7. An active draft/speculative configuration with behavioral support false must advertise no hybrid import/export.
8. checkpoint_pos_min=0 must survive export as 0.

The unpatched test must request Representation.OPAQUE. Keep the existing CANONICAL rejection as a separate representation-contract test if useful, but do not call it proof of unpatched fail-closed behavior.

## P1: make compound evidence exact

Keep upstream patch 0001 unchanged. Update local integration patch 0002 only as needed to report:

- sequence_bytes or checkpoint_offset;
- total file bytes;
- saved/restored checkpoint count;
- checkpoint bytes;
- newest checkpoint n_tokens;
- pos_min and pos_max;
- active required checkpoint state classes.

Export acceptance for a hybrid artifact:

- capability protocol is complete for the active configuration;
- the tested compound tuple is allowlisted;
- save metadata is complete and self-consistent;
- sequence_bytes + checkpoint_bytes equals total bytes;
- SCKP magic occurs exactly at checkpoint_offset;
- the full opaque payload digest covers both regions.

Import acceptance:

- outer KVX verification passes before staging;
- exact model, prompt, cache ABI, compound tuple, and checkpoint metadata match;
- restore response metadata equals manifest metadata;
- observed cache_n equals declared checkpoint_n_tokens;
- prompt_n equals the exact uncovered tail;
- the uncovered tail is within the tested ceiling;
- the verification probe is followed by a clean re-restore.

Any disagreement returns ok=false and native prefill remains the fallback.

Do not use arbitrary magic scanning as adapter evidence. It may remain in tests/runtime_matrix.py solely to classify a patch-0001-only runtime for test selection.

## P2: production 27B paired 256-token matrix

After P0 and P1 pass, parameterize the existing process-owning harness and run the exact production Qwen3.5/Qwen3.6 27B GGUF with its real Fred flags.

One retained JSON record must contain both legs:

- pinned unpatched llama-server;
- pinned patched llama-server with 0001 and the current 0002.

Both legs use the same model digest, exact token IDs, prompt digest, K/V dtypes, n_ctx, checkpoint policy, GPU split/offload, slot, and sampling settings.

Patched acceptance:

- first process exits and port closes;
- second PID differs;
- second process has cache_n=0 before restore;
- artifact digest is unchanged across the restart;
- save and restore metadata match exactly;
- cache_n equals checkpoint_n_tokens;
- prompt_n equals token_count minus cache_n and tail <= 8;
- persisted output tokens and probability vectors match native in-memory checkpoint reuse;
- active target/draft/spec requirements are explicitly covered.

Unpatched acceptance:

- no checkpoint protocol;
- no hybrid OPAQUE capability;
- explicit OPAQUE export is refused by the adapter;
- direct runtime save/restore may report sequence cells, but after the full restart cache_n=0 and prompt_n equals the full prompt;
- the required negative leg fails rather than skips if the wrong binary is supplied.

A skipped leg is not a passing matrix.

## P3: economics only after the gate

After the paired production record and adapter round trip pass, run 2K, 8K, and 32K with at least three clean repetitions per rung.

Measure:

- native cold prefill;
- native in-memory checkpoint reuse;
- save;
- outer verification;
- restore;
- tail prefill;
- total user-visible restore;
- sequence bytes;
- checkpoint bytes;
- total bytes;
- cache_n and prompt_n;
- RSS and VRAM;
- output parity against native reuse;
- tmpfs and NVMe separately.

Decision criterion:

total verified restore + tail prefill < native full prefill

Do not run 131K until 32K provides a credible size, latency, and memory projection.

## MTP/speculative gate

Serialization is not behavioral support.

Before enabling either state class:

- expose that the current launch requires it;
- run the process-owned restart with the real feature active;
- compare persisted restore against native in-memory reuse;
- remove or corrupt that required blob and require refusal or demonstrated behavioral failure;
- bind draft/spec model identity and flags into CacheABIIdentity.

Until then, false means withheld, not “enabled with a warning.”

## Required execution order

1. Add the explicit unpatched-hybrid OPAQUE-export red test.
2. Gate export on the same complete, configuration-aware protocol used by capabilities.
3. Replace arbitrary appendix scanning with exact offset/bounds evidence.
4. Require complete save metadata and exact restore-metadata equality.
5. Restrict hybrid support to tested compound tuples.
6. Enforce active draft/speculative requirements.
7. Run all offline tests and the existing OpenMythos live target-only suite.
8. Run the paired production-27B 256-token matrix.
9. Only after it passes, run 2K, 8K, and 32K economics.
10. Keep exact-boundary, 131K, canonical, cross-backend, vLLM, Transformers, and upstream submission deferred.

## Definition of the next milestone

The next milestone is complete only when:

- explicit OPAQUE export is impossible on an unpatched hybrid runtime;
- a compound artifact cannot be created from incidental magic or incomplete metadata;
- restore metadata is bound to the artifact and observed reuse;
- only a tested protocol tuple can enable hybrid transfer;
- active unproven draft/speculative state disables capability;
- one production-27B record contains both patched and unpatched process-owned legs.

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

- target-only hybrid checkpoint restart is proven on OpenMythos-Q6_K;
- compound adapter import works live on that tested path;
- protocol serialization and behavioral support are now separated;
- explicit unpatched-hybrid OPAQUE export is not yet proven fail closed;
- exact appendix placement and full restore metadata are not yet bound;
- production 27B remains untested;
- draft/speculative behavioral restoration remains untested;
- production economics remain unmeasured.
