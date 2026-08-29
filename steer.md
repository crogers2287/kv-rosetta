# KV Rosetta research steer: persist hybrid checkpoints

Status basis: default-branch head 1ec08a727d0a716c6cfeb16149c872f87dccdafc.

This steer supersedes the conclusion that hybrid and recurrent architectures are a dead end. Keep the current fail-closed behavior for unpatched runtimes, but continue the research by adding the missing persistence seam to llama.cpp and proving it on the actual 27B model.

## Mission

Make the fixed Hermes/system/tool prefix of the Qwen3.5/Qwen3.6 27B model persist across slot eviction and server restart without re-prefilling it.

The target is exact-prefix restoration. The target is not arbitrary recurrent-state slicing, approximate longest-common-prefix reuse, or cross-model translation.

A recurrent state after an exact token sequence is deterministic and can be restored. It cannot be reconstructed from ordinary KV cells or safely sliced to an arbitrary earlier prefix. That distinction defines the work.

## Corrected current truth

The existing evidence remains valid:

- Qwen2.5-3B can save, erase, restore, and reuse a llama.cpp slot state.
- q4_0 KV restore beats native prefill on the measured 3B ladder.
- The tested Qwen3.5/Qwen3.6 27B restores sequence cells but reports cache_n=0 and re-prefills.
- The failure is not caused by model size, MTP, flash attention, GPU offload, tensor splitting, slot count, or prompt length.
- The current adapter correctly withholds the opaque capability rather than reporting false success.

The interpretation needs correction:

- Hybrid/recurrent architecture does not make an exact saved prefix fundamentally unrestorable.
- It makes ordinary KV-only prefix matching insufficient.
- llama-server already creates common_prompt_checkpoint objects containing the state required to resume.
- The /slots save endpoint does not persist those checkpoints.
- The restore handler clears slot.prompt, which also clears checkpoints.
- The next request finds no checkpoint and deliberately resets n_past to zero.

Upstream issue:

- https://github.com/ggml-org/llama.cpp/issues/25913

The issue documents this exact save/restore defect and proposes two implementation paths. It is open, unassigned, and has no linked patch as of this steer.

Related in-memory checkpoint behavior:

- https://github.com/ggml-org/llama.cpp/issues/24055

Do not rediscover those reports. Reproduce them against our pinned runtime, then build from them.

## Why this remains worth pursuing

The actual checkpoint observed on the 27B was about 149.6 MiB at both roughly 252 and roughly 2,800 tokens. That suggests the recurrent checkpoint component is largely fixed-size per checkpoint rather than growing by 149.6 MiB per token.

llama-server may retain up to 32 or 64 checkpoints for general conversation reuse. KV Rosetta does not need to persist all of them for the first milestone. It needs one valid checkpoint at the exact end of the stable prefix.

The system prompt and canonical tool schemas are the best-case workload:

- token sequence is known exactly;
- prefix boundary is controlled;
- one endpoint checkpoint is sufficient;
- model, tokenizer, runtime, and cache ABI are already content-addressed;
- fallback remains native prefill.

## Non-negotiable safety rule

Do not remove the current hybrid architecture refusal merely because a patched binary exists.

Replace the static refusal only after the live runtime advertises a checkpoint-persistence format and passes retained save, restart, restore, full-reuse, and parity tests.

Unpatched llama.cpp remains unsupported for hybrid opaque restoration.

## R0: pin source truth and retain the failure

Work from the exact llama.cpp revision that produced the evidence before rebasing anything:

- installed runtime build and commit;
- server-context.cpp save handler;
- restore handler;
- common_prompt_checkpoint definition;
- server_prompt_cache_state implementation;
- checkpoint creation and matching logic;
- target, draft, and speculative state handling.

Create a concise source map in docs/hybrid-checkpoint-research.md with exact commit hashes, paths, symbols, and observed behavior.

Retain a live negative test that proves the unpatched runtime:

1. processes an exact 256-token prefix;
2. saves the slot;
3. erases it;
4. restores it;
5. reports restored sequence cells;
6. returns cache_n=0;
7. logs that full prompt processing was forced because checkpoint data is absent.

This negative control must remain runnable after the patch exists.

## R1: choose the smallest correct llama.cpp persistence patch

Audit both upstream issue #25913 options before writing the patch.

### Option A: versioned checkpoint sidecar

Save filepath plus a checkpoint sidecar containing, for every persisted common_prompt_checkpoint:

- n_tokens;
- id_task;
- pos_min;
- pos_max;
- length-prefixed data_tgt;
- length-prefixed data_dft;
- length-prefixed data_spec.

The file requires:

- fixed magic;
- explicit format version;
- byte order;
- bounded counts and lengths;
- exact prompt-token digest;
- sequence-state version;
- integrity digest;
- atomic temporary-file publication.

Restore must rebuild slot.prompt.checkpoints after slot.prompt.clear and token restoration.

### Option B: serialize the complete server prompt-cache state

The in-memory prompt-cache path already knows how to retain tokens and checkpoints together. Determine whether serializing server_prompt_cache_state avoids duplicating logic and prevents the disk path from drifting from the in-memory path again.

### Decision rule

Prefer reuse of the existing prompt-cache representation if it can be:

- versioned;
- bounded;
- validated before allocation;
- restored into one exact slot;
- kept backward compatible;
- tested without a large invasive refactor.

Otherwise implement the sidecar first.

Record why the chosen path is smaller and safer. Do not build both production paths.

## R2: persist one exact-boundary checkpoint first

The general server creates checkpoints according to --ctx-checkpoints and --checkpoint-min-step. That granularity may leave the newest checkpoint before the end of the stable prefix.

For the KV Rosetta use case, create or select a checkpoint that covers the exact exported prefix boundary. Do not synthesize fake pos_min metadata around state captured at another position. Upstream issue #25913 explicitly warns that this can silently decode on top of recurrent state that already consumed the token.

The initial save contract must either:

- force a real checkpoint at the current stable-prefix boundary before serializing; or
- return the exact checkpoint coverage and require the adapter to reprocess the uncovered tail.

Prefer a real endpoint checkpoint so the imported state can satisfy the existing full-prefix invariant.

Persist one checkpoint for the first proof. Add multiple-checkpoint persistence only after the one-checkpoint path is correct and a real workload requires it.

## R3: make the patch reproducible from KV Rosetta

Do not leave the only copy of the experiment as edits in Fred's llama.cpp checkout.

Commit:

- patches/llama.cpp/0001-persist-slot-prompt-checkpoints.patch;
- scripts/build_patched_llama.sh;
- the exact upstream base SHA;
- expected patch SHA-256;
- build flags;
- a version probe;
- startup flags used by the live tests.

The build script must refuse an unexpected upstream revision rather than applying fuzzily to unknown source.

The experimental binary must expose a machine-readable capability such as:

- slot_checkpoint_persistence = true;
- slot_checkpoint_format = 1.

Do not infer support from a version string, filename, or architecture.

## R4: extend KVX for a checkpoint-aware opaque bundle

Represent the persisted hybrid artifact as two independently hashed opaque segments:

1. llama.cpp sequence state;
2. llama-server prompt checkpoint state.

Use explicit roles such as:

- opaque.llama.sequence_state;
- opaque.llama.prompt_checkpoints.

The cache ABI and ArtifactKey must include:

- GGSQ sequence-state version;
- checkpoint format version;
- llama.cpp commit and build identity;
- target/draft/speculative configuration;
- K and V physical dtype;
- context and checkpoint flags;
- exact prompt token digest;
- model and tokenizer identities.

Do not label the bundle as plain ggsq/2. Use an explicit compound representation such as ggsq/2+llama-checkpoints/1.

ArtifactStore must publish both segments atomically under one ArtifactKey. A sequence file paired with the wrong checkpoint sidecar must fail verification before reaching llama.cpp.

Import must write the two restored files through bounded streaming, invoke the exact slot, and clean every temporary file on success and failure.

## R5: change capability probing, not the safety posture

Replace the blanket architecture conclusion with a two-part rule:

- ordinary attention architecture: sequence-state persistence may be sufficient;
- hybrid/recurrent architecture: checkpoint persistence is additionally required.

For hybrid/recurrent models, advertise opaque import/export only when the live runtime reports a supported checkpoint-persistence format.

Unknown architecture or unknown checkpoint format remains fail closed.

Update README and gguf.py wording only when this distinction has a retained test. Until then, the current refusal stays active.

## R6: prove the patched path at 256 tokens

Use the exact Qwen3.5/Qwen3.6 27B model that currently fails. Start without an external draft model or speculative decoder unless those features cannot be disabled.

Required sequence:

1. tokenize an exact 256-token stable prefix;
2. cold prefill and record outputs and probabilities;
3. create an exact-boundary checkpoint;
4. save sequence state and checkpoint state;
5. erase the slot;
6. rerun without restore and prove cache_n=0;
7. stop the server completely;
8. start the same patched build and configuration;
9. restore both state segments to an exact slot;
10. run the exact prompt;
11. require the expected checkpoint reuse;
12. compare against native in-memory checkpoint reuse;
13. record first divergent token and probability deltas.

The server restart is mandatory. Without it, the test does not prove persistence.

Success requires:

- checkpoint file integrity passes;
- prompt, model, runtime, and cache ABI match;
- the intended slot is used;
- n_checkpoints_restored is explicit and positive;
- cache_n and prompt_n match the declared checkpoint coverage;
- restored output matches native in-memory checkpoint reuse;
- missing, corrupt, wrong-version, wrong-prompt, and wrong-slot checkpoint files are refused;
- verification leaves a usable, uncontaminated prefix.

n_restored alone remains insufficient.

## R7: handle MTP, draft, and speculative state explicitly

common_prompt_checkpoint carries data_tgt, data_dft, and data_spec because target context alone is not always sufficient.

The first proof may disable external draft/speculative execution and mark that limitation in capabilities.

Before claiming support with MTP or a draft model:

- persist and restore data_dft;
- persist and restore data_spec;
- bind draft-model identity and speculative configuration into CacheABIIdentity;
- prove restored behavior against native in-memory checkpoint reuse;
- add a negative test where draft state is omitted.

Never silently restore only data_tgt while advertising the full configuration.

## R8: scale using break-first methodology

After the 256-token restart proof passes, run:

- 2K;
- 8K;
- 32K;
- 131K only after 32K evidence supports it.

At each rung record separately:

- cold native prefill;
- in-memory checkpoint reuse;
- checkpoint creation;
- sequence-state export;
- checkpoint-state export;
- integrity and identity verification;
- sequence-state restore;
- checkpoint-state restore;
- post-restore verification;
- total user-visible restore latency;
- sequence artifact size;
- checkpoint artifact size;
- total bytes;
- peak host RAM and VRAM;
- cache_n and prompt_n;
- output parity against native checkpoint reuse;
- first divergence.

Run at least three repetitions and report medians and ranges.

Test tmpfs and NVMe separately. The checkpoint load path may behave differently from the existing GGSQ restore path.

Use the actual production KV configuration. Do not extrapolate from the 3B.

## R9: measure the one-checkpoint design against server defaults

Compare:

- one exact endpoint checkpoint;
- llama-server's normal multi-checkpoint policy;
- no checkpoint persistence.

Measure RAM, disk, restore latency, and reusable prefix length.

Do not persist 32 or 64 copies of a roughly 149.6 MiB checkpoint unless measurements show a concrete benefit. The fixed-prefix workload should normally need one.

Determine whether checkpoint size remains approximately constant across context length and whether K/V sequence-state growth remains the dominant 131K cost. Report measured values rather than projecting from one sample.

## R10: prepare an upstream-quality change

Once local proof is retained:

- rebase the minimal llama.cpp patch onto current upstream;
- add upstream C++ or server integration tests;
- document backward compatibility;
- document missing-sidecar behavior;
- document draft/speculative handling;
- document checkpoint granularity;
- prepare a clean PR description referencing issue #25913.

Do not submit an upstream PR or comment without explicit authorization. Preserve the patch and evidence in KV Rosetta regardless of upstream timing.

## Parallel fallback path

While the persistent patch is being developed, retain the operational fallback:

- keep the stable prefix warm in llama-server's in-memory prompt cache;
- use cfrproxy affinity routing;
- replay through kvwarm after model startup;
- fall back to native prefill after restart or eviction.

This is not the final product, but it keeps Fred usable and provides the native checkpoint-reuse control required by the persistence tests.

Do not divert into another backend merely to avoid the hybrid checkpoint problem. The 27B production model is the research target.

## Work explicitly deferred

Until the 27B checkpoint-aware restart proof passes:

- canonical tensor extraction;
- CUDA-to-HIP transfer;
- vLLM adapter;
- Transformers adapter;
- cross-model mapping;
- 131K brute-force runs;
- additional model downloads solely to find an easier architecture.

The project already proved the ordinary-attention path. The next unknown with the highest value is persistent hybrid checkpoint state.

## Required execution order

1. Retain the unpatched 27B failure as a live negative control.
2. Map common_prompt_checkpoint and server_prompt_cache_state at the pinned llama.cpp revision.
3. Choose the sidecar or unified prompt-cache serialization path.
4. Implement one exact-boundary checkpoint save and restore.
5. Commit the reproducible llama.cpp patch and guarded build script.
6. Add runtime capability advertisement.
7. Add the two-segment checkpoint-aware KVX representation.
8. Prove a 256-token save, full server restart, restore, reuse, and native-checkpoint parity.
9. Add corrupt, mismatched, missing, multi-slot, and wrong-version refusals.
10. Add draft/speculative support or explicitly withhold it.
11. Run 2K, 8K, and 32K ladders with one checkpoint.
12. Decide whether 131K is economically and operationally justified.
13. Prepare the upstream patch only after the retained evidence is complete.
14. Resume canonical and cross-backend work only after the production 27B path is proven.

## Definition of the next milestone

The hybrid persistence milestone is complete only when:

- the unpatched runtime still fails closed;
- the patched runtime advertises an explicit checkpoint format;
- one exact stable-prefix checkpoint is persisted;
- sequence and checkpoint state are cryptographically bound in one ArtifactKey;
- a full llama-server restart occurs between save and restore;
- the exact 27B model reuses the declared prefix in the intended slot;
- restored behavior matches native in-memory checkpoint reuse;
- corrupted or mismatched checkpoint state never reaches runtime import;
- total restore latency and artifact size are measured;
- the complete experiment is reproducible from committed files.

Anything short of the restart test is an in-memory cache experiment, not persistent prefix restoration.

## Reporting discipline

Separate every statement into:

- proven by retained test;
- measured on Fred;
- confirmed by upstream source;
- inferred;
- untested;
- failed.

Do not repeat that hybrid restoration is impossible. State the narrower truth: current llama.cpp slot persistence omits the checkpoint state required by hybrid/recurrent models.
