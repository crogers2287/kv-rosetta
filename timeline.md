# Timeline — kv-rosetta

Newest entries at the top. `REQ-NNN` increments per-repo, never resets.

## 2026-08-28

### REQ-002 — Build out KV Rosetta as a model-agnostic sidecar (kvxd) behind cfrproxy

Source: chat ("look at this project and the context. i want to plan this out in detail so that we
can build it"; then "id like to make it model agnostic if possible. can we add it as a sidecar to
our model if we run it through a proxy or something like that? check the cfrproxy proj in gh").
Plan file: `~/.claude/plans/look-at-this-project-kind-widget.md`.

#### Decisions locked (operator, via question prompts)

| Decision | Choice |
|---|---|
| Sidecar home / language | kv-rosetta owns `kvxd`, Python. cfrproxy unmodified in v1. |
| Relationship to `cfrproxy/scripts/kvwarm.py` | kvxd supersedes it; replay-warm becomes its tier-0 floor. |
| Backend seams | All three in parallel (HF transformers, llama.cpp, vLLM). |
| Scope | Full, through the learned heterogeneous projector. |
| ROCm leg | Real CUDA↔HIP test, not provenance-only. |

#### Findings that changed the design

| Finding | Evidence |
|---|---|
| llama.cpp already exposes a sequence-state ABI | `llama_state_seq_get_data/set_data`, `LLAMA_STATE_SEQ_VERSION 3` — `~/llama.cpp/include/llama.h:874` |
| llama-server exposes HTTP slot save/restore | `POST /slots/:id_slot` — `~/llama.cpp/tools/server/server.cpp:286`; needs `--slot-save-path`, **not set** in `~/llama-swap/config.yaml` |
| vLLM 0.24.0 has the V1 KV connector API | `KVConnectorBase_V1` in `~/venvs/jlens-m36v`; `shared_storage_connector` no longer exists in this version |
| cfrproxy already implements the prompt-corpus layer | `internal/proxy/prefixcache.go` (content-addressed prefix manifests), `cachelog.go` (cache_n / cache_source / cache_reason), `routesticky.go` (affinity routing) |
| kvwarm already probes model identity from the live server | `model_path + ftype + chat_template + bos + eos` → adopted verbatim as KVX `identity.l0_sha256` |
| No new torch install needed | `~/venvs/jlens-m36v` = torch 2.11.0+cu130 / transformers 5.13.1 / vllm 0.24.0; `~/ComfyUI-rocm/venv` = torch 2.9.1+rocm6.4 / transformers 5.15.0 |
| KV quant differs across the fleet | CUDA 27B runs `kvarn4`/`kvarn6`, W6800 27B runs `q8_0` → opaque (tier-1) blobs are **not** interchangeable between them |
| transformers 5.x changed the Cache API | **Verified on `~/venvs/jlens-m36v` (transformers 5.13.1):** `key_cache`/`value_cache` are GONE. The accessor is now `cache.layers[i].keys` and `.values`, each `[batch, kv_heads, seq, head_dim]`; `DynamicLayer` also carries `is_initialized`, `dtype`, `device`. `cache.update(k, v, layer_idx)` still returns a `(keys, values)` tuple. The HF adapter MUST use `.layers[i].keys/.values`, not the 4.x names |

#### Items

| Item | Status | Evidence |
|---|---|---|
| 1. Plan the build in detail for a local model to execute | ✅ done | 12 milestones, frozen container spec, binary done-criteria per milestone |
| 2. Make it model-agnostic | ✅ designed | Capability tiers probed at runtime (PROMPT/OPAQUE/CANONICAL/MAPPED); identity read from the live server; every path degrades to native prefill |
| 3. Sidecar behind the proxy | ✅ designed | kvxd consumes `~/.cfrproxy/cache/**` manifests and adopts cfrproxy's fingerprint formula — zero proxy changes in v1 |
| 4. M0 repo hygiene | ✅ done | `.gitignore` extended; `timeline.md` created; `docs/research-findings.md` written |
| 5. M1 package + manifest kvx/0.2 | ✅ done | `kv_rosetta/manifest.py` + `cli.py`; golden fingerprint unchanged (see below) |
| 6. M2 KVX container + dtypes | ✅ done | `kv_rosetta/container.py`, `dtypes.py`; byte layout verified independently |
| 7. M3 adapter ABC + conformance | ✅ done | `kv_rosetta/adapters/base.py`, `conformance.py`, tier enum, registry |
| 8. M5 metrics + quality gate | ✅ done | `kv_rosetta/metrics.py`, `gate.py`; adversarial case verified |
| 9. M8 GGSQ v3 blob parser | ✅ done | `kv_rosetta/adapters/ggsq.py`; transposed-value field order checked against the C++ |
| 10. M9a RoPE apply/strip | ✅ done | `kv_rosetta/mappers/rope.py`; exact inverse verified |
| 11. M6a store + watcher + budget | ✅ done | `kv_rosetta/store.py`, `daemon/watcher.py`, `daemon/budget.py` |
| 12. Remaining test files (container, adapters, store, budget, ggsq) | 🔴 outstanding | Generation blocked by repeated upstream 502s; implementations are in place and independently verified |
| 13. Steer P0 hardening (identity, segments, integrity, gate, envelopes, capabilities, conformance) | ✅ done | Commits 29d55d0, 6e7c1fc, e433052, 180795c |
| 14. llama.cpp HTTP adapter + 256-token same-backend round trip | ✅ done | `cache_n=255 prompt_n=1`; token IDs identical; max top-5 probability delta 0.000e+00; 423ms → 97ms prefill. See `docs/research-findings.md` §10 |
| 15. Context ladder 2K / 8K / 32K | ✅ measured | 3B parity held at every rung; restore slower than prefill with f16 KV. See `docs/research-findings.md` §11 |
| 16. Steer P0-A..F (version labelling, slot binding, identity, streaming, conformance, CI) | ✅ done | Commits 6bc7566, b343c82, b40fe35, f479edc, 2375989, 953c8c3, 827d316 |
| 17. Diagnose 27B cache_n=0 | ✅ classified | Hybrid attention+recurrent architecture (`qwen35`), NOT MTP. One variable at a time; MTP, flash attention, tensor split, GPU offload and size all exonerated. See §13 |
| 18. Measure real q4 economics (3B), 3 repetitions with ranges | ✅ measured | q4_0 KV: 10.1 KB/token, restore ~5.4x cheaper than prefill at 8192, tmpfs and NVMe within noise. See `docs/research-findings.md` §14 and §16, and `bench/*.json` |
| 19. Quantized KV is not token-exact | ✅ measured | llama.cpp's OWN cache reuse diverges from a cold prefill at q4_0; model is deterministic; parity and native_cache_parity move together. Attribution controlled |
| 20. 27B q4 ladder | 🔴 blocked | Every large model on this host is qwen35/qwen35moe (hybrid). Only non-hybrid model above 3 GB is a 5.8 GB f16 build of the same 3B |
| 21. Hybrid failure mechanism | ✅ proven | Runtime's own trace: "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory)". Context checkpoints are server-side state that `llama_state_seq_save_file` does not carry. See `docs/research-findings.md` §15 |
| 16. CUDA↔HIP canonical transfer, HF reference adapter, vLLM connector | 🔴 outstanding | Gated behind the ladder |

#### Verification evidence (independent, not the generated tests)

| Claim | Method | Result |
|---|---|---|
| kvx/0.1 fingerprints survive the kvx/0.2 extension | Loaded the pre-change module and the new one side by side against the same manifest | Both `6e9e1fa7e16699f781403a2555844cde142a1bed4746dc0087c25f300b7b39d2` |
| `rope_applied` / `kv_tokens` excluded from identity | Two manifests differing only in those fields | Identical fingerprints |
| RoPE matches the HuggingFace convention | Hand-computed the pairwise rotation for D=4 at position 3 and compared | Exact match |
| `strip_rope` is the true inverse of `apply_rope` | Round trip at position offset 100000, theta 1e6 | Max error 4.4e-16 |
| The gate never admits on tensor similarity | Constructed near-parallel logits with the argmax moved on every position | cosine 0.999980, top-1 0.000, **admitted False**; `tensor_cosine` absent from the failure logic |
| KVX round trip is lossless and aligned | f32/f16/bf16 write→read, offset check, byte flip, truncation, mmap vs non-mmap | Bit-identical; offset 64-byte aligned; corruption caught; no raises on bad input |
| GGSQ transposed-value branch | Compared the parser against `llama-kv-cache.cpp:2277` | Reads `v_size_el` as uint32, not a uint64 row size — correct |
| CLI behaviour preserved | Ran `validate`, `fingerprint`, `compat` plus the new `inspect` | rc=0; fingerprint unchanged; `inspect` reports integrity ok |

#### Files
- `kv_rosetta/` — manifest, container, dtypes, metrics, gate, store, cli, adapters/, mappers/, daemon/
- `tests/` — 63 tests passing (`python3 -m unittest discover -s tests`)
- `docs/research-findings.md` — runtime seams, cfrproxy surface, tier design, GGSQ layout, environment

#### Notes on generating with a local model

Fanned out to Tiel-Coder 35B-A3B (2x W6800) via `fred:9069`. Five distinct failure modes, all
recorded in `docs/research-findings.md` §8: oversized prompts stall reasoning-native models;
lossy prompt compression corrupts specs; reasoning tokens consume the output budget leaving
empty `content` (and an empty file still compiles); long multi-file generations exceed the
proxy's upstream timeout; a cold model returns 502 under concurrent load.

The three format-critical keystones — `manifest.py`, `container.py`, `cli.py` — were written
by hand after repeated truncations, because every other module imports through them. Every
model-authored test file inspected so far contained at least one wrong assertion against
correct code, so each module's central claim was re-verified independently rather than trusted
to a green suite.

#### Environment recorded (do not re-derive)

- CUDA: 2× RTX 3090, `sm_86`. ROCm: Radeon PRO W6800, `gfx1030`.
- llama.cpp CUDA build `~/llama.cpp/build/bin/libllama.so` @ commit `ca3d5a3`. HIP builds exist
  only in unrelated forks under `~/tiel-dflash-*` — a same-commit HIP build is still required.
- Sandbox for sub-agents: `python:3.12-alpine`, `--network none`. **numpy is unavailable there**
  and the Docker registry is unreachable from this shell, so numpy-dependent modules are
  syntax-checked in-sandbox and unit-tested on the host.

### REQ-003 — Persist hybrid context checkpoints (research)

Source: steer branch `codex/steer-hybrid-checkpoint-persistence` (bdaa391), which supersedes
the conclusion that hybrid architectures are a dead end.

| Item | Status | Evidence |
|---|---|---|
| R0. Pin source truth at the exact revision | ✅ done | `docs/hybrid-checkpoint-research.md`: llama.cpp `ca3d5a3`, binary `b1-3e73446`, sequence-state version 2 emitted vs 3 declared |
| R0. Retain the unpatched failure as a live negative control | ✅ done | `tests/test_hybrid_negative_control.py`, 5 tests passing against a live `qwen35` server |
| Correct the earlier over-broad interpretation | ✅ done | findings §13/§15, README and `gguf.py` now state the narrower truth; refusal behaviour unchanged |
| R1. Choose the smallest correct llama.cpp persistence patch | ✅ decided | Base on upstream PR #26004 (appends a versioned `SCKP` payload inside the slot save file). Smaller than both issue options, no file-pairing failure mode, carries `data_tgt`/`data_dft`/`data_spec`, ships a server test. See `hybrid-checkpoint-research.md` §R1 |
| R2. Exact-boundary checkpoint | 🔴 next | PR #26004 persists only checkpoints that already exist; reuse is bounded by `--ctx-checkpoints`/`--checkpoint-min-step`. Synthesising `pos_min = 0` is a correctness trap, not a shortcut |
| R3. Reproducible patch + guarded build script | 🟡 partial | `patches/llama.cpp/0001-persist-slot-prompt-checkpoints.patch` (sha256 `baf44e7c…`) + `scripts/build_patched_llama.sh`, which refuses a tampered patch or an unexpected base, unshallows before verifying, and applies three-way. Not yet built |
| R3. Build the patched runtime | ✅ done | `/mnt/storage/llama-kvx-patched`, patch verified and applied three-way; `-DGGML_CCACHE=OFF` after a root-owned ccache dir broke the build; probe corrected to check `libllama-server-impl.so` |
| R6. Restart persistence, measured once | ✅ measured | Save, full server stop, fresh start, restore -> `cache_n=252 prompt_n=4`, parity held. Orchestrated by hand |
| R6. Restart persistence, harness-owned | ✅ **proven by retained test** | `tests/test_hybrid_restart_harness.py` owns both process lifetimes: verifies `/proc/<pid>` gone, new pid differs, fresh process reuses nothing before restore. ~58 s. Corrects an earlier overclaim that the in-process test proved the restart |
| R2. Exact-boundary checkpoint | ✅ resolved without a patch | Uncovered tail measured **constant at 4** across prompts 256/1024/4096 and checkpoint settings min-step 128/8, ctx-checkpoints 8/32 - not a granularity artifact, so forcing a boundary checkpoint would not help. Adapter invariant generalised to a bounded, self-consistent tail |
| Patched/unpatched test matrix | ✅ proven by retained test | `tests/runtime_matrix.py` classifies the runtime from SCKP format evidence; negative control skips (5 skipped) instead of failing (2 failures) against a patched binary |
| Artifact-size heuristics removed | ✅ done | Checkpoint presence asserted from the `SCKP` magic, not from a 200 MB threshold |
| Honest protocol semantics | ✅ proven by retained test | Serialization reported separately from proven support; draft/speculative now `false` (untested) rather than `true`; 10 offline gating tests cover absent/false/malformed/unknown protocols |
| Compound format + coverage contract | ✅ proven by retained test | Artifact labelled `ggsq/3+sckp/1` from its own bytes; coverage recorded and enforced (declared 252 == observed 252); truncated appendix refused, not degraded; capability lifted only by the advertised protocol |
| R3. Runtime capability advertisement | ✅ proven by retained test | Local patch 0002 adds `/props` fields (`sckp/1`, seq version, per-blob flags) and save/restore coverage metadata. Declared coverage 252 == observed reuse 252 |
| Binary provenance verified | ✅ proven | Running binary = commit `3e7344670` (seq version 2); HEAD `ca3d5a3` is 150 commits ahead (version 3); `~/llama.cpp` is a shallow clone, which fails ancestry checks silently |
| R2–R10 | 🔴 outstanding | Gated behind R1 |

#### Correction recorded

I had concluded hybrid restoration was impossible because recurrent state "has no prefix to
match against". Too broad. State after an **exact** token sequence is deterministic and
restorable; llama.cpp simply does not persist the context checkpoint those models resume
from. Source-confirmed at `ca3d5a3`: the save handler serialises only `prompt.tokens`, and
the restore handler calls `slot->prompt.clear()`, which drops `checkpoints`.

**REQ-003 status: IN-PROGRESS.**

**REQ-002 status: IN-PROGRESS** — core modules landed and verified; runtime adapters (M4a/b/c) next.

### REQ-001 — Start portable KV Rosetta core

Source: prior session (commit `7675906`, "feat: start portable KV Rosetta core").

#### Items

| Item | Status | Evidence |
|---|---|---|
| 1. Define the device-neutral KVX manifest boundary | 🟡 built | `kv_rosetta.py` — `ModelABI`, schema `kvx/0.1`, canonical layout `layer,kv,token,head,dim` |
| 2. Prove hardware is provenance, not identity | 🟡 built | `tests/test_kv_rosetta.py::test_hardware_backend_does_not_change_identity` — cuda and hip manifests share a fingerprint |
| 3. Classify cross-model transfer paths | 🟡 built | `compatibility()` returns exact / linear-candidate / learned-candidate |

#### Files
- `kv_rosetta.py` — manifest validation, ABI fingerprint, compatibility planner, CLI
- `tests/test_kv_rosetta.py` — 4 tests
- `README.md`, `pyproject.toml`

#### Verify
- `python3 -m unittest discover -s tests -v` — 4 tests, ok

**REQ-001 status: COMPLETE.**

## REQ-018 — Paired patched/unpatched restart matrix on the production 27B

Steer item 2 and 3. The prior restart evidence had only a positive leg, so it could not
separate "the patch persists checkpoints" from "this model reuses prefixes anyway". This
runs both halves against the same model, the same 256 prompt tokens, and two builds of the
same commit that differ only by the patches.

Model    `/mnt/storage/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf`
         arch `qwen35` (hybrid), content digest `9bf3b07e1fb6531e91d970384cc3bdbc...`
Builds   patched `b10665-ca3d5a3e1` / control `b151-ca3d5a3e1` - same base commit
         impl .so `1ddfe67d0e3425ac` vs
         `4b036b5259fc5ede`
Record   `bench/production-27b-matrix.json`

| | patched | control (unpatched, same commit) |
|---|---|---|
| advertises `slot_checkpoint_persistence` | yes, `sckp/1` | no |
| patch marker strings in impl .so | 4 | 0 |
| cold prefill | cache_n=0 prompt_n=256 | cache_n=0 prompt_n=256 |
| native in-memory reuse | cache_n=252 prompt_n=4 | cache_n=252 prompt_n=4 |
| fresh process, before restore | cache_n=0 prompt_n=256 | cache_n=0 prompt_n=256 |
| **after restore into a fresh process** | **cache_n=252 prompt_n=4** | **cache_n=0 prompt_n=256** |
| restore reports `n_restored` | 263 | 263 |
| artifact size | 465 MiB | 166 MiB |

**Both legs report `n_restored=263`. One reuses 252 tokens; the other reuses nothing and
re-prefills all 256.** That is the measurement behind the rule that a restore is never
called successful on `n_restored` alone - on this model the field is identical whether the
reusable state survived or not.

The 299 MiB difference between the artifacts is the checkpoint appendix; the
patched leg's restore response carries `checkpoint_n_tokens=252`, matching the 252 tokens
actually reused. The uncovered tail of 4 appears in native in-memory reuse on both builds,
so it is a property of the runtime's prefix matching, not of restoring.

Status: **proven by retained artifact** (`bench/production-27b-matrix.json`, one run on this
host). The runner refuses to write a record unless both legs complete, and fails rather than
skips if the control binary advertises checkpoint support.

Caveat: measured once, on one host, at one prompt length (256 tokens). The economic
question - whether restoring beats prefilling - is separate and not answered here.

## REQ-019 — Fail-closed export gate, then the 27B matrix re-run against it

Steer 53c390a, P0 then P1 then P2. Audited every assertion in that steer against the code
first; all held. Two were stale against the branch head rather than its stated basis: the
magic scan had already been tightened to require EOF termination (cdd698e), and the paired
27B record had landed (846c566).

**P0/P1** — `export()` reached the slot-save POST after checking only the requested
representation and the slot path. An unpatched hybrid runtime would answer with a plain
sequence-state artifact that restores, reports `n_restored`, and reuses nothing. 15 retained
tests in `tests/test_hybrid_export_gate.py` pin the refusals; the load-bearing assertion is
that no save POST is issued, since refusing afterwards still implies the runtime was
consulted. Compound labelling is now bound to `n_written - checkpoint_bytes` rather than to
a magic found anywhere, and `int(x or -1)` no longer rewrites position 0 as absent.

Patch 0002 gained `active_checkpoint_state_classes`. Verified on the rebuilt server: it
reports `["target"]` for a launch with no draft model. Without it, a target-only
configuration cannot be told apart from a speculative one and both must be refused.

**P2** — re-ran the paired matrix against the rebuilt binary, with the steer's acceptance
criteria enforced by the runner rather than merely recorded. Both legs pass:

| | patched | control |
|---|---|---|
| after restore | cache_n=252 prompt_n=4 | cache_n=0 prompt_n=256 |
| active state classes | ['target'] | None |

Now enforced, not just recorded: the port closes before the second process starts; output
tokens *and probability vectors* match native in-memory checkpoint reuse; cache_n equals the
declared `checkpoint_n_tokens`; cache_n + prompt_n equals the token count; the uncovered tail
is within the ceiling of 8; and every checkpoint metadata field is identical between the save
and restore responses.

Status: **proven by retained test** (offline gate) and **measured once on this host**
(the 27B matrix). Deliberately not done, per the steer: the context ladder stays deferred.

## REQ-020 — Correcting two evidence defects in the 27B record

Steer cd2afb0, audit corrections 1-3. Both defects verified against the retained record
before acting; both were real, and the first was a false claim of mine.

**Defect 1 - the probability parity was vacuous.** REQ-019 claimed restored probability
vectors matched native in-memory reuse. llama.cpp returns the per-token alternatives under
`top_logprobs` (carrying `logprob`) by default; `top_probs`/`prob` appear only when
`post_sampling_probs=true` is requested. The runner read `top_probs`, got nothing, and built
a list of empty dicts for both sides - so the equality compared nothing. The retained record
showed it plainly: eight empty objects. Token-ID and content parity in that record were real;
the probability parity was not, and should not have been claimed.

Now enforced: one vector per generated token, each nonempty, the generated token present in
its own vector, no more entries than requested, and logprobs equal within a declared 1e-6
tolerance. Both native and restored vectors are stored. Measured: 8 tokens x 5 alternatives,
native and restored **exactly** equal.

**Defect 2 - the record named a commit that lacked the runner.** It said `repo_commit
a0540a7`, but the acceptance logic landed in `6524714`: the modified runner was executed
before it was committed. The runner now refuses to start with a dirty worktree and records
its own SHA-256 (`1f8e36acc997bfdc`), the HEAD and worktree-diff digest of both
llama.cpp source trees, both builds' cmake flags, and the full launch argv of all four
processes. This record is from commit `56600b98b0ac`, clean tree.

**Defect 3 - model naming.** The tested file is `Qwen3.8-27B-UD-Q4_K_XL.gguf`, digest
`9bf3b07e1fb6531e91d97038...`, architecture `qwen35`. It is the file
loaded by the fleet's `qwen38-27b` llama.cpp entries, so it is the deployed production SKU
for llama.cpp serving. No Qwen3.5-27B is served on this host; Qwen3.6-27B variants exist
under other fleet entries and were not tested. The evidence is stated for this exact digest,
not transferred to any other 27B.

Result unchanged and now properly evidenced: patched `cache_n=252
prompt_n=4`, control `cache_n=0
prompt_n=256`.

Status: **measured once on this host**, from a clean committed runner.
Still outstanding from this steer: P0 (one predicate across capability/export/import),
P1 (streaming appendix validation), P2 acceptance items not yet covered, P3 ladder.
