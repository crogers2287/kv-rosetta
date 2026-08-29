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

## REQ-021 — One support predicate across capability, export and import

Steer cd2afb0, P0. Audited each assertion against the code first; all held.

Three surfaces answered "is this runtime supported" independently and disagreed.
`capabilities()` advertised OPAQUE for any complete protocol and only *appended a note*
about draft/speculative; `export()` additionally required a tested compound tuple and a
provably target-only launch; `import_()` checked neither. A caller who trusted
`capabilities()` and then exported got a refusal, and one who imported got no check at all.

`hybrid_support()` is now the single decision, called by all three. A note is not a gate:
an unproven active state class, an unreported set of active classes, or an untested compound
tuple now yields empty capabilities rather than an advertised one with a warning attached.
A retained test asserts capability and export agree across six configurations.

Also closed on the import side:

- The same predicate refuses before any restore POST, so unusable state never reaches the
  runtime.
- A compound artifact declaring zero or missing checkpoint coverage is refused before
  restore.
- Every checkpoint metadata field - count, bytes, n_tokens, pos_min, pos_max - must match
  between the manifest and the restore response; each mismatch is covered independently.
- `verify_reuse=False` can no longer bless a compound import. The whole claim is that a
  checkpoint survived, and only a probe shows that, so verification is forced on for this
  path.
- `_restore_pristine()` swallowed a failed re-restore with `except AdapterError: pass`. The
  verification probe appends its own token to the slot, so on failure the slot held the
  prefix *plus* that token while the caller was handed `ok=true` - the cache described was
  not the cache present. It now erases the slot and returns `ok=false`.

The launch's active state classes and the protocol tuple are bound into `CacheABIIdentity`
flags, so a cache written by a draft-carrying launch no longer shares an identity with a
target-only one.

Status: **proven by retained test** — 14 new tests in `tests/test_hybrid_support_predicate.py`,
342 offline total. Not yet done from this steer: P1 (streaming appendix validation),
remaining P2 acceptance items, P3 ladder.

## REQ-022 — Bounded appendix validation

Steer cd2afb0, P1. Audited and measured before changing anything: on a 400 MiB slot file,
validating a 112-byte appendix grew peak RSS by **400 MiB**. The production 256-token
patched slot is already 487,926,936 bytes and grows with the context, so this does not
survive the 2K/8K/32K ladder.

`parse_checkpoint_appendix()` and `checkpoint_appendix_at()` both called
`Path.read_bytes()`. They now parse by seeking: 4 bytes of magic, a 12-byte header, then
12 + 24 bytes of framing per checkpoint. Payload lengths are skipped by arithmetic and never
read or allocated, and each length is bounds-checked against the 16 GiB per-buffer cap and a
new aggregate cap before it is added to the offset — so a corrupt size field cannot make the
parser reserve anything.

Production export knows the boundary exactly, because the runtime reports `n_written` and
`checkpoint_bytes`, and uses `checkpoint_appendix_at()`. The general classifier still scans,
but in bounded 4 MiB chunks rather than by reading the file whole.

Re-measured after the change: **0.0 MiB** peak RSS growth on the same 400 MiB file.

Retained proof: a 4 GiB sparse fixture with a read-counting wrapper asserts validation reads
at most 256 bytes; a 16 GiB declared buffer length is rejected by arithmetic (the test would
exhaust memory rather than fail if it were ever read); a length pointing past EOF reports
truncated without reading. The sparse tests skip themselves if the filesystem does not keep
the fixture sparse, since the subject is the reader, not the writer.

Status: **proven by retained test** (24 tests in the file, 345 offline total) and
**measured once on this host** (400 MiB -> 0.0 MiB).

## REQ-023 — End-to-end KVX round trip, and a layout bug the offline suite could not see

Steer cd2afb0, P2's remaining acceptance item.

**A parser bug the tests were blind to.** The SCKP parser read 12-byte checkpoint records
(three int32). The patch writes 16: `n_tokens` is `int64_t`, `pos_min`/`pos_max` are
`llama_pos` (int32). Every fixture packed the same wrong shape, so 24 tests passed against
a layout the writer never produces. It took the live export to expose it - the adapter
refused its own artifact with *"no usable checkpoint appendix at the declared offset
174138116 (malformed)"*. The derived offset was correct; the record size was not.

This failed **closed**: a real appendix was rejected rather than a bad one accepted, so no
artifact was ever wrongly blessed. But the gate refused every genuine hybrid export, and the
tests asserting the parser worked were measuring their own assumption. The record size is now
a named constant pinned against framing captured from a real slot file, with a test asserting
the old 12-byte shape no longer validates.

**The uncovered tail is not a constant.** The first round trip failed with `cache_n=252
prompt_n=11` for 263 tokens. The artifact had been saved from a slot holding 256 prompt plus
8 generated tokens, while the checkpoint still covers 252. The raw legs show a tail of 4 only
because they replay a 256-token request. The tail is a function of how far the sequence ran
past the last checkpoint, so the ceiling of 8 does not generalise. Rather than widen the
bound, the matrix now exports what a prefix cache actually holds: prefill, then save before
generating.

**Result.** Both legs pass every P2 acceptance item:

| | patched | control |
|---|---|---|
| hybrid_support | True | False |
| capabilities export | `['opaque']` | `[]` |
| active state classes | `['target']` | none reported |
| adapter export | 315 MiB in 2.27s | refused, as required |
| adapter import | ok, cache_n=252 of 256, 4 reprocessed | no artifact |
| after restore | cache_n=252 prompt_n=4 | cache_n=0 prompt_n=256 |

**Economics, stated plainly.** At 256 tokens this is a correctness gate and a clear loss:

- native cold request: **0.540 s**
- raw runtime restore: 0.421 s
- tail completion after restore: 0.426 s
- **end-to-end adapter import: 3.795 s**

The end-to-end number is ~7x the cold request. It includes what a caller actually waits for:
outer KVX verification over 315 MiB, staging that payload to the slot
directory, the runtime restore, and the mandatory reuse probe. Save time is excluded, since
it is not paid on the request path. Nothing here suggests a win at 256 tokens; whether the
curve crosses is what the ladder is for.

Status: **measured once on this host**, from a clean committed runner
(`a9e617e54200`). The parser fix is **proven by retained test** (349 offline).

## REQ-024 — Close the compound-header bypass

Steer 233a897, P0. Audited against the code first; the finding is exact.

`import_()` checked compatibility against `header["blob"]["opaque_format"]` but selected
compound behaviour from whether `header["coverage"]["format"]` contained a `+`. Nothing
required the two to agree, and `container.verify()` checks integrity, not consistency. So a
correctly hashed artifact whose blob said `ggsq/3+sckp/1` - matching the live runtime, so it
passed the compatibility check - but whose coverage format was missing or plain reached a
compound-capable runtime with `is_compound=False`. That skipped the nonzero-coverage check,
the restore-metadata comparison, and forced reuse verification, so `verify_reuse=False`
could return success on `n_restored` alone: exactly the claim this project exists to refuse.

The blob format is now authoritative and the decision is an exact tuple comparison, not a
search for a plus sign - an unknown tuple must not read as compound support. Coverage must
equal the blob format exactly, in both directions; a compound blob must carry a coverage
object with complete nonzero fields; and a plain blob whose coverage claims compound is
refused too.

Coverage counts are also coerced defensively. A manifest is untrusted input, and
`int(value or 0)` raised out of the adapter boundary on `'many'`, `NaN`, a list, or a dict -
the report contract says a bad artifact yields `ok=false`, never an exception. Non-integers
now produce a refusal naming the offending fields.

Every case is covered with `verify_reuse` both true and false, using fixtures rebuilt and
re-hashed so they pass `container.verify()` - a corrupt fixture would be caught by integrity
alone and would prove nothing about format agreement.

Status: **proven by retained test** — 7 tests in
`tests/test_compound_format_agreement.py`, 356 offline total.

## REQ-025 — The sealed 256-token adapter gate

Steer 233a897, P1. Run from committed runner `fd4e9d01fe0a`
(runner digest `3581b5033e54e749`), clean worktree.

Both legs pass every acceptance item. The patched leg exports one
`ggsq/3+sckp/1` artifact through the public adapter and imports it into a fresh
process; the unpatched leg advertises nothing, refuses the export, and refuses the patched
artifact.

**Artifact identity** — container `9564029f75cf45cb`, payload
`744a57f7a5defef6`, key `d06a04f8ff22cd50`.
Total 330,575,504 bytes = 173,679,168 sequence +
156,894,416 checkpoint + 1,920 container
overhead. Coverage: 1 checkpoint, 252 tokens, positions 251..251.

**After the adapter import**: `cache_n=252
prompt_n=4`, tokens, content and probability vectors
all matching native in-memory reuse. Staged copies left behind: none.

**Endpoint evidence** is now mechanical, and it corrected a claim. The unpatched export made
**no calls at all**. The cross-import refusals made no restore call. Note that
`calls_during_capability_probe` shows a save: `state_version()` probes the emitted sequence
version by saving a slot, since no endpoint reports it - which is what I previously
misattributed to the export.

**The unpatched runtime refuses the patched artifact on cache ABI identity**, not on the
support predicate - `artifact fa9d63f33e30 vs expected 049ccd656b97`. That is an earlier and
stronger gate than the one the steer asked for, because the active state classes and
protocol tuple are bound into `CacheABIIdentity`. It holds with `verify_reuse` both true and
false.

**Phase attribution — the point of this run.** At 256 tokens the adapter path loses badly,
and now we know to what:

| phase | seconds |
|---|---:|
| staging (extract 315 MiB payload) | 1.295 |
| container verification (digest 315 MiB) | 1.054 |
| pristine re-restore | 0.275 |
| runtime restore | 0.250 |
| reuse probe | 0.212 |
| **total end-to-end** | **3.693** |
| *native cold request* | *0.588* |

The runtime restore is 0.250 s - faster than the 0.588 s
cold request. Everything above it is KV Rosetta's own overhead: staging and integrity
verification together cost 2.349 s, both linear in the
315 MiB payload. The loss is I/O over the artifact, not the model.

This is a correctness gate. It establishes no economic win at 256 tokens, and nothing here
projects one at any other length.

Status: **measured once on this host**, from a clean committed runner.

## REQ-026 — The 2K rung, on tmpfs

Steer 233a897, P2. Three clean repetitions, fresh processes per leg, from committed runner
`67efd9813f4c`. Slots on tmpfs to isolate compute and serialization from disk.

**Correctness holds.** All three patched repetitions restore to `cache_n=2044 prompt_n=4`;
all three unpatched controls to `cache_n=0 prompt_n=2048`. The uncovered tail is still 4 at
2048 tokens, the same as at 256 — it tracks the checkpoint boundary, not the prompt length.
Token, content and probability-vector parity against native in-memory reuse held in every
repetition.

**Artifact**: 604,960,596 bytes = 291,169,856 sequence +
313,788,820 checkpoint + 1,920 container
overhead, covering 2044 tokens in 2 checkpoints. Server peak RSS
17.06 GiB; runner peak RSS
61 MiB — the bounded appendix parsing holds at
this size.

**Economics: not cheaper, and the reason is ours.**

| phase (median of 3) | seconds |
|---|---:|
| staging | 2.375 |
| container verification | 1.987 |
| pristine re-restore | 0.473 |
| runtime restore | 0.465 |
| reuse probe | 0.212 |
| **adapter import + tail** | **6.528** |
| *native cold prefill* | *1.739* |

Repetitions are tight: 6.436, 6.544, 6.528 s against
1.710, 1.739, 1.750 s, ratios 3.76x, 3.76x, 3.73x.

The **runtime restore is 0.465 s against a 1.739 s
cold prefill** — the runtime half is already ~3.7x cheaper than prefilling. Staging plus
container verification cost 4.362 s, 67% of the
total, and both are linear in the 577 MiB payload. On tmpfs, so
this is memory bandwidth and SHA-256, not disk.

Two measured points, stated without extrapolation:

| tokens | cold | adapter+tail | ratio |
|---|---:|---:|---:|
| 256 | 0.588 | 4.103 | 6.98x |
| 2048 | 1.739 | 6.528 | 3.76x |

Cold prefill grew 3.0x over an 8x token increase; the adapter path grew 1.8x. The gap is
narrowing. Whether it closes is the next rung's measurement, not something to be projected
from two points.

**K/V cache dtype is not recorded, because this runtime does not report it.** The steer asked
for it explicitly and warned that Q4 weight quantization does not imply K/V dtype. Probed the
patched build's `/props` directly: the only dtype-ish key is `model_ftype`, which is exactly
the weight quantization being warned about. There is no `cache_type_k`/`type_k` anywhere in
props, settings, or params.

Consequence worth flagging: `CacheABIIdentity.k_dtype`/`v_dtype` are empty strings on this
runtime, so **the cache ABI digest does not currently bind the K/V cache dtype**. Two
runtimes differing only in KV quantization would produce the same digest. The exposure is
cross-runtime transfer, which this track has deferred, and the GGSQ payload does encode
per-layer `k_type`/`v_type` that could be read from the artifact bytes instead. Not fixed
here: it is outside this steer's stated order.

Status: **measured once on this host, three repetitions** (2K tmpfs).
**Untested**: the NVMe rung, 8K, 32K.

## REQ-027 — The 2K rung, re-measured with the adapter's own timers

Steer 0315e90, P0 then P1. Both of its measurement-attribution findings were real; I
confirmed each against the code before changing it.

**The previous 2K verdict was built on a borrowed timer.** REQ-026 reported
`adapter+tail = 6.528 s`, where the tail came from the completion after the *raw* endpoint
restore. The completion after the *adapter* import was checked for parity but never timed.
The two states do behave alike - measured now, the adapter tail is 0.430 s
against the raw path's 0.421 s - but parity does not turn a borrowed
timer into a measurement of the thing it stands in for. `compute_verdict()` now refuses
outright rather than substituting, and a retained test proves changing only the raw tail
cannot move the verdict.

**Phase attribution was incomplete**: 6.135 s reported against a
phase sum that previously left ~0.52 s unclassified — the `state_version()` probe inside
`opaque_format()`, plus the header read and identity checks. Those are now a named
`preflight` phase. This run reconciles to **0.0049 s unclassified**.

**Result — correct, and still not cheaper.** One clean repetition, tmpfs
(`/dev/shm/kvx-ladder`, fstype confirmed by `findmnt`), runner `053f2b9d5a00`:

| phase | seconds | share |
|---|---:|---:|
| staging | 2.404 | 39.2% |
| container verification | 1.986 | 32.4% |
| preflight | 0.542 | 8.8% |
| runtime restore | 0.515 | 8.4% |
| pristine re-restore | 0.477 | 7.8% |
| reuse probe | 0.207 | 3.4% |
| **adapter import + tail** | **6.664** | |
| *native cold prefill* | *1.772* | |

Correctness held: `cache_n=2044 prompt_n=4` patched, `cache_n=0 prompt_n=2048` unpatched,
with token, content and logprob parity against native reuse. Artifact 604,960,596
bytes = 291,169,856 sequence + 313,788,820 checkpoint +
1,920 overhead.

**Branch decision.** Ratio is 3.76x, above the steer's 1.25x threshold, so
repetitions and 8K are **not** the next work. The attribution points at one thing: container
verification and staging are **two separate full passes over the same
577 MiB payload**, together 4.390 s, 66% of the total, on tmpfs -
so memory bandwidth and SHA-256, not disk. Meanwhile the runtime restore is
0.515 s against a 1.772 s cold prefill: the
runtime half is already ~3.4x cheaper than prefilling.

Next per the steer: a one-pass verified staging primitive - validate the header, then hash
the payload while writing the staged file, compare the digest before any restore POST, and
delete the staged file on mismatch. That removes one of the two passes without weakening
fail-closed behaviour.

Status: **measured once on this host** (one repetition, by design - this was the
break-first run). **Untested**: NVMe, 8K, 32K, and the one-pass staging hypothesis.

## REQ-028 — One-pass verified staging

Steer 0315e90, following its measured 2K branch: ratio above 1.25x, so attack the largest
linear I/O pass rather than run repetitions or 8K.

The import made **two full passes over the same payload**: `container.verify()` hashed it,
then `extract_payload()` hashed it again while writing the staged copy. The second pass
already provided the guarantee — extraction compares the digest before publishing the file —
so the first was learning the same fact twice.

`verify_header()` now does everything `verify()` does except the payload read: structure,
header digest, blob fields, payload alignment, and that the file is long enough for what it
declares. Import calls it and relies on extraction's in-pass comparison, which happens before
any restore POST. Segmented artifacts are refused by `verify_header` and fall back to the
full check, since extraction does not cover per-segment digests.

Removing a check is only safe if nothing it caught gets through, so the tests assert the
refusals, not the speedup: a corrupted payload byte, a corrupted header byte and a truncated
artifact are each still refused with **no restore POST issued**, no staged copy or temp file
left behind, and a report returned rather than a `ContainerError` escaping the adapter
boundary — which it would have before, since `import_` only caught `AdapterError`.

**Measured, same model / prompt / hardware, tmpfs, one repetition each:**

| phase | two-pass | one-pass | delta |
|---|---:|---:|---:|
| container verification | 1.986 | **0.000** | −1.986 |
| staging | 2.404 | 2.222 | −0.182 |
| preflight | 0.542 | 0.525 | −0.016 |
| runtime restore | 0.515 | 0.438 | −0.077 |
| pristine re-restore | 0.477 | 0.443 | −0.034 |
| reuse probe | 0.207 | 0.232 | +0.024 |
| **adapter import + tail** | **6.664** | **4.371** | **-2.293** |
| *native cold prefill* | *1.772* | *1.734* | |
| ratio to cold | 3.76x | **2.52x** | |

Correctness unchanged: `cache_n=2044 prompt_n=4` patched, `cache_n=0 prompt_n=2048`
unpatched, full parity against native reuse. Phases reconcile to
0.0039 s unclassified.

**Still not cheaper**, at 2.52x. The remaining dominant phase is staging
at 2.222 s — 58% of the import — which is the
577 MiB payload being copied from the container into the slot
directory and hashed, both on tmpfs. The runtime restore is 0.438 s
against a 1.734 s cold prefill.

The options the steer parked "until the one-pass verified staging result is measured" —
direct-from-container restore, raw artifact plus sidecar manifests, reflinks, format
redesign — are now unblocked by measurement. Not started: choosing among them is the next
steer's call, not mine.

Status: **measured once on this host** (one repetition, matching the break-first design).
**Proven by retained test**: the refusals survive removal of the pass (8 tests, 395 total).
**Untested**: NVMe, 8K, 32K.

## REQ-029 — Cache dtype bound, version probe removed, sealed 256 gate re-run

Steer f081b53, steps 1-4. All three of its findings were real.

**P0 — the K/V cache dtype identity gap.** `/props` exposed no cache dtype at all, only
`model_ftype`, so `CacheABIIdentity.k_dtype`/`v_dtype` were empty and two launches differing
only in K/V cache type hashed identically. Patch 0002 now reports `target_cache_type_k/v`
from the resolved configuration the live context was created with, plus `draft_cache_type_k/v`
when a draft context exists. Verified live: a default launch reports `f16/f16`, one started
with `-ctk q8_0 -ctv q8_0` reports `q8_0/q8_0`, and `model_ftype` stays `Q4_K` in both.

The adapter binds them and refuses when absent, with no fallback to weight quantization — a
confident wrong dtype is worse than an empty one, because it compares *equal* across
genuinely different caches.

Two test stubs were returning a constant cache ABI digest and a constant identity dict,
which made every identity assertion vacuous. They now use the real implementations; that is
what exposed a differing-dtype artifact reaching a restore POST.

**P1 — the version probe is gone.** `state_version()` saved a whole slot to discover a number
a complete protocol states exactly. The advertisement is trusted only when the entire support
predicate holds; anything less still probes. Export derives the version from the emitted bytes
and refuses on disagreement. The support check also moved ahead of the identity and format
work, because an unpatched hybrid was refusing correctly but only *after* `opaque_format()`
had already saved a slot — probing a format it is not allowed to use.

Measured in this record: `calls_during_capability_probe` is `[]` and the import's first
endpoint call is the restore. No save POST occurs anywhere on the patched path.

**Phase attribution regressed twice, and the guard caught it.** Moving the support check left
`hybrid_support()` and `identity()` outside every named phase (0.612 s unclassified), and a
stray clock reset then discarded the identity window (0.510 s). The offline coverage test
could not see either, because its fixtures are small enough that the gap stays inside
tolerance. The runner now **fails** on an unreconciled record instead of printing the
remainder and carrying on — which is what "require reconciliation" has to mean, since the
first defect had already reached a committed record.

**Sealed 256 gate, re-run on the rebuilt binary** (patch 0002 `31f1f74b`, impl `93a4ddcf`):

| phase | seconds |
|---|---:|
| staging | 1.249 |
| preflight | 0.523 |
| pristine re-restore | 0.256 |
| runtime restore | 0.238 |
| reuse probe | 0.212 |
| container verification | 0.000 |
| **adapter import + tail** | **3.039** |
| *native cold request* | *0.595* |

Reconciles to 0.0005 s. Correctness unchanged:
`cache_n=252 prompt_n=4` patched, `cache_n=0 prompt_n=256` unpatched, which refuses export
and both cross-imports with **zero endpoint calls**. Recorded dtypes: `k=f16`,
`v=f16`, beside `model_ftype='Q4_K - Small'` so the record itself shows
they are different things.

Still not cheaper at 256 tokens. Staging remains the dominant phase.

Status: **proven by retained test** (413 offline). **Measured once on this host**: this gate.
**Untested**: the direct-raw 2K lower bound, NVMe, 8K, 32K.
