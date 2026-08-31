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

## REQ-030 — The direct-raw 2K lower bound is cheaper than cold prefill

Steer f081b53, P2. **The first economic win measured on this host.**

The public adapter loses because it copies and hashes the payload. This measures what
remains if that copy were gone: the raw `ggsq/3+sckp/1` state is fully validated *outside*
the timed window, stored under its content digest inside the slot directory, and restored in
place — no extraction, no byte copy.

| rep | runtime restore | reuse probe | pristine re-restore | tail | total | native cold | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.414 | 0.216 | 0.452 | 0.434 | **1.516** | 1.731 | 12.4% faster |
| 2 | 0.418 | 0.212 | 0.460 | 0.437 | **1.526** | 1.788 | 14.6% faster |

Two independent repetitions, both cheaper, with the margin well outside the ~3.5% spread
seen across every 2K cold measurement so far (1.71–1.79 s). The steer inferred a bound near
1.54 s and said the margin was small enough that it had to be measured rather than
projected. Measured: 1.516 s and 1.526 s.

For contrast, the public adapter on the same rung after one-pass staging landed:
4.371 s against 1.734 s cold,
2.52x. Removing the copy is worth about 2.9 s at 2K.

**Correctness is not weakened by restoring in place.** `cache_n=2044 prompt_n=4`; restore
metadata equal to the admitted coverage field by field; token, content and
probability-vector parity against native in-memory reuse; timed phases reconcile to 0.000 s.
The admitted 604,958,676-byte file is byte-identical afterwards — device, inode, size,
mtime_ns, ctime_ns and digest all unchanged — and no temporary file remains. The unpatched
control refuses from support evidence.

**Admission cost 3.1–3.3 s and is excluded from the verdict by design.** It is what makes
restoring in place safe — support predicate, tested tuple, proven active state classes,
present K/V dtypes, sequence framing, the exact prompt tokens the state carries, the SCKP
appendix at the *declared* offset rather than one found by scanning, and a digest over every
payload byte — and it happens once, off the request path. A filename or a prior hash alone
is not proof that the bytes restored are the bytes admitted, which is why the digest is
recomputed after the timed window rather than trusted.

This is a lower bound, not a design. Per the steer's branch it justifies designing the
smallest admitted raw-artifact contract; it is not itself that contract, and nothing here is
production.

Status: **measured twice on this host**. **Proven by retained test**: 413 offline.
**Untested**: NVMe, 8K, 32K, and any admitted-artifact contract.

## REQ-031 — The admitted-store 2K gate FAILS, and the cause is measured

Steer 9a883d9, P0–P2. The contract and path are built and every safety gate passes; the
economic gate does not.

**Result: 0 of 3 paired wins.** Median 2.182 s
(range 2.129–2.185) against median cold
1.730 s (range 1.713–1.733).
By the steer's decision rule this is a Fail: stop storage-format work and profile.

**Every correctness and safety gate passed.** `cache_n=2044 prompt_n=4` in all three
repetitions; restore metadata equal to the admitted metadata field for field; token, content
and nonempty probability-vector parity against native in-memory reuse; the admitted object
byte-identical across every timed window; **zero request-path payload bytes** with 965 bytes
of metadata read; and the unpatched control refusing with **no endpoint call at all**. The
failure is economic, not a safety failure.

**Profile — and my first hypothesis was wrong.** I guessed the regression was the
model-content digest and measured it instead: 1 ms. Splitting `resolve` into sub-phases
located it by measurement:

| phase | median | lower bound |
|---|---:|---:|
| **resolve_support** | **0.652** | *not in the timed window* |
| runtime restore | 0.429 | 0.414 |
| pristine re-restore | 0.441 | 0.452 |
| reuse probe | 0.205 | 0.216 |
| resolve_store | 0.000 | – |
| resolve_abi | 0.000 | – |
| resolve_identity | 0.001 | – |

The three phases the steer asked to profile — runtime restore, reuse probe, pristine
re-restore — all match the lower bound within noise. The entire regression is
`hybrid_support()`, whose cost is `gguf.architecture()` re-reading the model's metadata
block on every restore.

**This also corrects how the lower bound should be read.** That 1.516 s figure did not
include the support predicate in its timed window — it validated at admission and timed only
restore, probe and pristine. So the lower bound was never a like-for-like comparison with a
path that must fail closed on every request. The honest statement is that the restore
mechanics cost ~1.08 s and the fail-closed gate currently adds ~0.65 s on top.

The architecture of a given model file cannot change, so this cost looks cacheable by the
same file-stamp mechanism `weights` already uses. **Not implemented here:** the steer's Fail
branch says stop storage-format work and profile, which is what this is.

Status: **failed** (economic gate, 0/3). **Proven by retained test**: 457 offline, every
guard mutation-checked. **Measured three times on this host**: this gate.
**Untested**: NVMe, 8K.

## REQ-032 — The admitted-store 2K gate PASSES with the support check included

Steer 3cacbba. Both of its assertions were verified against the code before acting, and both
held — including the store-ownership gap, which was mine.

**The parser tax was the whole failure.** `read_metadata()` materialises every array element,
so asking for `general.architecture` built 150k Python strings from the tokenizer vocabulary:
~0.65 s per call on the 27B, paid on every restore by the fail-closed support check.

The support decision is unchanged and still runs on every restore. Only the evidence
retrieval changed — a cursor over an mmap steps past values arithmetically and allocates
nothing but the value asked for. **Measured: 0.65 s → 0.2–0.7 ms.**

Duplicate detection and request-path cost genuinely conflict, so they are split rather than
one quietly dropped. Catching a *later* conflicting duplicate still costs ~0.39 s, because
the tokenizer array must be stepped over. `architecture_exhaustive()` does that at
**admission**, where an ambiguous header is rejected once and never becomes an admitted
object; `architecture()` returns at the key and does not inspect later keys. That limit is
documented and pinned by its own test rather than left implied.

**Result — 3 of 3 paired wins:**

| rep | restore | tail | total | cold | |
|---|---:|---:|---:|---:|---|
| 1 | 1.230 | 0.467 | **1.696** | 1.734 | win |
| 2 | 1.125 | 0.425 | **1.551** | 1.731 | win |
| 3 | 1.085 | 0.424 | **1.509** | 1.737 | win |

Median 1.551 s (range 1.509–1.696)
against cold 1.734 s (range 1.731–1.737),
ratio 0.894.

| phase | before | after |
|---|---:|---:|
| **resolve_support** | 0.652 | **0.0022** |
| runtime restore | 0.429 | 0.431 |
| pristine re-restore | 0.441 | 0.459 |
| reuse probe | 0.205 | 0.222 |
| resolve_store / abi / identity | ~0.001 | ~0.001 |

Decision rule, every condition: `resolve_support` median 0.0022 s < 0.010 s;
median beats cold; 3/3 ≥ 2 paired wins; all correctness and safety gates pass —
`cache_n=2044 prompt_n=4` every run, restore metadata equal to admitted metadata, full parity
against native reuse, **zero request-path payload bytes**, the admitted object unchanged, and
the unpatched control refusing with **zero endpoint calls**.

**A request-path win is not a lifecycle win.** Admission costs 2.834 s
and saves 0.183 s per restore, so it breaks even after
**16 restores**. Below that, admitting is a loss.

Also fixed here, from the steer's audit of my own code: `AdmittedStore` documented a store
owned by the current user but checked only mode 0700, so a 0700 directory owned by someone
else satisfied it. It now requires `st_uid == geteuid()`.

Status: **measured three times on this host** (this gate). **Proven by retained test**: 472
offline, all four new guards mutation-checked. **Untested**: NVMe, 8K, and everything the
steer lists as prerequisites for calling this production-ready — the same-UID/root basename
race, model variants beyond the tested digest, and active draft/speculative state.

## REQ-033 — The 2K admitted path passes on persistent NVMe

Steer b76bc8c, P0 and P1. Its audit of the tmpfs record was accurate — the three paired wins
there were 2.2%, 10.4% and 13.1%, not a uniform 10.6% — so ranges are carried below rather
than a median alone.

**The record now identifies its own mount.** Naming the device from a pathname would have
been wrong on this host: `/mnt/storage` is a **FUSE-mounted SATA** volume, while the NVMe is
`nvme0n1` mounted at `/`. The runner walks `lsblk` toward the parents so LVM and partitions
resolve to a disk, and `--require-persistent` refuses tmpfs, ramfs, overlay, and any target
whose mount cannot be identified.

| field | value |
|---|---|
| resolved path | `/home/crogers2287/kvx-nvme-store` |
| mount source | `/dev/mapper/ubuntu--vg-ubuntu--lv` |
| mount target / fs | `/` / `ext4` |
| mount options | `rw,relatime` |
| backing device | `nvme0n1` (rotational: False) |
| same mount as model | False |
| available | 16.6 GiB |
| page-cache policy | natural page-cache state after admission and process restarts; no eviction forced |

**Result — 3 of 3 paired wins:**

| rep | restore | tail | total | cold | faster |
|---|---:|---:|---:|---:|---:|
| 1 | 1.120 | 0.427 | **1.548** | 1.739 | 11.0% |
| 2 | 1.048 | 0.428 | **1.477** | 1.723 | 14.3% |
| 3 | 1.061 | 0.425 | **1.486** | 1.731 | 14.2% |

Median 1.486 s (range 1.477–1.548)
against cold 1.731 s (range 1.723–1.739),
a **14.2%** margin — above the steer's 5% marginal threshold,
so this is a Pass rather than a marginal result. For comparison the tmpfs rung was
10.6%; NVMe is not slower here, and every phase
matches tmpfs within noise.

All correctness and safety gates held: `cache_n=2044 prompt_n=4` every run, restore metadata
equal to admitted metadata, token/content/probability-vector parity against native reuse,
**zero request-path payload bytes**, the admitted object unchanged, `hybrid_support` timed
inside the request path at 0.0028 s,
and the unpatched control refusing with **zero endpoint calls**.

**What this does and does not prove.** It is the immediate llama-server restart case: the
process dies, the file persists on NVMe, and the page cache may survive. Admission costs
3.883 s and saves 0.246 s per restore, so it
breaks even after **16 restores**. It is **not** evidence for a host
reboot or a long-idle cache eviction — that is the next experiment, and it is labelled
state-file cold-cache sensitivity rather than host restart.

Status: **measured three times on this host**. **Proven by retained test**: 476 offline.
**Untested**: state-cold behaviour, host restart, 8K.

## REQ-034 — The 8K rung passes 3/3, and the win grows with prefix length

Steer c867cda, P0 and P1. Its source review found two constants that made the harness
silently 2K-only, and both were mine: the acceptance check required `cache_n == 2044` and
`prompt_n == 4` literally, so a correct 8K run reporting 8188/4 would have been **rejected as
a failure**; and the record `kind` was hardcoded `admitted-store-2k-gate`.

Coverage is now derived from the admitted checkpoint metadata — `cache_n` equals the declared
coverage, `prompt_n` equals the uncovered tail, and the two sum to the prompt length. The argv
test is behavioural rather than a source-substring match, because the previous nonfunctional
context fix passed both `ast.parse` and a substring assertion.

**Result — 3 of 3 paired wins at 8,192 tokens:**

| rep | restore | tail | total | cold | faster |
|---|---:|---:|---:|---:|---:|
| 1 | 1.392 | 0.448 | **1.840** | 4.862 | 62.2% |
| 2 | 1.403 | 0.445 | **1.848** | 4.876 | 62.1% |
| 3 | 1.490 | 0.448 | **1.938** | 4.884 | 60.3% |

Median **1.848 s** (range 1.840–1.938)
against cold **4.876 s** (range 4.862–4.884)
— **62.1% faster**. Coverage 8188 + 4 = 8192 every run.

**The win grows with prefix length, measured not projected:**

| prompt | restored total | cold | faster | break-even |
|---|---:|---:|---:|---:|
| 2,048 | 1.486 s | 1.731 s | 14.2% | 16 restores |
| 8,192 | 1.848 s | 4.876 s | **62.1%** | **2 restores** |

This is the first measurement inside the stated 8K–32K agentic-prefix range, and it is the
result the product goal needed: a harness resending a large static prefix saves ~3 s per
request and repays admission after **two** restores.

Artifact: 1,007,783,892 bytes = 123,020 bytes/token,
against 295,390 at 2K. That non-linearity is raised as RA-003; the fixed recurrent component
amortises as the prefix grows, which is why the advantage widens.

All safety gates held: zero request-path payload bytes, the admitted object unchanged, the
unpatched control refusing with zero endpoint calls, ext4 on `nvme0n1` confirmed by mount
evidence rather than pathname, and phases reconciled.

Status: **measured three times on this host**. **Proven by retained test**: 490 offline.
**Untested**: 32K, canonical extraction, cross-dtype transfer.

## REQ-035 — The decoder reads bytes llama.cpp actually wrote

Every decoder test so far used fixtures I built from the writer's field order, which means
they shared my reading of it. That is exactly the shape of the 12-versus-16-byte checkpoint
error: self-consistent, and wrong. This decodes real output.

Produced on CPU (`-ngl 0`) from `library_of_alexandria_Q4_K_M.gguf`, so the fleet's GPUs were
never touched — 881/914 MiB free before and after. 49 prompt tokens, 1,808,028 bytes.

| check | source of truth | result |
|---|---|---|
| body consumed exactly | file size | **1,808,028 of 1,808,028 — nothing left over** |
| layer count | GGUF `block_count` = 36 | 36, 72 spans (36 k + 36 v) |
| row stride | GGUF `head_count_kv` × `embedding_length/head_count` = 2×128, f16 | 512 = 512 |
| materialised tensor | — | (49, 2, 128) float32, all finite, range [-75.31, 92.13] |

Every expected value is computed from the **GGUF**, never from the decoder's own output, so
the check can fail. The exact-consumption result is the sharpest of these: a wrong field
width anywhere in the attention section would leave a remainder or overrun, and this model is
non-hybrid so the body legitimately ends there.

One incidental finding: the unpatched `~/llama.cpp/build` emits **sequence-state version 2**,
not 3. The attention layout parsed identically, so that part of the format is stable across
those two versions on this evidence.

Retained as `tests/test_ggsq_live.py`, skipped unless `KVX_LIVE_STATE` and `KVX_LIVE_MODEL`
point at a real artifact and its model — so CI stays green without a GPU while the check
exists for anyone who can run it.

Status: **measured once on this host** (real-artifact decode). **Proven by retained test**:
the synthetic decoder suite, 589 offline. **Untested**: a real *hybrid* artifact, whose body
continues into recurrent state and checkpoints — that needs the 27B and a GPU window.

## REQ-036 — One cache file moves between ROCm/HIP and Vulkan, both directions

The project's headline claim, tested against the hardware rather than argued: the same cache
file written on one compute backend restores on another.

Three llama.cpp builds now exist at **one source revision** (`ca3d5a3e1`, build 151): HIP for
gfx1030, Vulkan (RADV), and the existing CUDA control. That matters more than it sounds. An
earlier attempt at this test restored a v3 state file into a binary compiled months earlier,
which rejected it for the **state-file version**, and the result read as "cross-backend does
not work" until the server log was read.

Both builds needed a workaround unrelated to this project: `~/.cache/ccache/tmp` is owned by
**root** and not writable, which fails every compile with `Permission denied`. Builds here
pointed `CCACHE_DIR` elsewhere rather than touching a root-owned directory. **This will break
other builds on this host until it is fixed** — the fix is `sudo rm -rf ~/.cache/ccache/tmp`.

Run: one vacant W6800 (rocm index 1 / `Vulkan2`), 128 prompt tokens, qwen2 Q4_K_M. The 3090s
were left alone — vLLM holds both, ~1.1 GiB free each.

| direction | reused | text | token ids |
|---|---|---|---|
| HIP → Vulkan | **127 / 128** | identical | identical |
| Vulkan → HIP | **127 / 128** | identical | identical |

The logprob vectors are *not* identical, and the first version of this record could only say
so as a boolean. That cannot separate twelfth-decimal kernel noise from a restore that put
wrong numbers in the cache, so the record now decomposes it:

| comparison | max logprob delta | top-1 agreement |
|---|---|---|
| restore of a foreign cache vs local cold prefill | 0.364 | 1.00 |
| two cold runs, different backends, no cache at all | 0.375 | 1.00 |

**Moving a cache across backends costs about what the two backends already differ by on their
own.** Neither number means anything without the other, which is why both are recorded.

Two measurement bugs were fixed to get there. The delta first folded top-k *membership*
changes into itself as infinity — honest, but it saturated and hid how closely every shared
alternative agreed; membership differences are now counted separately. And the run crashed
*after* both transfers succeeded, on an undefined name left by a refactor in the code that
assembles the record — ten minutes of GPU time spent and nothing written.
`tests/test_no_undefined_names.py` now scans every module and script for that class of fault.

### The missing control, added

The 0.364 above still carried two causes at once — the cost of restoring at all, and the cost
of restoring something *foreign*. The reader now saves and restores its **own** cache first,
by exactly the sequence the writer used, before touching the transferred artifact. It refuses
if that self-restore reuses nothing, since a control that measures nothing would still print
a number.

| comparison | HIP→Vulkan | Vulkan→HIP | membership changes |
|---|---|---|---|
| own restore vs cold prefill, same backend | 0.053 | 0.139 | **0** |
| foreign cache vs own cache | 0.373 | 0.397 | 4 |
| two cold runs, different backends, no cache | 0.375 | 0.375 | 2 |

Restoring a cache the backend wrote itself is very nearly exact — an order of magnitude
tighter than anything else here, and it does not even shuffle the tail of the top-k list.

The number that matters is the middle row against the bottom one. **Using a foreign cache
diverges by 0.373–0.397; the two backends already diverge by 0.375 doing identical cold work
with no cache involved at all.** Crossing the backend boundary therefore adds nothing
measurable beyond the arithmetic difference the backends carry anyway. Top-1 agreement is
1.00 everywhere and generated tokens are identical throughout.

Status: **measured once on this host** — HIP↔Vulkan, both directions, 128 tokens, four-way
decomposition. **Proven by retained test**: the revision guard, the divergence arithmetic, the
self-restore refusal, the undefined-name scan (728 offline tests, CI green). **Untested**:
CUDA↔anything, which needs a 3090 slot that vLLM currently holds; anything above 128 tokens on
this pair. Top-1 agreement is over 8 generated positions — a small sample, and the claim above
rests on the delta rather than on that agreement.

## REQ-037 — The same result at 8K, which is where the actual use case lives

RA-002 objected that every measurement so far sat at 128–2048 tokens while the goal — cached
system prompts, skills, MCP tool schemas for agentic harnesses — lives at 8K–32K. This is that
run, on the same vacant W6800, same pair of builds at `ca3d5a3e1`.

First, a bug that would have made this meaningless: the runner built its prompt from a **fixed
24 repetitions** of one sentence, a few hundred tokens, then sliced to `--prompt-tokens`. Any
request for more would have measured a short prefix and reported it under the requested
length. The text now scales with the request, and the run refuses if it still falls short.

| direction | reused | text | token ids | artifact |
|---|---|---|---|---|
| HIP → Vulkan | **8191 / 8192** | identical | identical | 288 MB |
| Vulkan → HIP | **8191 / 8192** | identical | identical | 288 MB |

The one-token shortfall is constant — llama.cpp always reprocesses the final token — so the
reuse fraction climbs from 99.2% at 128 tokens to **99.99% at 8192**. The win grows with
prefix length, which is the direction the use case pushes.

| comparison | HIP→Vulkan | Vulkan→HIP | 128-token run |
|---|---|---|---|
| own restore vs cold prefill, same backend | 0.175 | 0.750 | 0.053 / 0.139 |
| foreign cache vs own cache | 0.442 | 0.495 | 0.373 / 0.397 |
| two cold runs, different backends, no cache | 0.783 | 0.783 | 0.375 |

At this length **every one of these is the same order of magnitude**, and the foreign-cache
column is *smaller* than the backend-versus-backend floor. The divergence is dominated by
kernel and batch-shape arithmetic, not by moving the cache. Top-1 agreement is 1.00
throughout and the generated tokens are identical in every comparison.

Status: **measured once on this host** — HIP↔Vulkan at 8192 tokens, both directions, four-way
decomposition. **Proven by retained test**: the short-prefix refusal, the self-restore
refusal, the divergence arithmetic (728 offline tests, CI green). **Untested**: 32K, which
would be a ~1.1 GB artifact; CUDA↔anything, still waiting on a 3090 slot; any model other than
this qwen2 Q4_K_M. **Inferred, not measured**: that the constant one-token shortfall continues
to hold above 8K.

## REQ-038 — 32K transfers, and the size law predicts to the byte

The top of the range RA-002 named. Same vacant W6800, same builds at `ca3d5a3e1`, non-hybrid
qwen2 Q4_K_M. 32,000 rather than 32,768 tokens: this model's training context is 32,768 and
the slot caps there, so a full-length prompt plus eight generated tokens does not fit. That
arrived as a bare `HTTP 400` because the runner discarded llama-server's response body, which
had said so plainly — now fixed, the body is in the exception.

| direction | reused | text | token ids | artifact |
|---|---|---|---|---|
| HIP → Vulkan | **31,999 / 32,000** | identical | identical | 1.18 GB |
| Vulkan → HIP | **31,999 / 32,000** | identical | identical | 1.18 GB |

### The size law, stated before the run and then tested

From the 128- and 8,192-token artifacts: `bytes = 908 + 36,880 x tokens`. Applied to 32,000 —
**four times outside the range it was fitted on** — that predicts **1,180,160,908** bytes.

Measured: **1,180,160,908 bytes. Exact, zero error.**

The 908-byte constant matters beyond arithmetic. RA-003 argued the hybrid model's large fixed
term is recurrent/checkpoint state, which is sized per layer rather than per token. This model
has no recurrent state, and its fixed term is 908 bytes against the hybrid's ~449 MiB — which
is the cleanest confirmation of that mechanism available without re-measuring the hybrid.

### Divergence at 32K

| comparison | HIP→Vulkan | Vulkan→HIP |
|---|---|---|
| own restore vs cold prefill, same backend | 0.067 | 0.202 |
| foreign cache vs own cache | 1.226 | 1.304 |
| two cold runs, different backends, no cache | 1.163 | 1.163 |

The pattern holds at every length tested. Restoring a cache the backend wrote itself stays
tight; using the other backend's cache diverges by about what the two backends already
diverge by on cold work with no cache at all. Top-1 agreement 1.00 and identical generated
tokens in every comparison, at 128, 8,192, and 32,000 tokens.

Status: **measured once on this host** — HIP↔Vulkan at 32,000 tokens, both directions.
**Proven by retained test**: the size arithmetic is not tested (it is a fit over recorded
evidence, not code); the short-prefix refusal, self-restore refusal and divergence arithmetic
are (728 offline tests, CI green). **Untested**: CUDA↔anything, still waiting on a 3090 slot;
the hybrid model at any of these lengths; any second model. **Inferred**: that the size law
holds above 32K — it has only been tested up to the model's own context limit.

## REQ-039 — Artifact size derived from the format, exact on two models

REQ-038 fitted `908 + 36,880 x tokens` to two points and it predicted a third exactly. A line
through two points reproduces those two points by construction, so that was worth following
up rather than believing. This replaces the fit with the writer's own arithmetic.

A 128- and a 129-token artifact were saved back to back and one was decoded. The accounting
closes exactly:

```
12                      magic, version, n_token_count
+ 4  x header_tokens    the prompt's token ids, four bytes each
+ 8                     n_stream, cell_count
+ 12 x cells            per cell: pos, n_seq_id, one seq_id
+ 8                     v_trans, n_layer
+ n_layer x (24 + cells x (k_row + v_row))
= 4,721,548             measured 4,721,548
```

The term that had been missing is **four bytes per token for the prompt's token ids in the
header**. Arithmetic over the documented per-cell fields came out 4 bytes per token short of
the measured slope, and rather than absorb that into a constant it was found by decoding a
real file. Invisible at 128 tokens; 128 KB adrift at 32,000.

`kv_rosetta/sizing.py` reads geometry from the GGUF and predicts. Against every artifact this
project has written:

| model | cells | predicted | actual | error |
|---|---:|---:|---:|---:|
| qwen2 Q4_K_M | 128 | 4,721,548 | 4,721,548 | **0** |
| qwen2 Q4_K_M | 129 | 4,758,428 | 4,758,428 | **0** |
| qwen2 Q4_K_M | 8,192 | 302,121,868 | 302,121,868 | **0** |
| qwen2 Q4_K_M | 32,000 | 1,180,160,908 | 1,180,160,908 | **0** |
| **qwen3** 0.6B | 128 | 14,682,828 | 14,682,828 | **0** |
| **qwen3** 0.6B | 129 | 14,797,532 | 14,797,532 | **0** |

The last two are the ones that count. A second architecture, 28 layers against 36, 8 KV heads
against 2, **114,704 bytes per token against 36,880** — and the figures were printed before
the artifacts were written.

### Two silent-wrong bugs this found

**Head dimension.** The first version derived `head_dim = embedding_length / head_count`. On
qwen35 that is 5120/24 = 213.33, floored to **213**, against a declared `key_length` of
**256** — an estimate wrong by a fifth and entirely plausible. It surfaced only because
subtracting the attention estimate from a measured hybrid artifact left a *negative*
remainder. `attention.key_length` and `value_length` are now authoritative, keys and values
are sized independently, and an indivisible `embedding_length` is refused rather than floored.
The second model above would also have been wrong: it declares key_length 128 where the
derivation gives 64.

**A docstring that lied.** `state_bytes` said hybrid models were refused. Nothing checked.
A caller reading it would have taken an attention-only figure for a whole file, and the terms
are wrong for a hybrid twice over — the body continues into recurrent state, and not every
layer carries attention KV. Hybrid and recurrent architectures are refused by name now.

Status: **proven by retained test** — the arithmetic, both models' measured sizes as
constants, and every refusal (10/10 guards mutation-checked; 759 offline tests). **Measured
once on this host**: the six artifacts above. **Untested**: hybrid sizing, which needs a
decoded hybrid artifact to learn which layers carry attention KV; quantised KV types, which
are computed but never compared against a file written with one.

## REQ-040 — The space guard derives the size instead of scaling a flat rate

RA-003's first question, answered with the predictor from REQ-039.

`predict_space` scaled a flat `--bytes-per-token 295390.0`. That figure was obtained by
dividing a **hybrid** model's whole 2K artifact by its token count, which folds a fixed
per-layer recurrent term into a per-token rate. Given a model, it now derives the size from
the GGUF and the state-file layout instead.

| prompt | derived | flat rate | over-prediction |
|---|---:|---:|---:|
| 8,192 | 0.281 GiB | 2.254 GiB | **8.0x** |
| 32,000 | 1.099 GiB | 8.803 GiB | **8.0x** |

Eight times, not the 2.4x RA-003 estimated — because the rate was measured on a different
model as well as a hybrid one. The derived figures are the ones REQ-037 and REQ-038 actually
wrote, to within the 16 bytes of header token ids a saved slot carries beyond its cells.

A guard that refuses work which would have succeeded costs as much as one that admits work
that fails, and this was refusing eightfold.

Three things it deliberately does **not** do:

- A **hybrid** model still falls back to the flat rate. The derivation does not describe
  recurrent state and refuses rather than guess, which is the correct outcome — the rate was
  measured on that model, so the fallback is at least the right one.
- The record carries `basis` and `basis_note`, so a rate-based number can never be read as a
  derived one. That distinction is the whole point of doing this rather than lowering the
  constant.
- A model path that does not exist **raises**. It is a configuration fault, every later step
  of the run would fail on it, and answering a mistyped path with a plausible rate-based
  number would hide it.

Status: **proven by retained test** — six new cases covering both bases, the fallback, the
refusal, and the fit decision changing with the basis (765 offline tests, CI green). The
fallback is a `try`/`except` rather than an `if`-guarded raise, so `scripts/mutation_check.py`
cannot reach it; it was mutated by hand instead — widening the except to `Exception`, and
mislabelling the basis, each make a named test fail.

**Honest note on that script's coverage:** the mutation tool reports **3 of 21** guards
defended for `admitted_store_gate.py`. The other 18 live in the live-run path and only
execute against a running server on a GPU. That is a pre-existing gap, not one this change
introduced, and it is recorded here rather than left to be discovered from a passing summary.

## REQ-041 — Hybrid sizing derived, and a fail-open guard caught before it shipped

RA-003 argued from two fitted points that the hybrid artifact's large fixed term is recurrent
state. This decodes a real one and turns that argument into terms.

A 256- and a 257-token Qwen3.8-27B artifact were saved on the vacant W6800 and decoded. The
file closes to **zero leftover bytes**:

| part | bytes | what |
|---|---:|---|
| header | 12 + 4x260 | magic, version, and the prompt's token ids |
| attention | 16,783,760 | **16** layers, 256 cells, f16 rows of 2048 |
| recurrent | 156,894,356 | 1 cell, 48 present layers of 64 declared |
| total | **173,679,168** | measured 173,679,168 |

Both recurrent row sizes fall out of the GGUF's SSM metadata — conv state
`(d_conv-1) x (d_inner + 2 x n_group x d_state) x 4` = 122,880 and SSM state
`d_inner x d_state x 4` = 3,145,728. The attention section holds **16** layers, not the 17 the
recurrence rule implies: the NextN/MTP block is marked non-recurrent but is not in the KV
cache at all. The recurrent section is **byte-identical** between the two files and is 90.3%
of the artifact at 256 tokens — RA-003's claim, now a measurement.

### A decoder bug found by looking at real bytes

The recurrent section does **not** begin with `n_stream`.
`llama_memory_recurrent::state_write` writes `cell_count` first, unlike
`llama_kv_cache::state_write`. The layout inventory said otherwise, the decoder was written
to match the inventory, and the test fixture was written to match the decoder — three
artefacts agreeing with each other and none with llama.cpp. Eight tests failed once the
parser was corrected, which is the informative part: they had been green against a body no
writer produces. Same shape as the 12-versus-16-byte checkpoint record, found the same way.

### The guard that would have failed open

Wiring hybrid sizing into `predict_space` produced 291,169,840 bytes for the 2,048-token case.
RA-003's measured artifact is **604,958,676**. The difference is **2.0000x the recurrent
section plus 124 bytes**, consistent with the patched build appending two recurrent-only
context checkpoints — my probe used an unpatched build, so it wrote none.

Left as it was, the guard would have under-predicted that file by half. Over-predicting
refuses work that would have succeeded; **under-predicting runs the disk out mid-admission**,
which is the failure this guard exists to prevent. So `hybrid_state_bytes` now takes
`checkpoints` with **no default**, and:

- unstated → falls back to the conservative flat rate, and says why
- `checkpoints=0` → derives exactly, as measured on the two unpatched artifacts
- `checkpoints>0` → **refuses**. The 2.0000 multiple is arithmetic on one file, not a decoded
  appendix, and the 124 bytes of framing are unexplained. Predicting from it would dress a
  guess as a derivation, which is what every other term here exists to avoid.

Status: **proven by retained test** — every term, both measured hybrid artifacts, the layer
split, the cell_ext rule, and all the refusals (22/22 guards mutation-checked; 800 offline
tests). **Measured once on this host**: the two artifacts. **Inferred, not measured**: that
the RA-003 excess is two checkpoints — the arithmetic fits to 124 bytes but no appendix has
been decoded. **Untested**: any hybrid other than this one; quantised KV on a hybrid.

## REQ-042 — The checkpoint appendix decoded, and the last inference removed

REQ-041 left one thing inferred: that RA-003's 2,048-token artifact is a base plus two
context checkpoints. The arithmetic fit to 124 bytes, no appendix had been decoded, and
`hybrid_state_bytes` refused any nonzero checkpoint count rather than predict from it.

The patched build (`/mnt/storage/llama-kvx-patched`, same commit `ca3d5a3e1`) was run on
**CPU** with `-ngl 0`, so the 3090s were never touched. It wrote a 256-token artifact with one
checkpoint: 330,573,584 bytes, of which the derived base accounts for exactly 173,679,168.

The appendix terms come from `save_slot_checkpoints` in that tree: magic, version and count,
then per checkpoint an int64 `n_tokens` and two int32 positions, then `data_tgt`, `data_dft`
and `data_spec` each **length-prefixed and interleaved with its payload**, so an empty buffer
still costs its uint64 length. My first probe read the three lengths consecutively and got a
2.9 GB draft buffer and a 1 TB speculative one — the decoder already had it right.

`data_tgt` opens with **eight bytes whose meaning is not identified**, then a full recurrent
section. Their size is confirmed twice and the constant is named for what it is rather than
given a plausible label.

| file | checkpoints | predicted | measured |
|---|---:|---:|---:|
| 256-token, this host | 1 | 330,573,584 | 330,573,584 |
| 2,048-token, RA-003 | 2 | 604,958,676 | 604,958,676 |

**The second row is the one that counts.** It was produced weeks earlier by a different build
at a different prefix length with twice the checkpoints, and the appendix arithmetic was
settled on the first file alone. RA-003's excess is no longer an inference.

`predict_space` now derives hybrid sizes at any checkpoint count. One further honesty: a
saved slot's header has carried **four more token ids than cache cells** in every artifact
measured here, worth 16 bytes. That is an observation, not a derivation, so it is applied as a
named `HEADER_TOKEN_ALLOWANCE` that errs high — the direction that refuses work rather than
running a disk out mid-admission.

An aside worth recording: the flat rate predicts 604,958,720 at 2,048 tokens, 44 bytes from
the measurement. It was obtained by dividing that very artifact by its token count, so it is
near-exact at 2,048 and wrong everywhere else — 8x at 8K on the non-hybrid model. A constant
that reproduces its own origin is not evidence that it generalises.

### Decoder guard coverage closed

`scripts/mutation_check.py` reported **29 of 42** guards defended in the GGSQ decoder — the
module every other claim rests on. Thirteen refusals could be disabled without a test
noticing. All thirteen now have an input that reaches exactly the guard it names, which for
several meant stepping around an earlier check that shadowed it, and for the transposed-value
type id meant adding a `declared_v_type` to the builder: the first attempt searched the body
for the type id's bytes and took the last match, which landed unaligned inside the payload —
the hazard that builder's own docstring warns about.

**42/42 guards defended.** 817 offline tests, CI green.

Status: **proven by retained test** — the appendix terms, both measured artifacts, the
preamble length, every decoder refusal. **Measured once on this host**: the 256-token
one-checkpoint file. **Unknown**: what the eight preamble bytes hold. **Untested**: any
hybrid other than this one; checkpoint restoration, which is a separate claim from sizing.

## REQ-043 — Cross-tokenizer alignment, checked against two real vocabularies

The last unbuilt piece of the cross-model path. Two models with different tokenizers share no
token positions: "the naturalist" is three tokens in one vocabulary and five in another, and
token 2 of the first has no relationship whatever to token 2 of the second. What they share is
the bytes.

`kv_rosetta/mappers/align.py` turns each tokenization into half-open byte intervals and
weights every target token by the bytes it shares with each source token. Rows sum to one, so
pooling is a weighted mean and never rescales the vectors — pinned by pooling a constant cache
and getting the constant back.

Checked on real tokenizers rather than only on constructed cases: **qwen2.5** (151,643 tokens)
against **ornith-a1** (248,044), both present on this host. The same sentence is 17 pieces in
one and 16 in the other.

| | |
|---|---|
| qwen2.5 tail | `' a'  ' café'  ' au'  ' la'  'it'  '.'` |
| ornith-a1 tail | `' a'  ' café'  ' au'  ' lait'  '.'` |

ornith-a1's ` lait` covers qwen2.5's ` la` and `it` — three bytes and two of five, weighted
**0.6 and 0.4**. That is the case the module exists for, and it is now a retained test.

Three refusals carry the weight, because a misalignment is a quiet one-token shift rather than
a crash, and a shifted cache produces fluent wrong output:

- **Identical bytes on both sides**, or refuse. A leading space or a different unicode
  normalisation lands here. Equal lengths are not enough; the bytes are compared.
- **RoPE must be stripped.** Pooling post-RoPE keys averages vectors rotated by different
  position angles, and the mean encodes no position faithfully.
- **Pieces are bytes, never str, never empty.** Encoding here would pick an encoding on the
  caller's behalf; a zero-width span would match anything or nothing depending on comparison
  order.

One guard was written and then removed. "Target token overlaps no source token" cannot happen
once the byte-equality and empty-piece refusals hold, since the source spans tile the range
contiguously — an untestable guard is decoration, so the invariant is asserted by test across
a range of tokenizations instead, including a byte-fallback case that cuts a multibyte
character in half where the bytes still line up though the characters do not.

Status: **proven by retained test** — the arithmetic, every refusal (7/7 guards
mutation-checked), and the real qwen2.5/ornith-a1 alignment, skipped when those tokenizers are
absent (846 offline tests). **Untested**: alignment feeding an actual cross-model transfer —
that needs a fitted mapper and both models resident, and the divergence gate has the final say
on whether the result is usable at all.

## REQ-044 — The first cross-model translation, and the gate rejects it

The whole point of the gate, on real data at last: a cache captured from
`qwen38-27b` (qwen35, 16 attention layers, 4 KV heads) translated into `tiel-coder`
(qwen35moe, 10 attention layers, 2 KV heads) and offered to tiel.

### What made the pair promising

Every semantic axis matches and only capacity differs — head_dim 256 both, d_state 128 both,
d_conv 4 both, n_group 16 both. And they turned out to share a tokenizer exactly: identical
token ids and pieces, so no byte-span alignment was needed at all. If a linear cross-model map
were going to work anywhere, here.

### The fit

15,981 tokens from 8 varied passages, captured from both models on the vacant W6800. Held out
by **whole prompts** — a within-prompt split leaks, because adjacent tokens share context, and
an earlier run that split by token reported 0.98 where the honest number is 0.55.

| | |
|---|---|
| median held-out R² | **0.55** |
| above 0.5 | 13/20 |
| above 0.9 | **0/20** |
| source layers chosen | 1, 3, 5, 6, 7, 10, 8, 11, 12, 15 |

That layer selection came out **near-monotonic without being asked to be** — deeper target
layers independently picked deeper source layers. Real structure, not an artifact.

### The gate

The translated attention was spliced into tiel's own artifact, leaving its recurrent section
exactly as tiel wrote it. If translated attention cannot pass with everything else held
perfect, no amount of recurrent work would rescue it.

| run | reused | tokens match | top-1 | max logprob delta |
|---|---:|---|---:|---:|
| identity control | 764/768 | **yes** | **1.00** | **0.000** |
| **translated** | 764/768 | no | **0.00** | 2.42 |
| noise control | 764/768 | no | 0.00 | 7.99 |

The identity control reproduces the cold output byte for byte, so the splice, the restore and
the measurement are correct. Noise is degenerate, so the measurement is sensitive. Between
them, the translation **fails**:

```
cold/identity : " had unknowingly assembled the longest continuous record of coastal fo"
translated    : ", having spent his life reading the sea, had learned to"
noise         : ":\n\n<think>\nThis is a prompt that appears to be"
```

Fluent, grammatical, on topic, and disagreeing with the target model on **every generated
token**. This is the exact failure the gate exists for: wrong in the one way that does not
look wrong. Nobody reading that sentence would suspect the cache.

Translated does sit well clear of noise — 2.42 against 7.99 — so the map carries real signal.
It is simply nowhere near enough, which is what an R² of 0.55 predicts.

### Two invalid runs before this one

The first attempt reported all three variants producing identical output. It had
`--ctx-checkpoints 0` set, disabling the very feature the patched build exists to provide, so
on a hybrid model nothing was reused and every run was a plain cold prefill. **The noise
control is the only reason that was caught** — three identical outputs including noise is
impossible if anything was actually restored. The identity control is now a hard precondition:
if it reuses nothing, the run aborts rather than reporting numbers.

The same run also saved the target state after generating, giving 779 cells against 768
translated rows.

Status: **measured once on this host** — one prompt, 768 tokens, one fitted map.
**Proven by retained test**: the translator, the divergence arithmetic, the composition rules
(908 offline tests). **The honest conclusion**: a per-layer linear map on 16k calibration
tokens does not produce an admissible cache for this pair, and the gate says so. A linear map
is the floor, not the ceiling — C2C-style learned projectors are the next rung — but nothing
here supports admitting a translated cache today.

## REQ-045 — How close is close enough? The gate's tolerance, measured

REQ-044 established that a linear map at R² 0.55 fails. That says nothing about whether a
*better* map would pass, and building one is expensive. This measures the target first.

The method needs no new fitting: blend tiel's true cache with the translated one at a
sequence of ratios, splice each into tiel's own artifact, and find where the gate flips.
`alpha` is the fraction of the true cache; alpha 1.0 is the identity control and alpha 0.0 is
the pure translation from REQ-044.

| alpha | tokens match | top-1 |
|---:|---|---:|
| 1.00 | yes | 1.00 |
| 0.99 | yes | 1.00 |
| 0.95 | yes | 1.00 |
| 0.90 | yes | 1.00 |
| 0.80 | yes | 1.00 |
| **0.60** | **yes** | **1.00** |
| 0.30 | no | 0.33 |
| 0.00 | no | 0.00 |

**The flip is between 0.6 and 0.3** — far more tolerant than expected. A cache does not have
to be nearly exact; roughly 60% of the way there suffices on this prompt.

Converting a blend ratio to a fit quality: an alpha-blend has residual error
`(1-alpha)` times the map's, so `effective R² = 1 - (1-alpha)²(1-R²)`. At R² 0.55:

| alpha | effective R² |
|---:|---:|
| 0.80 | 0.982 |
| 0.60 | **0.928** |
| 0.30 | **0.780** |

**So the target is an R² somewhere between 0.78 and 0.93, and the linear map delivers 0.55.**
That is a real gap and a real number to aim at, rather than "it failed".

Two caveats that keep this from being a measurement of the threshold itself. The logprob delta
is **non-monotonic** across alpha — 5.19 at 0.8 against 1.68 at 0.6 — because on 12 generated
positions the top-k membership churns; top-1 agreement and token equality are the stable
signals here and delta is not. And this is one prompt with twelve generated tokens, so the
0.3-to-0.6 bracket is coarse.

Status: **measured once on this host** — one prompt, eight blend ratios, identity control
exact at alpha 1.0. **The useful conclusion**: cross-model translation for this pair needs
roughly R² 0.85, not 0.999. A per-layer linear map on 16k tokens reaches 0.55, so the approach
is short by a wide but finite margin - which is an argument for a better map rather than
against the idea. **Untested**: whether any map reaches 0.85 on this pair; whether the bracket
holds on other prompts or longer generations.

## REQ-046 — The bracket sharpened, and REQ-045's number corrected upward

REQ-045 bracketed the gate's flip point on one prompt and **twelve** generated tokens. This
repeats it with **forty-eight** generated tokens, ten blend ratios and two unrelated prompts,
each model loaded once.

| alpha | 1.0 | 0.7 | 0.6 | 0.55 | 0.5 | 0.45 | 0.4 | 0.3 | 0.0 |
|---|---|---|---|---|---|---|---|---|---|
| lighthouse | pass | pass | **fail** | fail | fail | fail | fail | fail | fail |
| orchard | pass | pass | pass | pass | **pass** | fail | fail | fail | fail |

**REQ-045 was optimistic and this supersedes it.** On twelve tokens the lighthouse prompt
passed at alpha 0.6; on forty-eight it fails there. More generated tokens is a strictly harder
test, because divergence that has not surfaced by token twelve still surfaces by token forty.
Any threshold measured on a short generation is an upper bound on quality, not a measurement
of it.

| prompt | lowest passing | implied R² | highest failing | implied R² |
|---|---:|---:|---:|---:|
| lighthouse | 0.70 | **0.960** | 0.60 | 0.928 |
| orchard | 0.50 | **0.887** | 0.45 | 0.864 |

**Required R² is 0.89 to 0.96 and prompt-dependent; the stricter reading is ~0.96.** The
linear map delivers 0.55. The gap is wider than REQ-045 concluded.

### The failure is a cliff, not a slope

Look at the orchard row either side of its flip: top-1 agreement goes **0.958 → 0.354** between
alpha 0.45 and 0.40. And the lighthouse prompt sits at 0.042 for every failing ratio from 0.6
down to 0.0 — a translated cache and a nearly-perfect one fail *identically*.

That is autoregression, not a property of the cache. One wrong token derails every token after
it, so top-1 agreement over a free generation measures **when the first divergence happens**,
not how good the cache is. It is the right instrument for admission - end-to-end behaviour is
what a user experiences - and the wrong one for grading a map, which needs teacher-forced
comparison at each position against the same prefix.

Status: **measured once on this host** — two prompts, ten ratios, 48 generated tokens,
identity control exact on both. **Corrects**: REQ-045's 0.78-0.93 bracket, which was taken on
twelve tokens. **The conclusion for the project**: cross-model translation for this pair needs
roughly R² 0.9-0.96, and a per-layer linear map on 16k tokens reaches 0.55. **Untested**:
whether a teacher-forced metric gives a smooth quality curve, which is what a map should be
tuned against; whether the prompt-dependence narrows over more prompts.

## REQ-047 — Teacher forcing, and what the translated cache is actually worth

REQ-046 found the gate's failure is a cliff: a nearly-perfect cache and a fully translated one
both scored 0.042 top-1, because generation is autoregressive and one wrong token derails the
rest. That measures *when the first divergence happened*, not how good the cache is, and gives
nothing to tune a map against.

Scoring each position against the same **forced** prefix removes the cascade.
`kv_rosetta.metrics.positionwise_agreement` does the arithmetic; the protocol is one request
per position, submitting `prompt + reference[:k]` and reading the single next-token
distribution.

| alpha | teacher-forced top-1 | mean \|Δlogprob\| | free-generation verdict |
|---:|---:|---:|---|
| 1.0 | 1.000 | 0.000 | pass |
| 0.8 | 1.000 | 0.313 | pass |
| 0.7 | 1.000 | 0.484 | pass |
| 0.6 | 1.000 | 0.508 | **fail** |
| 0.5 | 1.000 | 1.171 | fail |
| 0.4 | 0.958 | 1.678 | fail |
| 0.3 | 0.958 | 1.925 | fail |
| 0.2 | 0.875 | 2.220 | fail |
| 0.0 | **0.667** | 2.180 | fail |

### The translated cache is far better than the gate made it look

**0.667 against 0.042.** Under teacher forcing the fully translated cache predicts the same
next token as the true cache two thirds of the time. Under free generation the same cache
scored four percent. The cascade, not the cache, accounts for the difference — and any
judgement of map quality taken from a free generation is worthless for that reason.

`mean |Δlogprob|` is monotonic from 0.000 to 2.22 across the sweep, which is the gradient a
better map can be tuned against. It was not available before.

### But the admission bar is harsher than any R² target implies

Teacher-forced top-1 is 1.000 at alpha 0.5, and alpha 0.5 **fails** the free-generation gate.
So perfect agreement over 24 forced positions does not buy a passing generation over 48 free
ones. The real requirement is not "R² above 0.9" but **top-1 correct at every position over
the whole generation** - one wrong token in forty-eight is a failure, and the longer the
generation the harder it gets.

That reframes REQ-045 and REQ-046's R² targets: they are necessary, not sufficient, and they
understate the difficulty. A map at two-thirds per-token accuracy is not two thirds of the way
there.

Status: **measured once on this host** — one prompt, 24 forced positions, nine blend ratios,
identity exact at alpha 1.0. **Proven by retained test**: `positionwise_agreement`, including
that it varies smoothly with perturbation, which is the property the cliff metric lacks (915
offline tests). **Untested**: whether mean |Δ| below some value predicts a passing generation;
that needs the two metrics measured together across more prompts.

## REQ-048 — mean |Δ| does not predict admission, and the admission test is chaotic

REQ-047 proposed that teacher-forced `mean |Δlogprob|` could grade a map offline, so candidate
maps could be compared without a GPU generation run each. This tests that directly: both
metrics measured on the **same restored cache**, four prompts, eight blend ratios, 32 runs.

**It does not work, and the reason matters more than the result.**

| | |
|---|---:|
| passing runs, highest mean \|Δ\| | **0.667** |
| failing runs, lowest mean \|Δ\| | **0.171** |
| verdict | **overlapping** |

A mean |Δ| of 0.3 sometimes passes and sometimes fails. There is no threshold to set.

### Teacher-forced top-1 saturates and carries no information

Across all 32 runs, forced top-1 took exactly **one** distinct value: 1.000. At alpha 0.5,
where free generation fails everywhere, sixteen forced positions still agree perfectly. REQ-047
read a signal from this metric that at these ratios it does not carry - it discriminates
between alpha 0.4 and 0.0, and not at all above 0.5.

### The admission test itself is chaotic near the boundary

The `survey` prompt passes at alpha 0.9, fails at 0.8, 0.75 and 0.7, **passes again at 0.65**,
then fails at 0.6. That is not a threshold being crossed; it is a coin flip. One token grazes
a decision boundary, and whichever way it falls decides the whole generation.

| prompt | alphas that pass |
|---|---|
| lighthouse | 1.0 |
| orchard | 1.0 |
| ledger | 1.0 |
| survey | 1.0, **0.9, 0.65** |

**This undermines REQ-045 and REQ-046.** Both located a "flip point" in alpha and converted it
to an R² target. Those flip points were measurements of a chaotic quantity, so the R² figures
of 0.78-0.96 are not reliable. Three of four prompts here fail at alpha 0.9 - a ten percent
admixture of translation - which is far stricter than REQ-046's bracket and shows how much the
answer moves with the prompt.

### What follows for the gate

Exact token equality over a free generation is the right thing for a *user* to care about and
the wrong thing to *tune against*: it is brittle, prompt-dependent, and near-random in the
transition region. An admission rule needs a criterion that degrades smoothly - agreement over
the first few tokens, or a distributional bound - rather than one that a single unlucky token
decides.

Status: **measured once on this host** — 4 prompts, 8 ratios, both metrics on the same restored
cache, identity exact everywhere. **Corrects**: REQ-045's and REQ-046's R² targets, which rest
on a flip point that this shows is not stable. **The conclusion**: there is currently no cheap
offline number that predicts admission, so map quality cannot yet be graded without running the
target model. **Untested**: whether a first-K-token agreement criterion is stable enough to
serve as the gate.

## REQ-049 — Why the gate was chaotic: each prompt has one fragile token

REQ-048 found exact-match-over-48-tokens is near-random near the boundary and could not
explain it. Recording *where* the generation first diverges, instead of only whether it did,
explains it completely.

| alpha | 1 | 0.98 | 0.95 | 0.9 | 0.85 | 0.8 | 0.75 | 0.7 | 0.6 | 0.5 | 0.3 | 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lighthouse | 48 | 48 | 48 | 38 | 29 | 29 | 29 | 29 | 38 | 29 | 0 | 1 |
| orchard | 48 | 30 | 48 | 30 | 31 | 30 | 30 | 30 | 30 | 30 | 30 | 0 |
| ledger | 48 | 48 | 27 | 33 | 48 | 33 | 33 | 27 | 33 | 25 | 27 | 3 |
| survey | 48 | 48 | 48 | 48 | 22 | 22 | 22 | 22 | 22 | 22 | 22 | 6 |

**The divergence position is a property of the prompt, not of the perturbation.** survey
diverges at token 22 for every ratio from 0.85 down to 0.3. orchard sits at 30, lighthouse at
29, ledger between 25 and 33. Each prompt has a point where the model is nearly tied between
two continuations, and any disturbance flips it - while everything before that point survives
substantial cache error.

So the 48-token gate was measuring **"did you survive this prompt's coin-flip token"**, which
is a fact about the prompt. That is the whole of REQ-048's chaos, and REQ-045 and REQ-046's
alpha thresholds were readings of it.

### A shorter horizon is a stable criterion

| K | alphas where the four prompts disagree | alphas where all four fail |
|---|---|---|
| 8 | 0.3 only | 0 |
| **16** | **0.3 only** | **0** |
| 32 | seven of twelve | 0.7, 0.5, 0.3, 0 |
| 48 | four of twelve | everything below 0.85 |

**First-16-token agreement is stable across prompts and still rejects the translation.** The
pure map diverges at token 0-6 on every prompt - qualitatively different from any blend, which
survives to 22-33. The gate can tell those apart without depending on a fragile token.

The trade is real and worth stating plainly: a stable criterion is a **weaker** one. First-16
would admit an alpha 0.5 cache, which does produce different long-form output. Where to sit on
that curve is a product decision - how long a generation must stay identical - not something
the measurement can settle.

Status: **measured once on this host** — 4 prompts, 12 ratios, 48 free tokens, greedy and
therefore deterministic, identity exact on all four. **Explains**: REQ-048's chaos, and why
REQ-045/046's thresholds moved with the prompt. **The usable conclusion**: admission should be
first-K agreement with K well short of the fragile region, and K is a policy choice. **Untested**:
whether fragile points cluster at similar positions for other prompts or models, and whether
they can be predicted from the cold run's own logprob margins - if they can, the gate could
score a cache only at positions where the model is confident.

## REQ-050 — The fragile token is the one the model is least sure about

REQ-049 established that each prompt diverges at a fixed position regardless of how badly the
cache was perturbed, and that the 48-token gate was really testing whether that one token
survived. This tests the obvious explanation.

Cold generations from the target model alone - no map, no splicing - with the top-1 minus
top-2 logprob margin recorded at every position:

| prompt | fragile at | its margin | rank among 48 | median margin |
|---|---:|---:|---|---:|
| lighthouse | 29 | 0.514 | **0, the lowest** | 7.57 |
| orchard | 30 | 0.416 | **0, the lowest** | 7.70 |
| ledger | 27 | 0.195 | 1, second lowest | 5.30 |
| survey | 22 | 0.183 | **0, the lowest** | 7.95 |

**Four of four, three of them the single least certain position in the entire generation**, at
margins 15 to 40 times below the median. The fragile token is the token where the model is
nearly tied between two continuations, and a perturbed cache flips it because there is almost
nothing to flip.

### This makes the gate stable *and* strict

REQ-049 offered a trade: a short horizon is stable but weaker. This removes the trade. A
divergence at a position the reference model could not decide is not evidence about the cache -
it is evidence the model was indifferent. Skipping those positions, and scoring every one where
the model was confident, is both robust to the coin flip and no softer anywhere else.

`kv_rosetta.metrics.confident_agreement` does that: it scores only positions whose reference
margin clears a bar, reports how many were skipped so a thin score is visible, and **refuses
when nothing clears it** - agreement over an empty set is the vacuous pass this project is
written against.

Status: **measured once on this host** — 4 prompts, 48 positions each, target model alone.
**Proven by retained test**: the margin arithmetic and every refusal, including that a
certain-position flip still counts, so skipping the undecided does not soften the test
elsewhere (923 offline tests). **Untested, and the next thing to do**: whether the margin-aware
gate actually separates the translated cache from the blends on the alpha grid. It should - the
translation diverges at token 0-6, nowhere near the fragile region - but that is a prediction,
not a result.

## REQ-051 — The gate, settled: teacher-forced agreement over confident positions

REQ-050 predicted that skipping near-tied positions would make the gate stable without making
it weaker. Two things were needed to test it, and the first attempt at a rule failed.

### The simple rule is unsound

"Admit if the first divergence falls at an undecided position" can be checked against data
already collected - the first-divergence indices from REQ-049 and the margins from REQ-050 -
and it does not work. On lighthouse the **alpha 0.9** cache survived to token 38 and diverged
at a margin of 7.04, a confident position; the **worse alpha 0.85** cache diverged at the
near-tied token 29, margin 0.514. The better cache is rejected and the worse one admitted,
because the rule asks where you stopped and ignores how far you got.

It does get the important case right: at alpha 0 the translation diverges at margins of 2.41
to 7.46 on all four prompts - confident positions, real cache error - and is rejected.

### The sound combination

Free generation cannot be scored past the first divergence, since after it the candidate is
conditioned on different tokens. Teacher forcing keeps every position comparable; margin
skipping removes the positions where a flip says nothing. Both are needed.

| alpha | 1.0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.3 | 0.15 | 0.0 |
|---|---|---|---|---|---|---|---|---|---|
| lighthouse, confident-only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.867 | 0.867 | **0.733** |
| lighthouse, all positions | 1.000 | 1.000 | 0.969 | 0.969 | 1.000 | 0.969 | 0.813 | 0.844 | 0.719 |
| survey, confident-only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.968 | **0.903** |
| survey, all positions | 1.000 | 1.000 | 0.969 | 0.969 | 0.969 | 0.969 | 0.969 | 0.938 | 0.875 |

**Both confident-only rows are monotonic in alpha. Neither all-positions row is** - lighthouse
goes 0.969, 0.969, 1.000, 0.969 across the middle and 0.813, 0.844 at the bottom. Skipping one
or two positions out of thirty-two is what turns a wobbling number into an ordered one.

### The gate

**Admit when confident agreement is 1.000**: no disagreement at any position the reference
model was sure about. That admits every blend down to alpha 0.5, rejects the translation on
both prompts, and is monotonic so it cannot reverse on a better cache.

It saturates above alpha 0.5, so it decides admission and does not grade quality. Tuning a map
still needs `mean |Δlogprob|`, which is smooth across the whole range. Two instruments, two
jobs - the mistake in REQ-047 was expecting one number to do both.

Status: **measured once on this host** — 2 prompts, 9 ratios, 32 forced positions, margin bar
1.0, identity exact on both. **Proven by retained test**: the margin arithmetic and every
refusal (923 offline tests). **Remaining prompt dependence**: at alpha 0.3 lighthouse scores
0.867 and survey 1.000, so the two disagree at one ratio out of nine - much reduced, not
eliminated. **Untested**: the margin bar of 1.0 was chosen, not derived; whether a different
bar removes the last disagreement is unmeasured.

## REQ-052 — The margin bar, swept: skipping helps less than REQ-051 claimed

REQ-051 chose a margin bar of 1.0 and asserted that skipping near-tied positions "turns a
wobbling number into an ordered one". Capturing the raw forced distributions once and sweeping
the bar offline tests that properly, and it does not hold up.

| bar | positions skipped of 32 | monotonic | ratios where prompts disagree |
|---:|---:|---|---|
| **0.0** | 0 | yes | **1** (alpha 0.6) |
| 0.25 | 1 | yes | 4 |
| 0.5 | 1 | yes | 4 |
| 1.0 | 2 | yes | 2 |
| 2.0 | 3 | yes | 2 |
| 6.0 | 8 | yes | 2 |

**Every bar is monotonic on this grid, including no skipping at all**, and skipping nothing
gives the fewest prompt disagreements. REQ-051's monotonicity advantage was measured on a grid
sampling alphas 0.9, 0.8, 0.7 and 0.15; this grid samples 0.6, 0.4, 0.2 and 0.1 and the
wobble does not appear. The effect was a property of which ratios were sampled, not a robust
property of the metric.

The reason is simple arithmetic that REQ-051 did not check: **only one or two positions of
thirty-two fall below margin 1.0.** Skipping them can move agreement by about three percent,
which is the same order as the differences being compared. A correction built on moving one
position in thirty-two was never going to be stable.

| prompt | min margin | median | below 1.0 |
|---|---:|---:|---:|
| lighthouse | 0.420 | 7.94 | 2/32 |
| survey | 0.582 | 8.07 | 1/32 |
| orchard | 0.017 | 7.27 | 1/32 |

**No bar removes the prompt disagreement.** It persists at alpha 0.3-0.6 for every value
tried, so the residual variation is not something bar-tuning fixes.

### What survives from REQ-051

Two claims hold and one does not.

- **Teacher forcing is necessary** — free generation cannot be scored past the first
  divergence, and REQ-051's demonstration that a better cache can be rejected while a worse
  one is admitted stands on data.
- **The gate separates the translation from every blend** — 0.733 and 0.903 against 1.000.
  That is the result the project needed and it is unaffected.
- **Margin skipping is not established.** It is defensible in principle - a flip where the
  model was indifferent says nothing about the cache - but on this evidence it changes little
  and does not reliably improve anything. `confident_agreement` stays, since the principle is
  sound and the reporting is useful, but no claim rests on the skipping.

Status: **measured once on this host** — 3 prompts, 8 ratios, 32 forced positions, nine bars
swept offline from one capture. **Corrects**: REQ-051's monotonicity claim for margin skipping.
**Unchanged**: teacher forcing as the scoring protocol, and the separation of the translated
cache from the blends.

## REQ-053 — The whole pipeline, end to end, on the model the user actually runs

Capture, admit, restore and refuse - on `Tiel-Coder-35B-A3B` (qwen35moe, hybrid), on the
vacant W6800, with a live negative control.

```
saved      n_saved=512  checkpoints=1
admitted   e1dbb7a52cd9  needs patch: True  kv f16/f16  state v3
RESTORE    ok=True  cache_n=508  prompt_n=4  of 512 tokens
           "verified reuse: cache_n=508 of 512, 4 reprocessed"

check against the patched runtime : no objection
check against the STOCK runtime   : 2 objections -> refused
```

The stock runtime is the fleet's own binary. Offered the same artifact it is refused because
it reports neither a state-file version nor checkpoint persistence - and on that build the
restore would have been *accepted* and reused nothing, reporting the same `n_restored` as the
patched one. That is the failure this mechanism exists to convert into an error, demonstrated
rather than argued.

### Two defects the run found that no test had

**The store must be the slot-save-path.** Restore resolves an admitted object and hands the
server its filename with no copy on the request path, so a store in a different directory is
one the server cannot see. It failed with a bare HTTP 400.

**Model identity has to come from the caller.** The artifact records the model it was captured
from; llama.cpp's `/props` carries no identity at all, and the check read that as "cannot be
shown to match" - refusing a restore that then succeeded with 508 of 512 tokens reused. The
adapter derives identity from the weights file, so it is now supplied rather than looked up,
with `/props` as a fallback and absent-from-both still a refusal. It failed in the safe
direction, which is why it survived every offline test: the fixtures supplied an identity that
real props do not have.

Status: **measured once on this host** — one prompt, 512 tokens, one model, both binaries.
**Proven by retained test**: the requirements arithmetic and every refusal including the
caller-identity path (949 offline tests). **What this does not show**: any cross-model claim,
and any timing - this run measured correctness, not latency.

## REQ-054 — Restore against prefill on tiel-coder: the case for replacing kvwarm

REQ-053 showed the pipeline is correct. This measures whether it is worth using.

Patched build, vacant W6800, `Tiel-Coder-35B-A3B`, three repetitions per rung, medians. Total
restore latency is the restore call **plus** whatever the runtime still prefills afterwards -
reporting only the restore call would flatter it.

| tokens | artifact | cold prefill | restore total | of which call / residual | reused | speedup |
|---:|---:|---:|---:|---|---|---:|
| 512 | 135.6 MB | 443.8 ms | 246.9 ms | 106.4 + 140.4 | 508/512 | **1.80x** |
| 2,048 | 228.5 MB | 1,643.7 ms | 385.4 ms | 193.0 + 183.7 | 2,044/2,048 | **4.26x** |
| 8,192 | 348.7 MB | 6,029.9 ms | 410.9 ms | 244.6 + 145.6 | 8,188/8,192 | **14.68x** |

**The shape matters more than any single number.** Cold prefill grows linearly - 444, 1,644,
6,030 ms - while total restore stays nearly flat at 247, 385, 411 ms. The advantage therefore
widens with prefix length, which is the direction an agentic harness pushes.

### Against what kvwarm actually does

The observed kvwarm behaviour that started this work: up to 51 prefixes cycled into a server
with **2 slots**, every 900 seconds, 13 of the last 20 cycles complete re-prefills costing 15
to 84 seconds each. Those are recomputes of exactly the work a 411 ms restore reproduces.

The comparison is not quite like for like and should not be dressed up as one: kvwarm's
re-prefills were on the fleet's own quantised-KV configuration and included prompts longer
than 8,192 tokens. What this measures is that at the sizes tested, restoring is between 1.8
and 14.7 times cheaper than recomputing on the same machine and model.

### The caveat that matters

**Artifacts here live in `/dev/shm`, which is RAM.** Read time is a floor, not a typical
figure. The host's disks are at 99% and writing a 349 MB artifact there was not a risk worth
taking for a benchmark. Charging a notional NVMe read at 2 GB/s:

| tokens | +read | total | vs cold | speedup |
|---:|---:|---:|---:|---:|
| 512 | +71.1 ms | 318.0 ms | 443.8 ms | 1.40x |
| 2,048 | +119.8 ms | 505.2 ms | 1,643.7 ms | 3.25x |
| 8,192 | +182.8 ms | 593.7 ms | 6,029.9 ms | 10.16x |

Still decisive at the sizes that matter, and this is arithmetic rather than a measurement.

Status: **measured once on this host** — 3 rungs, 3 repetitions, medians, RAM-backed storage,
one model. **Not measured**: NVMe-backed reads, quantised KV, prefixes above 8,192, and
concurrent load - every figure here is from an otherwise idle GPU.

## REQ-055 — CUDA to ROCm: the last leg of the hardware claim

The one direction blocked all session, because vLLM held both 3090s. With one 3090 and one
W6800 free, the same artifact moves between **NVIDIA and AMD**.

Both binaries at `ca3d5a3e1` - the revision guard passes rather than being waived, which
matters here more than anywhere: a version mismatch is rejected for the state-file format and
is indistinguishable from "cross-vendor does not work".

| direction | reused | text | token ids | artifact |
|---|---:|---|---|---:|
| CUDA -> ROCm/HIP | **8,191 / 8,192** | identical | identical | 288.1 MB |
| ROCm/HIP -> CUDA | **8,191 / 8,192** | identical | identical | 288.1 MB |

The decomposition, which is the only reading that means anything:

| comparison | CUDA->HIP | HIP->CUDA |
|---|---:|---:|
| own restore vs own cold prefill | 0.750 | 0.229 |
| **foreign cache vs own cache** | **0.311** | **0.219** |
| two cold runs, different vendors, no cache | 0.718 | 0.718 |

**Using the other vendor's cache diverges by less than the two vendors already differ by**
doing identical cold work with no cache involved - 0.311 and 0.219 against a floor of 0.718.
On this run, moving a cache from an NVIDIA card to an AMD one costs *less* than the arithmetic
difference between the cards. Top-1 agreement is 1.000 in every comparison.

### The hardware claim, complete

| pair | status |
|---|---|
| ROCm/HIP <-> Vulkan, same AMD card | proven at 128, 8,192 and 32,000 tokens |
| **CUDA <-> ROCm/HIP, across vendors** | **proven at 8,192 tokens** |
| same backend, across a full process restart | proven, 252/263 after a cold stop |

One file, written on an NVIDIA GPU with CUDA, restored on an AMD GPU with ROCm, and back,
with identical output. That is the claim the project was named for.

Status: **measured once on this host** — one prompt, 8,192 tokens, one non-hybrid model
(qwen2 Q4_K_M), both builds at one revision, identity controls exact. **Untested**: CUDA at
32,000 tokens, CUDA with a hybrid model, and Vulkan-on-NVIDIA, which would close the matrix
entirely.

## REQ-056 — CUDA to ROCm at 32,000 tokens

REQ-055 proved the cross-vendor transfer at 8,192. This takes it to the top of the range the
use case lives in, on the same two builds at `ca3d5a3e1`.

| direction | reused | text | token ids | artifact |
|---|---:|---|---|---:|
| CUDA -> ROCm/HIP | **31,999 / 32,000** | identical | identical | 1.10 GiB |
| ROCm/HIP -> CUDA | **31,999 / 32,000** | identical | identical | 1.10 GiB |

99.997% reuse across vendors at 32,000 tokens. The one-token shortfall is llama.cpp always
reprocessing the final token, so it is constant and the reuse fraction only improves with
length.

**An operational note that belongs in the record rather than a footnote.** The harness reported
`fleet reclaimed the GPUs; unloading and retrying` twice during the second direction: its
built-in retry called llama-swap's unload endpoint to take memory back. That is the sanctioned
route and the operator had authorised taking the cards, but the effect is real - models that
were resident before the run are not resident after it, and llama-swap reloads them on the next
request rather than immediately. A benchmark that can evict production models should say so
where the numbers are, not somewhere else.

Status: **measured once on this host** — one prompt, 32,000 tokens, one non-hybrid model, both
builds at one revision, identity controls exact. **Untested**: CUDA with a hybrid model, and
Vulkan on NVIDIA, which would close the vendor/API matrix entirely.

## REQ-057 — The same KVX file across Ornith-9B, Qwen3.8-27B and Ornith-35B

The operator's question, answered directly: offer one artifact to three models and see what
happens. Every pairing, both same and cross, 512 tokens each.

| | attn layers | recurrent | KV heads | head_dim | d_inner |
|---|---:|---:|---:|---:|---:|
| Ornith-1.5-9B | 8 | 24 | 4 | 256 | 4096 |
| Qwen3.8-27B | 16 | 48 | 4 | 256 | 6144 |
| Ornith-1.5-35B (tiel) | 10 | 30 | 2 | 256 | 4096 |

**All three share a tokenizer** - identical token ids for the same text - and the 9B and 27B
share attention geometry *exactly*: 4 KV heads of 256, same rope base and rotary width. Their
per-token attention rows are the same size. Only the layer count differs.

| source \ target | ornith-9b | qwen38-27b | tiel-35b |
|---|---|---|---|
| **ornith-9b** | reused 508/512, output matches | **rejected** | **rejected** |
| **qwen38-27b** | **rejected** | reused 508/512, output matches | **rejected** |
| **tiel-35b** | **rejected** | **rejected** | reused 508/512, output matches |

**Six of six cross-model restores rejected. Three of three same-model restores exact. No
dangerous cases** - nothing was accepted, reused, and then produced the wrong model's output.

llama.cpp refuses them itself:

```
Unable to restore slot: No available space in KV cache or invalid slot save file
```

The artifact sizes make the reason plain even where the geometry matches: 122 MB, 347 MB and
142 MB for the same 512 tokens. Identical per-token rows do not help when one model has eight
attention layers and another sixteen - the body is a different length and the reader will not
take it.

### What this settles, and what it does not

It settles the literal question. **The same file is not usable across these models, and the
runtime says so rather than producing a wrong answer.** That is the safe failure, and it is
better than the alternative this project keeps guarding against.

It does not say translation is impossible - REQ-044 already measured that separately and the
gate rejected it. What it adds is that no *unmodified* artifact crosses, so any cross-model
path must go through the mapper and the gate, never through a direct restore.

**An unmeasured hazard, recorded rather than assumed away.** The server log shows
`restored 1 context checkpoint(s)` from a foreign file *before* rejecting its main body. The
checkpoint appendix carries no model identity, so it is loaded first and the state restore
fails after. Whether a rejected restore leaves those foreign checkpoints in the slot was not
tested here - the harness skipped the completion once the restore returned 400. If they
persist, a later completion could resume from another model's recurrent state, which is
exactly the silent-wrong case. **This needs a test before any claim of safety on the rejection
path.**

Status: **measured once on this host** — 3 models, 9 pairings, 512 tokens, same tokenizer
throughout, same patched build. **Untested**: whether a failed restore leaves foreign
checkpoints resident, and every pairing at longer prefixes.

## REQ-058 — Why the tensors are not readable across models, at the representation level

REQ-057 showed llama.cpp rejects a foreign artifact, and the log gives the exact check -
`state_read_data: mismatched layer count (16 instead of 8)` at `llama-kv-cache.cpp:2542`.
**That is not our handoff and not our transformers**: the file declares how many attention
layers it carries, the model has a fixed number of slots, and 8 layers of tensors are not 16.

But that is a *format* answer, and the real question is whether the tensors could be made
readable. This tests it on the most favourable pair on the machine.

**Ornith-1.5-9B and Qwen3.8-27B share everything except depth.** Both `qwen35`, both 4 KV
heads of 256, both rope base 1e7 over 64 dimensions, and the same tokenizer. A per-token
attention vector is 1024 values in **both**. Only the layer count differs: 8 against 16. So for
the first time a raw copy is even expressible - and the map, if fitted, is square.

| approach | median held-out R² | best | above 0.9 |
|---|---:|---:|---:|
| **raw depth copy**, no map at all | **−1.8146** | +0.3005 | 0/32 |
| fitted per-layer ridge, 1024 → 1024 | **+0.5324** | +0.6397 | 0/32 |
| *27B → tiel for comparison (heads 4→2, widths differ)* | +0.55 | — | 0/20 |

### Two findings, and the second is the important one

**A raw copy is far worse than predicting the mean.** Median R² −1.81. If the 9B and the 27B
shared a representation, copying layer to layer by normalised depth would score positive. It
does not. The vectors occupy the same *shape* and not the same *space*.

**Removing the geometric difficulty changed nothing.** 0.53 here against 0.55 for a pair that
also had to halve the head count and cross different widths. **The obstacle is not geometry.**
It is not the layer count, not the head count, not the file format, and not the handoff. Two
models trained separately encode the same text differently, and a per-layer linear map
recovers about half of it either way.

That is worth stating plainly because it redirects the work: making the file readable is
solved - splicing translated tensors into a target-geometry artifact produces a file llama.cpp
accepts, and REQ-044 did exactly that. What no amount of format work fixes is that the numbers
inside mean something different to each model.

### What would be needed

A per-layer linear map is the weakest thing that could have worked. Its ceiling here is ~0.64
on a single layer and ~0.53 typical, against a gate that wants agreement at essentially every
position. The next rung is a learned non-linear projector in the C2C direction, trained across
layers rather than one at a time. Nothing measured here says that is hopeless; everything
measured here says linear is not enough, and that the reason is representational rather than
structural.

Status: **measured once on this host** — 8,192 calibration tokens, held out by fraction, one
prompt pair, both directions of the depth mapping tried for the raw case. **Proven by retained
test**: none of this - it is a measurement, and the retained tests cover the machinery that
produced it. **Untested**: a non-linear map, and whether the raw copy improves at all with an
affine correction rather than none.

## REQ-059 — The paper's method, implemented properly, and what it does and does not buy

The operator asked for arXiv. [2608.03893](https://arxiv.org/abs/2608.03893) - *Cross-Model KV
Cache Transfer in LLM Families* - is the paper this project's README already cited, and reading
it properly showed **I had implemented only part of its method.**

It reports 56% of variance on keys and 32% on values from a single source layer, rising to
**79% and 65%** when the top-k most predictive source layers are **concatenated** as the input.
My fit used one source layer and reached 0.53, which matches their single-layer figure exactly.
The concatenation was the gap.

### What implementing it bought

Calibration was also quadrupled, from 8,192 to 32,768 tokens, because at k=8 the first attempt
had fewer training samples than features.

| pair | k=1 keys/values | k=3 | k=6 |
|---|---|---|---|
| 9B -> 27B, all 16 target layers | 0.536 / 0.469 | 0.580 / 0.496 | **0.585 / 0.502** |
| 9B -> tiel, all 10 target layers | 0.541 / 0.432 | 0.584 / 0.469 | **0.600 / 0.488** |
| *paper, Qwen3 14B -> 32B* | *0.56 / 0.32* | — | *0.79 / 0.65* |

**Single-layer reproduces the paper. Multi-layer does not.** Keys go 0.536 to 0.585 where the
paper goes 0.56 to 0.79. Values start *better* than theirs - 0.469 against 0.32 - and end far
worse.

### Three things tried that did not close it

**Per-head fitting**, which the paper also specifies, is **worse** here: 0.610 against 0.665 on
the same target layer. Head h of the source does not inform head h of the target, so
constraining the map to preserve head identity costs more than the better conditioning gains.
That is a real difference from the paper's setting, not a bug.

**More calibration** moved little. Four times the tokens took target-1 keys from 0.708 to
0.719.

**Lineage** turned out to matter, but not enough. The GGUF metadata settles what I had assumed:
`general.basename` is `ornith-1.5` for the 9B **and** for tiel, but `Qwen3.8-27B` for the 27B.
So every earlier cross-model number in this project - REQ-044 and REQ-058 included - was
measured **across lineages**, which is not the paper's setting.

**And with all layers scored, lineage makes no difference whatsoever.** Within-family reaches
0.600/0.488 at k=6; crossing lineages reaches 0.585/0.502. They are the same to within noise.

An interim reading of this entry claimed within-family was better by 0.06. That was taken when
only the first four target layers had been scored, and those are the layers that score best -
the very depth effect described below. Reporting a median over the easiest third of the data
was the mistake, and the full result withdraws it. **Two models sharing a lineage transfer no
better than two that do not**, which makes the obstacle look less like relatedness and more
like something every independently trained network has.

### The shape of the failure

R² falls sharply with target depth: 0.72 at layer 1, 0.44-0.52 at layer 10, values reaching
0.30. Shallow layers carry transferable structure; deep ones encode model-specific computation.
That is consistent across both pairs and is the clearest signal in the data about *why* a
linear map tops out.

Status: **measured on this host** — 3 model pairs, 32,768 calibration tokens each, held out by
fraction, k in {1,3,6,8}, whole-vector and per-head, ridge swept over four decades. **The
conclusion**: the paper's method is correctly implemented now and reaches ~0.64 on the
favourable within-family pair, against a gate that wants agreement at essentially every
position. **Untested**: whether the 9B->tiel median holds up over its remaining six layers -
only the first four are scored, and those are the ones that score best.

## REQ-060 — The conversion layer works: a 9B cache the 27B will load

The operator's plan: build a converter that reshapes heads and layers on the fly, make a KVX
per geometry, and work backwards to the conversion layer. `kv_rosetta/geometry.py` is that
converter, and this is it running against the two models.

**Acceptance and quality are separate questions and this separates them.**

| variant | loads | reused | output |
|---|---|---:|---|
| raw 9B file offered to the 27B | **no** — HTTP 400 | — | — |
| **geometry-converted** | **yes** | **508 / 512** | degenerate |
| 27B's own cache | yes | 508 / 512 | correct |

**The conversion layer does its job.** REQ-057's rejection - `mismatched layer count (16
instead of 8)` - is gone. A 9B cache reshaped by depth mapping into 16 layers of 4 heads is
loaded by the 27B and its prefix reused, 508 of 512 tokens. That is the milestone: the file is
*readable*, which it was not before, and readability is the precondition for measuring any map
at all.

**And it is not usable, exactly as expected.** The output:

```
27B's own       : " in late autumn and found their maps disagreed with the terrain by nea"
geometry-converted: " in the plateau in the plateau in the plateau in the pl"
```

Degenerate repetition. A reshape moves values into the right slots without making them mean
what the target expects, and this is what that looks like from the outside. Notably it is
*not* the fluent-but-wrong failure of REQ-044 - a pure reshape collapses rather than
hallucinating, which is at least a louder failure.

### A vacuous comparison I wrote and have to own

The harness reported `matches=True` for the converted run while the content plainly differed.
The completion request omitted `n_probs`, so `completion_probabilities` was absent, `toks()`
returned `[]` for both runs, and `[] == []` was compared. **A parity check that passed while
comparing nothing** - the precise failure this project was built around, in my own test.

The content strings are what the table above rests on. The `matches` field in that run's log
is worthless and the retained record says so.

Status: **measured once on this host** — one prompt, 512 tokens, depth-mapped conversion with
no fitted map. **Proven by retained test**: the converter itself, including that every output
value came from the input (981 offline tests). **What this shows**: geometry conversion is
solved and the remaining obstacle is entirely representational. **What it does not show**: any
quality claim - the fitted map on top of this reshape was not run here.

## REQ-061 — How the models actually consume the cache, and why R² is the wrong objective

The operator asked how each model uses the file. Reading `src/models/qwen35.cpp`, the key path
is:

```
Kcur = wk @ x
Kcur = RMSNorm(Kcur, attn_k_norm)     per-head normalisation with a learned gain
Kcur = ggml_rope_multi(...)           M-RoPE, then cached
```

### The RoPE question, settled rather than assumed

`ggml_rope_multi` is multi-section M-RoPE with `sections=[11,11,10,0]` on all three models, and
qwen35 uses the *interleaved* variant. That looked like it might invalidate every de-rotation
in this project. It does not: in `ops.cpp` all four component thetas advance unconditionally
each iteration (`theta_t *= theta_scale; theta_w *= ...; theta_h *= ...; theta_e *= ...`), so
for text-only input, where every position component is the token position, the sector selection
chooses between four **identical** values. **M-RoPE degenerates exactly to standard RoPE here.**
The existing strip is correct, now by verification rather than by luck.

### Keys live on a sphere

RMSNorm implies the cached keys have a constrained magnitude, and they do. Coefficient of
variation of the per-head key RMS, after stripping rotation:

| layer | source cv | target cv |
|---|---:|---:|
| 0 | 0.039 | 0.035 |
| 3 | 0.022 | 0.039 |
| 7 | 0.022 | 0.019 |

Two to four percent. The norm is essentially fixed, so a key is a direction and almost nothing
else - and a linear map's output does not lie on that manifold.

### The error is mostly magnitude, and R² rewards getting it wrong

| target | kind | R² | cosine | predicted/true magnitude |
|---|---|---:|---:|---:|
| 0 | k | 0.641 | **0.897** | 0.933 |
| 4 | k | 0.584 | **0.844** | 0.827 |
| 8 | k | 0.472 | **0.783** | 0.770 |
| 8 | v | 0.316 | 0.621 | 0.630 |

**Cosine runs far ahead of R².** The map recovers direction better than the variance figure
suggests, and systematically under-predicts magnitude by 7 to 37 percent - which is exactly
what ridge shrinkage does, because shrinking toward the mean minimises squared error.

Attention computes `softmax(q·k/sqrt(d))`. Under-scaled keys **compress the logits and flatten
the attention distribution**, and diffuse attention is what the degenerate repetition in
REQ-060 looks like from the outside. So the shrinkage that R² rewards is the very thing that
would damage the model's behaviour.

Rescaling each predicted key to the target's norm makes R² **worse** - 0.621 against 0.641 at
layer 0 - and that is the point: **R² is not how the model consumes the cache**, so optimising
it has been optimising the wrong thing. Whether the rescale helps the *gate* is a separate
question and the only one that matters.

Status: **measured on this host** — norms and error decomposition over 8,192 held-out tokens
at four target layers, source read from llama.cpp at `src/models/qwen35.cpp` and
`ggml/src/ggml-cpu/ops.cpp`. **Untested, and the next thing to run**: whether a
norm-corrected map passes the gate where the uncorrected one produced degenerate output. R²
says it is worse; the gate has not been asked.

## REQ-062 — Correcting key magnitude: the first intervention that moves the gate

REQ-061 established that cached keys are RMS-normalised, that the fitted map under-predicts
their magnitude by 7-37% through ordinary ridge shrinkage, and that attention computes
`softmax(q·k/sqrt(d))` so shrunk keys flatten the attention distribution. R² **rewards** that
shrinkage, because shrinking toward the mean is what minimises squared error. So R² has been
scoring the map on something the model does not do.

Rescaling each mapped key to the target's own per-head RMS, and putting all of it through the
gate:

| variant | reused | top-1 | mean \|Δ\| | output |
|---|---:|---:|---:|---|
| reshape only, no map | 508/512 | 0.042 | 1.394 | `" in the plateau in the plateau in the plateau"` |
| mapped, magnitudes as fitted | 508/512 | **0.000** | 1.897 | `" and found the maps were damaged."` |
| **mapped + norm correction** | 508/512 | **0.208** | 1.550 | `" in late autumn and found the weather had turned c"` |
| the 27B's own cache | 508/512 | matches | 0.000 | correct |

```
cold        : " in late autumn and found their maps disagreed with the terrain b"
mapped+norm : " in late autumn and found the weather had turned c"
```

**The norm-corrected output starts correctly.** It reproduces the target's own continuation for
several tokens before drifting, where the uncorrected map agreed on nothing at all and the bare
reshape produced degenerate repetition.

**0.000 to 0.208 top-1, from an intervention R² says is worse.** That is the whole argument of
REQ-061 demonstrated: optimising squared error optimises the wrong thing, because a key is
consumed as a direction with a constrained magnitude and not as a vector to be approximated in
the least-squares sense.

It is still nowhere near admissible - the gate wants agreement at essentially every position
and this gives one in five. But it is the first change in this whole line of work that moved
the number, and it came from reading `qwen35.cpp` rather than from more fitting. Everything
tried before it - top-k concatenation, per-head fitting, four times the calibration, matching
lineage, matching geometry - moved nothing.

**Operational note.** The harness again reported `fleet reclaimed the GPUs; unloading and
retrying` during this run, so models resident before it are not resident after. Recorded where
the numbers are, as in REQ-056.

Status: **measured once on this host** — one held-out prompt, 512 tokens, 24 generated, map
fitted on 32,768 calibration tokens. **What it establishes**: the objective was wrong, and
correcting it helps measurably. **What it does not**: admissibility, which remains far off.
**The obvious next step**: fit the map *under* a norm constraint rather than fitting freely and
rescaling afterwards - a projection onto the sphere the keys actually live on, rather than a
correction applied to a fit that was aiming somewhere else.

## REQ-063 — jlens assessed, and fitting on the sphere: a marginal gain

The operator asked whether `~/agentstuff/jlens` helps.

### What jlens offers, honestly

**No reusable code.** The repository imports an external `jlens` package
(`from jlens.fitting import fit`) that is **not installed in any venv on this host**, so the
lens-fitting implementation is unavailable. Its only in-repo gradient code is a smoke test -
`(out.hidden_states[-1] * probe).sum().backward()` in `m37j_phase0.py`, checking that
activation gradients are finite. Its findings concern router logits and domain signal on
Qwen MoE models, not KV representation. Its GPU program is blocked pending a dual-3090 window.

**One transferable idea, and it is the one that already worked.** A *Jacobian lens* fits in the
metric the model is sensitive to rather than in plain squared error. REQ-062's norm correction
is a crude instance: it took the gate from 0.000 to 0.208 precisely by matching what attention
consumes instead of what MSE measures. jlens names the principle; it does not supply machinery
for applying it here.

### Fitting on the sphere

Keys are RMS-normalised, so their metric is angular. REQ-062 fitted freely in the raw space and
dragged the result onto the sphere afterwards; this fits between unit directions and restores
magnitude at the end - aiming at the right thing from the start rather than correcting later.

| | median cosine |
|---|---:|
| raw fit | 0.8066 |
| **fit on unit directions** | **0.8161** |

**About one point of cosine.** Real, reproduced at every layer tested, and nowhere near enough
to matter. Per layer: 0.898 to 0.900 at layer 0 keys, 0.799 to 0.813 at layer 0 values, 0.595
to 0.609 at layer 8 values.

**A column in my own table that measured nothing.** "raw + rescale" reports the *same* cosine as
"raw" at every row, necessarily - rescaling a vector does not change its direction. The rescale
benefit is entirely in magnitude and shows up in the gate, not here. The column is a tautology
and is called one rather than left to look like a result.

### Where this leaves the map

Magnitude is now handled, and it was worth 0.208 of gate agreement. **Direction is the binding
constraint and ridge has hit its ceiling on it** - cosine sits at 0.78-0.90 for keys and
0.59-0.81 for values whichever way the fit is posed, and fitting on the correct manifold moves
it by a hundredth. Every remaining lever inside the linear family has now been tried:
single-layer, top-k concatenation, per-head, whole-vector, raw-space, unit-sphere, four
calibration sizes, three model pairs, two lineage relationships.

The next thing that could plausibly move direction is a more expressive map, which is the C2C
direction and a training problem rather than a closed-form one.

Status: **measured on this host** — 12 layer/kind combinations, 8,192 held-out tokens, six
ridge values per fit. **What jlens contributed**: the principle, already applied. **What it did
not**: any code, since its lens package is absent from this machine.

## REQ-064 — A non-linear projector, and the conclusion it forces

The last untried thing inside this approach. Every linear variant was exhausted and direction
was the binding constraint, so this trains a small network on the objective that matters.

The design is deliberately generous to the hypothesis: two hidden layers of 4,096 units with
GELU, **plus a linear skip connection**, so the model can represent any linear map and can only
improve on ridge. Trained directly on **cosine loss** rather than squared error - the metric
REQ-061 established is the right one - on 24,576 samples with the source layer ridge itself
preferred, and scored on identical held-out data.

| target | kind | ridge cosine | MLP cosine | gain |
|---|---|---:|---:|---:|
| 0 | k | 0.9001 | 0.9097 | +0.010 |
| 0 | v | 0.8125 | 0.8522 | **+0.040** |
| 4 | k | 0.8462 | 0.8539 | +0.008 |
| 4 | v | 0.7088 | 0.7314 | +0.023 |
| 8 | k | 0.7837 | 0.7895 | +0.006 |
| 8 | v | 0.6093 | 0.6222 | +0.013 |
| 12 | k | 0.8196 | 0.8235 | +0.004 |
| 12 | v | 0.6892 | 0.6997 | +0.011 |

**Median gain +0.0101.** Values benefit about three times as much as keys, and nothing benefits
enough to matter.

### What that forces

A network that can express any linear map, given the right objective and 24,576 examples,
recovers one additional point of cosine. **The ceiling is not the map's expressiveness.** It is
that the information required to reconstruct the target's keys is not present in the source's
keys.

That is not a surprising conclusion once stated. A cached key is `RMSNorm(W_k · h)` - a
learned, lossy projection of the hidden state into one model's own subspace. Two models trained
separately have different `W_k`, and the source's projection discards whatever its own `W_k`
happened not to preserve. No map from the source's keys can restore what was never in them.
Reconstructing the target's keys needs the *hidden states*, not another model's keys - and
anything that has the hidden states already has the text, at which point prefilling is the
direct route.

### The full record of what was tried

| lever | effect on the result |
|---|---|
| geometry match (heads 4→2 against 4→4) | none |
| lineage match (Ornith→Ornith against Ornith→Qwen) | none |
| top-k source concatenation, the paper's method | +0.05 R², plateaus |
| per-head fitting, also the paper's | **worse** |
| calibration 8,192 → 32,768 tokens | +0.01 |
| raw copy, no map | −1.81 R², far worse than the mean |
| fitting on the unit sphere | +0.01 cosine |
| **magnitude correction** | **0.000 → 0.208 gate agreement** |
| non-linear projector with a linear skip | +0.01 cosine |

One intervention mattered, and it came from reading how the model consumes the cache rather
than from fitting harder. Everything else moved nothing.

Status: **measured on this host** — 8 layer/kind combinations, 60 epochs each, held out by
fraction, ridge given its best source layer and four ridge values as a fair baseline.
**The conclusion**: cross-model KV translation from source KV alone is limited by information,
not by model class, and no map of this family will close the gap.

## REQ-065 — Round table consensus, and the one recommendation that did not survive checking

The operator asked for a panel. Four profiles were configured; **two produced nothing** - the
Architect returned HTTP 502 and the Skeptic returned an empty position - and the two that
answered, Engineer and Pragmatist, are **the same model** (`codex/gpt-5.6-luna`). So this is one
model answering twice, not a four-way consensus, and it is weighted accordingly.

### What they said

- **Abandon universal cross-model KV translation for production.** The 0.208 gate agreement
  against ~1.000 required, with geometry and lineage controls both failing, supports an
  information bottleneck rather than an inadequate projector.
- **The bottleneck conclusion is right for zero-training translation, but is not a theorem.** A
  trained bridge for deliberately co-trained families or shared trunks is not excluded - it
  would simply not be an architecture-independent format.
- **Hidden-state export does not rescue it.** The target must still run its layers to turn
  source hidden states into target KV, which is prefill with serialisation overhead. If the
  artifact carries the text, prefilling is the simpler route.
- **Ship model-specific portable KV instead**: fingerprints, content-addressed blocks, pinned
  hot prefixes, worker affinity, eviction and admission fixes, text/token fallback.

That matches the measured evidence and I accept it.

### The recommendation that did not survive checking

The Engineer's headline practical fix was to enable vLLM's Automatic Prefix Caching to stop the
warmer thrashing, with the Pragmatist adding that hybrid prefix caching is documented
experimental. Checked on this host:

```
vllm 0.27.1 (the version actually serving the 27B)
enable_prefix_caching default: True
launch path: no --no-enable-prefix-caching, no override
```

**It is already on, and the warmer still thrashes.** So APC is not the missing fix, and the
reason is what this project exists for: vLLM's prefix cache is in-memory and per-instance. It
evicts under pressure and is gone when the model unloads. Durable, restorable caches are
precisely the gap, and REQ-054 measured what closing it is worth - 14.68x at 8,192 tokens with
restore time flat where prefill grows linearly.

A panel's practical advice is worth exactly as much as checking it costs.

Status: **verified on this host** — vLLM version and default read from the serving venv at
`~/qwen38-27b-rtx3090/venv`. **Accepted**: the verdict on cross-model translation, which
matches REQ-064. **Rejected on evidence**: that enabling APC addresses the warmer, since it is
already enabled.

---

## REQ-066 — A mutation audit of every guard, and the two things it caught in itself

**Request:** "keep testing".

Ran `scripts/mutation_check.py` across all 22 modules: disable each `if`-guarded `raise` in
turn, require some test to fail. A guard no test notices the removal of is documentation, not
behaviour.

### First result was wrong, and the shape of the error matters

The first pass hand-mapped modules to test files and reported **170/212**, with
`container.py` at **0/23**. Zero is not a coverage level, it is a mapping error — the module
was paired with `tests/test_artifact_store` and `tests/test_store`, neither of which imports
it. Re-running against the modules that actually import it (`test_hardening`,
`test_conformance`, `test_compound_format_agreement`, `test_segments`,
`test_llamacpp_roundtrip`) gave **1/23**.

So the guess was wrong twice over: wrong that the number was 0, and wrong in the prediction
that the mapping error was the whole story. 1/23 is a real hole, in the one module that
decides whether bytes on disk become tensors handed to a runtime.

### Closed

| Module | Before | After |
|---|---|---|
| `manifest.py` | 4/10 | 10/10 |
| `container.py` | 1/23 | **23/23** |
| `store.py` | 2/5 | 5/5 |
| `admitted_store.py` | 9/12 | 12/12 |
| `dtypes.py` | 2/3 | 3/3 |
| `segments.py` | 0/1 | 1/1 |
| `identity.py` | 0/1 | 1/1 |
| `hybrid_schema.py` | 2/3 | 3/3 |
| `gguf.py` | 6/7 | 7/7 |
| `mappers/linear.py` | 10/12 | 12/12 |

### Four of the new tests were themselves vacuous

In `segments`, `hybrid_schema` and both `linear` cases, the guarded input **still raises the
same error class** once the guard is deleted — from a field-extraction `except` clause, or
from the underdetermined-fit check further down. `assertRaises(SegmentError)` passes either
way. Only asserting the message distinguishes the guard from its absence.

That is the same failure this audit exists to find, and it appeared inside the tests written
to close the audit. Caught only because every new test was put back through the mutation check
rather than trusted for being green.

Every module in the repo now reports 100%. 1042 offline tests pass, up from 999.

### The audit hung, and the hang was the finding

The `container.py` run sat at 0% CPU for 30 minutes. Removing `extract_payload`'s
`if not block: raise` does not produce a wrong answer — it produces a **non-terminating
loop**, because `read()` at EOF returns `b""` forever and `remaining` never decreases. The
harness had no timeout, so it hung at that mutation and reported nothing at all for the
module. `scripts/mutation_check.py` now bounds each run at 600s and reports a hang as `HUNG`,
distinct from a clean assertion failure: a hang is still a detection, but it is not the same
evidence.

Two copies of the audit were also running concurrently against the same file, each mutating
what the other read, and the killed run left `container.py` mutated in the worktree. Restored
from git and verified with `git status` before continuing. Worth stating plainly: an audit
tool that rewrites the source it audits will leave the tree dirty if it is interrupted.

The same mistake then repeated in a subtler form: the `gguf.py` audit was started while the
`container.py` audit was still running, and its test list included `test_container_guards`.
container's L261 was disabled at that moment, so the extraction test hit the infinite loop
described above and the gguf baseline timed out. Two mutation audits must not run at once, even
against different sources, if their test sets overlap at all. Re-run alone: **7/7**.

### Security-relevant guards, previously undefended

`store.path_for` and `ArtifactStore.path_for` refuse a path that resolves outside the store
root, and `AdmittedStore.resolve` refuses an object that resolves outside the store or is not
a regular file. Digest validation does not cover these: every path component is well-formed
hex and the escape happens during *resolution*, reached by planting a symlink where a model
directory or an object belongs. `os.open(dir, O_RDONLY|O_NOFOLLOW)` succeeds on Linux, so
`O_NOFOLLOW` does not cover the directory case either — only the `S_ISREG` check on the open
descriptor does.

Status: **proven by retained test** — every figure above is a mutation-check result, not a
line-coverage number.

---

## REQ-067 — Two untested matrix cells, and a determinism problem found by the control

**Request:** "keep testing", continued after the guard audit.

Closed two cells that had been on the untested list: **CUDA with a hybrid model**, and
**Vulkan on NVIDIA**. Both 3090s were free; the llama-swap fleet on ROCm0 (tiel-coder, 5819)
and the embedding server were left untouched throughout, and both 3090s were released after.

### What was measured

| cell | result |
|---|---|
| hybrid (`qwen35` 27B) on CUDA | `n_restored=453`, `cache_n=0` — reuses nothing |
| CUDA-written 186 MB state accepted by Vulkan | yes: `n_restored=453`, full `n_read` |
| Vulkan-on-NVIDIA determinism, 6 cold runs | **3 distinct outputs** |
| CUDA determinism, same model/flags/cards | **1 distinct output** |

Full write-up in `docs/research-findings.md` §24 and §25.

### The finding that was nearly reported backwards

The first observation was that Vulkan's output after restoring a CUDA-written cache differed
from Vulkan's own cold run — which reads as "restoring a foreign cache perturbs generation even
though it reuses nothing". That is a striking claim and it was wrong. Running the cold case a
second time showed the two cold runs differed from *each other*. The restore was never
implicated; the backend simply is not reproducible against itself here.

Six identical cold runs at temperature 0 with a fixed seed, slot erased and displaced between
each, no cache involved anywhere: Vulkan produced 3 distinct outputs, CUDA produced 1. One of
Vulkan's three is byte-identical to CUDA's, so it is variation rather than a systematic offset.

This matters beyond the immediate test. The acceptance harness uses identical text and token
ids as evidence that a restored cache reproduced the uncached result. That check is only as
strong as the determinism of the configuration it runs on, and that determinism had never been
measured. On Vulkan here it does not hold.

It does not invalidate REQ-036's HIP <-> Vulkan results — different model, AMD cards, and real
reuse with `cache_n` in the thousands rather than everything being recomputed. But whether AMD
Vulkan is deterministic under those flags is now an open question that the existing data cannot
answer. Logged rather than assumed either way.

### Two smaller corrections along the way

A first hybrid save reported `n_saved=0` and a 1,200-byte file — which looks exactly like the
known unpatched-build signature and would have been easy to write down as one. It was slot
routing: the completion had landed on a different slot than the one saved. Pinning `id_slot`
gave 453 cells and 186 MB.

`laguna-s-2.1` was picked as a non-hybrid candidate and does not load on upstream `ca3d5a3`
("wrong number of tensors; expected 76, got 69") — it is a fork-specific architecture. Every
locally available model is either hybrid or fork-specific, so the strongest form of the
Vulkan-on-NVIDIA test — one with actual prefix reuse to compare — still needs a model this
repo does not have on disk.

Status: **measured once on this host** for every number above. **Untested**: whether
`--parallel 1` or flash attention removes the Vulkan nondeterminism (batch composition under
4 slots and `kv_unified=true` is a plausible mechanism, stated as hypothesis); whether AMD
Vulkan shows the same; whether it holds for non-hybrid models.

---

## REQ-068 — Exhausting the options: two hypotheses killed, one matrix cell closed, one claim corrected

**Request:** "keep testing, stop waiting on me. you need to exhaust all options before you wait
on me to make a decision."

Fair. The previous tick ended by asking permission to download a model. Downloaded it and kept
going.

### 1. Both explanations for the Vulkan nondeterminism were wrong

REQ-067 proposed batch composition (4 slots, `kv_unified=true`) as the likely mechanism, with
flash attention as the other candidate. Tested both:

| Vulkan configuration, `qwen35` 27B | distinct outputs / 6 cold runs |
|---|---:|
| 4 slots (original) | 3 |
| `--parallel 1` | 3 |
| `--parallel 1 -fa on` | 3 |
| dense `qwen2` 3B instead | **1** |

All three failing configurations produce the *same three hashes*, so it is one specific
nondeterministic reduction, not general noise. Only changing the model removed it.

**This corrects REQ-067 and §24.** The claim "Vulkan is not run-to-run deterministic here" was
too broad. It is the hybrid/recurrent path *under* Vulkan. A dense model on the same card is
deterministic, which also means §17's HIP <-> Vulkan evidence was never in danger — it used a
dense model. §24 has been rewritten with the refuting table rather than quietly amended.

### 2. The missing matrix cell, closed

Downloaded Qwen2.5-3B-Instruct Q4_K_M (`qwen2`, prefix-reusable) — every model already on disk
was either hybrid or a fork-specific architecture, which is what blocked this last tick.

CUDA <-> Vulkan on **one** RTX 3090, both binaries at `ca3d5a3e1`, 2,048 tokens: **2047/2048
reused in both directions, content and token ids matching both ways**, 75.5 MB artifact. Every
prior cross-backend result changed the card as well as the API; this one holds the hardware
fixed, so the API is the only variable. Foreign-cache logprob divergence 0.353/0.456 against a
0.976 two-cold-runs floor, top-1 agreement 1.0 throughout. Full numbers in §26.

### 3. A wrong reference nearly recorded a failure

A hand-run check compared "CUDA after restoring a Vulkan cache" against "CUDA cold prefill" and
got differing text. Those are different computations and the comparison answers nothing. The
retained harness compares against the source backend's run, and under that reference both
directions match exactly. Reaching for the existing harness instead of trusting the hand-rolled
comparison is what turned an apparent failure into a clean pass.

Status: **measured once on this host** — §26's table, the determinism table above.
**Corrected**: REQ-067's "Vulkan is nondeterministic" narrowed to hybrid-on-Vulkan.
**Refuted**: batch composition and flash attention as mechanisms. **Untested**: whether AMD
Vulkan shows the same hybrid nondeterminism; whether it appears at other prompt lengths.

---

## REQ-069 — The nondeterminism isolated to one cell, and the vendor/API square completed

Continuation of REQ-068 without stopping to ask.

### Isolation

REQ-068 narrowed the nondeterminism from "Vulkan" to "hybrid on Vulkan" but could not say
whether it was Vulkan or simply not-CUDA. Two more measurements settle it:

| model | CUDA (NVIDIA) | HIP (AMD) | Vulkan (NVIDIA) | Vulkan (AMD) |
|---|---:|---:|---:|---:|
| `qwen35` 27B, hybrid | 1 | 1 | **3** | **3** |
| `qwen2` 3B, dense | — | — | 1 | 1 |

(distinct outputs across 6 identical cold runs, temp 0, fixed seed, no cache)

HIP with the hybrid model is deterministic, so it is not "everything except CUDA". Vulkan with
a dense model is deterministic on both vendors, so it is not Vulkan generally. **The single
nondeterministic cell is Vulkan x hybrid**, and two of AMD Vulkan's three output hashes are
byte-identical to two of NVIDIA Vulkan's — the same reduction orderings on unrelated hardware,
which points at the Vulkan recurrent-path kernels rather than a driver or a card.

### Vendor and API separated

| held fixed | varied | foreign-cache divergence | two-cold-runs floor |
|---|---|---:|---:|
| card (one RTX 3090) | API (CUDA <-> Vulkan) | 0.353 / 0.456 | 0.976 |
| API (Vulkan) | vendor (NVIDIA <-> AMD) | 0.974 / 0.965 | 0.960 |

Both at 2047/2048 reused, content and token ids matching in all four directions, top-1
agreement 1.0 throughout. Crossing vendors costs about exactly what the vendors already differ
by; crossing APIs on one card costs distinctly less than the APIs differ by.

Every previous cross-backend result changed the card and the API together. These two do not,
so the matrix is now closed on both axes independently.

Status: **measured once on this host**, all six determinism cells and both gate runs.
**Untested**: whether the Vulkan hybrid nondeterminism persists at other prompt lengths, and
whether it affects a *restored* hybrid cache (moot in practice, since hybrid restore reuses
nothing on an unpatched build).

---

## REQ-070 — Turning the determinism measurement into admission-quality evidence

The steer at `8915aff` (written against `45fcb3b`) added a P0 preflight requiring each
production reader to be shown reproducible against itself before any restored-versus-cold
verdict is taken, and judged the existing work insufficient: *"The commit retains the summary
but not an admission-quality runner or raw per-run record, so it does not allowlist any tuple."*

That is a fair reading. REQ-067–069 measured the right thing and kept hashes and counts, which
is research evidence, not an admission record. Built the runner rather than arguing the point.

### `scripts/reader_determinism.py`

Six or more cold completions on an explicitly emptied slot, retaining per run: raw token ids,
text, per-position probability vectors, slot routing, cache_n, timing — plus the launch argv,
process identity, binary and loaded-library digests, model and prompt digests, and the server's
own build attestation. Verdict is exact parity, declared in the source before measuring: one
text, one token sequence, one set of vectors, or the configuration is not allowlisted.

Fails closed on its own inputs. A run that reused cache is not a cold sample; a run with empty
probability vectors compares equal to any other such run; a set spread across slots is not six
repetitions of one configuration. Each refuses the whole set rather than dropping a run.
7/7 guards defended, 13 tests, none needing a GPU.

### What the raw records showed that the counts did not

| configuration | texts | token seqs | **prob vectors** | reproducible |
|---|---:|---:|---:|:--:|
| cuda-nvidia-hybrid | 1 | 1 | 1 | yes |
| hip-amd-hybrid | 1 | 1 | 1 | yes |
| vulkan-nvidia-hybrid | 3 | 3 | **6** | no |
| vulkan-amd-hybrid | 3 | 3 | **6** | no |
| vulkan-nvidia-dense | 1 | 1 | 1 | yes |
| vulkan-amd-dense | 1 | 1 | 1 | yes |

**All six Vulkan hybrid runs differ at the distribution level**, on both vendors. The text
collapses them into three, so REQ-067's "3 distinct outputs" understated the instability by
half. The steer's insistence on vectors over a summary count was not bureaucratic — it changed
the number.

Records retained at `docs/records/reader-determinism/`, all at build `b151-ca3d5a3e1` against
one prompt digest.

### Two bugs the runner found in itself

The first preflight attempt failed all three configurations with HTTP 501 on
`/slots/0?action=erase`: llama.cpp refuses every slot action unless `--slot-save-path` is set,
and the runner did not pass it although the manual servers had. Nothing is saved by this
runner, so the flag looked unnecessary. There is now a test asserting argv carries it.

The mutation audit also flagged `if __name__ == "__main__": raise SystemExit(main())` as an
undefended guard. It is an entrypoint, not a refusal, and no test can legitimately kill it —
a false positive in `scripts/mutation_check.py`, now skipped rather than reported forever.

Status: **proven by retained test** — the runner's refusals. **Measured once on this host** —
all six configurations, with raw records committed. **Not claimed**: this preflight allowlists
the four reproducible configurations for exact-parity comparison; it does not by itself
allowlist any restore tuple, which is the steer's next step.

---

## REQ-071 — P-1: proving a rejected restore is actually harmless

The steer's first execution step. Its premise: all six wrong-model restores returned HTTP 400,
but llama.cpp logged that it did work before rejecting, and no harness had issued a completion
afterwards — so nothing had established the target slot was untouched.

`scripts/slot_poisoning.py` establishes it. Baseline cold completion, offer the slot a foreign
model's state file, then re-run the identical completion **without erasing** — erasing would
destroy the evidence. Text, token ids and probability vectors must all match.

| target (CUDA, reproducible) | foreign artifact | rejected at | pristine |
|---|---|---|---|
| Qwen3.8-27B `qwen35` | Qwen2.5-3B `qwen2`, 443 KB | `state_read_meta` | yes |
| Qwen3.8-27B `qwen35` | Qwen3.5-4B `qwen35`, 53 MB | `state_read_data: mismatched layer count (8 instead of 16)` | yes |

The second case matters more: the reader consumed metadata and entered the data section before
failing, so this is not a cheap early-out, and the slot was still byte-identical afterwards
down to the vectors.

The runner will not render a verdict without a passing `reader_determinism` record for that
exact reader label — otherwise a post-rejection difference could not be attributed to the
restore rather than to the reader. Pointed at `vulkan-nvidia-hybrid`, it refuses before
starting a server. That guard is what connects REQ-070 to this result rather than leaving two
unrelated measurements.

**Not claimed:** neither foreign artifact carried an `SCKP` appendix, since neither writer was
a patched build, so the checkpoint-appendix-before-rejection path the steer specifically
described is untested. Establishing it needs a patched writer.

Status: **proven by retained test** — the runner's refusals (3/3 guards, 9 tests).
**Measured once on this host** — both poisoning cases, records at
`docs/records/slot-poisoning/`. **Untested** — the appendix path.

---

## REQ-072 — "Same geometry should work": tested, and largely right

**Request:** "i still think theres a way to make this model agnostic. at the very least models
with the same geometry should work."

Correct on both counts, and my summary of §20 was wrong. §20 tested `qwen35` (16 layers,
4 KV heads) into `qwen35moe` (10 layers, 2 KV heads) — a **geometry** change needing a learned
mapping, which is what failed. I had been citing it as "caches do not move between models". The
same-geometry case was never run. It needs no translation at all, because llama.cpp's state
format records shape, not model identity.

`scripts/cross_model_gate.py` measures it, with identity / noise / native controls.

| pair (same geometry) | reused | foreign top-1 vs target's own restore | max Δ |
|---|---:|---:|---:|
| Qwen2.5-3B-Instruct Q4_K_M → Q8_0 | 3087/3088 | **0.984** | 1.03 |
| Qwen2.5-3B **base** → **Instruct** | 3087/3088 | **0.930** | 3.56 |
| noise (values scrambled, header intact) | 3087/3088 | 0.000 | — |

128 teacher-forced positions. Quantisation variants transfer at or above this machine's own
reproducibility floor; a real fine-tune degrades measurably but is nowhere near the 0.0 that
cross-geometry produced.

### Five things had to be corrected before any number meant anything

1. **Free generation measured the cascade, not the cache.** At 128 freely generated tokens a
   model restoring its *own* cache agreed with its own cold prefill on 0.23 of positions.
   `gate.py` already declares `teacher_forced` the default protocol for this reason and this
   runner was not using it. Under teacher forcing the same case scores 0.977.
2. **The noise control was being rejected.** Scrambling everything after the token header
   destroyed per-layer type ids and row sizes; the server answered 400. A rejected noise
   control bounds nothing. It now scrambles only the spans the repo's own GGSQ parser locates.
3. **The threshold was stricter than the machine.** `min_top1 = 0.99` against an identity
   baseline of 0.969–0.977, so nothing could pass for reasons unrelated to the cache. The
   verdict now reports `baseline_top1` and `threshold_exceeds_baseline`.
4. **The gate scored against a cold prefill.** Restoring and prefilling are different
   computations. Scoring foreign-restore against own-restore isolates authorship.
5. **`probs()` crashed on a null logprob**, which llama.cpp emits for an alternative scored at
   probability zero. It killed a run after both servers had done their work. Shared helper, so
   this was latent for every gate in the repo.

Also corrected: the geometry check compared `key_length` as `None == None` for `qwen2`, which
omits it. Two models with different head dimensions would have passed on a pair of nulls.
`head_dim` is now derived from `embedding_length / head_count`, and an indivisible pair is
refused rather than floored — the 5120/24 → 213 error this project already made once.

One flaw was corrected and changed nothing: the identity cache was saved after the 32-token
native run while the foreign one was saved after a 1-token run, so the control carried 31 extra
cells. Matching them left every number identical. Recorded because assuming the reverse would
have been easy.

Status: **measured once on this host**, records at `docs/records/cross-model/`.
**Proven by retained test**: the gate's controls (7/7 guards, 28 tests). **Untested**: more
than one prompt, other geometry-sharing families, and whether a fine-tune pair closer than
base→instruct closes the gap.

---

## REQ-073 — One KVX file per model, compared through a lens: the converter is the identity

**Request:** build the same KVX file for a base model, a fine-tune and a dense model, see how
each uses the data, and work out how to build the translator that lets one file serve all.

Built `scripts/kvx_diff.py` (decode and compare per layer/kind/head) and
`scripts/kvx_fit_converter.py` (fit a per-head linear map, score it on held-out prompts against
a do-nothing baseline). Three state files over byte-identical token ids.

### What the lens shows

| pair | K cosine | K rel err | K norm ratio | V cosine | V rel err | V norm ratio |
|---|---:|---:|---:|---:|---:|---:|
| instruct Q4 vs instruct Q8 | 0.9972 | 0.079 | 1.0027 | 0.9833 | 0.186 | 1.0060 |
| instruct Q4 vs base Q4 | 0.9909 | 0.143 | 1.0003 | 0.9542 | 0.305 | 0.9831 |

Not scale — norm ratios are 1.000. V is about twice as far off as K. And the error accumulates
with depth: layer 0 is essentially identical (cosine 1.0000), the twenties are the worst.

### The translator, answered

Per-head ridge, 128->128, four prompts to fit, two held out whole:

| | fitted R² | do-nothing | gain |
|---|---:|---:|---:|
| K | 0.911 | 0.948 | −0.027 |
| V | 0.631 | 0.890 | −0.174 |

**Zero of 36 layers gained anything.** A ridge sweep over six orders of magnitude never reaches
parity and improves monotonically toward the identity, which is the signature of the identity
being optimal. So for same-geometry models the translator you asked for **is the identity map**,
and that is already what raw transfer does. Fitting anything else destroys a near-perfect
starting point — layer 0 goes from 0.998 to 0.488 on V.

The residual is the target model's own weight drift. The source cache carries no information
about it, so no function *of the source* can recover it.

### The lead this opens

Early layers agree between fine-tunes and late layers do not. That points at **partial-depth
reuse**: take the foreign cache for the layers where it agrees, recompute only the layers where
it does not. That trades a fraction of the prefill for correctness instead of trading
correctness for all of it, and it falls directly out of the depth curve rather than being a
guess. Untested; next.

Status: **measured once on this host**, records at `docs/records/kvx-diff/`. **Proven by
retained test** — both runners' refusals (6/6 and 3/3 guards, 26 tests), including that the
prompt-level split cannot be turned off and that the do-nothing baseline is always reported.

---

## REQ-074 — Partial-depth reuse tested and closed

Continuation of REQ-073's lead, which I called "the one I'd bet on". It lost.

`scripts/kvx_splice.py` builds hybrid state files byte-for-byte — named layers from the foreign
cache, the rest from the target's own — after checking span-by-span that the two payloads are
laid out identically. Both sweep endpoints reproduce a file already measured elsewhere.

| foreign layers | early end Δ | late end Δ | early as % of full | layers as % of stack |
|---:|---:|---:|---:|---:|
| 2 | 0.0615 | 0.0704 | 23% | 6% |
| 16 | 0.1436 | 0.2023 | 55% | 44% |
| 20 | 0.1445 | 0.2111 | 55% | 56% |
| 36 | 0.2637 | 0.2637 | 100% | 100% |

Deep layers do cost more than shallow ones, so REQ-073's depth reading was right about
direction. But the cost tracks layer count almost exactly past the first few layers, and those
first two cost 23% of the total damage for 6% of the stack — **worse** than proportional. There
is no knee, so there is nothing to build.

A control ruled out the tidy explanation. Two layers from the middle cost 0.0905 against 0.0615
from the bottom and 0.0704 from the top, so the entry cost is not a constant "mixing penalty":
position matters, and the middle is the worst place to swap. Every two-layer choice still lands
in the same band, which is what closes the idea regardless of mechanism.

Also added `mean_abs_logprob_delta` to `compare_forced`. Top-1 agreement over 64 positions moves
in steps of 1/64 and the max delta is one worst-case token; the first sweep's top-1 column was
unreadable noise (0.9531 at 12 layers, 1.0000 at 16) while the mean was cleanly monotonic. A
sweep needs a smooth quantity or a real trend reads as nothing.

Three attempts now agree: §20 could not translate across geometries, §29 found the optimal
same-geometry converter is the identity, §30 finds no subset of layers is cheap. The common
cause is that the difference is the target's own weight drift and the source cache holds no
information about it.

Status: **measured once on this host**, records at `docs/records/splice/`. **Proven by retained
test** — the splice's layout precondition and both sweep endpoints (5/5 guards, 17 tests).

---

## REQ-075 — The shared drive, and memory that does not throw away the prompt

**Request:** storing prompts is the main goal, storing memories would be nice, and it would be
nice if the KVX file acted as a shared drive any model can use.

Three measurements say the *tensors* cannot be the shared thing. Everything else in the file
can be, and that is the part the goal needs. `kv_rosetta/shared.py` splits a drive into
model-neutral **content** — system text, tool schemas, memory entries — and one cache
**attachment** per `(model identity, cache ABI)` that has warmed it.

Live, three models on one drive, 963 tokens:

| visit | attachment | cache_n | prefilled |
|---|---|---:|---:|
| 1 (all three models) | miss | 0 | 966 |
| 2 (all three models) | hit | 965 | 1 |

Three attachments, one content document, no cross-model leaks.

### Memory: measured the waste, then fixed it

Keying attachments to the whole content means editing a memory invalidates the unchanged system
and tools regions too. Measured what that costs before building anything, on a memory grown
from 823 to 889 tokens behind 642 unchanged tokens:

| | cache_n | prefilled | ms |
|---|---:|---:|---:|
| v2 cold — what exact-digest lookup forces | 0 | 892 | 110 |
| v2 via the drive's prefix match | 820 | 72 | 19 |

**5.8x**, for text the drive already held. `best_attachment` now returns the attachment sharing
the longest token prefix, with the reusable fraction — 822/889 predicted, 820 achieved.

Safe for one specific reason that does not generalise: llama.cpp checks the token prefix
itself, so a wrong guess costs a re-prefill rather than wrong output. That check covers tokens,
not weights, which is why a foreign *model's* attachment stays refused outright. Same model,
different text: the runtime protects us. Different model, same text: nothing does.

### Two bugs the tests found

The content digest covered token ids, names and roles but not the entry **text**, so editing an
entry left the identity unchanged and the tamper check could not fire. And a drive is not a
runtime's slot directory — llama.cpp resolves restore filenames only inside its own
`--slot-save-path`, which surfaced as an opaque HTTP 400 rather than a miss. `stage()` now
places the attachment where the runtime looks, hard-linking when the filesystems match.

Status: **proven by retained test** — 11/11 guards, 42 tests, including that a foreign model is
never offered an attachment by either lookup. **Measured once on this host** — the three-model
drive and the memory-growth timings.

---

## REQ-076 — One drive across dense, hybrid and MoE

**Request:** try it across dense and MoE models — does the same KVX share work with 9b, 27b and
tiel on the W6800?

Two corrections to the premise first, both stated rather than worked around. **There is no 9B
model** on this host or in the llama-swap config; Ornith 1.5 is 35B-A3B, and it fails context
creation on two 3090s. A dense Qwen2.5-3B stands in, which makes this a dense/hybrid/MoE
comparison rather than a size one. And **both W6800s were fully occupied** by the fleet's two
tiel instances (33.5 GB and 33.0 GB, ports 5818/5819), so this ran on the free 3090s rather
than unloading a live service.

### The tensor half, answered without a GPU

| model | arch | layers | kv heads | head_dim |
|---|---|---:|---:|---:|
| Qwen2.5-3B-Instruct | `qwen2` dense | 36 | 2 | 128 |
| Qwen3.8-27B | `qwen35` hybrid | 65 | 4 | 256 |
| Tiel-Coder-35B-A3B | `qwen35moe` hybrid MoE | 41 | 2 | 256 |

No pair shares a geometry, so no attachment can move between them. That is REQ-063's territory
and already measured at 0.00 top-1; running it again would have re-asked an answered question.

### The drive half, measured

| visit | model | attachment | cache_n | prefilled | ms |
|---|---|---|---:|---:|---:|
| 2 | dense 3B | hit | **675** | **1** | **18** (from 84) |
| 2 | hybrid 27B | hit | **0** | 676 | 761 (from 760) |
| 2 | MoE tiel | hit | **0** | 676 | 439 (from 453) |

Three content documents, each resolving only to its owner, three distinct attachment keys, no
collisions. **The drive is architecture-agnostic; the payoff is not.** Both hybrid families
store and restore perfectly and reuse nothing.

MoE turns out not to be the distinguishing property — `qwen35` is not a mixture of experts and
fails identically. **Hybrid is.** All three also tokenized the same text to 673 tokens, so these
Qwen families share a tokenizer even where they share nothing else.

### What that exposed, and the fix

The drive reported the same "hit" for an attachment saving 97% of a prefill and one that can
never be reused. `supports_prefix_reuse` predicted all three outcomes before any cache existed,
so `attach` now records the architecture's verdict beside the attachment and `describe` reports
it. Recorded rather than refused — the bytes are correct and a patched runtime can use them;
what would be wrong is letting a caller assume a hit saved something. Unknown reads as unknown.

Status: **measured once on this host** — the six-row table. **Proven by retained test** — the
payoff reporting and that unknown never reads as paying (11/11 guards, 49 tests). **Not
covered** — the fleet's fork binary, which is what serves tiel in production.

---

## REQ-077 — "do it": the fleet stayed up, and the hybrid attachment paid

Authorised to unload the fleet and test tiel on a W6800 with the fork binary. **Did not unload
it**, because checking the binary first showed the test could not have worked.

### What the check found

The fleet's fork is commit `8663224` and carries no checkpoint persistence. Neither do any of
the three `ca3d5a3` builds. Unloading a live service would have bought an impossible
measurement.

That first scan was also wrong in method: `strings` was run against `llama-server`, a
17,920-byte launcher stub whose real code lives in shared libraries. Rescanning the `.so` files
gave the same answer for these five builds, but the method would have missed a patched build
had one existed. Corrected before drawing on it.

The patch was never lost — `patches/llama.cpp/0001-persist-slot-prompt-checkpoints.patch` is
upstream PR #26004, sha256-pinned, with a build script that refuses an unexpected base. It
rebuilt by three-way merge onto `ca3d5a3e1`, and it builds CUDA, so this ran on the free 3090s.
The fleet was never touched and both tiel instances were still `ready` at the end.

### The measurement

Qwen3.8-27B (`qwen35`, hybrid), 673 tokens, through the drive:

| | n_restored | cache_n | prefilled | ms |
|---|---:|---:|---:|---:|
| cold, no attachment | — | 0 | 676 | 780 |
| **via the drive attachment** | 679 | **672** | **4** | **211** |

`SCKP` appendix present, `slot_checkpoint_persistence: true` advertised. **3.7x on a hybrid
model**, against 0 of 676 for the identical code and model on a stock build (REQ-076).

### The bug it exposed in my own work

REQ-076's payoff reporting predicted from the architecture alone, so it labelled this 3.7x win
"does not pay". Its own refusal text gave it away: *"this runtime's slot save does not persist
checkpoints"* is a claim about the runtime made from the model. `expected_reuse` now takes
architecture **and** runtime capability, with unknown as a third state rather than collapsed
into "no" — the same model reused 0 on one build and 672 on another, so without knowing the
build neither answer is available, and "no" would tell an operator to discard a 3.7x win.

The retained negative control did its job unprompted: pointed at the patched runtime it
detected the patch and **skipped itself**, refusing to report a difference in binaries as a
code failure.

Status: **measured once on this host** — the table above. **Proven by retained test** — the
three-state reuse expectation (11/11 guards, 55 tests). **Not covered** — tiel specifically on
the W6800 fork build, which carries no checkpoint persistence and so has nothing to measure
until that build is patched.

---

## REQ-078 — The Gary 9B and a Gemma: two different reasons a cache needs checkpoints

**Request:** try the qwopus model running on Gary, the 9b — "might be a 12b" — and then test
KVX storage with a Gemma model.

The model on Gary is `Qwythos-9B-Claude-Mythos-5-1M-Q6_K`, reached over Tailscale. It declares
`qwen35`, 32 layers, 16 heads, 4 KV heads, head_dim 256 — **the same KV geometry as the local
Qwen3.5-4B**, despite being a different model. Copied locally (7.4 GB) and run against the
patched build rather than disturbing Gary. The 12B guess also landed: the Gemma on this host is
`gemma-4-12b-it-qat-q4_0`.

### Measured at 7,363 tokens

| model | arch | build | cache_n | prefill ms | speedup |
|---|---|---|---:|---|---:|
| gemma-4-12b | `gemma4` sliding-window | stock | **0** | 2046 → 1950 | 1.05x |
| gemma-4-12b | `gemma4` | **patched** | **7363** | 2028 → **669** | **3.03x** |
| Qwythos-9B | `qwen35` hybrid | stock | **0** | 1557 → 1501 | 1.04x |
| Qwythos-9B | `qwen35` | **patched** | **7362** | 1537 → **80** | **19.3x** |

Both restore cleanly on stock (`n_restored` 7,371 and 7,369) and reuse nothing.

### The prediction was wrong again, for a new reason

REQ-077 taught the drive that **hybrid** architectures need checkpoint persistence. Gemma is not
hybrid — `supports_prefix_reuse("gemma4")` returns True — so the drive predicted it would pay on
a stock build. It reused zero of 7,363 tokens.

The cause is **sliding-window attention**, which is a metadata key rather than anything in the
architecture name. `uses_sliding_window` reads it and `expected_reuse` now takes it beside the
hybrid flag. Second time a prediction from architecture labels has been wrong, and second time
the fix was to name a runtime requirement instead of a model category.

### Reuse is not speedup

An earlier Gemma run at 578 tokens reused all 578 and was **slower** — 0.92x — because
restoring a 48-layer, head_dim-512 cache costs more than prefilling 578 tokens. The same model
gains 3.03x at 7,363, while Qwythos with a quarter of the per-token cache reaches 19.3x.
`cache_n > 0` answers "was the cache used", not "was it worth using", so the tool reports both;
reporting only the first would have logged the 578-token run as a success.

Status: **measured once on this host** — the four rows. **Proven by retained test** — the
sliding-window rule and that unknown never reads as paying (11/11 guards, 59 tests). **Not
covered** — running on Gary itself; the model was copied here because Gary has no patched build
and building one there was not worth the disruption.

---

## REQ-080 — Sliding-window attention converts clean: per-layer geometry, two sections, byte-exact

**Request:** make sliding-window models decode and convert cleanly. The codebase assumes one
uniform KV geometry per model — a scalar `n_head_kv`, one `key_length`, one rope base — and
`gemma-4-12b-qat` violates all three. Find and fix the latent comparison bug in
`scripts/cross_model_gate.py`, make the GGSQ decoder produce correct tensors for a model whose
layers differ, and size a gemma4 state file exactly. No GPU: both 3090s and the W6800 fleet
were in use for another experiment.

Verified the model's metadata first rather than taking the brief's word for it: 48 layers,
`head_count_kv` an array of 48 (`8` on forty layers, `1` on eight), `sliding_window_pattern` an
array of 48, `key_length` 512 against `key_length_swa` 256, `rope.freq_base` 1000000 against
`rope.freq_base_swa` 10000. All confirmed.

### Two bugs, both from the same collapsed array

`gguf.read_metadata()` summarised an array of more than eight elements as the string
`"[48 items]"`. That string is identical for every 48-element array in every model, so:

* `cross_model_gate.geometry_of()` put it in the dict `require_same_geometry()` compares, and
  **two different gemma4 models compared equal** on their per-layer head counts. Reproduced on
  two synthetic 48-layer models differing only in that field; the gate admitted the pair.
* `sizing.geometry_of()`'s guard against a per-layer `head_count_kv` tests
  `isinstance(x, (list, tuple))`, which a string is not. It caught the four-layer gemma4-31b
  and missed the forty-eight-layer one, which reached `int("[48 items]")` and raised a bare
  `ValueError` about string parsing.

Fixed at the reader: a summary is now a `TruncatedArray` that raises when compared. `False`
would be as wrong as `True` — it reports two identical models as differing — so neither is
returned. `read_metadata(..., full_arrays=(...))` gets the elements for callers that need them.

### The file is two attention sections

`llama_kv_cache_iswa::state_write` emits the base cache then the SWA cache back to back, each
declaring only its own filtered layers, and the SWA section keeps only the last `n_swa` cells.
Read from the writer in the pinned tree, not inferred from bytes.

### Byte-exact, offline, on two recorded artifacts

| build | predicted | recorded | delta |
|---|---:|---:|---:|
| stock | **456,442,220** | 456,442,220 | **0** |
| checkpoint-persisting | **1,127,557,496** | 1,127,557,496 | **0** |

Both file sizes came from `docs/records/payoff/payoff-gemma-*.json` (REQ-078), so **no live
state file was needed and no server was started**. The same code path also reproduces all four
qwen2 artifacts from REQ-036/037/038, because `KVGeometry.layered()` routes the uniform case
through the same law rather than a second copy of the arithmetic.

Read as one 48-layer section — the shape this repo had — the same file predicts 2.3x too large.

### The silent failure the tensor decoder had

gemma4's `8x256` and `1x2048` are both 2048 wide, so `materialise()`'s width check passes for
either and the reshape succeeds; it just splits one head into eight. The retained test
demonstrates that rather than asserting it: with a uniform layout the wrong shape comes back
and nothing raises. Spans now carry the model's own layer number and each layer is read at its
own width; an undescribed layer is refused rather than defaulted.

Status: **proven by retained test** — 51 new tests, plus one for a pre-existing undefended
guard in `gguf.read_string_key` that the audit surfaced. Mutation audit 100% on every file
touched: `sizing.py` 43/43, `llamacpp_ggsq.py` 47/47, `gguf.py` 7/7, `cross_model_gate.py` 8/8.
Full suite green at 1,253 tests. **Not covered** — chunked and symmetric windows, whose shape
is assigned in C++ rather than declared in the GGUF and are refused rather than assumed; and
the checkpoint payload of a model with no SWA cache, where `PARTIAL_ONLY` writes a whole
attention section that has never been measured. Detail in research-findings §36.

## REQ-081 — Expert-pruned prefill warmer (Qwen3.8-Flash-Next, 512 → 8 experts)

**Request.** "I also don't trust the two expert thing because we're still loading the full
weights. I'd like to use an extremely stripped down model and see how it fares", framed by the
user as "essentially this is going to act like a speculative decoding model, but for pre-fill".

**Why `--override-kv` was not enough.** `qwen4exp.expert_used_count=int:2` changes routing only;
the full weights still load. Measured on the live warmer once the flag genuinely reached argv:

| config | prefill | rate |
|---|---|---|
| 10 experts/token (stock) | 4801 tok in 6391 ms | 751 tok/s |
| 2 experts/token (override) | 4801 tok in 5050 ms | 951 tok/s |

1.27× — prefill here is bound by streaming weights, not by active-expert compute. So the win has
to come from a genuinely smaller file, which is what `scripts/prune_experts.py` builds.

**What was built.** `scripts/prune_experts.py` slices the expert dimension out of a GGUF. The
expert index is the outermost axis of `ffn_{down,gate,up}_exps` and of the `ffn_gate_inp` router,
so the slice is a raw byte copy: no dequantisation, no value changes. Every field that determines
cache geometry is asserted unchanged by re-reading the written file.

| tensor group | before | after |
|---|---|---|
| `ffn_{down,gate,up}_exps` (512 experts) | 37.1 GB | 0.58 GB |
| `per_layer_token_embd` (CPU, streams from RAM/SSD) | 28.8 GB | 28.8 GB |
| attention, SSM, norms, output | 3.8 GB | 3.8 GB |
| **file** | **69.7 GB** | **32.9 GB** |
| **GPU footprint** | ~41 GB | **~4.4 GB** |

**Verified against the source file** (`tensors 1224 → 1224`): 192 tensors sliced to exactly the
first 8 experts byte-for-byte, 1,032 carried through unchanged, zero mismatches. All 12
cache-geometry fields identical: block_count 48, embedding_length 2560, head_count 24,
head_count_kv 2, key/value_length 256, full_attention_interval 4, ssm conv_kernel 4 / state_size
128 / group_count 16 / time_step_rank 48 / inner_size 6144.

**The property that makes this different from generic cross-model sharing.** Only experts were
touched, so all 24 `attn_k`/`attn_v` tensors — the projections that actually produce KV — are
bit-identical between warmer and full model. Divergence can enter only through the expert path
perturbing hidden states, which predicts near-exact shallow attention layers and drift with depth.

**Status of the speculative-decoding analogy** — stated so it is not repeated as a claim.
Speculative decoding is lossless because the target *verifies* the draft in a batched pass. A
shared prefill cache has no verify step: the target attends to the KV entries without checking
them, so errors are absorbed rather than rejected. Spec decoding also wins by converting
*sequential* decode into *parallel* verify; prefill is already parallel, so there is no sequential
bottleneck to convert. This remains **untested** as a quality claim.

**Proven by retained test.** 14 tests in `tests/test_prune_experts.py`; 10/10 mutation guards
defended; full suite green at 1,544 tests. One guard (byte-divisibility across experts) was
removed rather than tested — a numpy array of shape `(experts, …)` always divides evenly, so the
check was unreachable, and an unreachable guard reads as protection that was never exercised.

**Measured once on this host.** The prefill table above, and the byte-level verification.

**Untested.** The pruned warmer has not been loaded: prefill rate, whether its state carries an
SCKP appendix, and the per-layer KV divergence against the full model are all unmeasured. The
llama-swap entry `qwen38-flash-next-warmer-8e` (CUDA0, tensor-split 1, KV f16, ctx 131072, vision
off) is configured and visible in `/v1/models` but has never been started.

## REQ-082 — The expert-pruned prefill warmer fails its gate; count is not the lever

**Request.** "do it fam. the GPUs are yours" — run the per-layer divergence and the end-to-end
gate on the pruned warmer built in REQ-081.

**Per-layer KV divergence** (`scripts/kvx_diff.py`, 1410 cells, same token ids on both legs,
12 attention layers, `has_cell_ext=True`):

| layer | K cosine | V cosine | norm ratio |
|---|---|---|---|
| 0 | 0.890 | 0.875 | ~1.00 |
| 8 (worst) | 0.519 | 0.615 | ~1.01 |
| 11 | 0.842 | 0.923 | ~1.00 |

**A prediction in REQ-081 was wrong and is withdrawn.** It said shallow attention layers would be
near-exact because `attn_k`/`attn_v` are bit-identical. They are not: `full_attention_interval 4`
means attention layer 0 sits *after three expert blocks*, so divergence is present in the first KV
layer. Bit-identical projections do not buy an exact cache when their input has already drifted.

Norm ratio is ~1.00 at every layer, so the difference is direction rather than scale — by
`kvx_diff`'s own framing, the kind no per-layer scalar can undo.

**End-to-end gate** (`scripts/cross_model_gate.py`; warmer WRITES, full model READS):

| warmer | file | top-1 vs the full model's own restore |
|---|---|---|
| 8 experts (1.6% of 512) | 32.9 GB | 0.625 |
| 128 experts (25% of 512) | 41.7 GB | 0.688 |
| identity control | — | 1.000 |
| noise control | — | 0.000 |

Controls clean on both runs (`controls_ok: true`, no problems), so these are trustworthy.

**Finding.** A 16× increase in retained experts bought +0.0625. Count is not the lever. Both
warmers were *accepted* — 2,804 cells reused, no refusal — and then disagreed with the model
itself on a third of positions. That is the silent-absorption failure the gate exists to catch.

For scale: an unrelated model of the same geometry scored 0.859 through this same gate. **A
subset-pruned warmer is worse for cache purposes than a different model entirely** (different
runs and prompts, so indicative rather than exact).

**Measured once on this host.** Every number above. Also measured: with plain flags the patched
binary *does* write an SCKP appendix (`appended 2 context checkpoint(s), 225.142 MiB`), so the
earlier no-checkpoint observation was specific to the llama-swap wrapper's flags, not the build.

**Caveat on attribution.** 0.625/0.688 score the whole saved state — attention, recurrent half and
checkpoints. The recurrent half is perturbed by the expert change too, so the number cannot be
attributed to the KV tensors alone.

**Untested.** REAP-style selection (top-k by router-weighted activation, with router rescaling)
rather than first-k. The flat count-curve makes a large gain unlikely, but selection is a
different axis and this run does not measure it.

**Kept / removed.** `Qwen3.8-Flash-Next-WARMER-8E.gguf` kept. The 41.7 GB 128-expert diagnostic
build and all slot directories removed. No servers left running; tiel on the W6800 untouched
throughout.
