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
| R3. Runtime capability advertisement | 🔴 outstanding | PR #26004 adds no `slot_checkpoint_persistence` field; support must not be inferred from a version string |
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
