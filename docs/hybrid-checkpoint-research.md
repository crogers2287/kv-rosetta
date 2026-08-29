# Hybrid checkpoint persistence: source map and negative control

Research record for the persistent-hybrid-checkpoint milestone. Everything here is pinned to
one llama.cpp revision so a later patch can be diffed against a known base.

## Pinned revision

| | |
|---|---|
| working tree HEAD | `ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc` (`~/llama.cpp`) |
| **source the running binary was built from** | **`3e7344670`** - the binary reports `build_info = b1-3e73446` |
| distance | 150 commits; the checkout was fetched forward without rebuilding |
| `LLAMA_STATE_SEQ_VERSION` at `3e7344670` | **2** |
| `LLAMA_STATE_SEQ_VERSION` at `ca3d5a3` | **3** |
| version observed in files the running server writes | **2** |
| upstream issue | https://github.com/ggml-org/llama.cpp/issues/25913 |
| upstream PR | https://github.com/ggml-org/llama.cpp/pull/26004 |
| related | https://github.com/ggml-org/llama.cpp/issues/24055 |

An earlier note here said only that "the binary predates the checked-out source", inferred
from the version mismatch. The chain is now verified end to end: the binary's `build_info`
names commit `3e7344670`, that commit declares sequence-state version 2, and the files the
running server writes carry version 2. HEAD is 150 commits ahead and declares version 3.

That gap is part of the cache ABI, not trivia: `llama_context::state_seq_load_file` requires
an exact version match, so a file written today is refused by a build from HEAD.

**`~/llama.cpp` is a shallow clone** (151 commits, boundary `3e7344670`). Shallow clones fail
ancestry queries *silently* - `git merge-base` returns nothing and `git log A..B` returns zero
commits rather than erroring - which is how an unverified base can slip past a check. The
build script unshallows before verifying anything.

## The correction this record exists to make

An earlier finding in `research-findings.md` §13 concluded that hybrid and recurrent
architectures cannot reuse a restored prefix, and framed it as a property of recurrent state.
That was too broad. The accurate statement is narrower:

> A recurrent state captured after an **exact** token sequence is deterministic and can be
> restored. What cannot be done is reconstructing it from ordinary KV cells, or slicing it to
> an arbitrary earlier prefix.
>
> **Current llama.cpp slot persistence omits the checkpoint state that hybrid and recurrent
> models require.** That is a missing persistence seam, not an impossibility.

The observed behaviour is unchanged; only the interpretation is corrected.

## Source map

### The structure that must be persisted

`common/common.h:1164` — `struct common_prompt_checkpoint`:

| field | note |
|---|---|
| `int64_t n_tokens` | tokens covered by this checkpoint |
| `int id_task` | optional, id of the creating task |
| `llama_pos pos_min`, `pos_max` | coverage boundary; must not be synthesised |
| `std::vector<uint8_t> data_tgt` | target-context state |
| `std::vector<uint8_t> data_dft` | draft-context state |
| `std::vector<uint8_t> data_spec` | speculative implementation state |

`data_dft` and `data_spec` exist because target state alone is not always sufficient. Any
claim of support with a draft model or speculative decoding has to carry both.

### Where checkpoints live

- `tools/server/server-task.h:569` — `std::list<common_prompt_checkpoint> checkpoints;` on
  `server_prompt`.
- `tools/server/server-task.h:573` — `server_prompt::clear()` calls `checkpoints.clear()`.
- `tools/server/server-task.h:597` — `server_prompt_cache_state` holds a `server_prompt`
  (checkpoints included) beside `server_prompt_data`, and its `size()` sums the checkpoints.
  This is the in-memory path that already keeps tokens and checkpoints together.

### Where the save path drops them

`tools/server/server-context.cpp` around line 2540, the `action=save` handler:

```cpp
packed = slot->prompt.tokens.serialize();
const size_t nwrite = llama_state_seq_save_file(
    ctx_tgt, filepath.c_str(), slot->id,
    reinterpret_cast<const llama_token *>(packed.data()),
    packed.size() / sizeof(llama_token));
```

Only `prompt.tokens` is serialised. `prompt.checkpoints` is never written.

### Where the restore path discards them

The `action=restore` handler deserialises the packed prompt, validates it, then:

```cpp
slot->prompt.clear();              // clears checkpoints
slot->prompt.tokens = std::move(restored);
```

So even a checkpoint that survived in memory is cleared, and nothing rebuilds it.

### Where the absence is detected

`tools/server/server-context.cpp:3338-3356`. For hybrid or recurrent memory the server does
not reuse a prefix from KV cells; it searches `slot.prompt.checkpoints` for one covering the
resume position. Finding none:

```cpp
SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data "
              "(likely due to SWA or hybrid/recurrent memory, see %s)\n", ...);
pos_next = 0;
n_past   = 0;
```

## Observed behaviour on the pinned runtime

Qwen3.5/3.6 27B (`general.architecture = qwen35`), 256-token exact prefix, one slot:

| step | result |
|---|---|
| cold prefill | `cache_n=0 prompt_n=256` |
| save slot | `n_saved=256`, `n_written=173,678,144` |
| erase | `n_erased=256` |
| control rerun, no restore | `cache_n=0` — the cache really was dropped |
| restore | `n_restored=256`, `n_read=173,678,144` |
| slot state after restore | `n_prompt_tokens=256` |
| next completion | **`cache_n=0 prompt_n=256`** |
| server trace | `forcing full prompt re-processing due to lack of cache data` |

With `-lv 10` the same run shows `created context checkpoint 1 of 32 (n_tokens = 252,
size = 149.626 MiB)` before the save, and `main/do_checkpoint = no, pos_min = -1, pos_max = -1`
on the post-restore request.

The checkpoint was about 149.6 MiB at both ~252 and ~2,800 tokens, which suggests it is
largely fixed-size per checkpoint rather than growing per token. That is one observation at
two points, not a measured relationship.

## Negative control

`tests/test_hybrid_negative_control.py` retains this failure as a runnable test. It must keep
passing while the runtime is unpatched, and it is the control the persistence work is measured
against. Run it with:

```bash
KVX_HYBRID_URL=http://127.0.0.1:8785 KVX_HYBRID_SLOTS=/path/to/slots \
  python3 -m unittest tests.test_hybrid_negative_control -v
```

## Status of each claim here

- **Confirmed by upstream source:** the struct, the save path, the restore path clearing
  checkpoints, and the reset condition — all read at `ca3d5a3`.
- **Measured on this host:** the 27B save/erase/restore/no-reuse sequence and the trace lines.
- **Inferred:** that persisting one exact-boundary checkpoint is sufficient for the
  fixed-prefix workload. Not yet proven.
- **Untested:** anything involving a patched binary, `data_dft`, or `data_spec`.

## R1: choice of persistence path

Audited before writing anything, as the steer requires. Three implementations exist, not the
two the steer listed.

### What is actually out there

| | approach | status |
|---|---|---|
| Issue #25913 option 1 | separate `filepath + ".ckpt"` sidecar file | proposal |
| Issue #25913 option 2 | serialize a whole `server_prompt_cache_state` | proposal |
| **PR #26004** (`Tough-Respawn`) | tagged payload **appended inside** the slot save file: `SCKP` magic, version, count, per-checkpoint fields and blobs, read back at the offset `llama_state_seq_load_file` returns | **open**, not draft, 1 commit, +195/-2 across 2 files, base `adb55e5`, updated 2026-08-16 |
| `headbouyJB`'s patch | the option-1 sidecar, `.ckpt` file | not a PR; reported working in issue comments |

### An attribution that must not be blurred

The working confirmations in the issue thread are for the **sidecar**, not for PR #26004.
`WinPooh32` quotes `restored 3 context checkpoint(s) from sidecar .../qwen3.6-27b.bin.ckpt` -
a `.ckpt` file, which is `headbouyJB`'s implementation. `headbouyJB` reports it running on
Qwen3.5 122B A10B with MTP, and describes it as an AI-led fix.

PR #26004 has no third-party confirmation visible in the thread. So "someone proved this
works on qwen3.6-27b" is true of the sidecar and **not** of the PR we are about to adopt.
Our own restart test decides it, not either author's report.

### Decision: base the work on PR #26004

Against the steer's decision rule:

| criterion | PR #26004 |
|---|---|
| versioned | yes - explicit `SLOT_CKPT_VERSION` beside a magic |
| bounded | yes - buffer reads reject `n > 1<<34` before allocating |
| validated before allocation | yes, same guard |
| restored into one exact slot | yes - reattached to the slot being restored |
| backward compatible | yes, in both directions: an old file has no magic and is skipped silently; a new file's trailing bytes are ignored by an old build, because `llama_state_seq_load_file` reports where its own payload ended |
| testable without a large refactor | yes - it ships `tools/server/tests/unit/test_slot_save.py` |

It is also smaller than either issue option, and being one file rather than two removes a
failure mode the steer explicitly worried about: a sequence file paired with the wrong
checkpoint sidecar. There is nothing to pair. That also simplifies the KVX side - the
artifact stays a single opaque blob rather than two segments needing atomic co-publication,
though the representation label must still change, since these bytes are no longer plain
`ggsq/2`.

It carries `data_tgt`, `data_dft` **and** `data_spec`, so it does not have the draft and
speculative gap the issue notes ("`ctx_dft` state is also not saved").

### What PR #26004 does NOT do, and remains our work

- **It persists whatever checkpoints already exist** - `if (slot.prompt.checkpoints.empty())
  return` - and does not force one at the stable-prefix boundary. The issue is explicit that
  reuse after a fix is bounded by `--ctx-checkpoints` and `--checkpoint-min-step`, so it
  "should match in-memory behaviour, not necessarily produce a full-prefix hit". Steer R2
  (an exact-boundary checkpoint) is therefore additional work, not something adopting the PR
  gives us.
- **No capability advertisement.** Steer R3 requires the binary to report something like
  `slot_checkpoint_persistence = true` so support is never inferred from a version string.
  The PR adds no such field.
- **It does not apply cleanly to our pinned revision.** PR base is `adb55e5`; we are pinned
  at `ca3d5a3`. It needs a deliberate rebase, and the guarded build script must refuse an
  unexpected upstream revision rather than applying fuzzily.

### The caveat that governs R2

From the issue, and it is a correctness trap rather than a performance note: synthesising a
checkpoint at `pos_min = 0` in the restore handler is **not** a valid shortcut. The matcher
takes the `cur.pos_min == 0` branch and decodes on top of a recurrent state that has already
consumed that token - silently corrupt output rather than merely slow. Any boundary-checkpoint
work must create a *real* checkpoint, never relabel one.

### Status of these claims

- **Confirmed by upstream source and API:** the existence, state, size, base SHA and file
  list of PR #26004; its magic, version, bounds check and the three blob fields; that it
  returns early on an empty checkpoint list.
- **Reported by third parties, for a different implementation:** the sidecar working on
  qwen3.6-27b and on Qwen3.5 122B with MTP.
- **Untested here:** everything. No patched binary has been built or run on this host.

## R3/R6: the patched runtime, built and proven across a restart

### Build

`scripts/build_patched_llama.sh /mnt/storage/llama-kvx-patched` on this host:

- patch sha256 verified, upstream pinned at `ca3d5a3e1`, applied by three-way merge;
- one real failure on the way: the build died with
  `ccache: failed to create temporary file ... Permission denied`, because
  `~/.cache/ccache/tmp` is owned by `root:root` from an earlier privileged build. The script
  now sets `-DGGML_CCACHE=OFF` - a reproducible build must not depend on the ownership of a
  shared cache directory it does not control;
- the resulting `llama-server` is only ~18 KB, because it is a thin launcher. The server
  implementation is in `libllama-server-impl.so`, which carries the `SCKP` magic twice and
  13 `context checkpoint` strings. An earlier probe that searched the executable found
  nothing and would have reported a successful build as unpatched; the script now probes the
  library and refuses if the magic is absent.

### In-process reuse

| step | unpatched | patched |
|---|---|---|
| cold | `cache_n=0 prompt_n=256` | `cache_n=0 prompt_n=256` |
| save | 173,678,144 bytes | **330,573,584 bytes** |
| control after erase | `cache_n=0` | `cache_n=0` |
| restore | `n_restored=256` | `n_restored=256` |
| next completion | `cache_n=0 prompt_n=256` | **`cache_n=252 prompt_n=4`** |

The 157 MB difference is the checkpoint: the server logs
`restored context checkpoint (pos_min = 251, pos_max = 251, n_tokens = 252, size = 149.626 MiB)`.

### The restart proof

The step that separates persistence from an in-memory cache experiment:

```
before restart  cold cache_n=0 | saved 263 cells, 316 MB
STOP SERVER     confirmed down
FRESH SERVER    no restore -> cache_n=0        <- a new process has nothing
restore         n_restored=263
AFTER RESTART   cache_n=252  prompt_n=4
parity          output matches the pre-restart cold run; token IDs identical
```

Retained as `tests/test_hybrid_checkpoint_restart.py`, alongside the negative control.

### The negative control did its job

Run against the patched binary it now **fails**, with the message written for exactly this
moment: *"if this is nonzero the runtime has been patched; update the capability rule and
retire this control deliberately."* A control that never fails proves nothing; this one
flipped when the world changed.

### What this does and does not establish

**Proven by retained test on this host:** a hybrid `qwen35` model reuses a restored prefix
after a full server restart, with output and token IDs matching a cold prefill.

**Not established:**

- **Full-prefix reuse.** `cache_n=252` of 256, because reuse is bounded by checkpoint
  granularity (`--ctx-checkpoints`, `--checkpoint-min-step`) exactly as upstream #25913
  predicts. Steer R2 - forcing a checkpoint at the exact stable-prefix boundary - is still
  open, and the four re-prefilled tokens are the visible symptom of it.
- **Economics.** One run, not three, and no ladder. Nothing here is a measurement of
  restore-versus-prefill cost.
- **The production model.** This was OpenMythos-Q6_K, a `qwen35` hybrid, not the specific
  27B the steer names.
- **Capability advertisement.** The patched binary reports nothing machine-readable, so the
  adapter deliberately still withholds the capability. A test asserts that it does: a
  patched binary that cannot be detected is still unsafe to assume.

## R2: the uncovered tail is constant, so there is no boundary to force

The steer's R2 assumed reuse would be bounded by checkpoint granularity, and preferred
forcing a real checkpoint at the exact stable-prefix boundary before serializing. Measured
first, on the patched runtime with a hybrid `qwen35` model:

| `--checkpoint-min-step` | `--ctx-checkpoints` | prompt | `cache_n` | `prompt_n` | uncovered |
|---|---|---|---|---|---|
| 128 | 8 | 256 | 252 | 4 | **4** |
| 128 | 8 | 1024 | 1020 | 4 | **4** |
| 128 | 8 | 4096 | 4092 | 4 | **4** |
| 8 | 32 | 256 | 252 | 4 | **4** |
| 8 | 32 | 1024 | 1020 | 4 | **4** |
| 8 | 32 | 4096 | 4092 | 4 | **4** |

The tail is **constant at 4 tokens** across a 16x range of prompt lengths and across a 16x
change in checkpoint density. It is therefore not a granularity artifact, and forcing a
checkpoint at the save boundary would not recover those four tokens.

That is the hybrid analogue of behaviour already measured on ordinary attention, where
exactly **1** token is always reprocessed. Both runtimes reuse everything except a short
fixed tail; only the size of the tail differs by architecture.

### Consequence for the adapter, and the C++ patch not written

The reuse invariant was `cache_n == L-1 and prompt_n == 1`. Correct for ordinary attention,
and it would reject every correct hybrid restore. The invariant is now:

    uncovered = L - cache_n
    1 <= uncovered <= max_uncovered_tail   and   prompt_n == uncovered

`prompt_n == uncovered` is what keeps this from degrading: it requires the runtime to have
reprocessed *exactly* the uncovered tokens and nothing more. `max_uncovered_tail` defaults
to 8 - large enough for the measured 1 and 4, small enough that "reuse" cannot quietly
become "reprocessed most of the prompt". A test shrinks the bound to 0 and confirms a good
restore is then refused, so the bound is enforced rather than decorative.

No llama.cpp patch was written for R2. The measurement showed the preferred option would not
help, and the steer's alternative - accept declared coverage and account for the tail - is
both smaller and sufficient. Recorded here because "we chose not to patch" is only defensible
with the numbers that made it the right choice.

**Caveat:** 4 is what this model and build do. The adapter measures the tail per restore and
bounds it; it does not hardcode 4 anywhere.
