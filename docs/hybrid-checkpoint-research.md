# Hybrid checkpoint persistence: source map and negative control

Research record for the persistent-hybrid-checkpoint milestone. Everything here is pinned to
one llama.cpp revision so a later patch can be diffed against a known base.

## Pinned revision

| | |
|---|---|
| llama.cpp source | `ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc` (`~/llama.cpp`) |
| installed server binary | reports `build_info = b1-3e73446` |
| emitted sequence-state version | **2** (the header at this source revision declares `LLAMA_STATE_SEQ_VERSION 3`) |
| upstream issue | https://github.com/ggml-org/llama.cpp/issues/25913 |
| related | https://github.com/ggml-org/llama.cpp/issues/24055 |

The binary predates the checked-out source. That gap is itself part of the cache ABI: a file
written at sequence-state version 2 is refused by a build expecting 3, because
`llama_context::state_seq_load_file` requires an exact match.

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
