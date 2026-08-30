# ggsq/3 + sckp/1 layout inventory

Source-derived byte layout of a llama.cpp slot state file, written to establish whether
canonical KVX tensors can be reconstructed from it without guessing. Every offset below
comes from reading the writer in the pinned tree at `ca3d5a3e`, not from inspecting bytes
and inferring a pattern — a self-consistent round trip is not evidence of a correct layout.

Sources: `src/llama-kv-cache.cpp` (`state_write`, `state_write_meta`, `state_write_data`),
`src/llama-memory-hybrid.cpp` (`state_write`), `src/llama-memory-recurrent.cpp`
(`state_write_data`), and patch 0001's `save_slot_checkpoints`.

All integers little-endian. `llama_pos` and `llama_seq_id` are `int32`.

---

## 1. File envelope

| offset | type | field |
|---|---|---|
| 0 | `char[4]` | magic `qsgg` (`0x67677371` LE) |
| 4 | `uint32` | sequence-state version (3 on the tested build) |
| 8 | `uint32` | `n_token_count` |
| 12 | `int32[n_token_count]` | packed prompt tokens |
| 12 + 4n | — | body begins |

The packed token array is `server_tokens::serialize()`: `LLAMA_TOKEN_NULL` (-1),
`SERVER_TOKENS_STATE_VERSION` (1), `[count][tokens]`, `[count][media_keys]`.

## 2. Body — and the first thing that matters

For a **hybrid** model (`qwen35`, the tested target) `llama_memory_hybrid::state_write`
emits **two sections back to back**:

```
    attention KV-cache section     (skipped when LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
    recurrent state section
```

**There is no delimiter, length prefix, or count between them.** The recurrent section can
only be located by fully parsing the attention section first. A decoder cannot seek.

This also explains the SCKP payloads: checkpoints are written with
`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, so **a checkpoint blob contains only recurrent
state**, while the slot file's main body contains both.

## 3. Attention KV-cache section

```
uint32  n_stream
repeat n_stream:
    uint32  cell_count
    if cell_count == 0: continue        # meta and data are both skipped
    -- meta, repeated cell_count times, in ascending cell-range order --
        int32   pos
        uint32  n_seq_id
        [llama_kv_cell_ext ext]         # only when has_cell_ext()
        int32   seq_id[n_seq_id]
    -- data --
    uint32  v_trans
    uint32  n_layer
    repeat n_layer:                     # all keys first
        int32   k_type                  # ggml type id
        uint64  k_size_row              # ggml_row_size(k_type, n_embd_k_gqa(il))
        bytes   cell_count * k_size_row
    if v_trans == 0:
        repeat n_layer:
            int32   v_type
            uint64  v_size_row          # ggml_row_size(v_type, n_embd_v_gqa(il))
            bytes   cell_count * v_size_row
    else:
        repeat n_layer:
            int32   v_type
            uint32  v_size_el           # ggml_type_size(v_type)
            uint32  n_embd_v_gqa
            repeat n_embd_v_gqa (as j):
                bytes cell_count * v_size_el    # column j across all cells
```

The meta section is **variable length per cell** (`n_seq_id` varies, and `cell_ext` is
present only for some architectures), so the data section's offset is not computable from
`cell_count` alone.

## 4. Recurrent section

**No `n_stream`.** `llama_memory_recurrent::state_write` writes `cell_count` first, unlike
`llama_kv_cache::state_write`. This document previously claimed the two sections shared an
outer shape; the decoder was written to match the document, and both were wrong until a real
qwen35 artifact was decoded. The section is `cell_count`, then meta, then:

```
uint32  s_trans                          # written as 0
uint32  n_layer
repeat n_layer, skipping null layers:    # R tensors
    int32   r_type
    uint64  r_size_row                   # ggml_row_size(r_type, n_embd_r())
    bytes   per cell range
... then the S tensors, with the same shape
```

Null layers are **skipped entirely with no marker**, so the reader must know which layers
carry recurrent tensors. That is model metadata, not file content.

## 5. SCKP checkpoint appendix (patch 0001)

```
uint32  magic 'SCKP' (0x504b4353 LE)
uint32  version (1)
uint32  count (writer rejects > 1024)
repeat count:
    int64   n_tokens                     # int64_t, NOT int32 - this cost us a day
    int32   pos_min
    int32   pos_max
    uint64 + bytes   data_tgt            # target-context partial state
    uint64 + bytes   data_dft            # draft context, empty when absent
    uint64 + bytes   data_spec           # speculative state, empty when absent
```

Measured on the 27B: one checkpoint, `n_tokens=252`, `pos 251..251`, `data_tgt`
156,894,364 bytes, draft and speculative empty — consistent with a target-only launch.

## 6. Identity versus portable data

| field | class |
|---|---|
| packed prompt tokens | **identity** — binds the artifact to an exact prompt |
| `pos`, `n_seq_id`, `seq_id[]` | portable structure, but slot-relative |
| `k_type` / `v_type` (ggml ids) | **portable** — dtype of the payload |
| `k_size_row`, `v_size_row`, `v_size_el` | **portable** — stride, and the only shape evidence |
| `v_trans` | **portable** — changes value layout entirely |
| key/value payload bytes | **portable tensor data** |
| recurrent R/S payloads | portable, but only meaningful with the layer map |
| `n_layer` | portable |

## 7. What the state file does NOT contain

This is the falsifier the steer asked me to test, and it partly fires.

1. **No head / head-dim split.** Keys record only `k_size_row`. The flat per-layer width is
   recoverable — `n_embd_k_gqa = k_size_row * block_size / type_size` — but splitting it into
   `(n_head_kv, head_dim)`, which canonical `layer,kv,token,head,dim` requires, is **not in
   the file**. It needs `n_head_kv` and `n_embd_head_k` from the GGUF.
   *Note:* `n_embd_v_gqa` **is** written explicitly, but only on the `v_trans` path.
2. **No layer index.** Layers are emitted in iteration order with no `il` recorded, and the
   recurrent section skips null layers silently. Mapping payloads to layers needs the model.
3. **No RoPE state.** Whether keys are pre- or post-RoPE is not recorded. The manifest field
   `position.rope_applied` exists in KVX precisely because the source cannot supply it.
4. **No `has_cell_ext()` flag.** Whether each cell carries a `llama_kv_cell_ext` is an
   architecture property. Guessing wrong desynchronises the entire parse.
5. **No section boundary** between attention and recurrent state.

**Assessment.** ggsq/3 carries enough to reconstruct *flat* per-layer K/V tensors with correct
dtype and stride, for a model whose architecture is known. It does **not** carry enough to
produce canonical `layer,kv,token,head,dim` on its own.

The missing values — `n_head_kv`, `n_embd_head_k`, the per-layer recurrent map, and
`has_cell_ext` — are all available from the GGUF of the same model, whose identity the
artifact already binds. Reading them is authoritative metadata, not model-specific inference,
so this is **not** a pivot-to-upstream case. But it must be recorded as a hard dependency:
**canonical extraction requires the source model's GGUF, not just the state file.** A state
file alone is not self-describing, and any future "portable" claim has to say so.

## 8. Not yet representable in the current KVX container

- `v_trans` transposed value layout has no segment role; it needs either a role or
  normalisation at decode time.
- Recurrent R/S tensors have roles (`recurrent`) but no layer-map segment to record which
  layers were skipped.
- The SCKP appendix has no canonical representation at all — it is opaque bytes today, and a
  hybrid model cannot reuse a restored prefix without it.
