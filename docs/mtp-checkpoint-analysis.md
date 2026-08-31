# MTP checkpoint state: what it is, what the patched runtime does with it, and what would make restoring it correct

Read-only source analysis. No server was started, no GPU used, no source file modified.

All line citations are against the working tree of `/home/crogers2287/tiel-kvx-patched` at base commit
`866322481` with the two KV Rosetta patches applied and one unstaged local edit to
`common/speculative.cpp` (see §1.1). Paths below are relative to that tree unless they start with
`/home/crogers2287/kv-rosetta`.

Every claim about behaviour carries a `file:line`. Where a sentence reasons past what the source
literally says, it says **inference** in that sentence.

---

## 0. The question, and the short answers

The live server reports:

```
active_checkpoint_state_classes       = ['target', 'speculative']
sckp_serializes_draft_state           = true
sckp_serializes_speculative_state     = true
supports_draft_checkpoint_state       = false
supports_speculative_checkpoint_state = false
slot_checkpoint_format                = sckp/1
```

and `kv_rosetta/hybrid_schema.py:132` refuses any checkpoint that declares draft or speculative
state.

| question | short answer | settled by source? |
|---|---|---|
| 1. What extra data does a checkpoint hold under MTP? | `data_dft` — a **full** KV dump of the draft context's nextn layer(s) for the slot's sequence. `data_spec` is **empty**: MTP does not implement `get_state`. | **yes** |
| 2. Is `supports_*=false` conservative, or a known gap? | Both, and they are separable. The flags are hardcoded literals — no probe. Separately there are two real, source-visible gaps (§3.2, §3.3). Neither is "the restored bytes are wrong". | **yes** for the flags; **yes** for the gaps' existence; **no** for their consequence |
| 3. Is the MTP draft head's state reconstructible from accepted tokens? | Its only cross-call scalar state, `pending_h`, is **rebuilt automatically** by the first target decode after restore. The draft context's **KV cache for the restored prefix is not**, and nothing on the restore path rebuilds it. | **yes** for `pending_h`; **yes** that the draft KV is not restored |
| 4. What proves or refutes correct restoration? | A 2x2 at temperature 0 comparing emitted text, plus a `/metrics` acceptance-rate delta. Text is the correctness observable; acceptance rate is the value observable. | design settled; **outcome must be measured** |

The headline for §3: **the thing that would make restoring MTP draft state worth doing is not
correctness, it is acceptance rate.** The source gives no path by which a stale draft context
changes the emitted token at temperature 0, because every emitted token is sampled from the
target's own logits (`common/sampling.cpp`, `common_sampler_sample_and_accept_n`). What a stale
draft context costs is drafts that get rejected.

---

## 1. What is in the file

### 1.1 The checkpoint record

`common_prompt_checkpoint` (`common/common.h:1164`) holds three independent byte blobs:

| field | line | written by | contents |
|---|---|---|---|
| `data_tgt` | `common/common.h:1173` | `update_tgt`, `common/common.cpp:2283` | target context sequence state |
| `data_dft` | `common/common.h:1174` | `update_dft`, `common/common.cpp:2301` | draft context sequence state |
| `data_spec` | `common/common.h:1178` | *no method on the struct* | speculative-implementation state, supplied by the caller |

`data_spec` has no `update_spec`/`load_spec`. It is filled by the server directly
(`tools/server/server-context.cpp:2339`) and consumed directly
(`tools/server/server-context.cpp:3483`). `clear_dft()` clears `data_spec` alongside `data_dft`
(`common/common.cpp:2359`), which is the only place the two are treated as one thing.

### 1.2 What the SCKP appendix serializes

`save_slot_checkpoints` (`tools/server/server-context.cpp:2383`) appends, after the llama state
payload, a `"SCKP"` header (magic `0x504b4353`, version 1, count) and then per checkpoint:
`n_tokens`, `pos_min`, `pos_max`, then length-prefixed `data_tgt`, `data_dft`, `data_spec`
(`:2402`–`:2405`). `load_slot_checkpoints` (`:2418`) reads them back in the same order
(`:2441`–`:2444`) and installs the list wholesale into `slot.prompt.checkpoints` (`:2453`).

The appendix carries **no identity binding**: no model hash, no cache-type record, no `n_embd`, no
speculative-type tag. `load_slot_checkpoints` validates magic, version, `count <= 1024`, and that
the reads did not run off the end (`:2432`–`:2447`). Nothing checks that `data_dft` was produced by
a compatible draft context.

### 1.3 What changes between `TIEL_SPEC=0` and `TIEL_SPEC=1 TIEL_SPEC_TYPE=draft-mtp`

Three things change; only one of them is the "speculative state" the flags are about.

**(a) `data_dft` becomes non-empty.** `create_checkpoint` unconditionally calls
`cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)`
(`tools/server/server-context.cpp:2337`). `update_dft` returns immediately when `ctx == nullptr`
(`common/common.cpp:2302`–`:2304`). With speculation off, `ctx_dft` is nulled at
`tools/server/server-context.cpp:1278`, so `data_dft` stays empty. With `draft-mtp` on, `ctx_dft`
is a real context (§2.1) and `data_dft` is populated.

How much: for a hybrid Qwen3.5-family target, the MTP draft context's memory is a **plain
`llama_kv_cache` filtered to the nextn layer(s)** — `mtp_on_hybrid_qwen` at
`src/llama-model.cpp:2429`–`:2432`, filter `il >= hparams.n_layer()` at `src/llama-model.cpp:2560`.
And `llama_kv_cache::state_write` **ignores the flags argument** — `GGML_UNUSED(flags)` at
`src/llama-kv-cache.cpp:2100`. So despite `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` being passed,
`data_dft` is the *complete* K and V for the nextn layer(s) over every cell of that sequence, not a
partial slice. That is the opposite of `data_tgt`, where `PARTIAL_ONLY` really does restrict the
write to the recurrent/SWA part (`src/llama-memory-hybrid.cpp:190`–`:195`).

**(b) `data_spec` stays empty for MTP.** `common_speculative_get_state` walks the implementation
list and returns the first implementation that produces state (`common/speculative.cpp:2961`–
`:2973`). The base class returns `false` (`common/speculative.cpp:175`). Exactly **one**
implementation overrides it: `common_speculative_impl_draft_eagle3`, whose `get_state` is at
`common/speculative.cpp:872` — inside the struct that spans `:426` to `:908` (the next struct,
`..._draft_dflash`, starts at `:910`). `common_speculative_impl_draft_mtp` spans `:1373` to `:1812`
and contains no `get_state`/`set_state` override.

Even for eagle3 the state is conditional: `need_boundary_stash()` returns true only for recurrent
or hybrid *targets* (`common/speculative.cpp:867`–`:870`), and the payload is one `llama_pos` plus
one `n_embd_dec`-wide float row (`:882`–`:886`).

So on this build, running MTP: **`data_spec.size() == 0` in every checkpoint.** The
`active_checkpoint_state_classes` entry `"speculative"` is derived from the launch flags, not from
the file — it is pushed whenever any speculative type is non-`NONE`
(`tools/server/server-context.cpp:4720`–`:4728`).

**(c) The target's recurrent memory is widened, but its serialized form is not.**
`cparams.n_rs_seq = params.speculative.need_n_rs_seq()` (`common/common.cpp:1724`), which is
`draft.n_max` whenever an MTP/EAGLE3/DFlash/DSpark type is enabled (`common/common.h:394`–`:400`).
`llama_memory_recurrent` then allocates `mem_size * (1 + n_rs_seq)` rows
(`src/llama-memory-recurrent.cpp:101`).

This does **not** change the on-disk size of `data_tgt`: `state_write` selects the sequence's
current rollback group and emits only that group's rows
(`src/llama-memory-recurrent.cpp:783`–`:807`), and `state_read` resets the index to 0 afterwards
(`src/llama-memory-recurrent.cpp:873`–`:875`). It *does* change `ctx_tgt_seq_rm_type` from
`COMMON_CONTEXT_SEQ_RM_TYPE_FULL` to `..._RS` (`common/common.cpp`, `common_context_can_seq_rm`,
the `llama_n_rs_seq(ctx) > 0` branch), which changes which rollback paths the server takes at
`tools/server/server-context.cpp:3183`–`:3187` and `:4038`–`:4039`. Checkpoint *creation* is
unaffected — `do_checkpoint` accepts both `FULL` and `RS`
(`tools/server/server-context.cpp:3581`–`:3584`).

### 1.4 The inversion worth naming

For an MTP launch with no separate draft model:

- the file **does** contain draft state (`data_dft`, sized by the nextn-layer KV), but `/props`
  does **not** list `"draft"` as active — because `has_dft()` tests only whether a separate draft
  **model file** was configured (`common/common.h:387`–`:389`), and the tiel launcher deliberately
  omits `--spec-draft-model` for `draft-mtp` (`/home/crogers2287/llama-swap/run-tiel-w6800.sh:64`);
- the file contains **no** speculative state (`data_spec` empty), but `/props` **does** list
  `"speculative"` as active.

Both advertised labels are therefore inverted relative to the bytes for this exact configuration.
The labels are honest about the launch; they are not a description of the artifact. Any consumer
that treats `active_checkpoint_state_classes` as an inventory of the file will be wrong in both
directions here.

---

## 2. The MTP draft context, and what the save/restore endpoints touch

### 2.1 The MTP draft context is a second context over the same model

With `--spec-type draft-mtp` and no draft model, `common_speculative_init_result` creates a context
from the **target** model with `ctx_type = LLAMA_CONTEXT_TYPE_MTP`
(`common/speculative.cpp:2609`–`:2618`, with `cparams.ctx_type` set at `:2578`). `cparams.ctx_other`
is set to `ctx_tgt` by the caller (`common/speculative.cpp:2587`) but is then **discarded** unless
the architecture is `GEMMA4_ASSISTANT` or a headless EAGLE3/DFlash —
`cparams.ctx_other = nullptr` at `src/llama-context.cpp:142`, reassigned only in the branches at
`:144`–`:160`.

Consequence: for a Qwen3.5-family MTP target, `llama_get_ctx_other(ctx_dft)` is null, so
`is_mem_shared` is false (`common/speculative.cpp:1466`). The draft context has its **own** KV
cache, independent of the target's.

### 2.2 The draft context is filled only by decoding

`common_speculative_process` runs after every target decode
(`tools/server/server-context.cpp:3858`, and again for mtmd chunks at `:756`). MTP's `process`
(`common/speculative.cpp:1527`) mirrors the target batch into `ctx_dft` with a catch-up
`llama_decode` (`:1562`–`:1621`) and then refreshes `verify_h` and `pending_h` from the target's
`h_nextn` rows (`:1624`–`:1640`).

There is no other writer of the draft KV cache. It is a pure function of the decodes that actually
ran in this process.

### 2.3 What `POST /slots/N?action=save` writes

`SERVER_TASK_TYPE_SLOT_SAVE` (`tools/server/server-context.cpp:2629`) calls
`llama_state_seq_save_file(ctx_tgt, ...)` at `:2658` — **target only** — then appends the checkpoint
list at `:2666`. The live draft context's state is never written.

### 2.4 What `POST /slots/N?action=restore` reads

`SERVER_TASK_TYPE_SLOT_RESTORE` (`tools/server/server-context.cpp:2692`) calls
`llama_state_seq_load_file(ctx_tgt, ...)` at `:2716`/`:2718` — **target only** — replaces
`slot->prompt`, then calls `load_slot_checkpoints` at `:2745`.

On the success path, `ctx_dft` is never mentioned. `slot->prompt.clear()` at `:2736` clears tokens
and the checkpoint list but touches no memory; only the *error* path calls `slot->prompt_clear()`
(`:2739`), and only that helper calls `mem.seq_rm(id, -1, -1)`
(`tools/server/server-context.cpp:337`), which does span both contexts (`common/common.h:998`–
`:1005`).

**So after a successful restore, the draft context holds whatever the slot's previous occupant left
in it, at positions that now belong to a different conversation.** That is the concrete state
mismatch. It is not hypothetical and it is not a flag: it is the absence of any write to `ctx_dft`
on `tools/server/server-context.cpp:2692`–`:2770`.

### 2.5 Where `data_dft` and `data_spec` are actually applied

Only on the mid-prompt checkpoint rollback, when a *new* request diverges from the cached prefix:
`it->load_tgt(...)`, `it->load_dft(ctx_dft, ...)`, `common_speculative_set_state(spec.get(), slot.id, it->data_spec)`
at `tools/server/server-context.cpp:3481`–`:3483`.

They are **not** applied by `action=restore`. A restore that is followed by a request which is a
strict extension of the restored prefix never touches them at all.

This is the shape of the finding: `data_dft` is exactly the missing draft-side KV that §2.4
describes, it is already in the file, and the plain restore path simply does not install it.
(**Inference**, from the two facts above: installing it would be the natural fix, subject to §3.3.)

---

## 3. Is `supports_speculative_checkpoint_state=false` conservative, or a known gap?

Three separable answers.

### 3.1 The flags themselves are conservative defaults, with no probe behind them

`supports_draft_checkpoint_state` and `supports_speculative_checkpoint_state` are JSON literals
`false` at `tools/server/server-context.cpp:4783`–`:4784`, inside `get_res_props`. Nothing computes
them; the surrounding comment says exactly this ("Draft and speculative restoration have never been
exercised here"). The same block hardcodes `sckp_serializes_*_state` to `true`
(`:4775`–`:4777`) as statements about the *format*.

Corroborating: the retained tests exercise `update_tgt`/`load_tgt` only —
`tests/test-recurrent-state-rollback.cpp:290`–`:292` and `:334`–`:336` use `update_tgt`/`load_tgt`
exclusively; the patch's own regression test
(`patches/llama.cpp/0001-persist-slot-prompt-checkpoints.patch`, `test_slot_restore_preserves_context_checkpoints`)
runs a `tinygemma3` preset with **no** speculative configuration, and
`tools/server/tests/unit/test_slot_save.py` contains no reference to `spec`, `draft`, or `mtp`.

So: nobody proved it. That is the literal answer to "conservative default".

### 3.2 There is nevertheless a real gap, and it is not in the bytes

The gap is §2.4: `action=restore` restores the target and leaves the draft context stale. This is a
gap in **restoration coverage**, not in the correctness of any restored blob. The MTP driver's own
`begin()` anticipates precisely this situation and classifies it as degradation, not error:

```
ctx_dft pos_max=%d < N-1=%d - process() hook may not have run on every prefill ubatch
(need_embd / logits=1 on every prompt position?). Drafts may degrade.
```

`common/speculative.cpp:1519`–`:1523`. It is `SPC_WRN`, and control falls through.

### 3.3 There is a second gap, and this one can kill the process

`common_prompt_checkpoint::load_dft` calls `llama_state_seq_set_data_ext` and **aborts** if the
return value does not equal the blob size — `GGML_ABORT("checkpoint size mismatch: ...")` at
`common/common.cpp:2349`–`:2351`. And `llama_context::state_seq_set_data` returns `0` on *any*
exception, including a magic mismatch (`src/llama-context.cpp`, the `catch` at the end of
`state_seq_set_data`, `return 0`).

So a checkpoint whose `data_dft` does not match the live draft context — different build, different
`--spec-draft-*` cache types, different nextn layer count, a file saved with MTP and restored on a
launch whose `ctx_dft` differs — reaches `GGML_ABORT` and takes the server down. `load_slot_checkpoints`
performs no compatibility check that would prevent this (§1.2).

This is only reachable via the divergence path (`tools/server/server-context.cpp:3481`), because
that is the only caller of `load_dft` for prompt checkpoints. It is not reachable by
`action=restore` alone. But it *is* reachable by "restore, then send a request that diverges", which
is an ordinary drive workload.

### 3.4 Stated plainly

- The flags: **conservative, unproven, no probe.** Settled from source.
- Gap A (§2.4, stale draft KV after restore): **real, and visible in the source as an omission.**
  Its *consequence* is argued in §4 to be acceptance rate rather than correctness; that argument is
  reasoning about the acceptance rule, not a line the source states.
- Gap B (§3.3, `GGML_ABORT` on a mismatched `data_dft`): **real and severe**, and it is an
  availability hazard rather than a correctness one. It applies to any cross-build or
  cross-configuration checkpoint file, which is exactly kv_rosetta's use case.

The source does **not** settle whether restoring `data_dft` on the plain restore path would produce
a draft context that is *equivalent* to one built by decoding. Nobody has run it.

---

## 4. Question 3, the crux: is the MTP draft head's state a pure function of accepted tokens?

Split it into the two things the driver carries.

### 4.1 `pending_h` — reconstructible, and in fact reconstructed automatically

`pending_h[seq]` is one `n_embd`-wide float row per sequence
(`common/speculative.cpp:1396`), initialized to zeros in the constructor (`:1479`). It is the
target's `h_nextn` row at the last position the driver has seen, and it is the only value carried
from one `process()`/`draft()` call to the next.

Its writers:

- `process()` sets it from the target's `h_nextn` for the last row of the batch just decoded —
  `common/speculative.cpp:1638`, reading `llama_get_embeddings_nextn_ith(ctx_tgt, ...)` at `:1634`;
- `accept()` sets it from `verify_h` at the accepted index — `common/speculative.cpp:1808`.

Both derive it from the target's own forward pass over accepted tokens. It is therefore a pure
function of the target's accepted prefix — nothing in `common_speculative_impl_draft_mtp` mixes in
draft-side history, sampler RNG, or previous-turn residue.

And it is refreshed before it can be read, on every path including restore:

1. `common_speculative_process` runs after **every** target decode
   (`tools/server/server-context.cpp:3858`), setting `pending_h` (`common/speculative.cpp:1638`).
2. The server guarantees at least one token is decoded even when the cache covers the whole prompt
   — `[TAG_PROMPT_LOGITS]`, `tools/server/server-context.cpp:3513`–`:3517`, which decrements
   `n_past` when `n_past == slot.task->n_tokens()`.
3. Only then does the slot enter `SLOT_STATE_GENERATING` and call `common_speculative_begin`
   (`tools/server/server-context.cpp:3950`–`:3951`), and only on a subsequent iteration does
   `common_speculative_draft` read `pending_h` (`common/speculative.cpp:1668`).

So `pending_h` at the first draft after a restore is the target's `h_nextn` at the last prompt
position, computed from the restored target cache — the same quantity a cold prefill would produce,
up to the numerical fidelity of the restored target cache itself. **Restoring it would be
redundant, not merely unproven.**

The rest of the driver's fields are within-call scratch: `verify_h`/`verify_h_rows` are rewritten
each `process()` (`common/speculative.cpp:1628`–`:1637`); `i_last` and `chain_h` are set inside
`draft()` (`:1671`, `:1674`); the samplers are reset per draft
(`common_sampler_reset`, `common/speculative.cpp:1665`). None survives a slot boundary in a way that
carries information.

### 4.2 The draft context's KV cache — **not** reconstructible from the restored target cache

This is the part that does not follow from accepted tokens alone, because "the accepted tokens" is
not what the restore hands you — it hands you a target KV cache and a token list, with no decodes.

`draft()` decodes into `ctx_dft` at position `dp.n_past` and upward
(`common/speculative.cpp:1667`, `:1696`, `:1766`). The nextn layer attends over the draft context's
cells below `n_past`. After a restore those cells were written by whatever previously occupied the
slot (§2.4). They are not absent — they are *wrong*, which is worse for reasoning about it, though
`llama_memory_seq_rm(mem_dft, seq_id, ckpt.pos_max + 1, -1)`
(`tools/server/server-context.cpp:3176`) does clear anything above the current checkpoint's
`pos_max` before each drafting round.

Note also that the first post-restore `process()` writes a draft KV cell using the **stale**
`pending_h` as its input embedding — `set_h(i_batch_beg[seq_id], pending_h[seq_id].data())` at
`common/speculative.cpp:1589`, before `pending_h` is refreshed at `:1638`. So the boundary cell in
the draft cache is polluted too.

Rebuilding that cache from the token list would mean re-decoding the prefix through `ctx_dft` —
i.e. paying most of the prefill the restore was meant to avoid. The alternative is installing
`data_dft`, which is already in the file (§2.5).

### 4.3 Why the honest verdict is "unnecessary for correctness, valuable for speed"

Every emitted token comes from `common_sampler_sample(gsmpl, ctx, idxs[i], ...)` where `ctx` is the
**target** context — `common/sampling.cpp`, inside `common_sampler_sample_and_accept_n`. A draft
token is used only as an equality test: `if (draft[i] != id) break;`. The loop then samples one more
token from the target and stops. The draft never contributes a token that the target did not itself
sample.

**Inference**, from that acceptance rule: a stale draft context changes *which* draft tokens are
proposed, therefore how many of them match, therefore how many target positions are consumed per
verification step. At temperature 0 the target's argmax at a given prefix is fixed, so the emitted
string is invariant to the draft's quality. Above temperature 0 the emitted string is still drawn
from the target's distribution position by position, but a different acceptance pattern consumes the
sampler's RNG stream differently, so the *particular* string will differ — for reasons unrelated to
cache fidelity. This is why §5 insists on temperature 0.

The one place this argument would not hold is a shared-memory draft context, where the draft writes
into the target's own KV. That is `is_mem_shared`, which is Gemma4-Assistant only
(`src/llama-context.cpp:144`–`:151`, `common/speculative.cpp:1466`) and is **false** for the
qwen35-family MTP configuration under discussion. The synthetic-acceptance replay path
(`server_sample_and_accept_synth`, `tools/server/server-context.cpp:4026`–`:4030`) is a second
exception; the tiel launcher configures no synth options
(`/home/crogers2287/llama-swap/run-tiel-w6800.sh` contains no `synth`), so it is inactive here.

---

## 5. What would prove or refute correct restoration

### 5.1 The claim under test

> With MTP enabled, restoring a slot from an SCKP file produces the same generated text as decoding
> the same prompt cold, and differs only in draft acceptance rate.

Refuting it means finding text divergence attributable to the speculative path. Proving it means
text identity plus a measured, explainable acceptance-rate gap.

### 5.2 Design: a 2x2, because restore fidelity and speculation must be separated

A hybrid target's restore is not guaranteed bit-exact on its own (this repo has prior records of
backend-dependent nondeterminism). So spec-on-vs-cold alone cannot attribute a divergence. Run four
cells on the same build and model, temperature 0, fixed seed, single slot, `--parallel 1`:

| cell | launch | prompt path |
|---|---|---|
| A | `TIEL_SPEC=0` | cold prefill of P, generate K |
| B | `TIEL_SPEC=0` | restore file F, generate K |
| C | `TIEL_SPEC=1 TIEL_SPEC_TYPE=draft-mtp` | cold prefill of P, generate K |
| D | `TIEL_SPEC=1 TIEL_SPEC_TYPE=draft-mtp` | restore file F, generate K |

F is produced by saving the slot immediately after prefilling P, in the same launch mode as the cell
that restores it (a `TIEL_SPEC=0` file for B, an MTP file for D — an MTP file has non-empty
`data_dft`, a spec-off file does not).

Also run a fifth cell, **E**, to reach gap B (§3.3): restore F in cell D's launch, then send a
request that **diverges** from P after a prefix long enough to land past a checkpoint, forcing the
`load_dft` path at `tools/server/server-context.cpp:3481`.

### 5.3 What to compare, and what each result means

**Correctness observable — the emitted token ids.** Compare with `logprobs` off and exact string
equality on `content`, plus the token id list.

| observation | reading |
|---|---|
| A == B and C == D | **proven** for this model/build: the speculative path adds no divergence over restore. |
| A == B but C != D | **refuted.** Restore is faithful with speculation off and not with it on; the difference is the draft path, and §4.3's argument is wrong somewhere. This is the outcome that would justify kv_rosetta's refusal on correctness grounds. |
| A != B | Inconclusive for the speculative question — the restore itself is not reproducing the target. Fix or characterise that before reading C vs D. |
| A == B, C == D, but C != A | Expected and harmless if it only shows up above temperature 0; at temperature 0 it would mean the target's argmax path differs with speculation on, which is a bug in the verification loop independent of restore. |

**Value observable — acceptance rate.** `/metrics` exposes
`spec_decode_num_draft_tokens_total`, `spec_decode_num_accepted_tokens_total` and
`spec_decode_num_drafts_total` (`tools/server/server-task.cpp:1554`–`:1565`); the per-slot totals
are also logged as `draft acceptance = ...` at `tools/server/server-context.cpp:677`–`:678`. Take
deltas across each cell.

Prediction to test (**inference**, from §4.2): D's acceptance rate starts materially below C's and
converges toward it as generation proceeds and `ctx_dft` refills by decoding. If D's acceptance rate
equals C's from the first verification step, then the stale draft cache costs nothing measurable and
restoring `data_dft` has no value — which would settle the question in the other direction, by making
the whole class moot rather than unsafe.

**Availability observable — cell E.** If the server survives and the divergence rollback completes,
`data_dft` round-tripped through the file into a live `ctx_dft` without a size mismatch. If it
aborts with `checkpoint size mismatch`, gap B is confirmed as reachable in practice and needs a
compatibility check in `load_slot_checkpoints` before any consumer forwards these blobs.

### 5.4 Cheap preliminary that needs no generation

Save a slot in each mode and read the reported coverage fields the KV Rosetta patch added —
`n_checkpoints_saved`, `checkpoint_bytes`, `checkpoint_n_tokens`
(`patches/llama.cpp/0002-advertise-checkpoint-capability.patch`, and
`tools/server/server-context.cpp:2681`–`:2687`). Then parse the SCKP appendix at
`n_written - checkpoint_bytes` (the offset arithmetic `scripts/direct_raw_lowerbound.py:102`
already does) and read the three length prefixes per checkpoint.

Expected under this analysis: `len(data_dft) > 0` and `len(data_spec) == 0` for every checkpoint in
an MTP file; both zero in a `TIEL_SPEC=0` file. That single byte-level read confirms or refutes §1.3
without a GPU-side experiment, and it is the first thing to run.

---

## 6. What this means for `kv_rosetta/hybrid_schema.py`

`CheckpointRecord.validate` refuses on `has_draft_state or has_speculative_state`
(`/home/crogers2287/kv-rosetta/kv_rosetta/hybrid_schema.py:132`–`:136`). Against the source:

1. **The two flags are not symmetric on this build.** For MTP, `has_speculative_state` should be
   `False` for every checkpoint — `data_spec` is empty (§1.3b). If kv_rosetta is deriving it from
   `/props`' `active_checkpoint_state_classes` rather than from the file's length prefixes, it is
   refusing on a launch label, and the label is wrong in this direction (§1.4). Deriving it from the
   `data_spec` length prefix would make the refusal accurate.

2. **`has_draft_state` is the one that is genuinely true**, and `/props` does not advertise it
   (§1.4). A gate that trusts `active_checkpoint_state_classes` will under-detect exactly the blob
   that carries risk.

3. **The refusal's stated reason is right for `data_dft` and for a reason the current comment does
   not give.** The comment at `hybrid_schema.py:44`–`:56` says the bytes are in the file and nothing
   has shown they restore. True. But the sharper reason is §3.3: `data_dft` applied to a mismatched
   draft context reaches `GGML_ABORT` and kills the server, and the SCKP appendix carries no
   identity to prevent it. For a format whose purpose is moving caches between builds, that is a
   stronger argument for refusing to forward `data_dft` than "unproven" is.

4. **Nothing here argues that restoring MTP draft state is needed for correct output.** §4 argues
   the opposite. If the §5 measurement comes back A==B, C==D, then the right disposition for
   `data_dft` is probably "drop it at ingestion and record that it was dropped", not "refuse the
   artifact" — the artifact's target state is fine and the draft blob is a speed optimisation with
   an availability hazard attached. That is a design call for whoever owns the gate, not a finding.

---

## 7. Ledger: settled from source vs left for measurement

**Settled from source, cited above:**

- Q1 fully. `data_dft` non-empty and full-KV under MTP (`common/common.cpp:2301`,
  `src/llama-kv-cache.cpp:2100`, `src/llama-model.cpp:2429`); `data_spec` empty under MTP
  (`common/speculative.cpp:175`, `:872`, MTP struct `:1373`–`:1812`); `data_tgt` size unchanged by
  `n_rs_seq` (`src/llama-memory-recurrent.cpp:783`, `:873`).
- Q2's first half. The `supports_*` flags are hardcoded literals with no probe
  (`tools/server/server-context.cpp:4783`–`:4784`) and no test covers the draft/spec blobs.
- Q2's second half, as far as existence goes. Gap A: restore never writes `ctx_dft`
  (`tools/server/server-context.cpp:2692`–`:2770`). Gap B: `load_dft` aborts on mismatch
  (`common/common.cpp:2349`).
- Q3's first half. `pending_h` is a pure function of the target's accepted prefix and is refreshed
  before first use on every path including restore (`common/speculative.cpp:1638`,
  `tools/server/server-context.cpp:3513`, `:3858`, `:3951`).
- Q3's second half. The draft context's prefix KV is *not* rebuilt by anything on the restore path.

**Not settled — must be measured:**

- Whether a stale draft context can change emitted text. §4.3 argues it cannot at temperature 0 from
  the acceptance rule, but that is reasoning, not a line of source, and it is exactly the thing the
  §5 2x2 exists to check.
- What the acceptance-rate cost of the stale draft cache actually is, and how fast it recovers.
- Whether installing `data_dft` on the plain restore path would yield a draft context equivalent to
  a decoded one. Nothing has run it; the blob's *content* validity across a process boundary is
  untested.
- Whether gap B is reachable with real kv_rosetta artifacts, and what `data_dft` sizes look like in
  practice.

**Explicitly not knowable from this tree:** whether the upstream author of PR #26004 had a reason
beyond "untested" for not restoring the draft context on `action=restore`. The patch's own comment
(`tools/server/server-context.cpp:2347`–`:2348`) explains only why checkpoints must be persisted at
all — "a recurrent state cannot be rewound" — and says nothing about the draft side.
