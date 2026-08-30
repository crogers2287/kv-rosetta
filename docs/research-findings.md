# Research findings

Everything here was verified on the development host on 2026-08-28. Each claim names how it
was checked. Nothing in this file is inferred from documentation alone.

## 1. The premise in the original README was half wrong

The seed README stated that runtime cache extraction was deferred because "neither llama.cpp
nor vLLM currently exposes a common stable cache-import ABI". Three usable seams exist today,
and none of them requires patching an engine.

| Runtime | Seam | Where |
|---|---|---|
| llama.cpp (in-process) | `llama_state_seq_get_data` / `llama_state_seq_set_data`, magic `GGSQ`, `LLAMA_STATE_SEQ_VERSION 3` | `llama.h:874`, constants at `llama.h:45-49` |
| llama.cpp (HTTP) | `POST /slots/{id_slot}?action=save\|restore`, gated on `--slot-save-path` | `tools/server/server.cpp:286`, server README §1150 |
| vLLM 0.24.0 | `KVConnectorBase_V1` — the connector receives real paged KV tensors | `vllm/distributed/kv_transfer/kv_connector/v1/base.py` |
| HF transformers | `past_key_values`, plus logits, which makes it the gate's ground truth | `DynamicCache` |

What is true is that none of these is a *common* ABI. llama.cpp hands out an opaque blob;
vLLM hands out live tensors; transformers hands out a Python object. The portable format has
to absorb that difference rather than assume it away — see §4.

Two caveats found while checking:

- `SharedStorageConnector` no longer exists in vLLM 0.24.0. The reference implementations
  that do exist are `example_connector.py`, `offloading_connector.py`, `multi_connector.py`,
  plus LMCache/NIXL/Mooncake backends.
- `--slot-save-path` is not set anywhere in the local llama-swap configuration, so the HTTP
  slot endpoints are currently inert on this fleet. Enabling them is a per-model flag in each
  wrapper script, not a code change.

## 2. cfrproxy already implements the prompt-corpus layer

The proxy is the one component that sees every prompt, and it already does the work KV Rosetta
would otherwise duplicate.

| Capability | Implementation |
|---|---|
| Content-addressed static-prefix manifests (system prompt + canonical tool schemas) | `internal/proxy/prefixcache.go` |
| Ground-truth cache telemetry: `cache_n`, `cache_lcp_n`, `cache_reprocessed_n`, `cache_source`, `cache_reason` | `internal/proxy/cachelog.go` |
| Conversation-level cache affinity routing | `internal/proxy/routesticky.go` |
| A sidecar that replays those manifests to keep prefixes resident | `scripts/kvwarm.py` |

Consequences for the design:

- **KV Rosetta adopts cfrproxy's fingerprint verbatim** rather than inventing one:
  `sha256(provider + "\0" + model + "\0" + system_sha256 + "\0" + tools_sha256)`.
  Adopting the key is what lets a sidecar consume the proxy's manifests with **zero proxy
  changes**.
- **kvwarm already solved model identity, model-agnostically.** It reads identity from the
  *running server* (`/props`) rather than assuming it:
  `sha256(model_path + model_ftype + chat_template + bos_token + eos_token)`. That is exactly
  the `identity.l0_sha256` field in the KVX header, formula unchanged.
- **kvwarm warms by recompute; KV Rosetta warms by restore.** That single difference is the
  entire delta between the two, and it is why a sidecar is the right shape.

One integration hazard, worth stating because it is silent: `cachelog.go` skips recording its
own warmup traffic by matching the literal client label `"kvwarm"`. A successor daemon using a
different label will have its replays recorded as real traffic, which refreshes every
manifest's mtime and freezes the LRU.

## 3. Capability tiers are the model-agnostic mechanism

A backend is supported at whatever tier it can *prove* at startup. There is no per-model
special-casing anywhere in the design.

| Tier | Means | Cross-backend | Cross-model |
|---|---|---|---|
| 0 `PROMPT` | replay the prompt to re-prefill | n/a | n/a |
| 1 `OPAQUE` | engine-native blob save/restore | only if cache config is byte-identical | no |
| 2 `CANONICAL` | KVX canonical tensors | yes | exact-ABI only |
| 3 `MAPPED` | canonical + learned mapping behind a quality gate | yes | yes, gated |

Every failure falls down the ladder, ending at a normal prefill. Tier 0 is kvwarm's existing
behaviour and remains the floor.

**Tier 1 is narrower than it looks on this fleet.** The CUDA 27B runs `kvarn4`/`kvarn6` KV
quantisation while the W6800 27B runs `q8_0`. Their opaque blobs are not interchangeable, so a
CUDA↔HIP transfer between them is a tier-2 problem, not a tier-1 one. An adapter refusing that
import is behaving correctly, not failing.

## 4. KVX container: one format, two payload encodings

`blob.encoding` is either `raw` (canonical `layer,kv,token,head,dim`, kv axis index 0 = keys,
1 = values) or `opaque` (engine-native bytes plus `opaque_format`, e.g. `ggsq/3`).

This single decision is what lets adapters at different tiers share one artifact format, and
what let three adapters be developed in parallel without forking the spec.

```text
0                        magic b"KVX1"                       4 bytes
4                        uint32 container_version = 1
8                        uint32 header_len
12                       header JSON, UTF-8, header_len bytes
                         pad 0x20 up to the next 64-byte boundary
align64(12+header_len)   payload
```

`blob.offset` lives inside the header whose length determines that offset, so writing is a
fixed-point iteration: serialise, measure, set the offset, re-serialise until stable (capped,
then an error rather than a hang).

Verified: bit-identical round trip for f32/f16/bf16; payload always 64-byte aligned; a single
flipped payload byte is caught by the sha256 check; `mmap=True` and `mmap=False` agree; opaque
bytes survive exactly; cross-encoding access raises; truncated and missing files return a
reason instead of raising.

## 5. The GGSQ v3 state blob layout

Read out of `src/llama-kv-cache.cpp` rather than guessed. All little-endian.

```text
state_write            (llama-kv-cache.cpp:2094)
  uint32  n_stream
  per stream:
    uint32 cell_count            (0 -> skip this stream entirely)
    meta block, then data block

state_write_meta       (llama-kv-cache.cpp:2244)
  per cell:
    int32  pos
    uint32 n_seq_id
    [optional fixed-size cell_ext block, only when has_cell_ext()]
    int32 * n_seq_id  seq ids

state_write_data       (llama-kv-cache.cpp:2277)
  uint32  v_trans
  uint32  n_layer
  per layer (keys):
    int32  k_type          (ggml type id)
    uint64 k_size_row
    cell_count * k_size_row bytes
  if v_trans == 0, per layer (values):
    int32  v_type
    uint64 v_size_row
    cell_count * v_size_row bytes
  if v_trans != 0, per layer (transposed values):
    int32  v_type
    uint32 v_size_el         <-- uint32 ELEMENT size, not a uint64 row size
    uint32 n_embd_v_gqa
    n_embd_v_gqa rows, each cell_count * v_size_el bytes
```

The asymmetry in the transposed branch is the single easiest thing to get wrong: it shifts
every subsequent offset if read as a uint64. Quantised KV types that cannot be dequantised
losslessly stay tier-1 only; the parser reports that rather than silently degrading.

## 6. Environment facts

Recorded so they are not re-derived.

- CUDA: 2x RTX 3090, `sm_86`. ROCm: Radeon PRO W6800, `gfx1030`. Both backends on one host,
  so the hardware-neutrality claim is testable here.
- **No new torch install is needed.** A CUDA environment (torch 2.11.0+cu130, transformers
  5.13.1, vllm 0.24.0) and a ROCm environment (torch 2.9.1+rocm6.4, HIP 6.4, transformers
  5.15.0) both already exist on the host.
- The system interpreter's transformers (4.46.3) cannot load `qwen3_5` model types.
- llama.cpp CUDA build present at commit `ca3d5a3`. The only HIP builds on the host belong to
  unrelated forks, so a same-commit HIP build is still required before trusting a CUDA↔HIP
  state transfer. Check `LLAMA_STATE_SEQ_VERSION` on both sides first.

### transformers 5.x changed the Cache API

Verified by instantiating a `DynamicCache` and inspecting it, not from docs:

- `key_cache` / `value_cache` **no longer exist**.
- The accessor is `cache.layers[i].keys` and `cache.layers[i].values`, each shaped
  `[batch, kv_heads, seq, head_dim]`.
- `DynamicLayer` also carries `is_initialized`, `dtype`, `device`.
- `cache.update(k, v, layer_idx)` still returns a `(keys, values)` tuple.

Any HF adapter written against the 4.x names is dead on arrival on this host.

## 7. The quality gate is the product

The project's central claim is that a translated cache must never be trusted because its
tensor reconstruction score looks good. That is enforced structurally, not by convention:
`tensor_cosine` is computed and reported but is never read by the admission logic.

Verified with a deliberately constructed adversarial case — two logit sets sharing a large
constant component so they are nearly parallel as vectors, but with the argmax moved on every
position:

```
tensor_cosine   = 0.999980
top1_agreement  = 0.000
admitted        = False
```

Admission requires held-out next-token agreement and KL divergence. The fallback is always a
normal prefill.

## 8. Notes on generating this code with a local model

The implementation was fanned out to a local Tiel-Coder 35B-A3B (2x W6800). Findings that
would otherwise have to be rediscovered:

- **Long prompts destroy reasoning-native models.** 6-7 KB specs produced zero tool calls
  across eight parallel agents; the model reasoned until its budget was gone. Moving the spec
  into a file the model reads with a tool, and cutting the instruction to ~700 bytes, fixed it.
- **Lossy prompt compression corrupts specs.** With compression on, agents received
  elision-marked spec text and burned every turn trying to recover the missing bytes with
  `cat`/`sed` fragments. Exact-bytes content must bypass compression.
- **The model writes good implementations and unreliable tests.** Every model-authored test
  file checked so far contained at least one wrong assertion against correct code: asserting
  log-probabilities sum to 1, exact float equality on `0.9000000000000004`, and asserting on a
  metric without passing the argument that makes it non-NaN. A green suite from a generated
  test file proves very little; each module's central claim was re-verified independently.
- **Reasoning tokens consume the output budget.** With a small `max_tokens`, `content` comes
  back empty because the whole allowance went to `reasoning_content`. An empty file also
  compiles, so a naive syntax check reports success — generated files must be checked for
  non-trivial length as well as validity.

## 9. Audit of commit fb3ecf7 against the hardening steer

The steer (`codex/steer-kv-rosetta-hardening`, commit 105f1fc) asserted that fb3ecf7 was
scaffolding with fail-open defects. Each claim was reproduced against the pushed code before
any fix was written. All of them were real.

| # | Defect | Reproduction | Severity |
|---|---|---|---|
| 1 | `holdout_tokens` was a maximum slice, not a minimum | `T=1` compared against a 64-token policy returned `admitted=True` | fail-open |
| 2 | Non-finite thresholds disabled the comparison they guard | `top1_agreement=nan`, `mean_kl=inf` admitted a candidate whose argmax was wrong at every position | fail-open |
| 3 | Untrusted fingerprints reached the filesystem | `path_for(fingerprint="../../../../tmp/pwned")` resolved outside the store root | path traversal |
| 4 | bf16 conversion truncated instead of rounding | `0x3F80C000` became `1.0` rather than `1.0078125` | silent, compounding precision loss |
| 5 | Integrity covered only the payload | Editing `kv.head_dim` from 4 to 8 in the header still returned `verify() == (True, "ok")` | fail-open |
| 6 | Conformance did not require a successful round trip | An adapter that always returns a failed `ImportReport` satisfies the suite | vacuous gate |
| 7 | GGSQ parsing took a bare body | `parse(blob)` has no magic or version check and cannot distinguish an in-process buffer from a `GGSQ` file | guesses the start offset |
| 8 | `Capabilities` exposed a single maximum tier | Cannot express "exports canonical, imports opaque only" | too coarse to be safe |

Defect 4 originated in the specification handed to the generating model, which said
"Truncate, do NOT round". The model implemented the spec correctly; the spec was wrong.

### Fixed so far

- **Gate fails closed.** `holdout_tokens` is a minimum; a shortfall is rejected as insufficient
  evidence rather than judged leniently on a small sample. All thresholds are validated on
  construction, so a NaN or infinite bound raises instead of silently disabling its check. A
  verdict now carries a `GateBinding` recording the source and target model identities, the
  artifact digest, the mapper id, the calibration digest and the policy version, because a
  verdict is a property of a specific translation and not of two arrays.
- **Store paths are contained.** Externally supplied digests must be 64 lowercase hex
  characters before they are used to build a path, and the resolved path is confirmed to be
  under the store root. The full digest is the filename: truncating it traded a real collision
  risk for tidiness.
- **Integrity covers the header.** A fixed-width digest placeholder is written into the header,
  hashed with the placeholder zeroed, then overwritten, so the header length and the payload
  offset stay valid. Editing any semantic field now fails verification.
- **bf16 rounds to nearest even**, with NaN, both infinities and signed zeros preserved.

Still outstanding from the steer: the segment table (P0 #2), composite artifact identity
(P0 #1), content-derived model and cache-ABI identity (P0 #4), explicit GGSQ envelopes
(P0 #6), directional capabilities (P0 #7), and a conformance suite that demands a real round
trip. Runtime adapters must not be built until those land.

## 10. First runtime proof: llama.cpp same-backend round trip

Everything before this was a property of the file format. This is the first evidence that
a real cache can leave a runtime and come back.

Setup: `llama-server` from `~/llama.cpp/build/bin` at commit `ca3d5a3`, CPU-only (`-ngl 0`,
both 3090s were at 23.5/24.5 GB serving the resident vLLM 27B, so there was no VRAM to take),
`library_of_alexandria_Q4_K_M.gguf` (Qwen2.5-3B), `-c 4096`, one slot, started with
`--slot-save-path`.

Method, following the break-first ladder at its smallest rung: a prompt tokenized to
**exactly 256 token IDs**, greedy decoding (`temperature 0`, `top_k 1`), 16 predicted tokens
with top-5 probabilities recorded.

| Step | Result |
|---|---|
| Cold run | `prompt_n=256 cache_n=0`, 423 ms prefill |
| Save slot 0 | 271 cells, 9,995,388 bytes, 8.6 ms |
| Erase slot 0 | 271 cells erased |
| Re-run after erase, no restore | `cache_n=0` — the cache really was gone |
| Restore from file | 271 cells, 9,995,388 bytes read, 7.0 ms |
| Run after restore | `prompt_n=1 cache_n=255`, 97 ms prefill |

Parity against the cold reference:

- generated text identical
- **token IDs identical** across all 16 generated positions
- **max top-5 probability delta `0.000e+00`**, first divergent position: none
- prefill 423 ms to 97 ms, a 4.4x reduction

The erase-and-rerun step matters: without it, a "successful restore" cannot be
distinguished from a cache that was never dropped.

### What this does and does not establish

It establishes: llama-server's slot endpoints are a usable cache-import ABI; an opaque KVX
artifact survives a save/erase/restore cycle; and a restored cache produces token-for-token
identical output rather than merely plausible output.

It does not establish: cross-backend transfer, cross-model transfer, canonical decoding of
the blob, or behaviour above 256 tokens. Those are separate rungs and none of them should be
described as working until each has its own retained test.

### Kept as a regression test

`tests/test_llamacpp_roundtrip.py`, skipped unless `KVX_LLAMA_URL` and `KVX_LLAMA_SLOTS` are
set. A one-time check does not count: the container, the identity formula and the adapter can
all drift, and this is the only test that would notice.

The adapter also passes the conformance suite that rejects five separately broken adapters
(9 tests run, 1 legitimate skip for the canonical representation it does not export).

## 11. The context ladder, and what it actually showed

Ran on the proven llama.cpp path, 2K / 8K / 32K, greedy decoding, parity checked at every
rung. Two models on the same code path.

**Qwen2.5-3B Q4_K_M, f16 KV, one 3090:**

| tokens | cold prefill | warm prefill | artifact | restore | restore cheaper? | parity |
|---|---|---|---|---|---|---|
| 2048 | 218 ms | 8 ms | 72 MB | 0.59 s | no | yes |
| 8192 | 621 ms | 9 ms | 288 MB | 2.50 s | no | yes |
| 32000 | 3413 ms | 17 ms | 1126 MB | 7.99 s | no | yes |

Parity held everywhere. But **restore cost more than recomputing the prefill at every rung**.
The in-server prefill after a restore is near-free (17 ms at 32K); moving 1.1 GB off disk at
~140 MB/s is not.

That result is narrower than it looks. A 3B prefilling at ~9,400 tok/s with f16 KV
(36 KB/token) is simultaneously the fastest prefill to beat and the largest artifact to move -
the worst case for a cache path on both axes at once. Holding everything else fixed, q4_0 KV
would cut the artifact to ~281 MB and restore to ~2.0 s, which beats the 3.41 s prefill. The
fleet runs kvarn4/q4_0, not f16.

**The 27B run did not answer the economics question, because the cache never came back.**

| tokens | cold | warm | cache_n | verdict |
|---|---|---|---|---|
| 2048 | 1546 ms | 1510 ms | 0 | no reuse |
| 8192 | 4745 ms | 4749 ms | 0 | no reuse |
| 32000 | 19286 ms | 19270 ms | 0 | no reuse |

`warm` equals `cold` because every token was re-prefilled. The server reported
`n_restored=201` and set `n_prompt_tokens`, and the very next completion still returned
`cache_n=0`. Also note 846 KB/token here against 36 KB/token on the 3B.

### The bug this exposed

The adapter reported `ok=True` on the strength of the server's restore count. **A restore
count is not evidence of a usable cache.** That is a fail-open defect of exactly the class
the steer is about, and it was in code I wrote.

`import_` now verifies reuse rather than trusting the count: it recovers the prompt token IDs
from the artifact, issues a one-token completion, and refuses unless `cache_n > 0`.

    VERIFIED   -> ok=False  "restore accepted (121 cells) but the cache is not reusable: cache_n=0"
    UNVERIFIED -> ok=True   (what the old code reported)

The retained round-trip test already asserted `cache_n > 0`, so it caught this on the 27B
while passing on the 3B. That assertion is the one that earns its place.

## 12. The llama.cpp state-file envelope, corrected against reality

Three constants written from the specification alone were all wrong, and one failed open.

| Assumed | Actual | Consequence |
|---|---|---|
| magic `b"GGSQ"` | `b"qsgg"` - `LLAMA_FILE_MAGIC_GGSQ` is `0x67677371` written as a little-endian uint32 | a real state file was detected as an in-process buffer and parsed into a plausible envelope with zero tokens and body_offset 4 |
| version 3 | the running binary writes **2**, while the checked-out header declares `LLAMA_STATE_SEQ_VERSION 3` | the installed binary (built 2026-08-19) predates the source at `ca3d5a3`; a file written today would be **refused** by a binary built from HEAD, since `state_seq_load_file` requires an exact version match |
| the array is a token list | it is `server_tokens::serialize()` reinterpreted as `llama_token *` | reading it as tokens yields `-1` as the first token and the server rejects the request with HTTP 400 |

The prompt payload layout, from `tools/server/server-common.cpp`:

```
LLAMA_TOKEN_NULL (-1)          format marker
SERVER_TOKENS_STATE_VERSION    currently 1
tokens      as [count][elements...]
media_keys  as [count][elements...]
media chunks
```

which accounts for every word of the observed header: `1 + 1 + 1 + 201 + 1 = 205`.
`decode_prompt_tokens` implements it, falls back to a plain list for older files (as
`server_tokens::deserialize` does), and returns nothing when media chunks are present,
because a text-only reuse of a multimodal prompt would silently drop the media.

A 2 KB slice of a real slot file is committed at
`tests/fixtures/llamacpp_state_seq_header.bin`. Every constant above is now pinned against
it rather than against a synthetic blob built from the same wrong assumptions as the parser.

### Runtime revision belongs in the cache ABI

The version-2-versus-3 split is the concrete case for what the steer required: an artifact
written by one build of the same runtime, at the same commit, is not necessarily readable by
another. Runtime revision is part of cache identity, not provenance.

## 13. The 27B failure, classified: hybrid architecture, not MTP

The steer named MTP as the leading suspect. It is not the cause. Variables were changed one
at a time, all on the same code path.

| Variable changed | Result |
|---|---|
| flash attention on, 3B, otherwise the known-good run | reuses, `cache_n=255` |
| the 27B's **exact** flags on the 3B: `-ngl 99`, `-fa on`, `--split-mode layer`, `--tensor-split 1,1`, `--parallel 1` | reuses, `cache_n=255` |
| a large **non-MTP** model (OpenMythos-Q6_K, 20.6 GB) with those same flags | **fails**, `cache_n=0` |

So GPU offload, flash attention, tensor split, slot count, prompt size and MTP are all
exonerated. Size is not the cause either: the 3B restored a 1.1 GB artifact at 32K and
reused it.

The discriminator is the architecture:

| | 3B, reuses | both failures |
|---|---|---|
| `general.architecture` | `qwen2` | `qwen35` |
| block_count | 36 | 64 and 65 |
| key/value length | 128 | 256 |
| bytes per token | 36 KB | 663-678 KB |

`LLM_ARCH_QWEN35` appears in `llm_arch_is_hybrid()` in `src/llama-arch.cpp`, alongside
jamba, falcon-h1, plamo2, granitehybrid, lfm2, nemotron_h, qwen3next, kimi-linear,
bailingmoe3, kimi-k3, qwen4exp, deepseek4 and minimax-01. These are hybrid attention plus
recurrent (SSM) models, which is also why the per-token state is roughly 2.6x what plain
attention KV of that geometry would be.

**Correction (see §17 and `hybrid-checkpoint-research.md`).** This section originally
concluded that hybrid restoration is impossible. That was too broad. A recurrent state
captured after an *exact* token sequence is deterministic and restorable; what cannot be done
is reconstructing it from ordinary KV cells or slicing it to an arbitrary earlier prefix.

The accurate statement is narrower: **current llama.cpp slot persistence omits the
checkpoint state that hybrid and recurrent models require.** `n_restored` equals the saved
cell count and `cache_n` is still zero because the sequence cells were restored and the
checkpoint the server actually consults was never saved. The observed behaviour below is
unchanged; only this interpretation is corrected.

### The fail-closed response

`kv_rosetta/gguf.py` reads `general.architecture` from the GGUF header - a few kilobytes,
never the tensor data - and `supports_prefix_reuse()` refuses hybrid and recurrent
architectures, mirroring llama.cpp's own two lists. An unknown architecture is refused
rather than assumed.

The adapter now withholds the capability instead of advertising it:

    3B qwen2       export=['opaque'] import=['opaque'] formats=['ggsq/2']  notes: none
    27B qwen35     export=[]         import=[]         formats=[]
                   note: opaque transfer withheld: qwen35 is a hybrid attention+recurrent
                   architecture ...

Advertising a capability the runtime accepts but cannot honour is the exact fail-open shape
this project exists to prevent. The earlier verified-reuse check already refused these
imports at run time; this moves the refusal up to capability probing, so the transfer is
never attempted and never has to be caught.

### Consequence for the economics question

The q4 ladder cannot be measured on either 27B available here, because neither can reuse a
restored cache at all. Measuring it needs a large **non-hybrid** model. The q4_0 projection
therefore remains arithmetic, not a result, and is still not promoted.

## 14. Measured q4 economics, and a quantization finding that matters more

The earlier q4_0 advantage was arithmetic. It is now measured, on the same 3B, changing only
the KV cache type. Storage media are reported separately and never mixed. Machine-readable
records: `bench/restore-tmpfs.json`, `bench/restore-nvme.json`.

**tmpfs (/dev/shm), q4_0 KV, medians of repeated runs:**

| tokens | native prefill | total restore | KB/token | restore cheaper? | parity | native cache parity |
|---|---|---|---|---|---|---|
| 256 | 372 ms | 133 ms | 10.4 | yes | **no** | **no** |
| 2048 | 1581 ms | 334 ms | 10.2 | yes | yes | yes |
| 8192 | 6516 ms | 1204 ms | 10.1 | yes | yes | yes |

**ext4 / NVMe, same configuration:**

| tokens | native prefill | total restore | KB/token | restore cheaper? | parity | native cache parity |
|---|---|---|---|---|---|---|
| 256 | 373 ms | 130 ms | 10.4 | yes | **no** | **no** |
| 2048 | 1501 ms | 421 ms | 10.2 | yes | yes | yes |
| 8192 | 6206 ms | 1183 ms | 10.1 | yes | yes | yes |

Total restore latency is the user-visible figure - artifact read plus integrity and identity
verification plus runtime restore plus reuse verification - not the server's restore time.

Two results:

**q4_0 KV reverses the economics.** 10.1 KB/token against f16's 36 KB/token, and restore
beats native prefill at every rung, by about 5.4x at 8192. The f16 measurement in §11, where
restore lost at every rung, was the worst case on both axes at once. Storage medium barely
matters at these sizes: tmpfs and NVMe are within noise of each other, so the cost is
dominated by verification and the runtime restore, not by the read.

**Quantized KV is not token-exact, and that is a property of the runtime, not of KV Rosetta.**
At 256 tokens the restored run diverged from the cold run. The benchmark now carries the
control that settles the attribution:

- two cold prefills with no cache reuse agree exactly, so the model is deterministic;
- llama.cpp's OWN prompt-cache reuse, with no KVX artifact anywhere in the picture, produces
  different output from a cold prefill: `'1892 the naturalist recorded'` against
  `' 1892 the naturalist'`;
- `parity` and `native_cache_parity` move together at every rung.

A fresh prefill quantizes values it has just computed; a reused cache reads back values that
were already quantized. The rounding differs, the logits differ, and greedy decoding can pick
a different token. Restore reproduces whatever the runtime's own reuse does - no better, no
worse.

The consequence is a trade, not a win:

| KV type | artifact | restore vs prefill | token-exact |
|---|---|---|---|
| f16 | 36 KB/token | slower at every rung | yes, max top-5 delta 0.000e+00 |
| q4_0 | 10.1 KB/token | ~5.4x cheaper at 8192 | **no**, diverges as the runtime's own cache does |

This is precisely the case the admission gate exists for. A quantized-KV cache must be
admitted on held-out agreement, never assumed exact, and "the runtime does this to itself
anyway" is an explanation of the divergence, not a licence to skip the gate.

### Still not measured

The 27B q4 ladder. Every large model on this host is `qwen35` or `qwen35moe`, all hybrid, and
none can reuse a restored prefix at all (§13). A survey of models above 3 GB found exactly one
non-hybrid: a 5.8 GB f16 build of the same 3B. Large-model economics therefore remain
unmeasured, and nothing here should be read as covering them.

## 15. The hybrid failure, confirmed by the runtime's own diagnostic

Section 13 classified the `qwen35` failure by elimination and by llama.cpp's architecture
list. That classification is now confirmed directly: run a hybrid model with `-lv 10` and the
server states the cause itself during the post-restore completion.

```
task 7 | forcing full prompt re-processing due to lack of cache data
         (likely due to SWA or hybrid/recurrent memory, see PR #13194)
task 7 | main/do_checkpoint = no, pos_min = -1, pos_max = -1
         created context checkpoint 1 of 32 (pos_min = 251, pos_max = 251,
         n_tokens = 252, size = 149.626 MiB)
```

The mechanism, from `tools/server/server-context.cpp` around line 3338:

1. For hybrid or recurrent memory the server does not reuse a prefix from KV cells alone.
   It searches `slot.prompt.checkpoints` for a **context checkpoint** covering the position
   it wants to resume from.
2. If none matches, `do_reset` is true: it logs the message above and sets
   `pos_next = 0`, `n_past = 0` - a full re-prefill.
3. Checkpoints are server-side in-memory state on the slot. `llama_state_seq_save_file`
   saves the llama_context's sequence state; it does not save the server's checkpoint list.

So a restored slot has every KV cell back - `n_restored` equals the saved count, and
`n_prompt_tokens` is correct - and still cannot reuse anything, because the structure the
server actually consults was never in the artifact. That is why `n_restored` was such a
misleading success signal, and why the verified-reuse probe was the right fix.

Note the checkpoint is 149.6 MiB for a 252-token prefix, and the server keeps up to 32 of
them. Any future support for hybrid models has to carry that state, not just the KV cells.

### What would have to change

The HTTP slot API exposes no way to save or restore context checkpoints, so no adapter can
work around it from outside. Supporting hybrid models over this seam needs llama.cpp to
persist checkpoints alongside the sequence state - which is a missing feature with an open
upstream issue (ggml-org/llama.cpp#25913), not a law of the architecture.

Until a patched runtime advertises a checkpoint-persistence format, the code keeps the
refusal: probe the architecture and withhold the capability. The pinned source map, the
correction to the earlier interpretation, and the retained negative control are in
`hybrid-checkpoint-research.md`.

The mechanism is **proven by the runtime's own trace and its source**, not inferred.

## 16. P2 completed for the measurable model: three repetitions, medians and ranges

Section 14 reported medians from two runs. The steer asks for at least three clean
repetitions with medians *and* ranges, plus throughputs, break-even context and peak
memory. Re-run at three repetitions per rung, both media, q4_0 KV, CPU inference.

**tmpfs (/dev/shm)**

| tokens | native prefill (range) | total restore (range) | speedup | ms saved | read | restore |
|---|---|---|---|---|---|---|
| 256 | 351 ms [347-386] | 138 ms [133-140] | 2.5x | 210 | 150 MB/s | 22 MB/s |
| 2048 | 1624 ms [1578-1633] | 376 ms [336-400] | 4.2x | 1224 | 245 MB/s | 70 MB/s |
| 8192 | 6678 ms [6569-6748] | 1163 ms [1145-1266] | 5.6x | 5412 | 297 MB/s | 91 MB/s |

**ext4 / NVMe**

| tokens | native prefill (range) | total restore (range) | speedup | ms saved | read | restore |
|---|---|---|---|---|---|---|
| 256 | 349 ms [342-410] | 146 ms [109-155] | 2.3x | 196 | 128 MB/s | 21 MB/s |
| 2048 | 1534 ms [1483-1561] | 408 ms [369-409] | 3.8x | 1126 | 233 MB/s | 65 MB/s |
| 8192 | 6685 ms [6171-6784] | 1212 ms [1154-1328] | 5.3x | 5357 | 255 MB/s | 91 MB/s |

- **Break-even: none in this range.** Restore wins at every rung on both media, so there is
  no crossing to interpolate. Reported as `null` rather than as a fabricated number.
- **Storage medium is not the variable.** tmpfs leads NVMe by 3-6%, inside the run-to-run
  ranges. The cost is verification and the runtime restore, not the read - which is why
  restore throughput (22-91 MB/s) is far below read throughput (128-297 MB/s).
- **Peak RSS 51-52 MB** for artifacts up to 84 MB, confirming the streaming path holds.
- Speedup grows with context because prefill is linear in tokens while the fixed costs of
  verification and the restore call are amortised.

### Two limits of these records, recorded in the records

- `llama-server /props` does not report `type_k`/`type_v`, so the KV type cannot be probed.
  The records carry `kv_type_declared` from the launch flag, corroborated by 10.1 KB/token
  against the 36 KB/token measured at f16 - not by the server's own word.
- `peak_vram_mb` comes from `nvidia-smi` and is host-wide, not attributed to this process.
  These runs used `-ngl 0`, so that figure reflects other models resident on the host and
  says nothing about this benchmark. It is kept because a GPU run would want it, and
  labelled so it cannot be misread.

Both caveats are written into `bench/*.json` alongside the numbers, so a later reader does
not have to reconstruct them.

---

## 17. Cross-backend, proven: one file across ROCm, Vulkan and a restart

The project's central hardware claim, tested rather than argued. Three llama.cpp builds at
**one source revision** (`ca3d5a3e1`): HIP for gfx1030, Vulkan (RADV), and CUDA.

Pinning the revision is not bureaucracy. A first attempt restored a v3 state file into a
binary compiled months earlier, which rejected it for the *state-file version* - and that
rejection is indistinguishable from "cross-backend does not work" until you read the server
log. The guard that prevents it reads the commit from `--version`, because an earlier
implementation scraped `strings` and reported two builds of one commit as a mismatch.

| prompt | HIP -> Vulkan | Vulkan -> HIP | artifact |
|---:|---|---|---:|
| 128 | 127/128 | 127/128 | 4.7 MB |
| 8,192 | 8,191/8,192 | 8,191/8,192 | 288 MB |
| 32,000 | 31,999/32,000 | 31,999/32,000 | 1.18 GB |

Text and token ids identical in every case. The one-token shortfall is constant - llama.cpp
always reprocesses the final token - so reuse *improves* with prefix length, from 99.2% to
99.997%.

**Logprobs are not identical, and a bare "they differ" would say nothing.** The record
decomposes it into three comparisons measured on the same run:

| comparison | 128 | 8K | 32K |
|---|---:|---:|---:|
| own restore vs own cold prefill | 0.05-0.14 | 0.17-0.75 | 0.07-0.20 |
| foreign cache vs own cache | 0.37-0.40 | 0.44-0.49 | 1.23-1.30 |
| two cold runs, different backends, no cache | 0.375 | 0.783 | 1.163 |

The second row tracks the third at every length. **Moving a cache across backends costs about
what the two backends already differ by doing identical work with no cache involved**, which
is the only comparison that means anything - and neither number means anything alone.

## 18. Artifact size, derived rather than fitted

A fitted law reproduces its own points by construction. This one is computed from the writer:

```
12                    magic, version, n_token_count
+ 4  x header_tokens  the prompt's token ids, four bytes each
+ 8                   n_stream, cell_count
+ 12 x cells          pos, n_seq_id, one seq_id   (+12 more when cell_ext is written)
+ 8                   v_trans, n_layer
+ n_layer x (24 + cells x (k_row + v_row))
```

Exact to the byte on seven artifacts across three architectures, including a 32,000-token file
predicted from terms checked only to 8,192, and a 2,048-token hybrid artifact this project did
not produce.

**The term that had been missing is four bytes per token for the prompt's token ids in the
header.** Arithmetic over the documented per-cell fields came out exactly 4 bytes/token short
of the measured slope. Rather than absorb that into a constant, a real artifact was decoded
and the term found. Invisible at 128 tokens; 128 KB adrift at 32,000.

Two silent-wrong bugs surfaced doing this. Deriving `head_dim` as `embedding_length /
head_count` gives 5120/24 = 213.33 on qwen35, floored to **213** against a declared
`key_length` of **256** - an estimate wrong by a fifth and entirely plausible. And
`state_bytes` claimed in its docstring to refuse hybrid models while checking nothing, so a
caller would have taken an attention-only figure for a whole file.

## 19. The hybrid state file, decoded

A 256-token Qwen3.8-27B artifact closes to **zero leftover bytes**: 16,783,760 for attention
(16 layers), 156,894,356 for recurrent (48 layers of 64 declared), 1,040 of header. Both
recurrent row sizes fall out of the GGUF's SSM metadata - conv state
`(d_conv-1) x (d_inner + 2 x n_group x d_state) x 4`, SSM state `d_inner x d_state x 4`.

The attention section holds **16** layers where the recurrence rule implies 17: the NextN/MTP
block is marked non-recurrent but is not in the KV cache at all.

**The recurrent section does not begin with `n_stream`.** `llama_memory_recurrent::state_write`
writes `cell_count` first, unlike the attention writer. The layout inventory said otherwise,
the decoder matched the inventory, and the fixture matched the decoder - three artefacts
agreeing with each other and none with llama.cpp. Eight tests failed once the parser was
corrected, which is the informative part: they had been green against a body no writer
produces. Same shape as the 12-versus-16-byte checkpoint record, found the same way.

The recurrent tail is **byte-identical** between a 256- and a 257-token file. It is 90.3% of a
short artifact and 6.8% of a 32K one, which is why a per-token rate taken from a hybrid
artifact over-predicts so badly.

## 20. Cross-model translation, attempted and rejected

Tried on the most favourable pair the host offers: `qwen35` (16 attention layers, 4 KV heads)
to `qwen35moe` (10 layers, 2 KV heads). Every semantic axis matches - head_dim 256, d_state
128, d_conv 4, n_group 16 - and they share a tokenizer exactly, so no alignment error is even
possible.

The fit: 15,981 tokens from 8 varied passages, held out **by whole prompt**. Median held-out
R² **0.55**, none above 0.9. The chosen source layers came out near-monotonic (1, 3, 5, 6, 7,
10, 8, 11, 12, 15) without being asked to, which is real structure.

**An earlier split by token rather than by prompt reported 0.98.** Adjacent tokens share
context; that number was leaked and it flattered the result by nearly a factor of two on the
early layers.

The gate, with translated attention spliced into the target's own artifact so its recurrent
state stayed exactly as the model wrote it:

| | reused | top-1 | max delta |
|---|---:|---:|---:|
| identity control | 764/768 | 1.00 | 0.000 |
| **translated** | 764/768 | **0.00** | 2.42 |
| noise control | 764/768 | 0.00 | 7.99 |

```
cold/identity : " had unknowingly assembled the longest continuous record of coastal fo"
translated    : ", having spent his life reading the sea, had learned to"
```

Fluent, grammatical, on topic, and disagreeing on **every** generated token. Nobody reading
that sentence would suspect the cache.

**Two runs before this one were invalid**, and the noise control is the only reason it was
caught. The first had `--ctx-checkpoints 0` set - disabling the feature the patched build
exists to provide - so on a hybrid model nothing was restored and all three variants were cold
prefills producing identical output. Three identical outputs *including noise* is impossible
if anything was restored. The identity control is now a hard precondition.

## 21. Eight iterations to a gate that could be trusted

Establishing that the gate itself was sound took longer than building it, and corrected itself
three times.

| stage | belief | what broke it |
|---|---|---|
| first result | translation fails | - |
| threshold | needs R² ~0.85 | measured on 12 generated tokens |
| sharpened | ~0.96, R² understates it | 48 tokens is a strictly harder test |
| predictor | mean\|Δ\| should grade maps offline | ranges **overlap**: 0.667 passing, 0.171 failing |
| chaos | the gate is near-random at the boundary | survey passed at 0.9, failed at 0.8, **passed at 0.65** |
| explanation | each prompt has one fragile token | first-divergence index is fixed per prompt |
| mechanism | that token is the least confident one | 4/4, three of them the minimum of 48 |
| settled | teacher-forced scoring | free generation cannot be scored past divergence |

The mechanism is the useful part. **The first-divergence position is a property of the prompt,
not of the perturbation** - survey diverges at token 22 for every blend ratio from 0.85 down to
0.3 - and that token is where the model is least sure, at margins 0.18-0.51 against medians of
5.3-7.9. So "did the generation match?" was really asking "did you survive this prompt's
coin-flip token?", which is a fact about the prompt.

**One claim from that sequence did not survive re-testing.** Skipping near-tied positions was
said to turn a wobbling number into an ordered one; sweeping the margin bar offline over a
second grid showed every bar monotonic *including no skipping at all*, and skipping nothing
gave the fewest prompt disagreements. Only one or two positions of thirty-two fall below the
bar, so the effect is the same size as the differences being compared. The reporting is kept
because the principle is sound; no claim rests on it.

What does hold: **teacher forcing is necessary**, and the gate separates the translation from
every blend at 0.733 and 0.903 against 1.000.

## 22. The pipeline end to end, and what restoring is worth

On `Tiel-Coder-35B-A3B`, the model the operator actually runs:

```
admitted   e1dbb7a52cd9   needs patch: True   kv f16/f16   state v3
RESTORE    cache_n=508 of 512, reuse verified
patched runtime : no objection
STOCK runtime   : 2 objections -> refused
```

The stock runtime is the fleet's own binary. Offered the same artifact it is refused, because
on that build the restore would be *accepted* and reuse nothing while reporting the same
`n_restored` as the patched one - measured at **252/256 against 0/256**.

Latency, three repetitions, medians, total counted as the restore call plus whatever the
runtime still prefills:

| tokens | artifact | cold | restore total | speedup |
|---:|---:|---:|---:|---:|
| 512 | 135.6 MB | 443.8 ms | 246.9 ms | 1.80x |
| 2,048 | 228.5 MB | 1,643.7 ms | 385.4 ms | 4.26x |
| 8,192 | 348.7 MB | 6,029.9 ms | 410.9 ms | **14.68x** |

**Cold prefill grows linearly; total restore stays nearly flat.** The advantage widens with
prefix length, which is the direction an agentic harness pushes. Artifacts here live in RAM,
so read time is a floor; charging a notional NVMe read at 2 GB/s still leaves 1.40x, 3.25x and
10.16x, and that is arithmetic rather than a measurement.

**Two defects only the live run found.** The store must *be* the slot-save-path, since restore
hands the server a filename with no copy on the request path. And model identity must be
supplied by the caller, because llama.cpp puts none in `/props` - the check refused a restore
that then succeeded with 508 of 512 reused. That one survived every offline test because the
fixtures supplied an identity real props do not have.

## 23. Where the goal stands

Two halves, opposite verdicts.

**Proven.** One artifact across ROCm, Vulkan and a full process restart, for one model, at up
to 32,000 tokens with identical output. Hybrid models cacheable at all. Sizing derived and
exact on three architectures. Restore 1.8x to 14.7x cheaper than recomputing.

**Not supported.** One artifact across *different models*. On the best pair available, with an
identical tokenizer, a linear map reaches R² 0.55 and the gate rejects it - and the translated
cache diverges from the target's own output within six tokens while writing fluent English.

A linear map is the floor rather than the ceiling, and the harness to evaluate a better one now
exists and is trustworthy. But nothing measured here supports admitting a translated cache, and
describing the format as model-agnostic today would be claiming the half that failed.

---

## 24. Nondeterminism is a hybrid-on-Vulkan interaction, not a Vulkan property

**Corrected.** This section first claimed that "the Vulkan backend produces different text for
identical work". That was too broad, and the follow-up measurement that would have supported it
refuted it instead. The nondeterminism is specific to the **hybrid `qwen35` model on Vulkan**;
Vulkan running a dense `qwen2` model on the same card is perfectly deterministic. The original
table is kept below because it is correct as far as it goes — it just does not license the
general claim that was drawn from it.

Same model, same flags, same two cards, same source revision (`ca3d5a3e1`), temperature 0,
fixed seed, slot erased and displaced between every run:

| backend | distinct outputs across 6 identical cold runs |
|---|---:|
| CUDA, 2x RTX 3090 | **1** (deterministic) |
| Vulkan, the same 2x RTX 3090 | **3** |

One of Vulkan's three outputs is byte-identical to CUDA's single output, so this is not a
systematic backend offset — it is genuine run-to-run variation.

### Why this matters more than it first looks

The acceptance harness treats *identical text and identical token ids* as evidence that a
restored cache reproduced the uncached result. On a backend that is not reproducible against
itself, that check can fail for reasons that have nothing to do with the cache, or pass by
luck. It is only as strong as the determinism of the configuration it runs on, and that
determinism has to be **measured, not assumed** — which it had not been before this.

This does not retroactively invalidate §17's HIP <-> Vulkan results. Those used a different
model on AMD cards and had real prefix reuse (`cache_n` in the thousands), so most of the
computation came from cache rather than being recomputed. But it does open a question that
section cannot answer from its own data: whether AMD Vulkan is deterministic under the flags
used there. Untested.

### Both hypotheses were tested, and both were wrong

Batch composition under four slots and `kv_unified=true` looked like the obvious mechanism, and
flash attention the next candidate. Neither survived:

| Vulkan configuration, `qwen35` 27B | distinct outputs across 6 cold runs |
|---|---:|
| 4 slots, `kv_unified=true` (original) | 3 |
| `--parallel 1` | 3 |
| `--parallel 1 -fa on` | 3 |

The three configurations do not merely all fail — they produce the *same three output hashes*,
so this is a small set of outcomes from a specific nondeterministic reduction rather than
general noise. Nothing about how the server is driven changed it.

### Isolated: it is Vulkan x hybrid, and nothing else

Varying the backend and the model instead, all at `ca3d5a3e1`, 6 identical cold runs each:

| model | CUDA (NVIDIA) | HIP (AMD) | Vulkan (NVIDIA) | Vulkan (AMD) |
|---|---:|---:|---:|---:|
| `qwen35` 27B, hybrid | 1 | 1 | **3** | **3** |
| `qwen2` 3B, dense | — | — | 1 | 1 |

Both vendor-native APIs are deterministic with the hybrid model. Vulkan is nondeterministic
with it **on both vendors**, and deterministic with a dense model **on both vendors**. Two of
AMD Vulkan's three output hashes are byte-identical to two of NVIDIA Vulkan's, so the same
small set of reduction orderings appears on unrelated hardware — which points at the Vulkan
recurrent-path kernels rather than at any driver or card.

So the mechanism is the recurrent/hybrid path under the Vulkan backend specifically: not slot
scheduling, not flash attention, not Vulkan in general, and not "everything except CUDA".
**§17's HIP <-> Vulkan evidence is not threatened**: it used a dense model, the case measured
deterministic here on both vendors.

### Re-measured with retained records, and the text was under-reporting it

The table above counts distinct output *hashes*. A summary count is not admission-quality
evidence, so `scripts/reader_determinism.py` re-ran all six configurations retaining every
run's raw token ids, text, per-position probability vectors, slot routing, launch record,
binary and library digests, and model/prompt digests. Records in
`docs/records/reader-determinism/`, all at build `b151-ca3d5a3e1` on one prompt digest:

| configuration | distinct texts | distinct token seqs | **distinct probability vectors** | reproducible |
|---|---:|---:|---:|:--:|
| cuda-nvidia-hybrid | 1 | 1 | 1 | yes |
| hip-amd-hybrid | 1 | 1 | 1 | yes |
| vulkan-nvidia-hybrid | 3 | 3 | **6** | no |
| vulkan-amd-hybrid | 3 | 3 | **6** | no |
| vulkan-nvidia-dense | 1 | 1 | 1 | yes |
| vulkan-amd-dense | 1 | 1 | 1 | yes |

**Every one of the six Vulkan hybrid runs has a different distribution.** The text collapses
them into three groups, so counting outputs under-reported the instability by half — on both
vendors, identically. The reproducible configurations are reproducible all the way down: one
text, one token sequence, one set of vectors.

That is the concrete argument for retaining vectors rather than a count. Had the gate been
built on text agreement, a Vulkan hybrid reader could have produced the "right" text from a
visibly different distribution, and the gate would have passed it.

The runner fails closed on its own inputs: a run that reused cache is not a cold sample, a run
with empty probability vectors compares equal to any other such run, and a set spread across
slots is not six repetitions of one configuration. Any of these refuses the whole set rather
than dropping a run from it. 7/7 of its guards are defended by a test, and 13 tests cover the
refusals without needing a GPU.

The methodological point survives the correction intact, and is the durable lesson: text
identity is only evidence when the configuration producing it has been shown to be
reproducible. That had never been checked. It now has been, and it is true for every dense
configuration tested and false for one hybrid one.

### How the finding was nearly missed, and then nearly reported backwards

The first observation was that Vulkan's output *after restoring a CUDA-written cache* differed
from Vulkan's own cold run, which reads as "the restore perturbed the model despite reusing
nothing" — an alarming and publishable-sounding claim. Running the cold case twice killed it:
the two cold runs differed from each other too. The restore was never implicated. Only the
control separated a property of the backend from a property of the cache.

Scope: this host, `ca3d5a3e1`, Qwen3.8-27B UD-Q4_K_XL (`qwen35`), `-c 8192`, 4 slots, 430-token
prompt, 24 predicted tokens.

## 25. Hybrid on CUDA: the same architectural refusal, on a backend where it was untested

`qwen35` hybrid restore had been measured on ROCm and Vulkan but never on CUDA. It behaves
identically, and it is a fresh instance of the rule that `n_restored` is not evidence:

```
save    : n_saved=453, n_written=186,595,276
restore : n_restored=453, n_read=186,595,276    <- reports complete success
next    : prompt_n=430,  cache_n=0              <- reuses nothing, re-prefills every token
```

The state file carries no `SCKP` appendix, so this build is unpatched, and the failure is the
documented one: a recurrent state is a function of the whole processed sequence and has no
common-prefix semantics. **The failure is architectural, not backend-specific** — three
backends now, one behaviour.

A first attempt at this measurement reported `n_saved=0` and a 1,200-byte file, which looks
exactly like the unpatched signature and would have been an easy thing to write down. It was
slot routing: the completion had landed on a different slot than the one being saved. Pinning
`id_slot` gave 453 cells and 186 MB.

### The container itself is backend-portable, even when the cache is useless

The 186 MB file written by the CUDA build was handed to the Vulkan build and **accepted**:
`n_restored=453`, `n_read` equal to the full file. Reuse was still 0, for the architectural
reason above. So format portability across backends and *usefulness* of the restored state are
independent properties, and the first one holds here while the second does not.

---

## 26. The vendor/API matrix closes: API isolated, vendor isolated

The cell that was missing. §17 proved HIP <-> Vulkan (API varies, vendor fixed to AMD) and
CUDA <-> ROCm (both vary). Neither isolated the API: no result held the *card* fixed and changed
only the compute API. This does, on a single RTX 3090, both binaries at `ca3d5a3e1`:

| direction | reused | content | tokens | artifact |
|---|---|---|---|---:|
| CUDA -> Vulkan | 2047/2048 (100.0%) | match | match | 75.5 MB |
| Vulkan -> CUDA | 2047/2048 (100.0%) | match | match | 75.5 MB |

Qwen2.5-3B-Instruct Q4_K_M (`qwen2`), 2,048-token prompt, one slot. The one-token shortfall is
the constant llama.cpp behaviour of reprocessing the final token, not a loss.

The three-way logprob decomposition, which is the only framing in which these numbers mean
anything:

| comparison (max abs logprob delta) | CUDA->Vulkan | Vulkan->CUDA |
|---|---:|---:|
| own restore vs own cold prefill | 0.077 | 0.350 |
| **foreign cache vs own cache** | **0.353** | **0.456** |
| two cold runs, different backends, no cache | 0.976 | 0.976 |

Top-1 agreement is 1.0 in every comparison. **Moving the cache between APIs costs less than the
two APIs already differ by doing identical work with no cache involved** — 0.353 and 0.456
against a 0.976 floor. That is the same pattern §17 found across vendors, now with the hardware
held constant so the API is the only variable.

### A wrong reference nearly turned this into a failure

Measured by hand first, a Vulkan-captured cache restored on CUDA produced text differing from
CUDA's own cold prefill, which reads as a failed transfer. It is not: reusing 3,087
Vulkan-computed KV rows and prefilling 3,088 tokens on CUDA are different computations, and
comparing them answers a question nobody asked. The retained harness compares the restored run
against the *source backend's* run and reports the decomposition above, and under that
reference both directions match exactly. The lesson is not that the manual check was sloppy but
that "different from what?" is the whole content of a claim like this one.

### Same API, different vendor

The complementary cell, run with the same binary so only the vendor changes:

| direction | reused | content | tokens |
|---|---|---|---|
| Vulkan/NVIDIA -> Vulkan/AMD | 2047/2048 (100.0%) | match | match |
| Vulkan/AMD -> Vulkan/NVIDIA | 2047/2048 (100.0%) | match | match |

| comparison (max abs logprob delta) | NV->AMD | AMD->NV |
|---|---:|---:|
| own restore vs own cold prefill | 0.174 | 0.077 |
| foreign cache vs own cache | 0.974 | 0.965 |
| two cold runs, different backends, no cache | 0.960 | 0.960 |

Top-1 agreement 1.0 throughout. Together with §26's first table the two variables are now
separated: **holding the card fixed and changing the API costs 0.353-0.456 against a 0.976
floor; holding the API fixed and changing the vendor costs 0.965-0.974 against a 0.960 floor.**
Crossing vendors costs about exactly what the vendors already differ by; crossing APIs on one
card costs distinctly less than the APIs differ by. Neither degrades top-1 agreement or the
emitted tokens at 2,048 tokens.

---

## 27. A rejected restore leaves the slot pristine — measured, not inferred from the 400

The steer's P-1: llama.cpp answers a wrong-model restore with HTTP 400, but it logs work done
on the way to that 400, and no test had ever issued a completion afterwards to see what the
slot looked like. A refusal is not safety until something checks.

`scripts/slot_poisoning.py` takes a baseline cold completion, offers the slot a state file
written by a *different model*, and — without erasing, which would destroy the evidence — runs
the identical completion again. Text, token ids and per-position probability vectors must all
match the baseline.

It refuses to render a verdict unless handed a passing `reader_determinism` record **for that
exact reader label**. On a reader that answers identical work differently, a post-rejection
difference cannot be attributed to the restore rather than to the reader, and the verdict would
be noise wearing a safety label. Demonstrated in the field, not just in tests: pointed at
`vulkan-nvidia-hybrid`, it refuses before starting a server.

| target (CUDA, reproducible) | foreign artifact | rejected at | slot pristine |
|---|---|---|---|
| Qwen3.8-27B (`qwen35`) | Qwen2.5-3B (`qwen2`), 443 KB | `state_read_meta: invalid seq_id-agnostic kv cell` | **yes** |
| Qwen3.8-27B (`qwen35`) | Qwen3.5-4B (`qwen35`), 53 MB | `state_read_data: mismatched layer count (8 instead of 16)` | **yes** |

Both returned HTTP 400. The second is the stronger case: the reader got past the metadata
section and into the data section before failing, so the rejection is not a cheap early-out —
and the slot was still byte-identical afterwards, vectors included. In both, the post-rejection
completion re-prefilled all 430 tokens (`cache_n=0`), which is what an untouched slot does.

**Limit, stated rather than glossed:** neither artifact carried an `SCKP` appendix, because
neither writer was a patched build. The specific checkpoint-appendix-before-rejection path the
steer described is therefore *not* exercised by these two runs. What is established is that a
model-mismatch rejection, including one that fails deep in the data section, leaves the slot
pristine on this reader. The appendix path needs a patched writer and remains untested.

---

## 28. Same-geometry cross-model reuse works — §20 answered a narrower question than it was quoted for

§20 is titled "cross-model translation, attempted and rejected" and has been cited, including
by me, as "caches do not move between models". It does not show that. It tested `qwen35`
(16 layers, 4 KV heads) into `qwen35moe` (10 layers, 2 KV heads) — a **geometry** change,
requiring a learned mapping, which is what failed. The case where two models already lay their
cache out identically was never run.

llama.cpp's state format records shape, not model identity, so that case needs no translation
at all: the file is simply accepted. Measured with `scripts/cross_model_gate.py`:

**Qwen2.5-3B-Instruct Q4_K_M writes → Qwen2.5-3B-Instruct Q8_0 reads**, 3,088-token prompt:

| | reused | top-1 vs target's own restore | text |
|---|---:|---:|---|
| identity (target reads its own cache) | 3087/3088 | — | reference |
| **foreign (other quantisation's cache)** | **3087/3088** | **1.00** | **byte-identical** |
| noise (same file, values scrambled) | 3087/3088 | 0.00 | `????????????????` |

The max logprob delta against the target's own restore is 1.03: the two agree on every emitted
token while differing in confidence behind it. Worth stating, because "identical output" and
"identical distribution" are not the same claim and only the first is established.

### Scored against the target's own restore, not against a cold prefill

A first version of this gate required the identity control to reproduce the *cold prefill*
output exactly, and failed the run when it did not. That was the wrong question. Restoring a
cache and prefilling a prompt are different computations, and here they disagree on 3 of 32
tokens **for a model reading its own cache** (`identity_top1_vs_native` 0.906). Scoring a
foreign cache against a cold prefill charges it for a difference that has nothing to do with
whose cache it is. The comparison that isolates the variable is foreign-restore against
own-restore — both restores, differing only in who wrote the bytes. This is the same row §17
found meaningful for cross-backend transfer.

### The noise control had to be rebuilt before any of this counted

The first noise control overwrote every byte after the token header, which destroyed the
per-layer ggml type ids and row sizes as well as the values. The server answered 400. **A
rejected noise control bounds nothing** — from outside it is indistinguishable from a control
that was never run, and it would have left a high foreign score with no floor under it. It now
scrambles only the tensor spans located by the repo's own GGSQ parser: 72 spans, 113.8 MB of
values replaced, 50,316 structural bytes preserved. The rebuilt control restores, reuses 3,087
tokens, and emits `????????????????`, which is what a floor is supposed to look like.

### The next rung: a genuinely different fine-tune

Qwen2.5-3B **base** writes, Qwen2.5-3B-**Instruct** reads. Same architecture and tokenizer,
same 36 layers / 2 KV heads / head_dim 128 / theta 1e6, different weight files — full
instruction tuning, not rounding. 128 teacher-forced positions:

| pair | foreign top-1 vs target's own restore | max Δ | free-running text |
|---|---:|---:|---|
| Q4_K_M → Q8_0 (rounding only) | **0.984** (126/128) | 1.03 | identical at 32 tokens |
| **base → Instruct (fine-tune)** | **0.930** (119/128) | 3.56 | diverges |
| noise floor (both) | 0.000 | — | `????????` |

**Same geometry is necessary but not sufficient, and the gap is measurable.** Quantisation
variants transfer at or above this machine's own reproducibility floor. A real fine-tune
degrades: 9 of 128 next-token predictions change, and the logprob delta triples.

### The threshold was stricter than the machine

The gate shipped with `min_top1 = 0.99`. The identity leg — the same model, same weights, same
prompt, differing only by restore-versus-prefill arithmetic — agrees on **0.969 to 0.977** of
128 teacher-forced positions. So 0.99 was unreachable *for a model reading its own cache*, and
nothing could ever have passed it for reasons having anything to do with the cache.

The verdict now reports `baseline_top1` and `at_or_above_baseline` next to the absolute number,
and flags `threshold_exceeds_baseline`. An absolute parity threshold is the wrong instrument
when the identity case cannot reach it.

### Free generation measured the cascade, not the cache

The first version of this gate scored freely generated tokens. At 32 tokens it reported
attractive numbers; at 128 it reported that a model restoring its **own** cache agreed with its
own cold prefill on **0.23** of positions. That is the autoregressive cliff — one divergent
token changes the next input and everything after it — and it says nothing about the cache.
`gate.py` already declares `teacher_forced` as its default scoring protocol for exactly this
reason, and this runner was not using it. Under teacher forcing the same identity case scores
0.977.

---

## 29. Where same-geometry caches differ, and why the converter is the identity

The obvious next question after §28: the fine-tune transfers at 0.930 and the quantisation
variant at 0.984, so what is the 0.054 made of, and can a converter remove it? Three state
files over byte-identical token ids, decoded to canonical `(token, head, dim)` float32 and
compared per layer.

| pair | K cosine | K rel err | K norm ratio | V cosine | V rel err | V norm ratio |
|---|---:|---:|---:|---:|---:|---:|
| instruct Q4 vs instruct Q8 (rounding) | 0.9972 | 0.079 | 1.0027 | 0.9833 | 0.186 | 1.0060 |
| instruct Q4 vs base Q4 (fine-tune) | 0.9909 | 0.143 | 1.0003 | 0.9542 | 0.305 | 0.9831 |

Medians over 36 layers. Three facts fall straight out:

1. **It is not scale.** Norm ratios are 1.000 to three decimals in every case, so the
   per-layer scalar that fixed the cross-*geometry* magnitude problem has nothing to do here.
2. **V is roughly twice as far off as K**, on both pairs. Values carry the divergence.
3. **It accumulates with depth.** Layer 0 is essentially identical (cosine 1.0000, relative
   error 0.003) and the error grows monotonically into the twenties before easing slightly.
   That is drift through the residual stream, not a fixed per-layer transform.

### A fitted linear converter is worse than doing nothing, at every layer

Per-head ridge regression, 128->128 per head, fitted on four prompts and scored on two the fit
never saw. Split by whole prompt, because §20 measured what a token-level split costs.

| | fitted R² (median) | do-nothing baseline | gain |
|---|---:|---:|---:|
| K | 0.911 | 0.948 | **−0.027** |
| V | 0.631 | 0.890 | **−0.174** |

**Zero of 36 layers show a positive gain, for K or V.** Layer 0 is the sharpest illustration:
the raw cache already scores 0.998, and fitting drops it to 0.826 (K) and 0.488 (V). The fit
destroys a near-perfect match.

Ruled out as a hyperparameter artifact — a ridge sweep across six orders of magnitude never
reaches parity, and improves monotonically as regularisation increases:

```
K   ridge 1e-6  -0.0595      V   ridge 1e-6  -0.2806
    ridge 1e-2  -0.0517          ridge 1e-2  -0.2803
    ridge 1.0   -0.0230          ridge 1.0   -0.2665
    ridge 10    -0.0193          ridge 10    -0.2457
```

Stronger regularisation is always better, and stronger regularisation means *closer to the
identity*. **The optimal linear converter between two same-geometry models is the identity
map** — which is exactly what raw transfer already does. There is no linear structure in the
residual for a converter to capture: it is the target's own weight drift, and the source cache
carries no information about it.

### What this means for "one file for everything"

The goal decomposes into two problems with opposite answers.

**Same geometry needs no converter.** The file is accepted as-is and works (§28). Building a
linear translator here is not merely unnecessary, it is actively harmful, and this section is
the measurement that says so rather than an argument that it should be.

**Different geometry needs one and none has been found.** §20 stands: a learned map across
differing layer and head counts reached 0.00 top-1 agreement.

### The lead this opens

Early layers are nearly identical between fine-tunes and late layers are not. That suggests a
*partial-depth* reuse — take the foreign cache for the layers where it agrees and recompute
only the layers where it does not — which would trade a fraction of the prefill for correctness
rather than trading correctness for all of it. Untested, and it is the next thing worth running.

---

## 30. Partial-depth reuse: the lead §29 opened, and it does not pay

§29 ended by proposing that since early layers agree between fine-tunes and late layers do
not, a cache need not be used whole — take the foreign layers where they agree and recompute
the rest. It was the most promising idea on the table. It does not work, and the shape of the
failure is worth more than the idea was.

`scripts/kvx_splice.py` builds hybrid state files: named layers from the foreign cache, every
other layer from the target's own, spliced byte-for-byte after checking the two payloads are
laid out identically field by field. Zero layers reproduces the target's own file and all 36
reproduces the foreign one, so the sweep is anchored at both ends.

Qwen2.5-3B base into Qwen2.5-3B-Instruct, 128 teacher-forced positions, scored against the
target reading its own cache:

| foreign layers | early end Δ | late end Δ | early as % of full | layers as % of stack |
|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | 0% | 0% |
| 2 | 0.0615 | 0.0704 | **23%** | **6%** |
| 8 | 0.0869 | 0.1308 | 33% | 22% |
| 16 | 0.1436 | 0.2023 | 55% | 44% |
| 20 | 0.1445 | 0.2111 | 55% | 56% |
| 28 | 0.2328 | 0.2544 | 88% | 78% |
| 36 | 0.2637 | 0.2637 | 100% | 100% |

Two readings, and the second is the fatal one.

**Deep layers do cost more than shallow ones.** Taking foreign layers from the top of the stack
is worse than taking the same number from the bottom, at every count — 0.153 against 0.090 at
ten layers. So §29's depth reading was right about direction.

**But there is no cheap subset.** Past the first few layers the cost tracks the layer count
almost exactly (55% of the damage for 56% of the layers), and the first two layers alone cost
23% of the total damage for 6% of the stack. The curve is *worse* than proportional, not
better. A knee is what would have made this worth building, and there is no knee.

### The entry cost is not simply "mixing"

Two layers from anywhere cost 0.06–0.09, which looked like a fixed penalty for a cache of mixed
provenance — something neither model would ever produce. A control says it is not that simple:

| two foreign layers taken from | mean Δ |
|---|---:|
| bottom (0, 1) | 0.0615 |
| **middle (17, 18)** | **0.0905** |
| top (34, 35) | 0.0704 |
| scattered (0, 17, 35) — three layers | 0.0925 |

Position matters — the middle is the worst place to swap, not the ends — so it is not a
constant. But every choice of two layers lands in the same band, and that band is already a
quarter of the damage of swapping all thirty-six. Whatever the mechanism, the practical
consequence stands: you cannot buy a small amount of foreign cache cheaply.

### Where this leaves the goal

Reusing a fraction *f* of the layers costs roughly *f* of the full penalty plus a substantial
entry fee, and requires a runtime that can restore some layers and recompute others — which
llama.cpp cannot do. A proportional benefit does not justify that, so this line is closed.

Standing back, the three attempts now agree on one thing. §20 could not translate across
geometries. §29 found the optimal same-geometry converter is the identity. §30 finds no
subset of layers is cheap. All three fail for the same reason: **the difference between two
models' caches is the target's own weight drift, and the source cache contains no information
about it.** No function of the source recovers it, whether linear, per-layer, or partial.
