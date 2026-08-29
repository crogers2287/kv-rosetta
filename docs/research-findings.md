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

**This is not a bug to fix.** A recurrent state is a function of the entire processed
sequence, so it has no common-prefix semantics: there is no prefix of it to match a prompt
against. `n_restored` equals the saved cell count and `cache_n` is still zero because the
restore genuinely succeeded and the reuse genuinely cannot happen.

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

The HTTP slot API exposes no way to save or restore context checkpoints, so this is not
something an adapter can work around. Supporting hybrid models over this seam would need
llama.cpp to include checkpoints in the sequence state, or to expose them separately. Until
then the honest position is the one the code now takes: probe the architecture and withhold
the capability.

The classification is therefore **proven by the runtime's own trace**, not inferred.
