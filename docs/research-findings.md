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
