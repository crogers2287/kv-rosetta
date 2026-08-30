# KV Rosetta

Portable KV-cache exchange across model families, inference runtimes, and
hardware backends.

KV Rosetta separates three things that serving stacks usually conflate:

1. **Prompt corpus** — canonical messages and tool schemas.
2. **KVX artifact** — device-neutral cache tensors plus exact model/token metadata.
3. **Adapter** — runtime-specific export/import for llama.cpp, vLLM, or another engine.

```text
prompt corpus ──compile──> source-native KV
                              │
                 exact reuse ─┼──────────────> target runtime
                              │
                              └─translate──quality gate──> target-native KV
                                              │
                                              └─fail──> normal target prefill
```

## Hardware contract

KVX uses a canonical little-endian `layer,kv,token,head,dim` layout. Hardware is
provenance, not identity: `cuda:sm_86`, `hip:gfx1030`, ROCm, CPU, and future
devices may exchange the same artifact. Each runtime adapter is responsible for:

- exporting its native paged/blocked cache into KVX;
- importing KVX into a target slot;
- converting dtype and layout efficiently on its own device;
- reporting whether zero-copy, device conversion, or host staging was used.

ROCm is the software stack, HIP is its programming/runtime layer, and RDNA/gfx1030
is a device architecture. KV Rosetta records all three without coupling the core
format to any of them.

## Cross-model paths

| Pair | Transfer path |
|---|---|
| Same model ABI, different GPU backend | Exact KVX import |
| Shared tokenizer and matched KV heads/dimensions | RoPE-stripped per-head ridge mapper |
| Different layers, heads, dimensions, tokenizer, or architecture | Learned projector plus token alignment |
| Any pair failing its held-out quality gate | Native target prefill |

The linear path follows [Cross-Model KV Cache Transfer in LLM Families](https://arxiv.org/abs/2608.03893).
The heterogeneous learned path is compatible with ideas in [C2C](https://github.com/thu-nics/C2C).

No translated cache is trusted because its tensor reconstruction score looks
good. Admission requires held-out next-token divergence and task-quality gates;
the fallback is always a normal target prefill.

## What is proven, measured, and not

Kept honest deliberately: this section distinguishes what a retained test proves from what
was measured once from what remains untested. Details and raw records are in
[docs/research-findings.md](docs/research-findings.md) and `bench/*.json`.

### Proven by retained test

- **A real llama.cpp round trip.** Save, erase, cold-control, restore, and full-prefix
  reuse on a live `llama-server`, with the erase-and-rerun control that distinguishes a
  restore from a cache that was never dropped.
- **Reuse is verified, never assumed.** A restore count is not evidence: on hybrid models
  the server reports every cell restored and then reuses none of them. Import recovers the
  prompt token IDs from the artifact, probes the intended slot, and refuses unless
  `cache_n == L-1` and `prompt_n == 1` - full-prefix reuse, not merely positive `cache_n`,
  which a different prompt sharing a prefix also produces.
- **The admission gate cannot be fooled by tensor similarity.** A candidate with cosine
  0.999980 and top-1 agreement 0.000 is rejected; `tensor_cosine` is reported but never
  read by the admission logic.
- **Identity is content-derived.** Renaming or relocating weights preserves identity;
  changing one byte does not. Digests are cached against `(device, inode, size, mtime_ns)`,
  so a 1.8 GB model costs 6.6 s once and 1.1 ms thereafter.
- **Artifacts are keyed on composite identity**, never the proxy prefix fingerprint alone -
  one prefix legitimately has many artifacts that differ only in encoding, cache ABI,
  runtime revision, or mapper.
- **Conformance is not vacuous.** Adapters that always fail import, advertise an import
  they cannot demonstrate, claim an opaque format they never write, ignore the expected
  cache ABI, accept a corrupted artifact, or derive identity from a filename are each
  rejected for that specific flaw.
- **The verifier never raises.** 203 malformed artifacts - truncations, bad magic, absurd
  header lengths, random bytes - each return a reason.

### Measured on one host

With `q4_0` KV on a 3B, three repetitions per rung, medians with ranges, tmpfs and NVMe
reported separately:

| tokens | native prefill | total restore | speedup |
|---|---|---|---|
| 256 | 351 ms | 138 ms | 2.5x |
| 2048 | 1624 ms | 376 ms | 4.2x |
| 8192 | 6678 ms | 1163 ms | 5.6x |

Total restore latency is the user-visible figure: artifact read, integrity and identity
verification, runtime restore, and reuse verification.

Two results matter more than the speedup:

- **KV quantization decides the economics.** At `f16` the artifact is 36 KB/token and
  restore is *slower* than recomputing the prefill at every rung. At `q4_0` it is
  10.1 KB/token and restore wins everywhere.
- **Quantized KV is not token-exact, and that is the runtime's behaviour.** llama.cpp's own
  prompt-cache reuse diverges from a cold prefill at `q4_0`, with no artifact involved. A
  fresh prefill quantizes values it just computed; a reused cache reads back
  already-quantized ones. Restore reproduces whatever the runtime does - which is exactly
  why the gate exists, and why "exact" is a claim only `f16` earns.

### Known limits

- **Hybrid and recurrent architectures need a patched runtime, and it now exists.** llama.cpp's
  slot save omits the *context checkpoint* those models resume from (upstream issue
  ggml-org/llama.cpp#25913). Measured on a 35B-A3B hybrid: **252 of 256 tokens reused** on a
  build carrying the patch against **0 of 256** on a stock one - both reporting the same
  `n_restored`, which is why a restore count is not evidence. An artifact records what its
  runtime must provide and the sidecar refuses rather than restoring into a build that will
  silently reuse nothing.
- **An artifact carrying checkpoints is not portable to a stock build.** The same hazard
  shimquant states of its own format: files built with the patch need the patch. This is
  recorded in the artifact rather than discovered at restore.
- **Cross-model translation does not pass the gate.** See below. Nothing in this repository
  should be read as delivering a cache that moves between *models*.
- **Large-model economics are unmeasured** for non-hybrid models: every model above 3 GB on
  the development host is `qwen35`/`qwen35moe`.

## Using it

```bash
python3 -m unittest discover -s tests -v      # offline suite

python3 -m kv_rosetta.cli validate  cache.kvx
python3 -m kv_rosetta.cli fingerprint cache.kvx
python3 -m kv_rosetta.cli compat source.kvx target.kvx
python3 -m kv_rosetta.cli inspect   cache.kvx   # header and integrity, no tensor load
```

Live tests need a `llama-server` started with `--slot-save-path`:

```bash
KVX_LLAMA_URL=http://127.0.0.1:8781 KVX_LLAMA_SLOTS=/path/to/slots \
  python3 -m unittest tests.test_llamacpp_roundtrip -v
```

Benchmarks emit machine-readable records under `bench/`, carrying the repo commit, runtime
build, model and cache-ABI identities, token digest, and their own caveats:

```bash
python3 scripts/bench_restore.py --url http://127.0.0.1:8781 \
  --slots /dev/shm/kvx-slots/ --medium tmpfs --kv-type q4_0 --repeats 3
```

### Cross-backend: proven

One artifact, written on one compute backend and restored on another, both directions, on
llama.cpp builds pinned to the same source revision - a mismatch is rejected for the
state-file *format* and is indistinguishable from a backend failure, which cost a whole
result before the guard existed.

| prompt | ROCm/HIP <-> Vulkan | text | token ids |
|---|---|---|---|
| 128 | 127/128 reused | identical | identical |
| 8,192 | 8,191/8,192 | identical | identical |
| 32,000 | 31,999/32,000 | identical | identical |

Logprob vectors are *not* identical, and the record decomposes why rather than reporting a
bare boolean: restoring a foreign cache diverges by about what two cold runs on different
backends already diverge by with no cache involved at all.

### Artifact size: derived, not fitted

`kv_rosetta/sizing.py` computes an artifact's size from the GGUF and the state-file layout,
every term read off `llama-kv-cache.cpp`. Exact to the byte on seven artifacts across three
architectures - qwen2 dense, qwen35 hybrid, qwen35moe hybrid-MoE - including a 32,000-token
file predicted from terms checked only up to 8,192, and a 2,048-token hybrid artifact this
project did not produce. The space guard derives instead of scaling a rate, which it had been
over-predicting by 8x.

### Cross-model translation: measured, and rejected

Attempted on the most favourable pair available: qwen35 and qwen35moe, identical head_dim
256, identical `d_state` 128, and - as it turned out - an identical tokenizer, so no
alignment error is even possible.

| | |
|---|---|
| calibration | 15,981 tokens, 8 varied passages, held out by whole prompt |
| per-layer linear map | median held-out R2 **0.55**, none above 0.9 |
| gate, teacher-forced agreement | **0.733** and **0.903** against 1.000 for every blend |
| first divergence from the target's own output | within **6 tokens** |

The translated cache produces fluent, grammatical, on-topic English that the target model
would not have written. That is the failure this project exists to catch, and the gate caught
it.

Establishing that the gate could be trusted took eight iterations and corrected itself three
times: exact-match over a free generation turned out to be chaotic near the boundary, because
each prompt has a single near-tied token - the least confident position in the whole
generation on three prompts of four - and a perturbed cache flips it. Scoring is now
teacher-forced, so one early mistake cannot condemn every position after it.

## Next

The proven half - durable, cross-backend, hybrid-capable caching for one model - is what this
would ship. Cross-model translation remains research: a linear map has lost on the best pair
available, no cheap offline metric predicts admission, and the recurrent half of a hybrid
cache is not addressed by the map at all.
