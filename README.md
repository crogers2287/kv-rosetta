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

- **Hybrid and recurrent architectures cannot reuse a restored prefix on an unpatched
  runtime.** Not because their state is unrestorable - state after an exact token sequence is
  deterministic - but because llama.cpp's slot save omits the *context checkpoint* those
  models need, and its restore clears the checkpoint list. The runtime says so itself:
  *"forcing full prompt re-processing due to lack of cache data (likely due to SWA or
  hybrid/recurrent memory)"*. Upstream issue ggml-org/llama.cpp#25913. Until a patched build
  advertises checkpoint persistence, the adapter probes `general.architecture` and withholds
  the capability; the failure is retained as a live negative control.
- **Large-model economics are unmeasured.** Every model above 3 GB on the development host
  is `qwen35`/`qwen35moe`, all hybrid.
- **Not implemented:** canonical extraction of llama.cpp state, cross-backend transfer,
  the HF and vLLM adapters, and any cross-model mapping. Nothing here should be read as
  covering them.

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

## Next

Gated behind a large non-hybrid model, which the development host does not have: the
large-model q4 ladder, then 131K, then canonical extraction, then same-revision CUDA-to-HIP
transfer. Cross-model mapping comes after all of those, behind the quality gate.
