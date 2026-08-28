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

## Current executable slice

```bash
python3 -m unittest discover -s tests -v
python3 kv_rosetta.py validate cache.kvx.json
python3 kv_rosetta.py fingerprint cache.kvx.json
python3 kv_rosetta.py compat source.kvx.json target.kvx.json
```

This branch defines and checks the portable boundary. Runtime cache extraction,
injection, mapping kernels, and storage are deliberately next: neither llama.cpp
nor vLLM currently exposes a common stable cache-import ABI, so pretending that
part exists would make the format unverifiable.

## Next implementation order

1. llama.cpp KVX export/import hooks, exercised on CUDA and HIP with one model.
2. Exact CUDA-to-HIP round trip with token-for-token logit parity.
3. Qwen-family matched-KV linear mapper and quality gate.
4. vLLM paged-cache adapter.
5. Heterogeneous Qwen27B-to-Tiel learned mapper.
