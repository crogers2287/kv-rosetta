# re-align.md — questions back to the steering model

The steer branch tells me what to do next. This file is the return path: where I record
where I think the work has drifted from the goal, and ask for the steer to be corrected.

I write here when I believe following the steer literally would produce good engineering
aimed at the wrong target. Being on-spec is not the same as being on-goal, and a steer that
is internally consistent can still be pointed away from what the project is for.

## Protocol

- Each entry gets an `RA-NNN` id, a date, and a status: **open**, **answered**, or
  **withdrawn**.
- An entry states the goal it is measuring against, what the steer is actually driving,
  the evidence for the gap, and a specific question. Not a complaint — a question with
  enough evidence attached to be answerable.
- **Reviewer: please answer open entries by updating `steer.md` on the steer branch**, and
  reference the `RA-NNN` id so I can mark it answered here. If a concern is wrong, saying
  so in the steer is a complete answer and I will withdraw the entry.
- I keep executing the current steer while an entry is open. Raising a question is not a
  reason to stop working, and I will not silently substitute my own plan for the steer's.

---

## RA-001 — The steers have optimised T1 while the product goal is T2/T3 · **open** · 2026-08-29

**Goal, as stated by the project owner today:**

> "the goal was to make the same cache file work across models and arches. like cuda and
> rdna2 or rocm. and vulkan seamlessly. the sidecar should have built in transformers so
> all models and arches can use the same kv storage files"
>
> "the goal is to make it so that when users are using things like agentic harnesses the
> heavy system prompts/skills/mcps/tools etc can all be cached instead of prefilled every
> time."

**What the steer track has driven:** roughly fourteen consecutive steers of single-runtime,
single-model, opaque-blob hardening on one host — checkpoint persistence, fail-closed
identity, an admitted store, and request-path economics. That work is correct and I stand
behind it: a cache that cannot survive a restart on one machine cannot be ported to another,
and we found a real blocker there (hybrid context checkpoints, upstream #25913).

**But the current adapter is built to refuse exactly what the goal asks for.** This is by
design in the steers, not an oversight on my part:

- `CacheABIIdentity` binds K/V cache dtype, so a cache written under f16 is refused by a
  q8_0 runtime — the CUDA fleet runs `kvarn4/kvarn6` while the W6800 runs `q8_0`, so today
  those two machines can never share a file.
- The compound-tuple allowlist admits exactly `ggsq/3+sckp/1`, one runtime's private format.
- The admitted store is explicitly local-only and pins device and inode.

**Module evidence for the gap** (checked, not recalled):

| needed for the stated goal | status |
|---|---|
| `adapters/llamacpp_ggsq.py` — decode opaque blob → canonical tensors | **missing** |
| `mappers/linear.py`, `learned.py`, `align.py` — the "built-in transformers" | **missing** |
| `adapters/hf.py` — transformers reference, ground truth for logits | **missing** |
| `adapters/vllm_connector.py` | **missing** |
| `transfer.py` — plan → export → map → import → gate → fallback | **missing** |
| `daemon/server.py` — the sidecar a harness would actually call | **missing** |
| `container.py`, `segments.py`, `dtypes.py` — canonical format substrate | present |
| `gate.py`, `metrics.py` — the admission gate | present |
| `mappers/rope.py` | present |

So the format and the safety machinery exist. The conversion layer — the part that makes one
file work on CUDA, ROCm and Vulkan — is essentially unbuilt.

**Questions for the steer:**

1. Is cross-backend portability still in scope? If yes, `adapters/llamacpp_ggsq.py` (decode
   GGSQ → canonical) is the smallest unlock: llama.cpp's state layout is backend-independent
   by source review — it stores ggml type ids and row strides, not device buffers — so the
   blocker is the cache-config mismatch, which canonical decode is what survives.
2. Is `daemon/server.py` in scope? Without it none of this is reachable by a harness. There
   is no service; `watcher.py`, `budget.py` and `resolve.py` exist but nothing serves them.
3. If both are out of scope for now, please say so in the steer so I stop treating them as
   the goal, and I will withdraw this entry.

---

## RA-002 — We have measured at 256 and 2048 tokens; the use case lives at 8K–32K · **open** · 2026-08-29

An agentic harness prefix — system prompt plus skills plus MCP tool schemas — is typically
8K–32K tokens. Every economic number retained so far is from 256 or 2048.

**Measured on this host, same model:**

| prompt | cold prefill | runtime restore | ratio |
|---|---:|---:|---:|
| 256 | 0.595 s | 0.238 s | 2.5× |
| 2048 | 1.731 s | 0.412 s | 4.2× |

Over an 8× token increase, prefill grew 2.9× and restore grew 1.7×. Two points is a trend,
not a law — I am not projecting a number — but it points the same way the goal does: the
advantage widens with prefix length, and we have never measured the length that matters.

The steer currently gates 8K behind the state-cold sensitivity run and a recorded space
budget, which is reasonable. One hard constraint found while sizing it: the 2K artifact is
605 MB, so 8K ≈ 2.4 GB and 32K ≈ 9.7 GB. Admission copies the file, so 32K needs ~19 GB
transient against **16.6 GB free** on the NVMe target. **32K will not fit there** — it needs
tmpfs or a different device, and that should be decided before the rung is ordered rather
than discovered mid-run.

Also worth recording against the goal: the 16-restore break-even that looked like a caveat is
not one here. A harness restores the same prefix on every request, so it clears that
threshold within minutes.

**Question for the steer:** should the 8K rung be reprioritised as the first measurement that
speaks to the actual use case, and where should the 32K artifact live given the space limit?

---

## Answered

_None yet._
