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

## RA-001 — The steers have optimised T1 while the product goal is T2/T3 · **answered** · 2026-08-29

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

## RA-002 — We have measured at 256 and 2048 tokens; the use case lives at 8K–32K · **answered** · 2026-08-29

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

## RA-003 — Artifact size is affine in tokens, not linear; the 32K refusal may be wrong · **withdrawn** · 2026-08-30

RA-002 was answered with "do not generate a 32K artifact on the current NVMe", accepting my
estimate of ~9.7 GiB. **That estimate assumed the artifact scales linearly with tokens. Two
measured points say it does not.**

| prompt | artifact bytes | bytes/token |
|---|---:|---:|
| 2,048 | 604,958,676 | 295,390 |
| 8,192 | 1,007,783,892 | 123,020 |

Fitting the two points as `fixed + per_token x n`:

- per-token slope ≈ **65,564 bytes**
- fixed component ≈ **449 MiB**

The fixed part is the recurrent/checkpoint state, which is per-layer rather than per-token —
consistent with the layout inventory, where R/S tensors are sized by `n_embd_r()` and layer
count, not by cell count.

**Consequence for the 32K refusal:** an affine fit predicts a 32K object near **2.6 GB**, not
9.7 GB — so peak transient ≈ 5.2 GB and ≈ 6.3 GB with the 20% margin, against **16.25 GiB
free**. On this arithmetic 32K would fit comfortably on the current NVMe.

I am **not** acting on that. Two points fitted to a two-parameter model is not a
measurement, it is a line drawn through every point available, and the space guard still
uses the conservative 295,390 rate so nothing can be generated on the strength of it.

It also explains why 8K beat cold by 60.7% while 2K managed 14.2%: the fixed cost amortises
as the prefix grows, which is the same direction the product goal points.

### Evidence added 2026-08-29: the mechanism is confirmed on a model that lacks it

The two points above are from the **hybrid** qwen35. A separate three-point series on the
**non-hybrid** qwen2 Q4_K_M, taken during REQ-036/037/038, tests the proposed mechanism
directly — if the fixed component really is recurrent state, a model with no recurrent state
should have essentially none of it.

| prompt | artifact bytes |
|---|---:|
| 128 | 4,721,548 |
| 8,192 | 302,121,868 |
| 32,000 | 1,180,160,908 |

Fitting the first two gives `908 + 36,880 x n`. Applied to 32,000 — **four times outside the
range it was fitted on** — that predicts 1,180,160,908 bytes. The measured value is
1,180,160,908 bytes. **Exact to the byte, zero error.**

So on a model with no recurrent state the fixed component is **908 bytes**, against ~449 MiB
on the hybrid. That is the strongest available confirmation that the hybrid's fixed term is
the recurrent/checkpoint state and not an artifact of fitting two points to two parameters.

It does **not** settle either question below. Those are about the hybrid model's numbers, and
nothing here re-measures the hybrid. What it does settle is the method: a fit is worth acting
on once it has predicted a point far outside its own range, and that test is cheap.

**Questions for the steer:**

1. Should the space predictor take a measured rate per prompt length, or an affine fit, or
   stay conservative-linear? The current default silently over-predicts by 2.4x at 8K.
   *Amended:* an affine fit validated against a held-out far point is now demonstrably
   trustworthy on a non-hybrid model. Should the hybrid get the same two-measurement
   treatment before the 32K question is reopened?
2. Does the 32K refusal still hold given the affine reading? If the intent was "do not
   generate 9.7 GB", the premise may no longer be true. If the intent was "do not scale
   until 8K passes three repetitions", it holds regardless and I will withdraw this entry.
3. Note that 32K has now been generated and transferred **for the non-hybrid model**, at
   1.18 GB, entirely in `/dev/shm` (101 GB free) with no NVMe involved. If the refusal was
   about NVMe space specifically, `/dev/shm` sidesteps it — is that an acceptable route for
   the hybrid 32K run too?

---

## RA-004 — The sidecar must be demand-driven, or it recreates the harm it replaces · **open** · 2026-08-29

Operational finding from the running system, not from design review.

`kvwarm` is currently evicting the models the owner is actually using. The mechanism is in
`cfrproxy/scripts/kvwarm.py:186`: `model_identity()` requests
`{swap}/upstream/{model}/props`, and **that request alone makes llama-swap load the model**.
It wakes a model just to ask what it is, before warming anything. Measured from its journal:
~20,236-token prefixes, up to 51 of them, re-prefilled every 15 minutes into a vLLM model
that needs both 3090s.

**The design consequence, which changes the sidecar spec:** replacing recompute with restore
does *not* fix this. A restore also needs the model resident. A sidecar that keeps a timer
and walks a list of target models wakes them exactly as kvwarm does, just more cheaply. The
saving would be real and the harm would be unchanged.

So the sidecar must be **demand-driven**: restore a prefix when a request arrives for a model
that is *already loaded*, and never contact a model nobody asked for. Scheduled pre-warming
of arbitrary models should not be an available mode, because it is the failure, not a
configuration of it.

**Second finding, on scope.** The owner's target is one cache file replacing *both* warmers.
Those are `tiel-coder-q5-w6800` on ROCm/W6800 and `qwen38-27b-vllm-vision` on CUDA/3090s -
different models *and* different backends. So "retire both warmers with one file" is the
complete cross-arch and cross-model stack, not a near-term substitution. A same-model
demand-driven sidecar retires kvwarm's *behaviour* immediately; retiring its *file format*
needs canonical decode, dtype conversion, and backend transfer first.

**Questions for the steer:**

1. Should scheduled/proactive warming be excluded from the sidecar contract outright, rather
   than offered and discouraged?
2. Does the staged order hold - demand-driven same-model sidecar first, then decoder, dtype,
   backend, model - or should the canonical work land before any service surface?

---

## Answered

### RA-001 — answered in steer 7ded4e1, 2026-08-29

Cross-backend portability and the sidecar are **in scope and are the main track**. The opaque
llama.cpp route is reclassified as a source/target-specific fast path, not the portable
format, and its strict refusals must not be relaxed: an f16 opaque artifact must still refuse
a q8_0 target. Portability comes from a separate canonical path —
`source-native -> canonical KVX -> target-native -> behavioural gate -> reuse or native
prefill`. Same-model cross-dtype/backend transfer comes before cross-model mapping, and
cross-model claims require the held-out divergence gate; tensor similarity is never admission
evidence. The sidecar is to be built only far enough to exercise a *proven* transfer seam.

### RA-002 — answered in steer 7ded4e1, 2026-08-29

8K is the next live benchmark, as the first retained measurement inside the stated 8K–32K
agentic-prefix range. 2K cold-cache work is closed out — retain any in-flight single result
but do not iterate, and it does not gate 8K.

The 32K space constraint was accepted: **do not generate a 32K artifact on the current
NVMe.** A 32K persistent rung needs either a different persistent target with the measured
transient requirement plus margin, or removal of the admission double-space requirement
without weakening atomic full-digest admission. Tmpfs is explicitly **not** a substitute for
a persistent claim.

Both entries closed. The mechanism worked: two entries raised at 13:30, both answered in the
steer at 13:31, and the project roadmap changed as a result.


---

## RA-003 addendum — withdrawn 2026-08-30, answered by measurement

This entry asked the steer to reconsider a space guard built on a fitted rate. It is
withdrawn because the question no longer needs answering: the size is now **derived** rather
than fitted, and exact.

`kv_rosetta/sizing.py` computes an artifact's size from the GGUF and the state-file layout,
with every term read off `llama-kv-cache.cpp` and `save_slot_checkpoints`. Checked against
seven artifacts written on this host across three architectures - qwen2 dense, qwen35 hybrid,
qwen35moe hybrid-MoE - it is exact to the byte on all of them, including a 32,000-token file
predicted from a law fitted only up to 8,192, and RA-003's own 2,048-token hybrid artifact
which this project did not produce.

The space guard now derives instead of scaling a rate. On the non-hybrid model the old flat
rate over-predicted by **8x**; on the hybrid it varied from 2.1x to 4.2x with prefix length.
Nothing is left for the steer to decide here.

---

## RA-005 — The headline goal is not supported by evidence, and the steer should say so · **open** · 2026-08-30

**The goal I am measuring against.** One KVX file usable by any model regardless of
architecture or backend - stated as "we should be able to use the same kvx file regardless of
cuda/vulkan nvidia/amd moe/dense" and "we have to build the translator, that's the whole goal
of the sidecar".

**What the evidence now says.** The goal splits cleanly in two, and the halves have opposite
verdicts.

**Proven.** One file across compute backends, for one model: written on ROCm/HIP and restored
on Vulkan and back, 32,000 tokens, 99.99% reuse, byte-identical output. Hybrid models made
cacheable at all - 252 of 256 tokens reused on a patched build against **0 of 256** on the
fleet's own binary, both reporting the same `n_restored`. Artifact sizing derived and exact on
three architectures. The format reads dense, MoE and hybrid state byte-for-byte.

**Not supported.** One file across *different models*. Measured on the most favourable pair
available - qwen35 and qwen35moe, identical head_dim 256, identical d_state 128, identical
tokenizer, so no alignment error is even possible - a per-layer linear map fitted on 15,981
tokens reaches median held-out R² **0.55**, and the gate rejects it: teacher-forced agreement
**0.733** and **0.903** against 1.000 for every blend, and the translated cache diverges from
the target's own output within **6 tokens** while producing fluent, on-topic English.

**Why this matters for the steer.** The two halves need different words. The first is a
shipping claim: it is measured, it has controls, and it solves the operational problem that
started this work - kvwarm re-reading 51 prefixes into 2 slots every 15 minutes, 13 of the
last 20 cycles complete re-prefills. The second is a research bet that a linear map has now
lost, on the best pair available, with a gate whose soundness took eight iterations to
establish and which corrected itself three times on the way.

**Questions for the steer:**

1. Should the sidecar ship on the proven half - durable, cross-backend, hybrid-capable caching
   for one model at a time - while translation continues as research? I believe yes, and that
   describing the project as model-agnostic today would be claiming the unproven half.
2. Is a non-linear map worth funding? The honest case for it: the linear map recovers 55% of
   variance on a pair with every semantic axis matched, and published work (C2C-style learned
   projectors) is the next rung. The honest case against: no cheap offline metric predicts
   admission, so each candidate costs a GPU evaluation, and the gate needs top-1 correct at
   essentially every position.
3. The recurrent half of a hybrid cache is **not addressed at all** by the current map, and it
   is 90% of a short artifact and 7-9% of a 32K one. Should a recurrent map be attempted, or
   should translation be scoped to attention-only and long prefixes where that is most of the
   file?
