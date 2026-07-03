The record follows. This is my final output (returned as text per the task; not written to disk).

---

# M0 Closure — Weight & KV Carousel RFCs

Status date: 2026-07-03. Branch `ah/carousel` @ 36c4c97 (torch-spyre fork). This record closes M0 (premise + go/no-go on paper + roofline + the one device unknown that was reachable). Evidence tags: `GT-torch` (torch-spyre @ cf67411 live-code verdicts), `GT-dt` (deeptools @ codex/ah-comms-collectives phase-0 probes), `roofline` (roofline.py, executed this session), `device` (M0 rho probe, this session, single shared accelerator, solo).

---

## 1. M0 verdict — per carousel

### 1a. Weight Carousel (prefill matmul `Y[S,N]=X[S,K]@W[K,N]`, M-split over P cores)

| M0 axis | Verdict | Basis |
|---|---|---|
| **Premise vs live code** | **REFUTED (headline) / narrowed** | `GT-torch` Q1: the only live cost model `_matmul_split_cost` (work_division.py:654–704) counts each weight operand **once** (`bytes_total=(M*K+K*N+M*N)*b`, :683) and amortizes broadcast up to cohort 8. The emitted prefill plan is an M×N co-split with `max(m,n)≤8` → W is replicated **m-fold (≤8), never ×P=32**. The ×32 DRAM-replication strawman the RFC attacks does not ship. |
| **Validators** | **PASS (5/5 weight; 12/12 suite)** | reference.py, executed, on disk: `tile_exact`, `schedule_cover`, `r_invariance`, `kt_invariance`, `padding`. Proves the r-family and K-slab rotation are bit-exact data-movement rewrites, not numeric changes. |
| **Roofline go/no-go** | **NO-GO S≤512 / GO S≥1024 — but array-fill, not ×P wall-clock** | `roofline` compute-gate `rho ≥ u·C·b/(2S)`: at u=0.30, S=512 → rho_min=**176 GB/s > 166 GB/s ring → NO-GO**; S=1024→88, S=2048→44, S=4096→22 → **GO**. HONEST: "GO" means the ring hides transport *because the array is under-filled* (~29.5% PT-util, device-established), not an ×P wall-clock win. The carousel's prefill value is seam-transparency + array-fill; the DRAM-byte saving is a byte-count win the cost model cannot even see. |
| **rho** (ring per-link BW) | **Loose LOWER bound ~11 GB/s; consistent with 166 GB/s spec; not independently confirmed** | `device` below. |
| **lambda** (per-hop latency) | **NOT ISOLABLE without per-hop instrumentation** | `device` below. |

**Weight M0 = conditional NO-GO.** Headline ×P DRAM thesis is dead at production S; the residual array-fill/seam plan is real but unquantified and unselectable (no cost term). The measured rho does **not** overturn the S=512 NO-GO: the probe's confirmed floor (11 GB/s) sits *below* even the S=1024 threshold (88 GB/s), so the go/no-go rests on the 166 GB/s **spec** rho, which the probe is consistent with (via the HBM-dominance argument) but does not independently prove clears 88 or 176.

### 1b. KV Carousel (decode attention, BS=1, KV sharded along sequence over all P channels)

| M0 axis | Verdict | Basis |
|---|---|---|
| **Premise vs live code** | **UNCONFIRMED in-backend; partially BLOCKED** | `GT-torch` Q2: no KV cache, no paged attention, no RoPE, **no flash/online-softmax** in torch-spyre. SDPA is *decomposed* (dense QK^T `matmul` → `torch.softmax` over full Skv → attn@V `matmul` → **dead** `torch.empty` logsumexp). The LSE ring-fold merge operator **does not exist in the lowering** — this is a proposed lowering, being built upstream as a flash-attention script (frontend, in progress), not a live-path edit. GQA is materialized to 32 heads (decompositions.py:534), so "only H_kv=8 channels active" is an upstream DMA fact, invisible here. |
| **Validators** | **PASS (7/7 KV; 12/12 suite)** | reference.py, executed, on disk: `flash_matches_softmax`, `fold_bit_determinism`, `fold_order_invariance`, `block_placement_invariance`, `bkv_invariance`, `adversarial_spread` (±80-logit overflow proof), `gqa_each_head_once`. Proves the sharded flash + LSE ring-fold reproduces single-pass softmax attention exactly, independent of placement / block size / fold order. |
| **Roofline go/no-go** | **GO in principle above a topology-set crossover L_min; ceiling capped to fill-factor** | `roofline` at rho=166 GB/s, lambda-swept (lambda unmeasured): crossover L_min = A(linear) 91–1068 / B(rs+allgather) 130–2055 / C(recursive-half) **15–198** tokens over lambda 0.25–4.0 us. Step-speedup asymptotes to P/H_kv=**4×** by L≈2048 (3.71× @ L=1024, 3.85× @ L=2048). **A3 caveat:** the 4× "all P channels stream" ceiling is realizable only via **core-ownership** of shards, not HBM-channel pinning (`GT-dt` A3: HBM is one flat `memId=-1` space) → the ceiling collapses from 4× to whatever fill-factor core-ownership actually delivers (unconfirmed; needs the SENCORES active-core BW-sweep proxy). |
| **KV crossover recompute** | **Cannot pin — the lever (lambda) did not come back** | The crossover is L-independent overhead / per-L saving, and the merge payload is tiny (16.6 KB), so **lambda — not rho — sets it**. lambda is not isolable this session (below), so L_min stays a lambda-parameterized band. The measured rho lower bound does not sharpen it (rho is sub-dominant for the KB-scale merge). Best available: topology C, L_min ≈ 27 tokens at the strawman lambda=0.5 us — but that is a strawman, not a measurement. |

**KV M0 = NO-GO now.** Furthest from reality: needs a new frontend lowering (flash, in progress) **and** a new backend primitive (on-chip fold) **and** device confirmation of the memory-bound premise — none present.

### 1c. Device unknown — the M0 rho probe (`device`)

The two device unknowns left for M0 were rho and lambda. One was reachable.

- **Relayout fired (verified):** fused SwiGLU [1,512,4096] with the `stcdp_range` carrier produced **2 `OnChipMoveSTCDPOpLx` SDSCs**, `bytes_moved=13,107,200` each → **26.2 MB** over the ring; PLANNER=0 baseline = **0** relayout SDSCs. This confirms `GT-dt` P1 (rotation/reshard accepted as STCDPOpLx) and P3 (LX→LX, no HBM bounce) *on device*, not just by code-read.
- **rho ≈ 11.1 GB/s — LOOSE LOWER BOUND, LOW confidence.** Method: A/B median wall-clock **and** profiler-bundle device time (they agree to 0.2%): ON (relayout) 10.746 ms wall / 10.218 ms device; OFF (PLANNER=0) 13.110 ms / 12.578 ms; delta ≈ **2.36 ms**. `rho = 26.2 MB / 2.36 ms ≈ 11.1 GB/s`. **Why it is only a bound:** the delta is dominated by the **HBM round-trip removed**, not the ring move. At the BiRing spec (166 GB/s/dir) the 26.2 MB move is ~0.16 ms — under 7% of the delta; the other ~93% is avoided HBM traffic (52.4 MB round-trip ⇒ ~22 GB/s effective HBM). True per-link rho is therefore well above 11 GB/s, consistent with the ~166 GB/s/dir spec; the A/B method **cannot resolve it** because the ring move is sub-dominant in the delta it measures.
- **lambda: NOT ISOLABLE without per-hop instrumentation.** The borrowed `_C.so` *is* a profiler build (kineto/AIUpti symbols, emits device events), but its finest device-event granularity is the **fused inductor bundle** — one event for the whole SwiGLU, no isolated `STCDPOpLx` event. So even `rho = remap_bytes / STCDPOpLx_device_time` is not computable, let alone a per-hop split. lambda needs a move-distance / hop-count sweep with per-op isolated timing, which this bundle-granularity build cannot provide.
- **Blocker note (non-fatal):** a 60,001 ms "lost completion" synchronize warning fires on **warmup only** (known flex profiling/streams thread-lock); the 20 timed iters are clean and wall-vs-device deltas agree to 0.2%, so it does not contaminate the A/B (constant overhead cancels).

---

## 2. Hard Gap Ledger (updated)

### DONE — frontend + validators (committed, `ah/carousel` @ 36c4c97)

| Piece | State |
|---|---|
| Weight: `WeightCarouselPlan` / `enumerate_weight_carousel_plans` (r-family) / `weight_carousel_cost` {dram_bytes, link_occupancy, lx_highwater, t_est} / `seam_layout` | committed, numerics-neutral, r=P↔r=1 as endpoints of one cost family. |
| KV: `kv_placement` (prefill-contiguous + decode block-cyclic, seam-free compose) / `lse_combine` / `ring_fold` (A/B/C topologies) / `decode_step_plan` | committed; fold is a **composition** of STCDPOpLx move + SFP `lse_combine` (no fused primitive assumed). |
| `reference.py` numerical validators | **12/12 PASS** (5 weight + 7 KV), on disk. |
| `roofline.py` go/no-go gate + `comm_cost.py` ring model | committed; roofline **executed** this session (§1 tables). |
| `probes.py` P1–P5 harness + `TS_ENABLE_CAROUSEL` emission gate (off by default) | committed; P1/P3 now **device-fired** (§1c). |

*Rationale: the entire frontend — plan, cost family, numerically-exact reference, and the go/no-go roofline — is written, committed, and validated where runnable. This is the whole M0/M1 frontend deliverable.*

### DEVICE-GATED — one measured, one blocked

| Gap | State | One-line rationale |
|---|---|---|
| **rho** (P5) | **MEASURED as a loose lower bound (~11 GB/s); consistent with 166 GB/s spec, not independently confirmed** | The only device lever reachable this session; the A/B delta is HBM-round-trip-dominated so it bounds rho from below but cannot resolve the per-link rate — for a tight rho you need an isolated STCDPOpLx device event, which the bundle-granularity profiler does not emit. |
| **lambda** (P5) | **BLOCKED on profiler granularity** | The number that sets the KV crossover and the fold-topology ranking; not isolable without per-hop AIUpti records or a hop-count sweep with per-op isolated timing — this build's finest event is the fused bundle, so lambda stays the strawman 0.5 us. |
| P3 K_t LX-flip point | still SCAFFOLD | LX→LX confirmed on device (0 bounce SDSCs), but the exact K_t slab size where STCDPOpLx flips to STCDPOpHBM (the double-buffer feasibility ceiling) needs a payload sweep, not yet run. |
| A3 streaming-ceiling proxy | still SCAFFOLD | The realized 4× (vs fill-factor) must come from a SENCORES active-core BW sweep; unmeasured. |

### DEEPTOOLS-GATED — ranked by leverage

| Gap | State | One-line rationale |
|---|---|---|
| **P2 overlap-gate extension** (dsmperf.cpp:3733–3736) | **the concrete near-term win** | `overlapInpFetchWithCompute` is hard-gated to Conv2D/SparseConv2D consumers; matmul/BMM are PriOps but not Conv, so a SHUFFLE↔matmul overlap never fires. Eligibility is already met (seam-transparent reshard keeps `layoutDimOrder`, so `assignCanOverlapInpFetch` stays true) → **widen the consumer gate to PriOp/matmul** and both carousels stop paying the sequential producer→move→consumer barrier. Small, bounded, exact. |
| **On-chip LSE ring-fold primitive** | **the one genuinely new primitive** | `GT-dt` Fold: no single-AIU move-then-reduce exists (`STCDPOpLx` is pure movement; reduce lives only in the cross-chip collective layer needing `_WORLD_SIZE/_RANK`). The KV merge must be composed as move + separate SFP `lse_combine` per hop — on the critical path of every layer every step, and as a composed op it also eats the P2 barrier. Largest single backend lift. |
| **A3 channel affinity** | **hard limit / bigger allocator ask** | HBM is one flat `memId=-1` space (dataOpDsc.h:184); the compiler cannot pin a persistent HBM region to an LPDDR channel. Not a patch — reframing the KV "all P channels stream" premise onto **core-ownership** streaming (a device fact) or a much larger channel-aware allocator. Caps the KV ceiling from 4× to fill-factor. |
| **Weight-carousel HBM cost term** (`_matmul_split_cost`, torch-spyre) | **required for the plan to be selectable at all** | The cost model counts each weight operand once and amortizes broadcast to cohort 8, so a DRAM-byte saving is invisible to Pass-2 — a carousel is never picked without a new m-fold weight-replication HBM term. M4 blocker, and worthless before an M2 array-fill win exists to reward. |

---

## 3. Bottom line

**Distance to production.**

- **Weight carousel — one cheap decision away from stop-or-go, and its headline thesis is already dead.** Frontend at M1 (12/12 validators, reshard path device-fired), but roofline puts the ×P wall-clock win at NO-GO for S≤512 and "GO-because-the-array-is-slow" (array-fill, not bytes) for S≥1024, and the cost model cannot reward a byte saving. Blocking distance to M4: an M2 array-fill measurement (needs a tight rho + K_t flip, both blocked on the bundle-granularity profiler) → then the P2 gate (M3) → then the HBM cost term (M4). **Pursue only if M2 shows an array-fill win; do not spend the backend edits before that.**
- **KV carousel — multiple milestones out.** Numerics done (7/7), but it needs a new frontend flash lowering (in progress upstream), a new on-chip fold primitive (absent), and a device-confirmed memory-bound premise (unmeasured, and the 4× ceiling is A3-capped to fill-factor). Realistically two milestones behind the weight carousel. **Park behind a device-confirmed premise; do not start relying on the fold until it is scoped as a standalone deeptools investment.**

**Single recommended next action: extend the P2 overlap gate on the deeptools fork** — widen `overlapInpFetchWithCompute` (dsmperf.cpp:3733–3736) from its Conv2D/SparseConv2D consumer list to PriOp/matmul consumers. It is the **smallest, most exact** backend change; eligibility is already satisfied (seam-transparent reshards keep `layoutDimOrder`); it is the sole open perf gate the **weight** carousel needs; it is on the critical path of the **KV** fold; and it directly benefits the **comms-collectives** work that shares the same STCDPOpLx overlap machinery. Everything else (tight rho, lambda, the flash lowering, the HBM cost term, the fold primitive) is either profiler-blocked or a larger build that should not start until an M2 win is measured.