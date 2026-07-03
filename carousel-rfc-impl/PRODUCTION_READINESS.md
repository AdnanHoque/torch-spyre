# Carousel RFC — Production Readiness Gate

Status date: 2026-07-02. Scope: the two carousel RFCs (Weight Carousel — prefill matmul; KV Carousel — decode attention). This is a **gate document**, not an endorsement. Every claim below is tagged with its evidence source: `GT-torch` (torch-spyre @ cf67411 verdicts), `GT-dt` (deeptools @ codex/ah-comms-collectives phase-0 probes), `roofline` (hand-computed, not executed), or `impl` (component deliverable notes).

## Milestone ladder (used throughout)

| Milestone | Definition |
|---|---|
| **M0** | Premise + GO/NO-GO validated on paper + roofline: does the win exist, is the premise confirmed. |
| **M1** | Frontend plan + cost model + numerical reference, buildable and unit-tested. No device, no backend, numerics-neutral. |
| **M2** | Device probes run: rho, lambda, duplex, LX-flip point, streaming ceiling measured; premise **device-confirmed**. |
| **M3** | Backend primitives landed and running in deeptools (async SHUFFLE overlap; on-chip fold) + lowering emits a real SHUFFLE on the compile path. |
| **M4** | Wired into the planner (new HBM/replication cost term), e2e on device, perf win demonstrated, shipped behind a flag. |

---

## 1. M0 GO / NO-GO

### Weight Carousel — **NO-GO on the headline thesis; unproven CONDITIONAL-GO on the residual**

- **The ×P DRAM-replication premise is FALSE as shipped.** `GT-torch` Q1: the only live cost model (`_matmul_split_cost`, work_division.py:654) counts each weight operand **once** (`bytes_total=(M*K+K*N+M*N)*2`, :683) and amortizes broadcast up to cohort 8. The emitted prefill plan is an M×N co-split with `max(m,n)≤8`, i.e. W is replicated **m-fold (≤8), never ×32**. The ×32 endpoint the RFC attacks is a strawman that does not ship.
- **Roofline kills the wall-clock claim at production prefill S.** `roofline`: compute-bound gate is `rho_min = u·C_array·b/(2S)`. At u=0.30, S=512 → rho_min = **176 GB/s > 166 GB/s ring** → transport not hidden → **NO-GO**. GO appears only at S≥1024 (88), and there "GO" means *the ring isn't the bottleneck because the array is slow* — the ~29.5% PT-util array-under-fill (device-established, `GT-torch` Q1) — **not** an ×P wall-clock win.
- **The cost model cannot even reward a byte saving.** `GT-torch` Q1: counting each operand once makes any DRAM-byte reduction invisible to Pass-2; a carousel that saved bytes would still never be selected without a new HBM term.
- **Residual thesis that survives:** array-fill + seam-transparency via the group-carousel r-family as *one cost-model plan family* (r=P = today, r=1 = full carousel). This is real but **unquantified** and requires device + a new cost term. Verdict: pursue only if M2 shows an array-fill win.

### KV Carousel — **NO-GO now; premise UNCONFIRMED in-backend and partially BLOCKED**

- **The entire substrate is absent.** `GT-torch` Q2: there is no KV cache, no paged attention, no RoPE, and **no flash / online-softmax** in torch-spyre. SDPA is *decomposed* (dense QK^T `matmul` → `torch.softmax` over full Skv → attn@V `matmul` → a **dead** `torch.empty` logsumexp). The LSE ring-fold merge operator **does not exist in the lowering**. This is a *proposed* lowering, not a modification of a live path.
- **The ×4 ceiling premise is unconfirmable in-backend and rests on a BLOCKED mechanism.** GQA is materialized to 32 heads (`GT-torch` Q2, decompositions.py:534), so "only H_kv=8 channels active" is an upstream placement/DMA fact, not visible here. And `GT-dt` A3: the compiler **cannot pin an HBM region to an LPDDR channel** (flat `memId=-1`). The "all 32 channels stream" claim can only rest on **which CORE (LX memId) owns a shard**, not HBM channel placement — and that must be device-measured, not asserted.
- **The fold primitive does not exist.** `GT-dt` Fold: single-AIU move-then-reduce is absent; `STCDPOpLx` is pure movement, reduce lives only in the cross-chip collective layer.
- **Win that survives in principle:** memory-bound decode (intensity ~2G/b) at long L can gain by streaming from all P cores; roofline puts the KV-term speedup approaching P/H_kv=×4 by L~2048, break-even ~L 512–1024 — **but entirely gated on unmeasured lambda.**

**Net M0:** Weight carousel's stated win does not exist; a narrower array-fill win is plausible-but-unproven. KV carousel's win is real in principle but the furthest from reality — it needs a new frontend lowering **and** a new backend primitive **and** device confirmation of the premise.

---

## 2. Gap Ledger

### Weight Carousel

**DONE (frontend) — but read the execution caveat.**

| Piece | State |
|---|---|
| `WeightCarouselPlan`, `enumerate_weight_carousel_plans` (r-family), `weight_carousel_cost` {dram_bytes, link_occupancy, lx_highwater, t_est} | `impl` buildable, numerics-neutral. **Hand-verified only — NOT executed** (disk writes declined). |
| `seam_layout` (M-split phase p in→out, seam_transparent=True) | `impl` hand-verified. |
| `comm_cost.py` ring-occupancy model (RING_SUPPLY=64 link-us, `price_carousel`, radius sweep r=1→31·|W|/rho=21931us, r=32→0) | `impl` **executed via stdin — validated.** |
| `reference.py` numerical validator (tile-exact bit-for-bit vs tiled, schedule-cover permutation, r-/kt-invariance allclose, padding) | `impl` **executed, 12/12 PASS, on disk at /tmp/carousel-wt.** The only fully-run frontend artifact. |
| `roofline.py` compute-bound gate | `impl` **hand-computed, NOT executed.** |
| `TS_ENABLE_CAROUSEL` env gate (off by default) | `impl` hand-verified. |

Honest caveat: of the frontend set, only `reference.py` and `comm_cost.py` were actually run. `weight_carousel.py` and `roofline.py` were never executed; none of these files are committed or pre-commit-clean. "Unit-tested" is true for the validator, aspirational for the rest.

**DEVICE-GATED** (a subagent cannot produce any of these — no accelerator):

| Probe | Missing number | Rationale (one line) |
|---|---|---|
| **P1** rotation/multicast accept | Confirm planner emits `STCDPOpLx`/`reqMulticast` on a *forced* M-split→rotate reshard | Path is LIVE (`GT-dt` P1) but P1-the-harness is scaffold: needs `SPYRE_SDSC_DIR` + `force_split_dbg.py` to pin an exact δ. |
| **P3** LX-local no-bounce | The `STCDPOpHBM` flip point = max K_t slab that fits the LX double-buffer | Path is LIVE (`GT-dt` P3) conditional on LX capacity; the K_t ceiling is the whole feasibility knob and is unmeasured. |
| **P4** duplex | Measured duplex factor (assumed 2 → RING_SUPPLY=64) | Sets the ring supply denominator; needs the symmetric ±1 shuffle built (blocked on the split-forcing hook). |
| **P5** rho / lambda | Measured ring rho and hop latency **lambda** | lambda is a **STUB (0.5us)** and is *the* number deciding whether the marginal wall-clock win exists at all; OLS fit runs offline, device sweep is scaffold. |
| roofline boundary | Measured rho vs the 176 GB/s S=512 threshold | Decides whether even the residual array-fill plan clears the ring; ~29.5% PT-util is already device-established, rho is not. |

**DEEPTOOLS-GATED** (a subagent can write the patch; it cannot land+run it):

- **P2 — async SHUFFLE↔matmul overlap.** `overlapInpFetchWithCompute` (dsmperf.cpp:3725) is hard-gated to Conv2D/SparseConv2D consumers (:3733); matmul/BMM are PriOp but not Conv, so the hook does not fire (`GT-dt` P2). **Ask:** widen the consumer gate to matmul/BMM. **Rationale:** without it the relayout is a sequential barrier (producer→move→consumer) and the already-marginal wall-clock win is dead. This is the open perf gate.
- **A1 — rotation δ=±1.** **NOT a gap.** `GT-dt` P1: ±1 rotation is a strict subset of an arbitrary `coreIdToWkSlice_` permutation and is accepted on the live path. Closed.
- **PLANNER-GATED (torch-spyre, not deeptools) — the reward term.** `_matmul_split_cost` cannot price m-fold/P-fold weight replication (`GT-torch` Q1). **Ask:** add an m-fold weight-replication HBM term so Pass-2 can *prefer* a carousel. **Rationale:** without it the plan is invisible to the planner and will never be selected — this is the M4 blocker.

### KV Carousel

**DONE (frontend).**

| Piece | State |
|---|---|
| `kv_placement` (prefill contiguous + decode block-cyclic, composes via `KVSegment`) | `impl` pure arithmetic, testable. |
| `lse_combine` (exact fp32 stable merge, dead-lane guarded, associative→deterministic) | `impl` the real SFP-op semantics. |
| `ring_fold` (composes SHUFFLE + `lse_combine`/hop; prices A_linear / B_reduce_scatter / C_halving) | `impl` **self-test NOT executed** (writes blocked). Correct by reasoning, unverified by run. |
| `decode_step_plan` (Q-bcast→local→fold→normalize→O-proj; ×(P/H_kv) ceiling + zero-relayout O-proj) | `impl` hand-verified. |
| `price_fold` in `comm_cost.py` (3 variants; linear occ=0.78us lowest, reduce_scatter head-split endpoint, recursive cut-through vs store-forward) | `impl` **executed via stdin — validated.** |
| `reference.py` (flash==softmax, bkv/placement/fold-order invariance, bit-determinism, adversarial ±80 overflow proof, gqa_each_head_once) | `impl` **executed, PASS** (part of the 12/12). |

Honest caveat: `fold.py._self_test()` was never run this session — the numerical fold correctness is reasoned, not executed. Run it before relying on it.

**DEVICE-GATED:**

| Probe | Missing number | Rationale |
|---|---|---|
| **P5** lambda | Measured hop latency | `HOP_LATENCY_US=0.5` is a strawman and is *the* crossover between fold topologies **and** the L break-even; roofline prints the full lambda sweep precisely because it's unmeasured. |
| **A3-proxy / P5 sweep** streaming ceiling | The real ×(P/H_kv) lift via the SENCORES active-core BW-sweep | Since HBM channels can't be pinned (`GT-dt` A3), the ×4 must be proven through core-ownership streaming — a device fact, currently a first-order estimate. |
| memory-bound premise | Measured ~25% aggregate LPDDR ceiling at H_kv=8 | The whole motivation; unconfirmed in-backend (GQA materialized upstream). |

**DEEPTOOLS-GATED (deepest of the two carousels):**

- **On-chip LSE ring-fold primitive — ABSENT.** `GT-dt` Fold: no single-AIU move-then-reduce; reduce lives only in the cross-chip collective layer (needs `_WORLD_SIZE/_RANK`). **Ask:** either add an intra-AIU fused move-then-reduce, or accept composed `STCDPOpLx` (move) + separate SFP `lse_combine` per hop. **Rationale:** the fold is on the critical path of *every layer every step*; as a composed op it also eats the P2 barrier, which can erase the win — this is the single largest backend lift.
- **P2 barrier again.** The same Conv-only overlap gate blocks overlapping the fold move with SFP compute.
- **A3 channel-affine placement — BLOCKED, no ask achievable.** Compiler cannot pin an HBM region to an LPDDR channel (flat `memId=-1`). **Ask:** none as HBM placement; the premise must be *reframed* to core-ownership streaming. **Rationale:** the ×4 "all 32 channels stream" claim cannot rest on channel pinning; it rests on core-ownership, which is a device measurement, not a compiler capability.
- **FRONTEND-GATED (torch-spyre) — the flash lowering itself does not exist.** `GT-torch` Q2: SDPA is decomposed to dense matmul + `torch.softmax`; `local_flash_partial_pass` is a `NotImplementedError` stub. This is a whole new lowering (online-softmax + GQA fan), the biggest single missing piece, and it is *predicated on the still-unmeasured premise.*

---

## 3. Distance to production + sequencing

### Where each carousel sits

- **Weight Carousel: at M1 (frontend), with a NEGATIVE M0 on the headline thesis.** Numerics are validated (reference.py 12/12), the reshard path it needs is LIVE (P1, P3), and its only hard backend gap is a bounded C++ edit (P2 gate widening). But roofline says the wall-clock win is marginal-to-absent at production S, and the cost model can't reward it. **Distance to M4: gated on M2 (lambda/duplex/LX-flip) → then decide.**
- **KV Carousel: at M1 partial, with a NO-GO M0.** Placement/fold-numerics are done, but the flash lowering (frontend) is a stub, the fold primitive (backend) is absent, and the channel-affinity mechanism is blocked. **Distance to M4 is large:** new frontend lowering + new backend primitive + device-confirmed premise. Realistically two milestones behind the weight carousel.

### Recommended sequencing

1. **Weight carousel M2 FIRST, before any backend work.** Run P5 (lambda, rho) and P3 (LX K_t flip) on a device. Roofline already says the ×P wall-clock story is dead at S=512; measure whether the residual array-fill plan clears the ring at S≥1024 with real lambda. **If M2 is negative, stop — do not spend the P2 backend edit.**
2. **Only if M2 is positive:** land the P2 Conv→matmul overlap-gate widening (M3), then the m-fold weight-replication cost term in Pass-2 (M4). The cost term is worthless before an M2 win exists to reward.
3. **KV carousel: PARK behind a device-confirmed premise.** Do **not** start the flash-attention lowering (a large frontend build) until the memory-bound decode ceiling is device-confirmed via the core-ownership BW proxy (A3 is blocked; the HBM-channel premise is dead). Separately, treat the on-chip fold primitive as a standalone deeptools investment decision — it is the gating backend lift and the flash lowering is worthless without it.
4. **Do NOT target vLLM / paged-attention** for the KV carousel (RFC's own caveat: static compile-time placement is incompatible with a runtime page table).

### What a subagent fleet CANNOT deliver (blunt)

- **Any device number.** lambda, rho, duplex, the LX K_t flip point, the ×4 streaming ceiling, the ~25% LPDDR decode premise, every A/B wall-clock. No accelerator, no measurement. All of P1–P5 are **scaffold** until a human runs them on a pod with `SPYRE_SDSC_DIR` + `force_split_dbg.py` + a stable device.
- **A landed, running deeptools edit.** A subagent can write the P2 gate-widening patch and (KV) draft the fold primitive, but **cannot verify either on-device** — prior stream notes show these builds are infeasible here (old flex API, senlib skew, segfaults, pod-refresh needed). "C++ written" ≠ "backend gap closed."
- **The flash-attention lowering proven on device.** A subagent can draft the frontend decomposition, but "does it compile, run, and match numerics on the accelerator" is a device question.
- **A planner that actually prefers the carousel.** The cost-model HBM term can be written, but "does Pass-2 now pick it and is it faster on real Granite shapes" is device-only.
- **Model-graph facts.** A-RoPE / KV layout confirmation needs the FMS/Granite graph, outside this backend.
- **On-disk, committed, pre-commit-clean artifacts, this session.** Every component's disk write was declined; `roofline.py` and `fold.py`'s self-test were never executed. Treat the frontend as *reviewed source returned as text*, not as a green test suite — except `reference.py` (12/12, on disk) and `comm_cost.py` (stdin-run).

**Bottom line:** the frontend scaffolding for both carousels is largely written and, where run, correct. Neither carousel has a confirmed win. The weight carousel is one cheap device probe away from a go/no-go decision and its wall-clock thesis is probably already dead — pursue it only for array-fill, only if M2 says so. The KV carousel is a multi-milestone build resting on an unmeasured premise and an absent backend primitive; do not invest in its lowering until the premise is measured and the fold primitive is scoped.