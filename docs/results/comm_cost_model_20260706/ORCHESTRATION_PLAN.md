# Orchestration Plan — Granite HBM-Spill Removal + Ring Carousels

**Date:** 2026-07-05 · **Driver:** Fable (main loop, no subagents — see §0) · **Epic:** [torch-spyre#3049](https://github.com/torch-spyre/torch-spyre/issues/3049)
**Scope:** fold the Epic (7 comm classes / 6 phases), Codex's landed lanes, our device-grounded M0 findings, the two carousel RFCs, and the production flash-attention de-spill ask into one prioritized, pod-parallel build plan.

---

## 0. Execution-model note (why no subagents)

The request was "a workflow with **fable** subagents." Verified this session: **Fable is not provisioned for spawned children in this environment.** Both the Workflow tool and the Agent tool emit a hard downgrade record `{from: claude-fable-5 → to: claude-opus-4-8}` for every subagent, regardless of the `model` override. Only the main conversation loop runs Fable. To honor "run on Fable only," this orchestration runs **inline on the Fable main loop**, with real parallelism obtained by dispatching **concurrent background jobs to the 3 pods** (the pods do the compute; Fable coordinates). If Opus subagents are acceptable later, the Workflow fan-out is ready to relaunch.

---

## 1. TL;DR verdict

1. **Both carousels lose to cheaper, device-verified levers on their own goals — do not implement the rotations.** (Details §3.) This is our own prior M0/ROI conclusion, re-verified against live pod state today.
2. **Build the survivors** — the high-value variations the RFCs' machinery actually justifies:
   - **S1 · Ring cost-model correction** (Bet 2, `ah/ring-cost-term`): the "afternoon-check" outcome — price per-link **transfer count** (36 vs 130 GB/s band), not burst size. **No blocker, highest ROI, already 7/7 unit-tested.** Finish the same-core-free refinement + Granite device-measure. → Epic **Phase 6**.
   - **S2 · Cross-bundle co-bundling substrate** (Move 3): keep a producer shard LX-resident *into its consumer program* so the multicast operates on true shards instead of re-reading HBM. This is the **wall blocking both flash value-correctness and Codex's attention all-gather**, and the correct home for the confirmed 130 GB/s uniform-shift transport. → Epic **Phase 4** + the production flash ask.
   - **S3 · P2 overlap-gate extension** (`dsmperf.cpp:3733–3736`): widen `overlapInpFetchWithCompute` from Conv2D/SparseConv2D consumers to matmul/PriOp. Smallest exact backend change; eligibility already satisfied for seam-transparent moves; on the critical path of S2 and the fold. → enables overlap for the whole collectives lane.
   - **S4 · LSE ring-fold reduce lane** (Bet 3): the uncontested arithmetic-collective (`Epic Phase 5`), math-validated 13/13. Sequence **after** S2 (it needs the same move-half plumbing). Note K3 refuted: the fold is a *contention-pattern* linear chain (~58 µs fixed/move dominates), not the 130 GB/s uniform shift — cost it honestly.
3. **Production flash de-spill = S2, not a layout relayout.** The flash HBM→LX conversion is structurally done but **value-incorrect** because `mul(K)` is a separate device program that re-reads K from HBM in the consumer bundle; the fix is co-bundling (S2), not the restickify path. The `amax(dim=-1)` sparse-reduction wall (`optimize_restickify`) is a **reduce**, tracked to the `coarse_tile` Stage-1 Lk-tiling work, *not* to Codex's restickify-on lane (refuted).

---

## 2. Verified current-state matrix (live, 2026-07-05)

Devices confirmed free this session: CDX `/dev/vfio/80` (128c), DEV `/dev/vfio/31` (192c), CLC `/dev/vfio/25` (128c).

### 2a. Epic comm-class taxonomy

| Class | Epic phase | Status | Evidence (this session) |
|---|---|---|---|
| **Scatter / permutation** | 1 | **Landed, MEASURED** | Codex PR1 ~1.065× kernel (14.726→13.821 ms/iter S512). Fresh CLC baseline 12.55 ms/iter kernel clean. |
| **Broadcast** | 3 | Representable, not productionized | DLDSC cardinality probes pass; placement is the gap. |
| **Multicast** | 3 | Representable, not productionized | Same; cohort descriptors expressible. |
| **Gather** | 4 | Representable, not productionized | — |
| **All-gather / replicate → KERNEL operand** | 4 | **BLOCKED (the wall)** | CDX M4: **232 ALLCLOSE-False / 26 True**, 12 self-ring `BusFence`. Value-correct⊕capacity-safe never both. `runDcgForInputFetchNeighbor` pinned `DsTypes::INPUT` vs operand `KERNEL`; empty coord maps in `fillDataInfo`. |
| **Reduce** | 5 | Uncontested lane, math-only | Bet 3 LSE fold 13/13 math; not compiler-wired. |
| **All-reduce** | 7 | Future | After reduce+broadcast stable. |

### 2b. Our three bets

| Bet | Maps to | Status |
|---|---|---|
| **Bet 1** flash-in-a-loop | production flash | **Blocked upstream** at `optimize_restickify` (the `amax(dim=-1)` multi-stick→single-stick reduction). Real dep = `coarse_tile` Stage-1 Lk-tiling (commit `cf67411`). Codex restickify-on does **not** help (refuted). |
| **Bet 2** ring cost term | Phase 6 / **S1** | **DONE + 7/7**, `ah/ring-cost-term` = `work_division.py` +30/−6. `_cohort_penalty` caps multicast at peak/130, scatter at peak/36. **Not device-measured**; missing same-core-free + true-stride fidelity. |
| **Bet 3** LSE ring-fold | Phase 5 / **S4** | Math-validated 13/13; needs S2's move-half + an SFP `lse_combine`. |

### 2c. Carousel RFCs (device-grounded verdicts)

| RFC premise | Verdict | Basis (device unless noted) |
|---|---|---|
| Weight: prefill 30× under roofline from 32× weight hauling | **Refuted** | Real gap was a layout artifact (shared-weight unit-BMM), closed 29.8%→**72.7%** util, MLP-proj 3.122→**1.024 ms**, zero new transport. |
| Weight: compute-bound at full-seq prefill | **NO-GO at S≤512** | roofline ρ_min 176 > 166 GB/s ring at Granite's prefill block; GO only S≥1024; C is a placeholder. |
| Weight: ×P DRAM-byte win | Real but **unselectable** | No cost term prices weight replication — planner can't see it. |
| **Uniform p→p+1 shift runs at ring speed** | **Confirmed** | ρ_eff **54 / 90 / 130 GB/s** @ 4/8/16 MB, R²≥0.9985; distinguishing var = per-link transfer count (1 vs 4–9), **not burst** (burst ↑50–100× = 1.0×). |
| LX→LX no LPDDR bounce (P3) | **Confirmed** | −147 µs on fused SwiGLU (~1.017×); ~1.7% of the layer. |
| KV: decode BS=1 is BW-bound (K1) | **Confirmed** | HBM one shared ~170 GB/s pipe across 32 cores. |
| KV: channel-affine "all 32 channels stream" | **Refuted as stated** | HBM is one flat `memId=-1` space; no channel pinning. Survives only via **core-ownership** of shards (fill-factor unconfirmed). |
| KV: fold transport de-risked by fast ring (K3) | **Refuted** | 130 GB/s is the *uniform shift*; the fold is a *linear-chain contention* pattern, ~58 µs fixed/move dominates. |
| KV: asymptotic 4× | Conditional | Step-speedup → P/H_kv=4× by L≈2048 (3.71× @1024) **iff** core-ownership fill realized; crossover L_min is **λ-set** and λ is **not isolable** (strawman 0.5 µs → L_min≈27 tokens, topology C). |

**Physics constants (use these in any cost work):** ρ_multicast/uniform-shift ≈ **130 GB/s** (16 MB), ρ_scatter/range-relayout ≈ **36 GB/s** (measured effective, R²≥0.9999), fixed **F ≈ 7.3 µs per STCDP execute**, **λ per-hop = 0** modeled (never isolated — flag if a plan leans on it), perfmodel ringBw 128 GB/s/dir, spec 166, HBM one shared ~170 GB/s pipe.

---

## 3. Why the rotations are shelved (one paragraph)

The Weight Carousel's motivation is refuted at the root (the under-roofline gap was layout, not weight bytes, and is already closed by a pure-inductor fix), its one surviving premise (130 GB/s uniform shift) is worth only ~1.7% in that vehicle, and its DRAM-byte payoff is invisible to the planner — a proven mechanism with a refuted motive and an unselectable payoff. The KV Carousel needs a new frontend lowering **and** a new backend fold primitive **and** an unconfirmed memory-bound premise, its channel-affinity story is refuted (flat HBM), and its transport is the slow contention pattern, not the fast shift. **Harvest the one real finding (uniform-shift transport) into S2 and close the RFCs as designed.** The intellectual content survives as S1–S4; the rotations do not.

---

## 4. Pod-parallel build plan

Three lanes, minimal cross-dependency. **Do not** touch Codex's dirty deeptools checkouts (CDX deeptools 17 dirty files, CLC 1 dirty `LayoutAllgatherRestickify.cpp`) — they are intentional live work. Use isolated workspaces; read-only on shared checkouts. Acceptance is **counter-first / kernel-trace**, never wall-time alone (Epic rule).

### Lane CLC — `/dev/vfio/25` — Costing + Granite audit (Phase 6 + Phase 2) — **START HERE, no blocker**
1. **S1a — same-core-free refinement.** In `work_division.py::_cohort_penalty`, reduce the effective cohort to the number of **distinct cross-core** destinations (same-core replicas are free local copies — Codex's self-ring `BusFence` proves it). Add a unit test alongside the 7 in `tests/tensor/test_matmul_split_cost.py`. *Off-device, pure Python.*
2. **S1b — true coordinate stride.** Carry the real coordinate stride (0,4,8,…,28) into the priced edge, not the cohort **count** (0,1,…,7) — this is the fidelity gap that made Codex's flash plan value-wrong. Converge the descriptor with the backend-plan JSON (`consumer_operand_ds_type`, `group_count`).
3. **S1c — device-measure.** On Granite S512, confirm the planner now **picks** the on-chip move and quantify beyond the structural 1.065×. Triage the current **relayout-ON `rc=1` failure at Granite scale** (baseline clean at 12.55 ms/iter).
4. **Phase-2 spill audit.** Complete the non-weight HBM spill inventory from existing SDSC artifacts; classify each by comm class; mark covered / blocked / future. Acceptance: before/after SDSC tables + runbook.

### Lane DEV — `/dev/vfio/31` (192c) — Flash value-correctness (production ask)
1. **Clean oracle first.** `test_flash.py`'s CPU assert is **not** a clean oracle in-env (baseline mismatched 75.1%); the "SUCCESS" runs use `PATCH_MODE=no_h2d,skip_cpu_ref`. Stand up a trustworthy numpy/CPU reference (fixed seed, fp32 carry) before any value claim.
2. **Root-cause the value gap = S2 (co-bundling).** The flash HBM→LX is structural only; the value error traces to `mul(K)` being a separate device program that re-reads K from HBM in the consumer bundle. This is the same wall as CDX's all-gather-into-KERNEL. Coordinate with Lane CDX.
3. **The `amax` reduce wall.** `optimize_restickify` gathering the sparse `amax(dim=-1)` real_max/denominator is a **reduce**, not layout relayout — depends on `coarse_tile` Stage-1 Lk-tiling (`cf67411`), which also removes the `kv_block=Lk/1` "no coarse Lk tiling" limitation. Track that upstream pass, not the restickify lane.

### Lane CDX — `/dev/vfio/80` — All-gather-into-KERNEL + co-bundling substrate (Phase 4 + S2)
1. **S3 — overlap gate.** Extend `overlapInpFetchWithCompute` (`dsm/dsmperf.cpp:3733–3736`) from Conv2D/SparseConv2D to matmul/PriOp consumers. Eligibility already holds for seam-transparent moves (`isSrclayoutChangeStcdp` stays false → `assignCanOverlapInpFetch` true). Smallest exact backend edit; unblocks overlap for the whole lane.
2. **S2 core — generalize `runDcgForInputFetchNeighbor` off `DsTypes::INPUT`-only** to `KERNEL`, and populate/consume the coordinate maps `fillDataInfo` currently finds empty. One backend capability that moves **both** the M4 operand-broadcast and the attention all-gather forward.
3. **Co-bundling redesign** — the frontend/SDSC-wiring half: co-bundle `mul(K)` (and the flash partials) into the consumer program so LX persists producer→consumer. This is Move 3 and the home for the 130 GB/s finding. High effort (deeptools + inductor).

### Cross-lane (after S2 lands)
- **S4 — LSE ring-fold** as `STCDPOpLx` move + SFP `lse_combine` per hop; the reduce lane (Phase 5). Reuse S2's move-half + Codex's two-stage loop-scoped staging. Cost it as a contention chain (not uniform shift).

---

## 5. Dependency order

```
S1 (cost model, CLC)  ─ no blocker ─────────────► ships selectability now
S3 (overlap gate, CDX) ─ small, exact ──────────► unblocks overlap for all moves
S2 (co-bundling, CDX+DEV) ─ the wall ───────────► unblocks flash value + attention all-gather
        └────────────► S4 (LSE fold reduce lane, Phase 5)
Phase-2 spill audit (CLC) runs in parallel throughout (read-only artifacts).
```

## 6. Honesty ledger

- PR1 scatter 1.065× — **MEASURED** (trace). Fresh CLC baseline 12.55 ms/iter — **MEASURED**; relayout-ON at Granite scale currently **rc=1 FAIL**.
- Flash 0-HBM/97-LX/32-plans — **STRUCTURAL**; value **skipped** (`PATCH_MODE=no_h2d,skip_cpu_ref`), 19:19 run today.
- M4 operand-broadcast — value-correct **xor** capacity-safe; 232/258 ALLCLOSE-False; 12 BusFence — **MEASURED**.
- Uniform-shift 54/90/130 GB/s, scatter 36 GB/s, F≈7.3 µs — **MEASURED** (R²≥0.998).
- λ per-hop — **NOT ISOLABLE** (strawman 0.5 µs); any plan leaning on it is unverified.
- Weight roofline NO-GO S≤512; layout-artifact fix 29.8%→73% — **MODELED + MEASURED**.
- Bet 2 cost term — **DONE + 7/7**, not device-measured. Bet 3 LSE fold — **MATH 13/13**, not wired. Bet 1 — **BLOCKED upstream**.
- Carousels — motivation **refuted**; transport finding **confirmed**; net: shelve rotations, build S1–S4.

---

## 7. Round-2 device-run results (2026-07-05, gated runs A + B — MEASURED)

### A. Granite plan-pick (CLC `/dev/vfio/25`, rc=0, 30.65 ms/iter median) — S1 verdict CORRECTED
The real in-compiler planner selections (inductor DEBUG, `cost_model work_division`):

| matmul | M·N·K | selected | `rhs_loaded_once` | cost µs |
|---|---|---|---|---|
| buf6 QKV | 512·6144·4096 | m=4 n=8 | True | 593.7 |
| buf24 out-proj | 512·4096·4096 | m=4 n=8 | True | 399.6 |
| buf33 MLP gate+up | 512·25600·4096 | m=4 n=8 | True | 2350.8 |
| buf36 MLP down | 512·4096·12800 | m=8 n=4 | True | 1210.5 |
| buf14 attn scores | B32·512·512·128 | m=16 n=2 | **False** | 146.0 |
| buf22 attn AV | B32·512·512·128 | m=32 n=1 | **False** | 150.3 |

- **The production cost model already prices operand-identity** via an `rhs_loaded_once` flag — it has evolved past both `cf67411` (my analytical base) and `ah/ring-cost-term`. My pure-Python "selection-neutral" finding was against a stale base.
- The 3 big K=4096 GEMMs land on `m=4/n=8` (matches analytical; ring-inert, cohorts ≤8). The divergences are the **batched attention matmuls** (`rhs_loaded_once=False`): there the operand is genuinely distinct, so cohort>8 wins — and **ring-cost-term's "both operands are identical multicasts" formula would MIS-PRICE them.**
- **Corrected S1 action:** do NOT ship `ah/ring-cost-term` as-is. Fold the measured 36-vs-130 band into production's already-identity-aware pricing, gated on `rhs_loaded_once`. S1 still does not re-route Granite prefill selection → **S2 remains the unlock.** Run also confirmed `ReStickifyOpHBM` spills present (corroborates the Phase-2 audit).

### B. Flash value (DEV `/dev/vfio/31`) — NOT value-correct: causal-boundary bug (adversarially verified)
Clean fp32 oracle, real device execution (relayout OFF, no `PATCH_MODE`), warmup call to absorb the stall:
- **Warmup-stall NaN ruled out:** the warmup call is NaN (the benign `synchronize ... lost completion`, 8×60s), but the **second cached call is FINITE** (absmax 9.80).
- **But flash is NOT value-correct.** `torch.allclose(0.1)` FAIL; `beyond_tol_frac=0.0099`, `maxabs=9.28`. The error is **entirely at small query positions**: `lq<8` are 74–86% wrong, first-512 bucket 7.7%, ~0 beyond ~512; **uniform across all 32 heads**; at bad locations the **device outputs ≈0** (out mean 0.005 vs |ref| 0.233).
- **This is a genuine device defect, not precision/oracle:** CPU-fp16 flash matches the same oracle to 0% beyond tol. The tail is a **causal-window boundary bug** (device zeros the first ~few-hundred query rows).
- **Root cause (hypothesis, well-supported):** the sparse `amax(dim=-1)` real_max/denominator FIXME + untiled `Lk` (`kv_block=Lk//1`; work-division could not tile Lk — would need 256 cores > SENCORES=32) mishandle the boundary where the causal window is tiny. → the **reduce / coarse-tile lane (Bet 1)**, NOT the collective lane. **Flash de-spill value-validation is gated on fixing base flash at the causal boundary first.**
- Note: the workflow subagent's headline ("flash unblocked / dirty-oracle tail") was **over-optimistic and is refuted** by the per-position characterization above.
- **Correctness is Jamie's lane** (root cause: `unsqueeze(-1)` broadcast loses `stride_map` through `TensorArg` → SDSC dense-stride miscompute; `op_spec.py:40`, `spyre_kernel.py:497`, `superdsc.py:231`). OUR flash scope = performance + collectives (§8).

---

## 8. Flash performance + on-chip LX collectives (the production ask) — ANALYTICAL, MEASURED-grounded

**Headline (counter-intuitive):** the LX collective *loses per-handoff in isolation* — a flash operand is only ~1 MiB/head, so `F + band` (15.4 µs shift / 36.4 µs scatter) exceeds the 12.3 µs HBM round-trip. **The win is NOT per-transfer bandwidth.** It is (a) **eliminating the spill** once the producer is co-bundled (operand never touches HBM), and (b) **parallelism**: HBM is ONE shared ~170 GB/s pipe that *serializes* all 32 handoffs (394.8 µs for 64 MiB), while LX rings on disjoint core-groups **overlap**. Do not sell it on wall-time-per-transfer.

**De-spill map:** relayout-ON flash = 550 SDSC with **32 `ReStickifyOpHBM` → 32 `matmul_operand_broadcast` plans** (one/head; `all_gather_replicate`, group_count=4, 8 replicas/group). Each edge: `scaled_keys→QK^T` and `exp_scores→AV` re-read from HBM because the producer is a *separate device program* (LX doesn't persist across programs).

**Perf (measured constants F=7.3 µs; shift 54/90/130 GB/s; scatter 36 GB/s; HBM one ~170 GB/s pipe):**
- Isolated: LX loses (−3 µs shift / −24 µs scatter vs 12.3 µs HBM).
- **Chip-wide: LX wins because HBM serializes and rings overlap** — shift band DOP2=1.6× … DOP32=**26×**; scatter band breaks even only at DOP≥4. **Co-bundling flips the band from the losing 36 GB/s scatter to the winning 130 GB/s shift** — it is the load-bearing fix, not optional.
- **Reduce lane (LSE fold) is 98% fixed-cost** (31 hops × 7.3 µs = 226 µs; payload negligible). **Lk-tiling never wins on throughput in prefill** (untiled 15.4 µs ≪ any tiled+reduce ≥253 µs) — tiling is forced only by LX capacity or the decode merge. Minimize hop count for the fold.
- **Decode seq-shard KV: P/H_kv = 4× is a compute/on-chip-parallelism ceiling, NOT an HBM-BW multiplier** (KV read stays pipe-bound). Fixed merge tax 226 µs = 70% overhead at L=4k → 7% at L=128k ⇒ seq-shard for long context, head-split for short. (Honest correction to the KV-carousel RFC's ×(P/H_kv) BW claim.)

**Build sequence (frontend contract already exists at `layout_allgather_restickify.py:79`, stamped `realized=False`, gated behind 3 default-off `config.py` flags):**
1. **S2 co-bundling** — `bundle.py generate_bundle:195`: fuse `scaled_keys`/`exp_scores` producer into the QK^T/AV consumer program so LX persists (attacks refusal-A). Verify `ReStickifyOpHBM` 32→0. *Unblocked, highest value.*
2. **S2 capacity** — Lk-chunked loop/tile-scoped broadcast so each chunk ≤2 MiB/core (defeats refusal-B / `L3DlOpsScheduler.cpp:1701`); flip `realized=True` (`lx_relayout.py:134/173`); default the flags for flash. *This is where the 32 handoffs become fast-band LX broadcasts.*
3. **S4 reduce lane** — remove the `producer_has_partial` skip (`lx_relayout.py:440-446`), add a `reduce_scatter` topology + `make_lse_ring_fold_contract`. Gated on S2. Enables value-correct Lk reduction + decode merge.
4. **Dividend + decode** — reduce-over-Lk-within-head, scatter-over-heads → output head-split = k_fast out-proj input (zero relayout). Decode KV-carousel: Q broadcast + block-cyclic KV, 1 hop/step. Safety: same-core = free local copy, never self-ring (BusFence).

**Caveats (verify before committing):** the "HBM serializes 32 handoffs" and "F per fold-hop" are analytical (measured constants, modeled contention). ~~the `exp_scores→AV` operand is larger...~~ **RESOLVED (§9 recon):** all 32 plans broadcast `Tensor1` = the matmul RHS (K/V, ~1 MiB/head), *not* `exp_scores` (which is the LHS and stays local) — the 1 MiB assumption holds.

---

## 9. Codex sync (2026-07-06) + re-scoped high-ROI workstream

### 9a. Where Codex is now (MEASURED, both branches)
- **deeptools `2162efb3e` "prototype matmul operand LX collectives"** (`L3DlOpsScheduler.cpp` +85/−18, `LayoutAllgatherRestickify.cpp` +121) and **torch `e4ae1053e` "keep computed clone relayout sources LX eligible"** now **realize the operand broadcast** (`realized=true`, `physical_lowering_status=lowered_loop_scoped_kernel_neighbor`).
- **Granite S512 (backend2162, CLC, today):** two attention operand-broadcast edges realized → **kernel 12.55 → 11.92 ms = measured 1.053×.** Remaining HBM restickifies are weight-format (out of scope).
- Codex's clone-source-LX-eligible patch **is** a concrete co-bundling instance (keeps the producer output LX-pinned so the consumer's `Tensor1` target is LX, not HBM).
- **Real flash SDSC recon (32 plans):** `all_gather_replicate` into `KERNEL` `Tensor1`, `group_count=4`, `replicas/group=8`, `producer_chunks/group=8`, `256` logical transfers, `realized=True`, and **`estimated_tensor_bytes=0`** (backend never prices the move).

### 9b. THE performance finding (MEASURED from the plan's 256 transfers)
The realized broadcast is an **all-to-all within four contiguous 8-core groups** (0–7 / 8–15 / 16–23 / 24–31): hop distances **1–7**, **max physical link occupancy = 16/link**, 32 same-core (free) + 224 cross-core. **This is worse than the 36 GB/s scatter floor (4–9/link) and ~7–13× off the 130 GB/s uniform-shift ceiling.** ⇒ **the 1.053× is entirely HBM-round-trip elimination; the ring transport runs near worst-case.** The mechanism is realized; the *schedule* is naive.

### 9c. Re-scoped workstream — the highest-ROI gaps Codex has NOT solved
Codex owns the operand-broadcast **mechanism** (realized, measured). Our uncontested, high-value space:

| # | Gap (Codex has not touched) | Why high-ROI | First step |
|---|---|---|---|
| **G1** | **Hardware-aware ring *schedule*** — reschedule the realized all-to-all (16/link) as a **uniform-shift carousel** (p→p+1, 1/link) | up to **~7–13×** on the ring-transport term; this is where the confirmed 130 GB/s carousel-transport premise finally pays off (the *schedule*, not the weight-rotation lowering that stays shelved) | route the 256 transfers as a rotation over the arc instead of all-to-all; contiguous-group placement so cross-core rides +1 hops; verify max-occupancy → 1 |
| **G2** | **Reduce lane / LSE ring-fold (S4)** — Codex's commits are *all* operand-broadcast; **zero** reduce/fold | the only path to value-correct `Lk` reduction + the KV-carousel decode merge; Epic Phase 5 uncontested | remove `producer_has_partial` skip (`lx_relayout.py:440`), add `reduce_scatter` topology + `make_lse_ring_fold_contract`; cost as F-dominated linear chain, minimize hops |
| **G3** | **A SEPARATE communication cost model** — `estimated_tensor_bytes=0`; nothing prices the move, so the planner can't tell the 16/link all-to-all from a 1/link shift | without it, G1's win is **unselectable** (same failure as the shelved carousel); it's what makes the planner prefer the good schedule | **build it as its own model, NOT folded into the matmul cost model** (see 9d) |
| **G4** | **KV-carousel decode** — sequence-sharded KV + Q broadcast + LSE merge; Codex is prefill-only | long-context decode ceiling (compute-parallelism, not HBM-BW — §8); composes with G2 | block-cyclic placement fn + Q broadcast; consume G2's fold |

### 9e. G1 modeled — the ring *topology* is REFUTED for attention; the real lever is EXECUTE COUNT (F)

Honest "afternoon check" of the ring/carousel schedule (`g1_ring_model.py`, MEASURED F=7.3 µs/execute, ~140 GB/s raw link). The 16→1 link-occupancy drop is real *structurally*, but **it does not buy time for the attention operand** because F dominates at 128 KiB/chunk:

| operand | naive all-to-all (1 exec, 16/link) | ring carousel (7 exec, 1/link) | winner |
|---|---|---|---|
| **1 MiB/head (attention K/V — the real case)** | **22.3 µs** | 57.7 µs | **naive (ring 2.6× SLOWER)** |
| 2 MiB | 37.3 µs | 64.2 µs | recursive-doubling (3 exec) 35 µs |
| 5 MiB | 82.2 µs | 83.9 µs | bidir-ring 48 µs |
| 16 MiB (MLP-scale) | 247 µs | 156 µs | bidir-ring 89 µs (**2.8× over naive**) |

- **Crossover: the ring beats the 1-execute naive only for operands > 5.2 MiB/head.** Attention K/V is 1 MiB ⇒ ring loses. My earlier "~7–13×" was the raw *bandwidth* ratio and **ignored F**; corrected, the win evaporates at attention scale. **The uniform-shift carousel is the right optimization for *large* operands (MLP/weight-scale), not attention.**
- **The dominant variable at attention scale is EXECUTE COUNT, not topology.** Each execute costs F=7.3 µs; 6 extra ring executes (44 µs) dwarf the ~15 µs of contention they remove. If the current `loop_scoped_input_fetch` pays F *per transfer* (56 cross-core → **461 µs**), then batching to few executes is a **~20× win** — far bigger than any ring reshaping. So **redirected G1 = determine + minimize the operand-broadcast execute count** (is `loop_scoped_input_fetch` one STCDP execute or per-iteration?), not reschedule the topology.
- Bidirectional/recursive-doubling (fewer steps) beat the plain 7-step ring when a ring *is* used — keep them for the large-operand case.

**Revised priority:** **G3 (separate comm cost model)** first — it's what would have *predicted* this F-domination and is the prerequisite for selecting any schedule correctly; then **redirected-G1 = execute-count minimization** in the loop-scoped lowering (potential ~20× if per-transfer today); **G2 (reduce lane)** next; ring-topology (original G1) deferred to large-operand consumers only; G4 after G2.

### 9d. Design principle — the communication cost model is SEPARATE from the matmul cost model
The current Bet-2 term folds a cohort penalty into `hbm_us` **inside** `_matmul_split_cost` — conflating two different resources. The comm cost model must be its **own** model the planner consults for relayout/collective edges, pricing what the matmul model cannot see:
- **per-link transfer count** (1 vs 4–9 vs 16) → band (130 / 36 / worse), not burst size;
- **fixed F ≈ 7.3 µs per STCDP execute**; **same-core = free**; **hop count** (F-dominated for small payloads like the LSE fold);
- **`link occupancy` as a first-class costed resource** (so concurrent collectives serialize correctly).
The matmul cost model prices compute/HBM/PSUM per work-division; the comm cost model prices the on-chip movement. They compose at the seam, they do not merge. (The `estimated_tensor_bytes=0` gap is the concrete symptom: the backend emits the plan but nothing costs it.)
