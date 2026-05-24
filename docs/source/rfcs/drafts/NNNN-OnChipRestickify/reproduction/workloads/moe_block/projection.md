# Projected whole-MoE-block speedup from on-chip core-to-core handoffs

Offline projection (2026-05-24). No device run. Combines the **MEASURED** anchor
(`CoreToCoreDataMovementRecipe.md` §9) with the eligible-handoff sizes from
`edges.md` and AIU bandwidths (`reference_aiu_architecture.md`). Every number is
labeled **MEASURED-anchor** or **PROJECTED**.

## The anchor (MEASURED) and a reconciliation it forces

**MEASURED anchor (recipe §9):** eliminating a same-stick HBM handoff saves
**~0.029 ms per MB** of handoff tensor, minus **~0.005 ms STCDP setup** per edge;
net-positive above **~1 MB**. (From the `(a+b.t()+c.t())@d` proof: baseline 1.557
ms vs same-core STCDP 1.313 ms at S=2048, etc.)

**Physical cross-check (PROJECTED, important).** The HBM bus is 166 GB/s shared.
A handoff via HBM is a round-trip (`2 x bytes`), so the *physical* bandwidth
ceiling on the saving is `2 MB / 166 GB/s = 0.0120 ms/MB`. The empirical anchor
(0.029 ms/MB) is **~2.4x larger** than this bandwidth ceiling. That is expected —
the measured anchor is an *end-to-end* saving that also captures 32-core HBM
contention, descriptor/launch overhead, and serialization that the raw-bandwidth
number ignores. **But it means applying 0.029 ms/MB blindly overshoots when the
handoff dwarfs compute** (it can "save" more than the entire handoff costs). I
therefore report a **range**:

- **Low (conservative):** physical HBM round-trip ceiling, 0.012 ms/MB. Pure
  bandwidth, no contention credit.
- **High (measured):** empirical anchor 0.029 ms/MB, capped so the total saving
  cannot exceed the total HBM handoff time present in the baseline.

The truth sits between: MoE is bandwidth-bound (recipe §12.3 names MoE explicitly
as a top on-chip target), so the high end is plausible, but the low end is the
floor I would defend without a device measurement of these exact shapes.

## Eligible vs blocked handoff data (from `edges.md`)

Only **same-stick** edges are addressable today. The eligible same-stick edges in
the MoE block are M8/M9 (gate/up bmm-out -> SwiGLU act), M11 (down bmm-out ->
combine), M2 (rmsnorm -> dispatch broadcast), and the attention residual/norm
edges. The big **layout-changing** edges (M6/M7 dispatch->bmm.in, M10
act->down-bmm.in) are **BLOCKED** on the Compute-CB transpose and are *not*
counted as savings.

| shape (B,Sq,H,INTER,E,K) | eligible same-stick (addressable) | blocked layout-changing | block compute floor* |
|---|---|---|---|
| default `1,128,2048,8192,8,2` | **43.0 MB** / 5 edges | 25.2 MB | ~4.3 ms |
| small `1,128,1024,4096,8,2` | **21.5 MB** / 5 edges | 12.6 MB | ~1.1 ms |
| prefill `1,512,4096,14336,8,2` | **310 MB** / 5 edges | 185 MB | ~60 ms |
| decode `1,1,4096,14336,8,2` | **0.61 MB** / 5 edges | 0.36 MB | ~0.1 ms |

\* PROJECTED compute floor = (3 expert bmms + 4 attn projections) FLOPs at 72
TFLOPS fp16 peak and a blended **35% efficiency** (a deliberately rough placeholder
— MoE bmms at small Tk are bandwidth-bound, so true efficiency is likely lower,
which would *raise* the handoff share and the speedup). Flagged as the softest
assumption in this projection.

**Roughly 63% of the moved MoE activation bytes are on eligible same-stick edges;
37% are on blocked layout-changing edges (the activation->matmul.in transposes).**

## Projected whole-block speedup

baseline ~= compute_floor + HBM round-trip time of (eligible + blocked) handoffs;
on-chip ~= baseline - saving on eligible edges only.

| shape | baseline (PROJECTED) | speedup range (low..high) | net faster |
|---|---|---|---|
| **default** (1,128,2048,8192) | ~5.1 ms | **1.11x .. 1.19x** | 11% .. 19% |
| **small** (1,128,1024,4096) | ~1.5 ms | **1.19x .. 1.39x** | 19% .. 39% |
| **prefill** (1,512,4096,14336) | ~66 ms | **1.06x .. 1.10x** | 6% .. 10% |
| **decode** (1,1,4096,14336) | ~0.13 ms | **~0.95x (regression)** | handoffs <1 MB, below the net-positive threshold; on-chip should be **gated OFF** |

### Realistic headline range

**PROJECTED whole-MoE-block speedup: ~1.1x to ~1.4x (10-40% faster) at the
mid-size hidden dims (1k-2k H, 128-512 tokens) where MoE is bandwidth-bound,
tapering toward ~1.06-1.10x at large prefill (compute O(N^3) dwarfs the O(N^2)
handoff, recipe §9 ii) and to a slight regression at decode-with-1-token (handoffs
fall below the ~1 MB net-positive floor).** The mid-size win is the sweet spot the
recipe's measured crossover predicts (peak relative speedup at 1024, 1.22x).

## Key assumptions and honesty flags

1. **Layout-changing edges are excluded from savings (the honest discount).** The
   single largest MoE activation (the 16.78 MB act tensor at default, 117 MB at
   prefill) feeds the down-projection bmm on its contracted axis (M10) — a
   `stickDimOrder_` flip that needs `ReStickifyOpWithPTLx`, which **faults
   Compute-CB on device today**. Same for dispatch->gate/up.in (M6/M7). I counted
   ~37% of MoE handoff bytes as blocked. If the transpose primitive ever lands,
   the eligible fraction jumps and the speedup roughly doubles its handoff credit.

2. **The act tensor is double-counted in reality, half-credited here.** The
   16.78 MB act tensor is on an eligible edge (M8/M9 produce-side, bmm-out->silu)
   AND a blocked edge (M10 consume-side, act->down.in). I credited the produce-side
   (eligible) and discounted the consume-side (blocked) — net ~half of that tensor's
   round-trip is saved.

3. **Attention is fused SDPA, not bmms.** Per `project_bmm_aware_split`, Spyre
   runs attention as one fused SDPA SDSC, so the score->softmax same-stick
   cross-core edge (Agent C's best single target) is *inside* the fusion and is
   not a separate HBM handoff to eliminate here. The attention contribution to the
   saving is only the residual/norm same-stick edges (small). Almost all the MoE
   block's on-chip leverage is in the **MoE MLP**, not attention.

4. **MoE is bandwidth-bound at low token counts (the favorable regime).** Each
   expert sees Tk tokens; at Tk=128 the bmms are skinny-M and operand-fetch-bound,
   so the handoff is a larger fraction of wall time -> the high end of the range is
   realistic. At large prefill the bmms become compute-bound and the relative win
   shrinks (it does NOT disappear — absolute bytes saved grow, recipe §9 ii).

5. **The compute floor (35% efficiency) is the softest input.** A lower true
   efficiency raises the handoff share and the projected speedup; this projection
   is conservative if MoE bmms run below 35%.

6. **Anchor calibrated on a different graph.** The 0.029 ms/MB anchor came from a
   single fused add+matmul proof, not an MoE bmm. The reconciliation (capping at
   the physical HBM ceiling) is my correction for transplanting it; flagged as
   PROJECTED.

7. **Per-size LX allocation / 2 MB-per-core limit (recipe §9 iii).** A *cross-core*
   round-trip needs 3 live LX regions and cannot fit a >0.66 MB/core slice. The MoE
   eligible tensors sharded across 32 cores are well under that per-core
   (16.78 MB / 32 = 0.52 MB/core at default), and a production *single* cross-core
   move is only 2 regions — so the eligible MoE edges fit. At prefill the per-core
   act slice is 117 MB / 32 = 3.67 MB/core > 2 MB LX, so a single eligible edge
   would need tiling across the move; flagged as an integration constraint, not a
   correctness blocker.

## Bottom line

The MoE block is a strong on-chip target precisely because it moves far more
activation data than attention (the E× expert replication + INTER expansion):
tens to hundreds of MB of handoff per block. **About two-thirds of that data is on
same-stick edges addressable by the proven `STCDPOpLx` primitive today; one-third
is on layout-changing activation->matmul.in edges that are blocked on the faulting
transpose.** Crediting only the addressable two-thirds, the projected whole-block
speedup is **~1.1-1.4x in the bandwidth-bound mid-size regime**, narrowing at
large prefill and turning slightly negative at single-token decode (where it
should be gated off). Landing the transpose primitive would roughly double the
addressable byte share and push the high end further.
