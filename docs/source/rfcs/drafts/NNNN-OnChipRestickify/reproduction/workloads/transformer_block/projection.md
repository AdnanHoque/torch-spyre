# Whole-block on-chip speedup projection (OFFLINE)

Grounded projection of the on-chip core-to-core speedup for the full transformer
block in `transformer_block_workload.py`, using the device-MEASURED saving anchor
and AIU bandwidths. **No device run.** Every number is labeled MEASURED-anchor or
PROJECTED.

## Inputs

**MEASURED anchor** (from `CoreToCoreDataMovementRecipe.md` §9 + the task brief):

- **~0.029 ms saved per MB** of eliminated same-stick handoff tensor.
- **minus ~0.005 ms STCDP setup** per folded handoff.
- **net-positive above ~1 MB**; below ~1 MB the STCDP overhead can regress
  (recipe §9(i): the S=512 micro-graph regressed to 0.95x because the per-core
  slice fell below one stick).

**Spec bandwidths** (MEASURED hardware spec, recipe §1):
HBM 166 GB/s (= MB/ms), RIU BiRing 166 GB/s/dir, LX ~4.5 TB/s aggregate.

**Sanity floor (PROJECTED).** A single eliminated HBM round-trip moves
`2 x bytes` off-chip through the shared 166 GB/s pipe, i.e. `2*MB/166` ms/MB =
**~0.012 ms/MB**. The MEASURED anchor (0.029 ms/MB) is **~2.4x larger** than this
naive bandwidth floor -- because the eliminated restickify is a *whole extra SDSC
launch* (descriptor build + DCC dispatch + serialized HBM contention across 32
cores), not just the raw transfer. I therefore use 0.029 ms/MB as the headline
and the 0.012 ms/MB bandwidth floor as a conservative lower bound.

## Addressable handoffs (from `edges.md`)

Only **same-stick** handoffs are addressable today. The big-byte ones on the
critical path (sizes at hidden=2048, seq=512, fp16):

| Edge | Bytes | Class | Counted? |
|---|---|---|---|
| E3 Q/K/V linear -> SDPA (x3) | 2.000 MB each | cross-core RING | yes |
| E5 O linear -> residual add | 2.000 MB | cross-core RING | yes |
| E6 residual add -> RMSNorm | 2.000 MB | same-core | yes |
| E7 RMSNorm tail chain | 2.000 MB | same-core | yes |
| E11 gate linear -> SwiGLU mul | 5.375 MB | cross-core RING | yes |
| E12 up linear -> SwiGLU mul | 5.375 MB | cross-core RING | yes |
| E14 down linear -> residual add | 2.000 MB | cross-core RING | yes |

**Excluded and why (honest):**

- **Layout-changing (BLOCKED):** E1/E4/E10/E13 (`activation -> linear input`,
  stick flips `out`->`in`) total **11.375 MB** of handoff that we **cannot** do
  today -- they need the `ReStickifyOpWithPTLx` transpose that faults Compute-CB.
  These are large and on the critical path, so the blocked bucket is *bigger* than
  the addressable one. This caps the achievable win.
- **Attention-score -> softmax:** the single best cross-core target in
  `/tmp/real_edge_analysis.md` is **intra-SDSC** here (PyTorch SDPA fuses to one
  Spyre attention kernel), so it is *not* an HBM handoff in this block and is
  **not counted**. It would only appear if SDPA were unfused.
- **Sub-1-MB handoffs** (RMSNorm reduction scratch `[B,S]`): below the
  net-positive threshold, excluded.
- **Weight restickifies (E2):** prelayout bucket, no runtime primitive.

## Projection (PROJECTED from MEASURED anchor)

Saving per edge = `max(0.029*MB - 0.005, 0)` (anchor) or `2*MB/166` (bw floor),
summed over the addressable same-stick edges. Whole-block denominator is a
PROJECTED block compute time from matmul FLOPs at ~72 TFLOPS fp16 x ~35% MAC
efficiency (skinny seq=512 GEMMs; deliberately conservative).

| Shape (H/nh/S/I) | sum anchor save | sum bw floor | block compute (proj) | **% opt** | **% cons** |
|---|---|---|---|---|---|
| 2048/16/512/5504 (sweet spot) | 0.673 ms | 0.298 ms | 2.14 ms | **~20%** | **~5%** |
| 4096/32/512/11008 | 1.391 ms | 0.596 ms | 8.39 ms | **~12%** | **~3%** |
| 2048/16/2048/5504 (long seq) | 2.826 ms | 1.193 ms | 9.59 ms | **~19%** | **~5%** |

- **% opt** = anchor saving with **80%** of it on the critical path:
  `0.8*save / (compute + 0.8*save)`.
- **% cons** = bandwidth-floor saving with **40%** on the critical path.

### Headline range

**PROJECTED whole-block on-chip speedup: ~5-20%** at the mid-size sweet spot
(hidden 2048, seq 512), with the realistic expectation **~10-15%** once partial
critical-path overlap is accounted for. The win **tapers to ~3-12%** at hidden
4096 because matmul cost (O(N^3)) grows faster than handoff bytes (O(N^2)) -- the
exact taper the recipe §9(ii) measured (speedup peaked at 1024, declined at 4096).

## Assumptions (all PROJECTED, flagged)

1. **Critical-path overlap is the dominant uncertainty.** A handoff saving only
   helps end-to-end if the HBM round-trip is *serialized* with compute. Some
   handoffs (e.g. residual `add` feeding the next norm) are squarely serial;
   others may partly overlap with adjacent kernels. I bracket this with 80%
   (optimistic) and 40% (conservative) on-critical-path fractions. **This is the
   biggest lever and is not measured at the block level** -- the orchestrator's
   real A/B will pin it down.
2. **The anchor transfers from the micro-graph to the block.** The 0.029 ms/MB
   anchor was measured on `(a+b.t()+c.t())@d` add->add handoffs (recipe §9). I
   assume it holds per-MB for the block's linear-output -> elementwise handoffs,
   which are the *same* same-stick class (granite `[4]bmm->[5]mul`). Per-MB
   linearity is an inference; the block handoffs are larger (2-5 MB) so they sit
   comfortably above the 1 MB net-positive gate.
3. **Block compute denominator is PROJECTED, not measured.** ~72 TFLOPS x 35%
   efficiency is a conservative MAC estimate for skinny GEMMs; if real efficiency
   is lower the denominator shrinks and the **% speedup rises**. The orchestrator
   should replace this denominator with the measured baseline `median_ms` from
   `transformer_block_workload.py` (the harness prints it).
4. **Cross-core RING edges are net wins, not just HBM-elim.** Per recipe §9 the
   cross-core round trip tracked the same-core STCDP within ~1% (ring transfer is
   cheap vs the matmul), so I price RING and same-core edges at the same per-MB
   anchor. Production cross-core moves are single STCDPs (2 LX regions: producer +
   consumer) and fit at all these sizes (the 3-region proof round-trip 2 MB/core
   limit does not apply).
5. **Only same-stick edges counted.** The layout-changing bucket (11.375 MB, E1/
   E4/E10/E13) is the bigger half and is explicitly **excluded** -- the projection
   is a floor on what the *proven* primitive buys, not the full handoff budget.
   If the transpose primitive lands, the addressable bytes roughly double and the
   speedup range would roughly double too (PROJECTED).

## Where the win lands (consistent with recipe §12.3)

- **Largest relative win at mid-size hidden (~2048) and shorter seq** -- the
  measured sweet spot. Bandwidth-bound regimes (decode, MoE) would show more.
- **Tapers at hidden 4096 / long seq** -- compute dwarfs the handoff fraction.
- **The MLP edges (E11/E12, 5.375 MB) dominate the saving** -- they are the
  largest same-stick handoffs and recur every block.

## Bottom line

PROJECTED **~10-15% whole-block speedup** at the sweet spot from the proven
same-stick primitive alone (range **~5-20%** across the overlap bracket),
**~3-12%** at larger hidden dims. This is gated by (a) critical-path overlap (the
orchestrator's A/B resolves this) and (b) the fact that the *layout-changing*
half of the handoff budget (the `activation->linear-input` transposes) stays
**blocked** until the Compute-CB transpose primitive lands. Replace the PROJECTED
compute denominator with the harness's measured `baseline median_ms` and the
overlap fraction with the measured on-chip `median_ms` to convert this into the
real A/B number.
