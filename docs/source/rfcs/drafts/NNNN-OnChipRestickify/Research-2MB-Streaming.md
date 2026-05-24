# Streaming large cross-core activation slices through the fixed 2 MB LX budget

> **CORRECTION (2026-05-24).** This doc's "stream through a fixed ~2×128 KB buffer"
> recommendation addresses only the *move* staging. A later code review (see
> `StreamingImplementationPlan.md` CORRECTION + `PerformanceResults.md`) found the
> on-chip handoff also needs the producer *output* and consumer *input* LX-resident,
> which at >4k exceed 2 MB regardless of the move buffer. So move-tiling is necessary
> but **not sufficient** — genuine >4k needs producer/consumer tiling (a fused
> pipeline). Read the recommendation here as the move-tiling sub-component only.

Design document. Research + design only: **no device, no code run; files read only.** Every
hardware/code claim is grounded in a file that was read; inferences are flagged **[INFER]**.

**Sources read:** `CoreToCoreDataMovementRecipe.md` (§1, §7c, §9, §11); memory
`reference_aiu_architecture.md`, `project_ring_aware_restickify.md`, `project_bmm_aware_split.md`;
`/tmp/tier-up/torch_spyre/_inductor/codegen/onchip_bridge.py`;
`/tmp/tier-up/torch_spyre/_inductor/scratchpad.py`; `config.py`;
`/tmp/rt-verbose/debug/sdsc_2_add/senprog.txt` (cross-core proof, S=2048);
`/tmp/dg-verbose/debug/sdsc_2_add/senprog.txt` (degenerate, zero ring);
`/tmp/bench_onchip_multisize.txt`; `/tmp/real_edge_analysis.md`; `onchip_handoff.py`.

---

## 1. Problem statement and the exact LX math

Per-core LX is **2 MB** (`scratchpad.py`: "scratch pad is 2MB = 2<<20 bytes"). Usable ≈ 0.8× =
1.68 MB after backend reserve (`config.py`: `dxp_lx_frac_avail=0.2`). A same-stick cross-core
STCDP bridge buffer costs, per core (`per_core_slice_bytes`, onchip_bridge.py):

```
slice = rows * max(chunk, stick) * 2B,  chunk = split_dim/num_cores,  stick = 64 fp16 = 128 B,
rows  = product of the non-split dims
```

Square S×S, out:32 (rows=S, chunk=S/32):

| S | chunk | per-core slice | single move (2 regions) | round trip (3 regions) |
|---|---|---|---|---|
| 2048 | 64 | 256 KB | 512 KB ✓ | 768 KB ✓ proven |
| 4096 | 128 | 1 MB | 2 MB (zero DL headroom) | 3 MB ✗ NOFIT proven |
| 8192 | 256 | 4 MB | 8 MB ✗ | 12 MB ✗ |

A single move caps near S≈4096; the 3-region proof round trip already NOFITs at 4096 (recipe §9 iii;
`allocate_lx_bases` raises). Real hidden/seq dims (and MoE/attention intermediates) exceed 4k.
Goal: move an arbitrarily large per-core slice cross-core inside fixed 2 MB.

## 2. The pivotal finding: STCDPOpLx already STREAMS

Cross-core proof senprog, S=2048, every core: `L3_MVLOOPCNT|(64<<10)` then ONE `L3_LDU|((31-i)<<14)`,
mirrored by `L3_STU`; 64+64 total = 32 cores × 2 STCDPs. Degenerate = 0/0. The ring move is a
hardware MVLOOPCNT loop bounded by the reserved `lxSize_`/PieceInfo, not by op limits. So streaming
is a footprint+schedule change, not a new primitive. **[INFER]** STCDP honors `dimToSize_` < slice.

Concretely:
- ≤4k: drop the proof scratch, use a 2-region single move — fits with no streaming.
- >4k: shrink the reserved region (stream) and/or split the consumer dim finer.

## 3. Solution families

### Family 1 — Streaming / row-tiling through a fixed buffer
Split each PieceInfo into K = ⌈slice/T⌉ row-tiles; cycle K STCDPs over ONE fixed in+out buffer pair.
- LX: 2T + DL tensors. T ≥ 1 stick (128 B); 64–128 KB is ample. 4096 needs 1 MB → 8 tiles of 128 KB.
- Ring: K transactions/core; total bytes unchanged ⇒ ~166 GB/s/dir; cost = K sync barriers.
- Correct: tiles partition the slice. Inductor: smaller `dimToSize_`, K dscs, mixed_schedule(K).
- Deeptools: same gate; MVLOOPCNT already loops, so likely no new op. **[INFER]** one buffer reused.

### Family 2 — Double-buffer / ping-pong
2 tile buffers/side overlap ring(k+1) with DL-consume(k). LX = 4T. But schedule is all data-ops then
DL → no overlap today. Defer until #1 lands; marginal extra buffer.

### Family 3 — Reduce regions (drop scratch)
3 regions only because reversed scratch is a proof artifact (recipe §9 iii). A real move = 2 regions
(producer+consumer): 2 MB@4096 fits if DL not LX. Fits ~3k clean. Pure inductor. Recommended baseline.

### Family 4 — Finer work-division
out:64 sequential or 2-D m×n halves slice/axis; MUST match consumer numWkSlicesPerDim_. Stacks with #3.

### Family 5 — HBM fallback
buffer+DL > 2 MB → stock HBM. Safe degradation; threshold = slice + DL tensors.

| # | LX footprint | ring tx/core | correct | inductor | deeptools |
|---|---|---|---|---|---|
| 1 Tile | 2T + DL | K=⌈slice/T⌉ | yes | pieces+sched | gate |
| 2 Double-buf | 4T + DL | 1 (overlapped) | yes | +interleave | gate |
| 3 Drop scratch | 2×slice | 1 | yes | yes | gate |
| 4 Co-split | slice/N | N | match shard | yes | gate |
| 5 HBM fallback | 0 | 0 | yes | yes | none |

## 4. Recommendation by size regime (crossovers)

Reference: empirical ~0.029 ms/MB on-chip saving; ring 166 GB/s/dir vs HBM 166 GB/s shared,
saving = bytes/HBM − bytes/ring. Crossovers from slice = S/32 × 2B vs 1.68 MB usable.

- **S < ~1k:** HBM. STCDP overhead > saving (0.95×@512), sub-stick chunks.
- **~1k ≤ S ≤ ~3k:** #3 two-region single move, no scratch, both fit ≤512 KB. Today's primitive.
- **~3k < S ≤ ~6k:** #3 with DL forced to HBM, or #4 finer co-split. 2 MB tight.
- **S > ~6k:** #1 stream 2×128 KB fixed buffer, K=⌈slice/128KB⌉; fall to #5 if buffer+DL>2 MB.
- **Round trip:** cap ~3k (3 regions); production is a single move so this is not the real wall.

## 5. Recommended schedule/datadscs_ sketch
Reuse `make_datadsc`; set PieceInfo `dimToSize_[rows]=T`, `dimToStartCordinate` = i·chunk + t·T.
Build K dscs `0..K-1` over `allocate_lx_bases(2, T)` (one in+out buffer), schedule `mixed_schedule(K)`.
Same STCDPOpLx; lxSize_=2T not full slice. No new datadscs_ shape, no new op.

## 6. Inductor-synthesizable vs needs-deeptools
Inductor: pieces, K-schedule, 2-region drop-scratch, co-split, HBM fallback, per-size bases. Deeptools:
mixed-bundle gate (existing patch). **[INFER]** K data-ops likely no new op (MVLOOPCNT exists).

## 7. Open questions
1. Does STCDPOpLx honor a `dimToSize_` smaller than the full slice over one reused buffer, or
   does correctness need K distinct buffers / a producer-completed barrier? (verify on device).
2. Can the K-tile buffer (2T) coexist with the consumer DL op's own LX tensors under 0.8× = 1.68 MB?
3. K-barrier vs HBM crossover: where does sync overhead make HBM win (~6-8k)? measure.
4. Tile the mb dim or the out-chunk? mb keeps stick whole; out-chunk risks sub-stick.
5. When even streaming + DL won't fit, prefer #5 HBM fallback or #4 finer co-split? both correct.
