# Asymmetric Cross-Core Reshard — Implementation Plan

Generalize the on-chip handoff synthesizer from SYMMETRIC redistribution (32->32
matching boundaries, or the reversed `i<->31-i` mirror) to ASYMMETRIC (N->M with
unequal, non-aligned piece boundaries) so the proven same-stick cross-core ring
move (`STCDPOpLx`) works on the REAL Granite block edge: producer `batchmatmul`
sharded `{out:8, in:4}` across 32 cores -> consumer `mul` sharded `{out:25}`
across 25 cores. SAME-STICK only; layout-changing stays the separate blocked-
transpose frontier (the `ReStickifyOpWithPTLx` Compute-CB fault).

Research/design only: no device, no torch.compile, no live-code edits. All paths
absolute; line numbers absolute. Inferences flagged **[INFER]**.

---

## 0. Headline verdict

**Asymmetric same-stick reshard is PURE-INDUCTOR. No new deeptools data-movement
support is required.** The STCDP frontend already implements a fully general
overlap-cell engine. `DcgFE::createSubPieces(STCDPOpLx*)`
(`/home/adnan/dt-inductor/deeptools/dcg/dcg_fe/pcfg_gen/stcdpOp.cpp:227-240`)
loops over every output piece x every input piece, calls `doesPiecesOverlap`
(`:2736-2801`, a rectangle-intersection test on `dimToStartCordinate`/
`dimToSize_`), and for each non-empty intersection calls `insertSubPieces`
(`:2803-2880`) which computes the exact overlap rectangle (max-of-starts, min-of-
ends, `:2862-2880`) and registers one LX->LX sub-move keyed by src memId -> dst
memId in `dtTable_` (`:2885-2909`). So a single STCDP already moves an arbitrary
stick-aligned sub-rectangle of producer-core i's piece to a sub-rectangle of
consumer-core j's piece. The only inductor work is to FEED it producer pieces +
consumer pieces at their NATIVE sizes; DCG computes the cells. This is a strict
generalization of the equal 32x32-cell builder in
`/tmp/transpose_fix/onchip_bridge.py` (`_reshard_cell_pieces`, `:318-347`). The
dxp gate+dispatch patch (recipe section 5) remains the only deeptools dependency.

---

## 1. The overlap-cell algorithm (granite 8->25 worked)

### 1.1 The real edge (verified, no inference on the addresses)
`/tmp/granite_inductor/inductor-spyre/sdsc_fused_add_linear_mul_rms_norm_6_m56h1rzb`:
bmm OUTPUT idx2 @HBM 16850944 == mul INPUT idx1 @HBM 16850944; out segment
1359936 B. Producer bmm `numWkSlicesPerDim_={x:1,out:8,in:4}`, OUTPUT layout
`[out,x]` stick `out`. `in` is the contracted axis: out-band b = cores[4b..4b+3]
summing over `in` (band0=[0,1,2,3], band7=[28,29,30,31] verified). After the in-
reduction the OUTPUT (only `out`) lives on 8 representative cores -> producer
partition = **8 pieces, owner(p)=4p [INFER: rep=first core; confirm via the bmm
OUTPUT PieceInfo memId]**. Consumer mul `{mb:1,out:25}`, INPUT `[out,mb]` stick
`out`, 25 pieces, owner(c)=core c. Both stick `out` -> same-stick (STCDP, no
transpose); shards 8 vs 25 -> genuine cross-core ring.

### 1.2 Cell construction (any N->M, unequal sizes)
Cut logical `out` (length L): prod p covers [p*Lp,(p+1)*Lp), Lp=ceil(L/8); cons c
covers [c*Lc,(c+1)*Lc), Lc=ceil(L/25), both stick-aligned. cell(p,c)=
[max(starts),min(ends)); keep non-empty. Each cell -> one STCDP sub-move:
src=owner(p), dst=owner(c), dimToStartCordinate/dimToSize=cell on out, full on
mb/x. Single tiling axis -> ~8+25-1=32 cells. 2-D co-split -> grid. Worked: L=1600
=25 sticks, Lc=1, Lp=4; c0[0,256) subset p0; c4[256,320) straddles p0/p1 -> 2
cells. fail-closed if cell not whole-stick (DCG :2792 exact, :407 no-gap).

---

## 2. Synthesizer changes (file-by-file)

### 2.1 `onchip_bridge.py` — add `build_asymmetric_reshard_bridge`
The symmetric builders (`_piece_info` `:119-151`, `build_same_layout_bridge`
`:289-307`) assume one fixed chunk = `iter_sizes[split]/num_cores` and a 1:1
logical-slice<->core map. Asymmetric needs explicit, unequal pieces. Add:

- `_partition_pieces(stick_dim, owners, lengths, starts, layout, iter_sizes,
  base)`: N pieces; piece k has dimToStartCordinate[stick_dim]=starts[k],
  dimToSize_[stick_dim]=lengths[k], full on other dims, memId=[owners[k]]. N may
  differ from M. (Equal-cell `_reshard_cell_pieces` in /tmp/transpose_fix is the
  special case.)
- `build_asymmetric_reshard_bridge(prod_owners, prod_lens, prod_starts,
  cons_owners, cons_lens, cons_starts, layout, stick_dim, iter_sizes, src_base,
  dst_base, num_cores)`: IN labeledDs = 8 producer pieces, OUT labeledDs = 25
  consumer pieces, ONE STCDPOpLx datadsc, `mixed_schedule(1, num_cores)`,
  coreIdsUsed_=0..31. Cells are left to DCG `createSubPieces`; do NOT pre-cell.
  Fallback: if DCG rejects unequal IN/OUT counts, pre-expand to cells with
  `_reshard_cell_pieces` generalized to NxM.

### 2.2 `onchip_realize.py` — add `realize_asymmetric_handoff`
Read both `numWkSlicesPerDim_`. Gate: same-stick AND `is_same_shard`==False
(`:213-226`). Map producer (out,in) -> 8 out-owners (in contracted), consumer ->
25 owners. `allocate_lx_bases(2, slice~170KB)` fits 2MB; fail-closed else. Reuse
`apply_lx_flip`/`detect_onchip_edge`/fold (`:259-409`); mixed_schedule(1).

### 2.3 selection: bundle.py detects bmm-out idx -> mul-in idx HBM match, dispatch
asymmetric (8 vs 25) vs same-shard.

## 3. STCDP feasibility verdict — PURE-INDUCTOR
:229 out x in; :2792 rect overlap; :2862-80 cell; :2885 memId; :264 pMemID!=cMemID
-> L3LU/L3SU ring. ~32 dataops>2 hits isNotAN_reqFullUnroll (:803), fine. No
equal-N, no aligned boundaries. PURE-INDUCTOR; reuse dxp patch only.

## 4. Correctness + validation

**Value correctness.** cell(p,c) carries exactly out-band p's overlap with
consumer band c; cells partition `out`, so every consumer element arrives from
the right producer fragment, once. STCDP is byte copy (same-stick) -> bit-exact.

**Offline structural gate.** assert union of cells == [0,L), no gap/overlap, each
whole sticks; reconstruct the 25 consumer pieces from cells -> must match cons
partition. (Mirrors DCG `checkSubPieceCoverage` :407.)

**Device.** spliced granite bmm->mul: value-correct vs CPU + L3_LDU>0 (cross-core
ring) + negctrl (remove senprog -> fail). Compare dg=0/rt=64 ring counts.

## 5. Scope / effort / risk

~250 LOC pure-inductor (2 builders + realize + detect), symmetric verifies the
common axis. PURE-inductor: 0 deeptools changes beyond the existing gate. Risks:
(a) in-reduction rep-core owner [INFER]; (b) sub-stick cells if L not 25/8-stick;
(c) NxM>2D grid count; (d) 2MB LX. Same-stick ONLY (layout-change blocked).
Composes: stream tiling for >4k cell, dynamic addr for variable L.
