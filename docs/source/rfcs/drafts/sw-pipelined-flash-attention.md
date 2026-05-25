# SW-pipelined flash attention on Spyre — design & build plan (Phase 1)

*Working design for overlapping attention's data movement with its compute via a
co-scheduled mixed SuperDSC. Phase 0 verdict: GO, Route A, intra-op pipeline
overlap device-confirmed, combined upside ceiling ~1.7–2.3× at seq512 (compounds
the on-chip handoff). This doc turns that into a concrete build.*

## 1. The pipeline structure (what overlaps what)

Two flavors, build the easy one first.

### Phase 1A — pipeline over **independent tiles** (no online softmax)

Tile attention over the **query / batch-head** dimension (seq_q or bh). Each tile
is a *complete, independent* attention (its own QKᵀ → full softmax → ·V), so there
is **no online-softmax dependency** between tiles — the easy case. Pipeline them:

```
for tile t (over bh or seq_q):
   ASYNC_DMAI : load Q_t, K, V                       } overlap with
   COMPUTE    : S_{t-1}=Q_{t-1}·Kᵀ ; softmax ; O_{t-1}=P_{t-1}·V   } tile t-1
```

The movement (loads + the cross-core score handoff) for tile *t* runs on the DMA
pipelines while tile *t-1* computes on COMPUTE. This delivers the overlap upside
*without* materializing-vs-flash changes — it still materializes each tile's score,
just overlaps the tiles. **This is the Phase-1 target.**

### Phase 2 — true flash (tile over KV + online softmax)

Tile over seq_k with running max/sum rescaling so the full O(seq²) score is never
materialized. Gets the overlap *and* O(seq) memory (longer seq, the >4k regime).
Harder (online softmax + the streaming/Ask-3A residency); deferred.

## 2. Spyre realization (Route A: mixed SuperDSC + double-buffering)

Put the tile loop in **one** SuperDSC so the compiler sees the whole dependency
graph and can map independent tiles' steps onto the 3 pipelines
(COMPUTE / ASYNC_DMAI / ASYNC_DMAO):

- **`dscs_`** : the per-tile DL ops (QKᵀ, softmax, PV).
- **`datadscs_`** : the per-tile loads + the cross-core score handoff (`STCDPOpLx`).
- **`coreIdToDscSchedule`** : interleave tile *t*'s load/move with tile *t-1*'s
  compute, with sync barriers only at true (intra-tile) dependencies.
- **`numBuffers_ = 2`** (double-buffer): the mechanism that already overlaps a
  single op's HBM load behind its compute — extended to span tiles.

The container is built by the existing mixed-SDSC machinery
(`onchip_realize.py` / `onchip_bridge.py` / the bundle fold) plus a codegen change
to group the tile OpSpecs into one SuperDSC.

## 3. The deeptools co-design ask (the one missing piece)

Today the double-buffered chunk scheduler overlaps a single op's **HBM operand
loads** with its compute. The Phase-1 ask: **extend double-buffering to span the
cross-core move and the next tile's load**, so tile *t*'s `STCDPOpLx` + loads
overlap tile *t-1*'s compute. This is the backend (deeptools) piece — the
torch-spyre side feeds it the grouped, tiled SuperDSC. It is the same mechanism,
broader scope; that's why Phase 0 favored Route A over OP_ORDERING.

## 4. Phased build

- **P1.1 — characterize the gap (device, now).** Build a minimal tiled mixed
  SuperDSC; confirm value-correct; measure/inspect whether the current scheduler
  **overlaps the tiles or serializes them** (expected: serial). Output = the
  concrete failing/serial test case + senprog evidence for the co-design.
- **P1.2 — codegen the container.** Group consecutive tile OpSpecs into one
  SuperDSC with `numBuffers_=2` + the interleaved `coreIdToDscSchedule`. Runs
  value-correct (serial until P1.3).
- **P1.3 — deeptools overlap (co-design).** Extend double-buffering across the
  move; measure attained overlap vs the P1.1 serial baseline and the Phase-0
  ceiling.
- **P1.4 — pipeline over real attention tiles**; full A/B vs baseline + the on-chip
  handoff (does overlap compound the on-chip win toward the ~1.7–2.3× ceiling?).

## 5. Risks / open questions

- The overlap itself is gated on the P1.3 deeptools change (co-design); P1.1–P1.2
  are the in-wheelhouse foundation + the test case that justifies it.
- DSC "all-cores-identical" contract vs. tiles doing different work per pipeline —
  P1.1 must confirm a per-tile interleaved schedule is even expressible.
- Tiling granularity vs the 2 MB/core LX budget (per-tile Q/K/V/score residency).
- Phase-1A still materializes each tile's score (O(tile·seq) memory); the O(seq)
  win needs Phase-2 online softmax.
