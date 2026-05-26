# SW-pipelined flash attention on Spyre — design & build plan (Phase 1)

*Working design for overlapping attention's data movement with its compute via a
co-scheduled mixed SuperDSC. **Phase 1A delivered**: the LX-resident softmax-chain
pass is built and device-measured at **1.88× vs production torch-spyre SDPA** on
stock dxp, value-correct, automated, on the `attention-overlap` branch (commit
`935fd62`). See §6. Phase 1B (the deeptools overlap co-design) is in-flight on
the `deeptools-overlap` branch.*

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

## 6. Phase 1A — delivered

The LX-resident softmax-chain pass is built, measured, and committed on this
branch.

- **Headline.** production SDPA **2.557 ms** → pass-on **1.358 ms** = **1.88×**
  on stock dxp, value-correct (max_err 7.6e-5 vs CPU reference, both legs
  VALIDATE_OK). Best of three runs each, harness-native A/B.
- **Mechanism.** A new pass (`torch_spyre/_inductor/onchip_softmax_chain.py`)
  detects maximal same-shard same-core activation chains (identical
  work-division *and* identical per-core HBM bases) and flips producer-output
  *and* every consumer-input LX-resident at coordinated bases, with a
  liveness-aware first-fit packer over the usable LX window. Sibling of
  `onchip_realize` — pure persistence, no data-op, stock dxp.
- **Wire-in.** Runs after `realize_onchip_handoff` in
  `torch_spyre/_inductor/codegen/bundle.py`. Gated by
  `config.onchip_softmax_chain`, env `SPYRE_ONCHIP_SOFTMAX_CHAIN=1`, default off.
- **Commit.** `935fd62` "Realize same-core SDSC chain LX persistence" (3 files,
  +353 lines).
- **Env caveat.** The validation harness used an editable install + a
  `PYTHONPATH=/tmp/attn-boot` shim to pick up the worktree's `torch_spyre/`
  ahead of the installed copy in `/home/adnan/dt-inductor/.venv`. This is
  environment plumbing for the shared single-accelerator setup, not a pass
  limitation — a fresh `pip install -e .` from this worktree fires the pass
  automatically with `SPYRE_ONCHIP_SOFTMAX_CHAIN=1`.
- **Next step (Phase 1B).** The deeptools overlap co-design — extend
  double-buffering across the cross-core move so loads + handoff overlap the
  prior tile's compute. In iteration on `deeptools-overlap`; framed as
  in-flight, not delivered.
