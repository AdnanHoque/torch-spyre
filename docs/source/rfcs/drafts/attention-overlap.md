# Attention compute–movement overlap (warp specialization) — research branch

Research / iteration branch for a new direction: **hide data movement behind
compute** on Spyre by co-scheduling it across the device's independent execution
pipelines — the flash-attention / warp-specialization move — starting with prefill
attention. This branch is based on the on-chip handoff work (`tier0-tier1-onchip`)
because the overlap path extends the same mixed-SuperDSC machinery.

> Full scoping, the measured Phase-0 packet, and the backend co-design detail live
> in the internal briefing repo
> (`github.ibm.com/Adnan-Hoque1/spyre-onchip-core-to-core`, `frontiers/attention-overlap.md`).
> This file is the torch-spyre-side working tracker and is kept free of
> backend-internal specifics.

## The idea (one paragraph)

The on-chip handoff eliminated the HBM round-trip for a producer→consumer
activation handoff. The next lever is **overlap**: run the remaining data movement
(KV loads, the cross-core handoff) *concurrently* with compute instead of before
it, so memory time is hidden behind matmul time. On GPUs this is warp
specialization (FlashAttention / CUTLASS): producer warps issue async copies while
consumer warps do MMA, software-pipelined through a multi-stage buffer. The Spyre
device has the independent execution pipelines to do the same; the gap is in the
scheduling, not the silicon.

## Phase 0 verdict (measured): GO, extend the mixed-SuperDSC path

- **The premise holds — device-confirmed.** The pipelines already overlap a single
  op's own HBM load behind its compute (a large matmul runs at ≈ `max(compute,
  load)`, not the serial sum). So the hardware can overlap; the work is to extend
  that across the producer→move→consumer handoff.
- **Real, compounding upside.** Overlap attacks *different* bytes than the on-chip
  handoff (which removed the score HBM round-trip, measured ~1.29×): it hides the
  residual Q/K/V/O loads + the cross-core move behind compute. Combined upside
  ceiling ≈ **1.7–2.3× at seq=512**, more at longer prefill (ceilings, not
  attainment).
- **Regime improves with sequence length** — longer prefill pushes on-chip
  attention toward compute-bound, the ideal regime for overlap.
- **Chosen route:** extend the existing double-buffered, mixed-SuperDSC scheduling
  to span the cross-core move (vs. a runtime multi-stream alternative that can't see
  the intra-bundle ops without bundle-splitting + an unverified hop).

## Next steps (torch-spyre side)

1. **Codegen: group consecutive OpSpecs into one SuperDSC** so the producer, the
   cross-core move, and the consumer compile into a single program the scheduler can
   pipeline. Builds on the mixed-SDSC machinery on this branch
   (`torch_spyre/_inductor/onchip_realize.py`, `onchip_bridge.py`, the bundle
   fold).
2. **Splice prototype** of a tiled attention overlap using the reproduction harness
   (the seq=512 attention splice is the starting point), to measure attained
   overlap vs. the Phase-0 ceiling.
3. **Backend double-buffering across the cross-core move** — this is the backend
   (deeptools) co-design piece, tracked in the internal design repo; not torch-spyre
   work. The torch-spyre side feeds it the grouped SuperDSC.

## Status

Research/iteration branch off `tier0-tier1-onchip`. Phase 1 first risk to de-risk:
the cross-move double-buffer extension (backend co-design). Device experiments run
solo on the shared accelerator.
