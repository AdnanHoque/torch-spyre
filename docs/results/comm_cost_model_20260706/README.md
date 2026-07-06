# Communication Cost Model (G3), Planner Seam, and LSE Reduce Lane (G2)

This directory holds the analysis and orchestration plan for the on-chip LX
communication-collective cost-model work on this branch
(`ah/comm-cost-model-g3`). It targets Epic
[torch-spyre#3049](https://github.com/torch-spyre/torch-spyre/issues/3049)
(remove avoidable Granite HBM activation spills with on-chip LX collectives).

## What's on this branch (code)

| Commit | Adds |
|---|---|
| `31e76c7` | **G3** — `torch_spyre/_inductor/comm_cost.py`, a standalone communication cost model (+ `tests/tensor/test_comm_cost.py`, 20 tests) |
| `fb07bb4` | Docstring/adapter cleanup; the **flag-gated planner seam** in `work_division.py`; `comm_edge_from_plan` adapter in `lx_relayout.py` |
| `d73c6c7` | **G2** — the LSE ring-fold **reduce lane**: `lse_fold_ref.py` (value oracle), `layout_allgather_restickify.py`, a config gate, and the `lx_relayout.py` reroute (+ `tests/tensor/test_lse_fold.py`, 16 tests) |

**Everything is flag-gated OFF by default** — the branch is a zero-change no-op
until a flag is turned on:

- `SPYRE_COMM_COST_SEAM` — the planner seam (comm cost priced as a separate
  additive edge term instead of folded into `hbm_us`).
- `SPYRE_LX_PLANNER_RELAYOUT_REDUCE` — the LSE reduce lane realization.

## Files here

- `ORCHESTRATION_PLAN.md` — the full device-grounded plan (verified current-state
  matrix, the RFC go/no-go, the Codex sync, the G1 finding, and the pod-parallel
  build plan). The single source of truth for the workstream.
- `g1_ring_model.py` — the seed schedule model (pure Python; the calibration
  `comm_cost.py` reproduces). Run it standalone to see the schedule costs.
- `orchestration/` — the multi-agent workflow scripts used to produce the
  analysis (recon/gate, build, device runs, G3, G2, CLC validation). Reproducibility
  artifacts; they embed pod-specific paths from the dev environment.

## Key findings (one line each)

- **The carousel RFCs are shelved.** Weight-carousel's motivation was a layout
  artifact already fixed a cheaper way, NO-GO at production S≤512, and its
  DRAM-byte win is unselectable. KV-carousel's channel-affinity premise is false
  (flat HBM) and its fold rides the slow ring pattern.
- **The deep reason (G1, MEASURED):** on-chip moves are **F-dominated** (~7.3 µs
  fixed per STCDP execute), not bandwidth-bound, for the operands we actually
  move; and the ring-vs-naive crossover (~5.2 MiB) sits **above** the ~2 MiB LX
  capacity — so the bandwidth-optimal ring carousel **never wins** for an
  LX-resident broadcast.
- **G3 prices what the matmul model can't:** per-link transfer count → band
  (36 vs 130 GB/s), F-per-execute, hop count, same-core-free, and the LX-capacity
  ceiling — as a **separate** model composed additively at the planner seam.
- **G2 is a value-correctness lane, not a speedup:** the tiled online-softmax must
  fold; priced honestly as F-dominated tree-fold (~37 µs vs ~230 µs plain ring at
  P=32), with a head-split layout dividend into the out-projection.

## Status

- **Validated (MEASURED):** `comm_cost` 20/20 and `lse_fold` 16/16 pass on the Mac
  (standalone) and in the real pod torch env on `adnan-clc` (36/36).
- **Device-gated (pending):** three flags stay OFF until a healthy AIU confirms
  them — (a) `SPYRE_COMM_COST_SEAM` reproduces the device-verified Granite
  work-division selections; (b) the backend SFP `lse_combine` primitive for G2;
  (c) S2 co-bundling. All AIU devices were degraded at authoring time.
