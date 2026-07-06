# Flash attention AIU staged layout relayout checkpoint - 2026-07-06

## Result

This run used the pushed `ah/comms-collectives` Torch and Deeptools branches with the layout-changing attention flags enabled. It exercised the full Torch -> DXP -> runtime compile probe on `adnan-spyre-dev-pf`.

Outcome:

- Runtime command returned `rc=0`.
- Script printed `SUCCESS` with CPU reference skipped by the established compile-probe patch.
- SDSC count: `550`.
- `ReStickifyOpHBM` text count: `0`.
- `ReStickifyOpLx` text count: `64`.
- `matmul_operand_broadcast` contract count: `32`.
- Backend plan count: `32`.
- Total logical transfers represented in backend plans: `8192`.
- Every backend plan is `matmul_operand_broadcast / all_gather_replicate / gather_then_restickify / lowered_gather_then_restickify`.

This is stronger than the DXP-only replay: the frontend kept the attention handoffs on LX (`ReStickifyOpHBM=0`, `ReStickifyOpLx=64`) and the backend lowered the layout-changing matmul operand movement with staged gather plus local restickify.

## Required env flags

The run only emitted the useful staged path when both frontend and backend lanes were enabled:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=1
```

Without `SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1` and `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1`, the same script still records matmul operand metadata but keeps the relevant handoff as HBM-backed `ReStickifyOpHBM`, so Deeptools never gets an LX-pinned operand to relayout.

## Artifacts

- `command.txt`: exact run metadata and environment notes
- `run.stdout`, `run.stderr`: raw command output
- `sdsc_and_plan_summary.json`: operation counts and sample SDSC paths
- `sample_sdsc/`: representative SDSCs from the run
- `backend_plans/`: all 32 backend plan JSON artifacts

## Caveats

This is still a compile-probe/value-skipped run because the current flash test has an independent broadcast/zero-stride correctness issue. The result proves structural elimination of the in-scope attention HBM restickifies and successful staged backend lowering on AIU, not final numerical correctness.
