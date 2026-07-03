# CDX DLDSC Collective Investigation Update - 2026-07-03

This note preserves the collective findings that were not explicit in the main summaries at branch tip `origin/ah/comms-collectives` `0572df5f`. It was drafted in a detached pod worktree; no push was performed.

## Existing Artifact Coverage

The current bundle already records these required points:

- Remaining Granite classes: `layout_allgather_restickify` for the `mul/ReStickifyOpLx -> batchmatmul` edge and `all_gather_replicate` / `matmul_operand_broadcast` for the matmul operand path are listed in `README.md` and `collective_status_checkpoint_20260703.md`.
- Count-4 fission standalone pass: `collective_status_checkpoint_20260703.md` records `count=4` passing, and `flash_layout_allgather_20260703/grouped_allgather_sweep/split_count_summary.tsv` shows `FlashGroupedAllgatherSplitDataOpsIn16Start0Count4` and `FlashGroupedAllgatherSplitDataOpsIn16Start64Count4` with zero exit codes through DataOp generation, `senpcfg`, `dcc-opt` senprog, `dcc-opt` smc, and `senulator -v store`.
- Count-8 grouped failure: the same checkpoint and TSV record `FlashGroupedAllgatherSplitDataOpsIn16Start0Count8` failing the large grouped-program path with DCC/program-capacity and store-verification failure.
- Split-env configuration: `flash_layout_allgather_20260703/one_edge_105_isolation_result_20260703.md` records the wrapper shape where Torch sees `DXP_LX_FRAC_AVAIL=0` and the DXP subprocess sees `DXP_BACKEND_LX_FRAC_AVAIL=1`.

## New Explicit Checkpoint

These current CDX-lane observations were missing or only implicit in the artifact notes:

- Monolithic layout-allgather count-8 runtime result: the unfissioned/full count-8 grouped `layout_allgather_restickify` path reaches DXP/runtime but is value-wrong when used as the flash/Granite relayout materialization. This is the same class as the one-edge `105_batchmatmul` result: DXP compilation and AIU launch succeed, then numerical correctness fails with `inf` mismatches.
- Fission-4 DXP result: count-4 fission is a positive compiler/lowering result, not only a proposed granularity. The standalone split-count sweep passes both the start-0 and nonzero-start count-4 cases through DXP/DCC lowering and store verification.
- Fission-4 runtime result: threading the count-4 fission shape into the CDX runtime lane currently segfaults in `JobPlanBuilder::executeAllocate`. Treat this as a runtime plan allocation/integration blocker, separate from the standalone DXP pass.
- Split-env wrapper issue: the wrapper boundary is part of the experiment. If `DXP_LX_FRAC_AVAIL=0` and `DXP_BACKEND_LX_FRAC_AVAIL=1` are collapsed or exported to both sides, Torch planning and backend allocation stop testing the intended split-env condition.

## Raw Artifact Gap

The exact CDX backtrace/log for the `JobPlanBuilder::executeAllocate` segfault has not been copied into this artifact directory yet. When the CDX lane stabilizes, preserve that log next to this note so the runtime crash can be tied to a concrete run directory.

## Current Interpretation

The investigation now has two distinct backend blockers:

- Same logical grouped/layout all-gather, monolithic or count-8, can compile and launch in the flash path but remains value-wrong.
- Count-4 fission keeps the standalone DXP/DCC/store path under the known backend limits, but the integrated runtime path hits `JobPlanBuilder::executeAllocate`.

The remaining Granite classes are still `layout_allgather_restickify` and `all_gather_replicate` / `matmul_operand_broadcast`; PR1 scatter does not cover them.
