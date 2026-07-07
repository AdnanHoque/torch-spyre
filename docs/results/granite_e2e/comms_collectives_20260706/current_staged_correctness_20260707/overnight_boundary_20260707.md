# Overnight DLDSC Gather/All-Gather Boundary Update - 2026-07-07

## Short Status

We made real progress on the bounded `matmul_operand_broadcast` / all-gather-restickify path, but it is not complete yet. The important new boundary is that the current clean path can now emit a post-mutated SuperDSC that is descriptor-equivalent to the old value-correct run, but the current full-LX test environment is not a clean value oracle: the same synthetic probe also fails when relayout is disabled.

That means the remaining wrong values should not be attributed to the communication descriptor alone until the current baseline path is green again.

## What Was Proven

Old green reference:

- Run: `/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506/gather_restickify_M16_225240`
- Result: `ALLCLOSE True`, `MAX_DIFF 0.001953125`, `MISMATCH 0 / 4096`
- Artifact copied here: `overnight_boundary_20260707/old_green_run.log`

Current clean bounded path after local debug fixes:

- Run: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/current_staged_allocfix_20260707_103515/staged_M16`
- Result: compiles and runs, but `ALLCLOSE False`, `MAX_DIFF 4.0`, `MISMATCH 4096 / 4096`
- Post-mutated SuperDSC matches old green after normalizing only naming suffixes.
- Artifacts copied here:
  - `overnight_boundary_20260707/old_green_post_sdsc.json`
  - `overnight_boundary_20260707/current_descriptor_equiv_post_sdsc.json`
  - `overnight_boundary_20260707/current_descriptor_equiv_still_failing_run.log`

Current relayout-off baseline in the same full-LX setup:

- Run: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/current_baseline_no_relayout_20260707_105041/staged_M16`
- Result: `ALLCLOSE False`, `MAX_DIFF 4.0`, `MISMATCH 2048 / 4096`
- This means the current M16 full-LX environment is not a clean communication correctness oracle.
- Artifact copied here: `overnight_boundary_20260707/current_relayout_off_baseline_run.log`

## Fixes / Experiments Applied Locally

The local Deeptools debug delta is archived at:

- `overnight_boundary_20260707/deeptools_local_debug_delta.patch`

The meaningful changes tried were:

1. In `dxp/SdscRelayoutInsertion.cpp`, fix final restickify output piece address derivation so final pieces land at the logical destination offsets instead of compact per-core offsets. This made the final descriptor match the old green output-address pattern.
2. In `dxp/SdscRelayoutInsertion.cpp`, force `STCDPOpLx::enSubPieceReuse = false` on the serialized chunk data-op, matching old green.
3. In `dxp/SdscRelayoutInsertion.cpp`, when rewriting the consumer LX allocate node, use `target_kernel_tensor.coreIdToWkSlice_` rather than `computeCoreIdToWkSlice_`. For matmul operands, the tensor distribution can be `out` while the consumer compute split is `mb`; using compute ownership for the tensor allocation is wrong.
4. In DCC, tried the old local `LX -> LX` transfer lowering support and the corresponding register-assignment allowance. This did not change the failing value result in the current environment, so it is not sufficient by itself.

## Important Artifact Comparison

After fixes 1-3, the current post-mutated SuperDSC has:

- `datadscs_`: equivalent to old green
- `coreIdToDscSchedule`: equivalent to old green
- `dscs_`: equivalent to old green after the allocation-coordinate fix

The remaining generated artifacts still differ at the bundle level. Old green uses constant SDSC symbol bindings in `bundle.mlir`; current main uses dynamic input-arg symbol bindings. The current full-LX relayout-off baseline failing suggests this dynamic/full-LX environment issue must be separated from the communication implementation.

## Current Assessment

Progress toward the broader Granite communication goal:

- Scatter/permutation PR1 remains the production-ready part.
- Bounded gather/all-gather is structurally represented and can emit the intended mixed `STCDPOpLx + ReStickifyOpLx` rows.
- The descriptor-level bugs for final addresses and tensor-vs-compute allocation ownership are now identified.
- Value correctness for bounded gather/all-gather is still blocked by current-environment/runtime lowering behavior. Baseline-off failing means we need a clean green baseline before using this probe to judge communication correctness.

## Next Steps

1. Re-establish a clean value oracle on current main/full-LX, or intentionally run the bounded gather/all-gather value tests in the old known-good post-2829 environment.
2. Once baseline is green, rerun `M16/M32/M64` bounded `matmul_operand_broadcast` with the descriptor fixes.
3. If relayout still fails while baseline passes, diff generated `spyrecode.json` and DCC module/program output against the old green run.
4. Only after bounded value correctness is green should we expand this to the flash/Granite spill-removal paths.

This keeps the scope aligned with the North Star: the communication substrate handles bounded resident tiles; WSR owns full-tensor tiling/streaming.
