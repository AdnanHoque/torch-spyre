# DLDSC coordinate-start fix for copy collectives

Date: 2026-07-06

## What changed

The Deeptools DLDSC relayout insertion path was creating `PieceInfo::dimToStartCordinate` from raw `coreIdToWkSlice_` slice ordinals. That is correct only for unit-sized slices. `PieceInfo` start coordinates are element coordinates, so a coarser source split must start at `slice_ordinal * piece_size`.

Example: a producer with two `y` slices feeding a consumer with four `y` slices has producer starts `0` and `2`, not `0` and `1`.

Without this, the producer pieces overlap logically. DCG then creates an STCDP transfer table row for a source subpiece that has no consumer subpiece, and `determineInnerLoopOrder()` previously dereferenced the empty consumer list.

## Why this matters

This is the backend-derived cardinality piece for DLDSC LX relayout. With this fix, the generic relayout path can represent and lower these copy-style communication cases:

- 1 producer shard to partitioned consumers: scatter/fanout by coordinate refinement.
- many producer shards to one consumer: gather/fan-in.
- one producer shard to a subset of consumers: multicast/replicate subset.
- many producer shards to replicated consumers: all-gather/replicate.

This is still copy movement only. Reduce/all-reduce remain a separate arithmetic-combine class and should not be modeled as copy relayout.

## Code locations

Torch branch: `gather-restickify`, SHA `c2faa793a91d`.

Deeptools branch: `gather-restickify`, SHA `57c6f040b02f`.

Patch archived here:

- `deeptools_57c6f040b_coordinate_start_fix.patch`

## Validation

Focused Deeptools relayout gate:

```text
./dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
2 tests passed
```

The cardinality test now covers:

- existing direct remap/scatter-style case,
- gather/fan-in case,
- one-source subset multicast case,
- coarser-to-finer fanout case,
- all-gather/replicated-consumer case.

All-gather/restickify utility gate:

```text
./util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
27 tests passed
```

Torch-side DLDSC contract gate:

```text
python3 -m pytest tests/inductor/test_lx_relayout_dldsc.py
23 tests passed
```

Flash structural compile probe after the fix:

```text
returncode: 0
stdout: SUCCESS
plan_count: 32
plan_kind_counts: {"matmul_operand_broadcast": 32}
plan_realization_strategy_counts: {"loop_scoped_input_fetch": 32}
plan_physical_lowering_counts: {"lowered_loop_scoped_kernel_neighbor": 32}
plan_communication_pattern_counts: {"all_gather_replicate": 32}
ReStickifyOpHBM_total: 0
```

Note: the quick `ReStickifyOpLx_total` field in `flash_structural_summary.json` was produced by raw JSON string counting and over-counts op rows. Use `ReStickifyOpHBM_total`, plan count, and plan lowering/classification as the reliable structural checks for this run.

## Current status after this fix

The DLDSC copy-relayout substrate is now materially stronger: scatter/permutation, gather/fan-in, broadcast/multicast-style replication, and all-gather/replicate are all represented in tests and pass backend lowering gates.

Remaining work for the broader Granite goal:

- Run a fresh full Granite block profile with this Deeptools head to see which non-weight HBM spills remain.
- Continue separating weight/prelayout restickifies from activation spills.
- Treat reduce/all-reduce as a distinct arithmetic movement primitive, not a copy collective.
- Keep flash value correctness separate until the zero-stride/broadcast-view lowering bug is fixed upstream.
