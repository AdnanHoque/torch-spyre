# Bounded Broadcast Gather/Restickify Checkpoint

Date: 2026-07-07

This directory archives the first green bounded `broadcast` positive test for
the Deeptools `matmul_operand_broadcast` path using the staged
`STCDPOpLx + ReStickifyOpLx` carrier.

## Branch Heads

- Deeptools branch: `Adnan-Hoque1/deeptools:ah/comms-collectives`
- Deeptools commit: `071e293cf39dc58bd3b07bcda369a68520be62dd`
- Previous clean head: `23010446ed4cc91c80288cc1047f6c50c47d6c88`

The commit is SSH-signed and includes:

```text
Signed-off-by: Adnan Hoque <adnan.hoque1@ibm.com>
```

## What Changed

Production code changes are intentionally narrow:

- Allow the existing staged gather/restickify materialization path for
  `matmul_operand_broadcast` plans when:
  - `DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1`;
  - the plan requires layout conversion;
  - the plan is `all_gather_replicate`, or is explicitly marked
    `realization_strategy = gather_then_restickify` with communication pattern
    `broadcast` or `multicast`.
- Fix the layout-dimension rename helper so a size-1 source dimension can map
  to a size-1 target dimension. Without this, a valid bounded broadcast fixture
  failed with:

```text
source layout dimensions cannot be grouped to target dim x
```

Test changes add a physically coherent two-consumer broadcast fixture:

- one producer core owns the source LX cell;
- two consumer cores both request that same logical source cell;
- Deeptools gathers over LX and locally restickifies into the BMM operand
  layout before the consumer matmul.

This replaces the earlier invalid experiment that simply relabeled a sharded
all-gather fixture as broadcast.

## Validation

Focused Deeptools run on `adnan-cdx-spyre-dev-pf`:

```bash
cd /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools
ninja -C build-deeptools dxp_unit_test util_unit_test -j 16
build-deeptools/dxp/dxp_unit_test \
  --gtest_filter="DxpTestFixture.MatmulOperandBroadcastPatternBroadcastGatherRestickifyCompiles:DxpTestFixture.MatmulOperandBroadcastPattern*FailsClosed:DxpTestFixture.MatmulOperandBroadcastChunkCapFailsClosed:DxpTestFixture.PartialViewGather*:DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
build-deeptools/util/util_unit_test \
  --gtest_filter="LayoutAllgatherRestickify.*"
```

Result:

- DXP focused tests: `9/9` passed.
- Utility tests: `32/32` passed.

The backend plan artifact reports:

```text
communication_pattern: broadcast
realization_strategy: gather_then_restickify
realized: true
physical_lowering_status: lowered_gather_then_restickify
stages: source_operand_shards, grouped_broadcast, local_layout_conversion,
        gather_then_restickify, bind_matmul_kernel_operand
```

## Archived Files

- `deeptools_bounded_broadcast_gather_restickify_071e293cf.patch`: portable
  patch for the Deeptools change.
- `deeptools_bounded_broadcast_commit_summary_071e293cf.txt`: commit summary.
- `bounded_broadcast_plan_071e293cf.json`: emitted backend plan proving the
  lowered broadcast path.
- `focused_dxp_tests_071e293cf.log`: focused DXP unit test log.
- `focused_util_tests_071e293cf.log`: focused utility test log.

## Current Read

This moves `broadcast` from "classified but blocked" to "bounded positive
backend path exists" for the staged matmul operand gather/restickify carrier.
It does not complete the full Granite Epic by itself:

- `multicast` still needs a positive physical fixture, although the production
  gate now allows it when explicitly marked `gather_then_restickify`.
- `reduce` / `all-reduce` remain future arithmetic communication classes.
- oversized full Granite activations should still be handed to WSR/tile
  scoping rather than solved with a custom full-tensor streaming path here.
