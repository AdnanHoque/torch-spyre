# Bounded Multicast Gather/Restickify Checkpoint

Date: 2026-07-07

This directory archives the first green bounded `multicast` positive test for
the Deeptools `matmul_operand_broadcast` path using the staged
`STCDPOpLx + ReStickifyOpLx` carrier.

## Branch Heads

- Deeptools branch: `Adnan-Hoque1/deeptools:ah/comms-collectives`
- Deeptools commit: `3a4349e62baff978faa21b8cbad376a524658398`
- Previous checkpoint: `071e293cf39dc58bd3b07bcda369a68520be62dd`

The commit is SSH-signed and includes:

```text
Signed-off-by: Adnan Hoque <adnan.hoque1@ibm.com>
```

## What Changed

This checkpoint is test-only relative to `071e293cf`.

The previous commit already enabled the staged gather/restickify production
gate for `broadcast` and `multicast` when the frontend explicitly requests:

```text
realization_strategy = gather_then_restickify
```

This commit adds a physically coherent multicast fixture:

- two producer cores own two distinct source groups;
- four consumer cores form two destination groups;
- each producer group fans out to two consumer cores;
- the group dimension is placed on source `x` and target `out`, matching the
  existing `x -> out` layout rename used by this BMM operand path;
- Deeptools emits staged LX gather plus local `ReStickifyOpLx` before binding
  the consumer matmul operand.

An earlier attempt put the multicast group on target `x`; DCC rejected that
fixture because the generated target `x` element-array metadata only had one
lane. Moving the group onto target `out` gives a coherent bounded fixture
without changing production lowering.

## Validation

Focused Deeptools run on `adnan-cdx-spyre-dev-pf`:

```bash
cd /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools
ninja -C build-deeptools dxp_unit_test util_unit_test -j 16
build-deeptools/dxp/dxp_unit_test \
  --gtest_filter="DxpTestFixture.MatmulOperandBroadcastPattern*GatherRestickifyCompiles:DxpTestFixture.MatmulOperandBroadcastPattern*FailsClosed:DxpTestFixture.MatmulOperandBroadcastChunkCapFailsClosed:DxpTestFixture.PartialViewGather*:DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
build-deeptools/util/util_unit_test \
  --gtest_filter="LayoutAllgatherRestickify.*"
```

Result:

- DXP focused tests: `10/10` passed.
- Utility tests: `32/32` passed.

The multicast backend plan artifact reports:

```text
communication_pattern: multicast
realization_strategy: gather_then_restickify
realized: true
physical_lowering_status: lowered_gather_then_restickify
```

## Archived Files

- `deeptools_bounded_multicast_gather_restickify_3a4349e62.patch`: portable
  patch for the multicast test change.
- `deeptools_bounded_multicast_commit_summary_3a4349e62.txt`: commit summary.
- `bounded_multicast_plan_3a4349e62.json`: emitted backend plan proving the
  lowered multicast path.
- `focused_dxp_tests_3a4349e62.log`: focused DXP unit test log.
- `focused_util_tests_3a4349e62.log`: focused utility test log.
- `multicast_positive_plan_generate_3a4349e62.log`: single positive test run
  used to generate the archived plan artifact.

## Current Read

This moves `multicast` from "fail-closed plus utility-plan covered" to
"bounded positive backend path exists" for the staged matmul operand
gather/restickify carrier.

Remaining communication-substrate work:

- validate the same classes from real generated Torch/Granite SDSCs, not only
  synthetic Deeptools fixtures;
- keep oversized/full-activation cases as WSR/tile-scoping handoff;
- treat `reduce` / `all-reduce` as a later arithmetic communication class.
