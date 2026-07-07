# Overnight Status - 2026-07-07 Late Checkpoint

## Short Version

We made real progress, but the full goal is not complete.

Green today:

- scatter/permutation PR1 path;
- bounded generic DLDSC core-work-div relayout, including copy-cardinality cases that look like broadcast and gather;
- partial-view gather;
- flash/attention all-gather plus layout conversion through staged `STCDPOpLx + ReStickifyOpLx`;
- bounded BMM-operand broadcast plus layout conversion through staged `STCDPOpLx + ReStickifyOpLx`;
- saved full flash DXP replay after the default chunk-policy and tensor-contract split guard fixes.

Not done:

- multicast with layout conversion before BMM;
- reduce/all-reduce;
- full Granite spill removal across oversized activations without WSR.

## Branches

- Torch artifact branch: `ah/comms-collectives`
- Deeptools implementation branch: `ah/comms-collectives`
- Deeptools clean head: `071e293cf39dc58bd3b07bcda369a68520be62dd`
- Torch prototype head on CDX: `bced14b49acf4fae92ef4df07d2f5229806c672b`

## Validation Run

Clean focused run on `adnan-cdx-spyre-dev-pf`:

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

Artifacts:

- `fanout_physical_fixture_probe_20260707/clean_focused_after_fanout_probe/rc.txt`
- `fanout_physical_fixture_probe_20260707/clean_focused_after_fanout_probe/dxp_focused.log`
- `fanout_physical_fixture_probe_20260707/clean_focused_after_fanout_probe/util_focused.log`
- `bounded_broadcast_gather_restickify_20260707/focused_dxp_tests_071e293cf.log`
- `bounded_broadcast_gather_restickify_20260707/focused_util_tests_071e293cf.log`
- `bounded_broadcast_gather_restickify_20260707/bounded_broadcast_plan_071e293cf.json`

## Fanout Finding

The generic `test_core_work_div_incompt` fixture already validates bounded
cardinality changes in the core-work-div relayout path. Its cardinality test
covers:

- full producer to sliced consumers;
- sliced producers to one full consumer;
- sliced producers to replicated full consumers;
- replicated full consumers with a dynamic outer dimension.

This is evidence that the DLDSC contract can express these copy-cardinality
classes and Deeptools can realize them when the physical source and destination
contracts are internally consistent.

## Failed Experiment And Follow-Up

I attempted to enable staged BMM-operand broadcast/multicast by reusing the
32-core `test_matmul_operand_broadcast_chunk_cap` all-gather fixture.

That was the wrong fixture. It has real source shards spread over 32 producer
cores. Relabeling it as broadcast claims one source core owns the full fanout
payload, but the physical LX residency still comes from the 32-core sharded
operand. DCC then sees invalid STCDP source/destination information.

Artifacts:

- `fanout_physical_fixture_probe_20260707/dirty_broadcast_multicast_attempt.diff`
- `fanout_physical_fixture_probe_20260707/fanout_compile_attempt.log`
- `fanout_physical_fixture_probe_20260707/probe_valid_Broadcast_dst1/test.log`

Failure modes:

```text
DtException: maxGrpId <= sysDef.maxGroupID
DtException: Invalid start address or buffer offset.
```

The key lesson is that broadcast/multicast need a physically valid fanout
fixture, not a relabeled all-gather fixture.

Follow-up completed later on 2026-07-07:

- Added a physically coherent bounded two-consumer broadcast fixture.
- Enabled the staged gather/restickify path for `broadcast`/`multicast` only
  when the frontend explicitly marks the plan as
  `realization_strategy = gather_then_restickify`.
- Added a unit-dim layout grouping fix so a size-1 source dimension can map to
  a size-1 target dimension.
- Re-ran focused DXP/util tests green at Deeptools `071e293cf`.

## Current Architecture Read

There are two different mechanisms in play:

1. Generic DLDSC core-work-div relayout.
   This handles bounded copy relayout between a producer tensor distribution
   and consumer compute distribution. It is already green for several
   cardinality changes.

2. Layout-changing matmul operand relayout.
   This is the flash/attention path that gathers LX pieces and then locally
   restickifies into the BMM operand layout. It is green for
   `all_gather_replicate`, but not yet proven for broadcast/multicast.

Keeping these separate matters. The first path says "copy the tensor view to
the consumer ownership." The second path says "copy pieces and transform layout
before feeding a BMM operand." The second has stricter physical-address and
schedule constraints.

## Next Work

Next useful implementation step:

1. Build a small, physically valid staged matmul operand fanout fixture:
   - broadcast: one source core really owns the full source cell;
   - multicast: one source core per group really owns that group's cell;
   - target compute uses several destination cores per source group;
   - source/target tensor metadata and LX start addresses agree.
2. Only then enable staged physical lowering for broadcast/multicast.
3. Add fail-closed validation so invalid fanout contracts fail before DCC
   attempts to schedule impossible STCDP rows.

Do not build a custom full-tensor streaming system here. If a valid bounded
tile works but the full Granite activation is too large, that is a WSR/tile
scoping handoff, not a communication-substrate problem.
