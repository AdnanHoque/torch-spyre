# Fanout Physical Fixture Probe - 2026-07-07

## Purpose

This probe separates two questions that were getting conflated:

1. Can DLDSC metadata represent copy-cardinality changes such as one-to-many broadcast and many-to-one gather?
2. Can the layout-changing matmul operand physicalizer safely realize broadcast/multicast through the staged `STCDPOpLx + ReStickifyOpLx` carrier?

The answer today is different for the two paths:

- Generic core-work-division LX relayout: bounded copy-cardinality cases are green.
- Flash/attention matmul operand staged relayout: all-gather/replicate is green; broadcast/multicast are not enabled yet because the attempted physical fixture was invalid.

## Current Heads

- Artifact branch: `ah/comms-collectives`
- Artifact head before this probe: `3fa7d21a391d174847a181e14cb7a8a71d1e00e6`
- Deeptools branch: `ah/comms-collectives`
- Deeptools head: `23010446ed4cc91c80288cc1047f6c50c47d6c88`
- Torch branch: `gather-restickify`
- Torch head: `bced14b49acf4fae92ef4df07d2f5229806c672b`

## What Passed

The small `test_core_work_div_incompt` fixture validates bounded cardinality changes in the generic DLDSC relayout path.

Command:

```bash
cd /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools
ninja -C build-deeptools dxp_unit_test util_unit_test -j 16
build-deeptools/dxp/dxp_unit_test \
  --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*:DxpTestFixture.MatmulOperandBroadcastChunkCapFailsClosed:DxpTestFixture.PartialViewGather*"
build-deeptools/util/util_unit_test \
  --gtest_filter="LayoutAllgatherRestickify.*"
```

Result:

- DXP focused tests: `6/6` passed.
- Utility movement-plan tests: `32/32` passed.

Artifacts:

- `clean_focused_after_fanout_probe/rc.txt`
- `clean_focused_after_fanout_probe/dxp_focused.log`
- `clean_focused_after_fanout_probe/util_focused.log`

The key DXP test is `CoreWorkDivIncomptLxRelayoutCardinality`. It rewrites the physical source and consumer maps consistently and checks:

- full producer to sliced consumers;
- sliced producers to full consumer;
- sliced producers to replicated full consumers;
- the same replicated case with dynamic outer dimension.

That is enough to say the generic DLDSC relayout path can express and realize bounded copy fanout/fanin cardinality changes when the physical tensor contract is internally consistent.

## What Failed

I tried to enable `broadcast` and `multicast` in the staged matmul operand carrier by reusing the existing `test_matmul_operand_broadcast_chunk_cap` fixture. That fixture is a 32-core all-gather/replicate matmul operand test.

The speculative patch is archived as:

- `dirty_broadcast_multicast_attempt.diff`

The direct compile attempt failed:

- broadcast: `DtException: maxGrpId <= sysDef.maxGroupID`
- multicast: `DtException: maxGrpId <= sysDef.maxGroupID`

Artifact:

- `fanout_compile_attempt.log`

Then I capped destination cores down to one to check whether this was only a transfer-count problem. Even with a single destination core, broadcast failed with:

```text
DtException: Invalid start address or buffer offset.
```

Artifact:

- `probe_valid_Broadcast_dst1/test.log`
- `probe_valid_Broadcast_dst1/plans/20_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`

## Root Cause

The 32-core all-gather fixture cannot be turned into a valid broadcast/multicast physical test by changing only the classification label and core maps.

The original fixture has real shards on 32 producer cores. Relabeling it as broadcast says one source core owns the full fanout payload, but the LX allocation geometry still comes from the original 32-core sharded operand. That means the generated STCDP rows can reference logical source bytes that the selected source core does not physically own.

Put simply:

- all-gather fixture: every source core owns one real piece;
- broadcast fixture: one source core must own the piece that every destination consumes;
- relabeling all-gather as broadcast does not create that physical source residency.

That is why the staged physicalizer reached DCC with invalid LX addresses. This was a bad fixture, not a proof that broadcast/multicast are impossible.

## Current Supported Classes

| Class | Current State | Evidence |
| --- | --- | --- |
| Scatter/permutation | PR1 path, bounded DLDSC relayout | existing PR1 artifacts |
| Partial gather/subview | Green | `PartialViewGather*` focused DXP tests |
| Generic broadcast-like cardinality | Green for bounded generic core-work-div relayout | `CoreWorkDivIncomptLxRelayoutCardinality` |
| Generic gather-like cardinality | Green for bounded generic core-work-div relayout | `CoreWorkDivIncomptLxRelayoutCardinality` |
| Flash all-gather/replicate with layout conversion | Green in saved full flash replay | `replay_payloads/artifact_payload_20260707_overnight` and `tensor_contract_split_guard_artifact_20260707` |
| Broadcast/multicast with layout conversion before BMM | Not enabled | this probe |
| Reduce/all-reduce | Not in scope yet; arithmetic primitive needed | no implementation |

## Next Implementation Step

Do not keep modifying the 32-core all-gather fixture for broadcast/multicast.

The next useful implementation step is one of:

1. Build a genuinely small staged matmul operand fanout fixture where source LX residency matches the broadcast/multicast contract.
2. Add an explicit fail-closed validation in the staged physicalizer that rejects fanout contracts when the selected source core does not physically cover each destination cell.

The first option proves the physicalizer. The second option prevents false positives while that proof is being built.

The bounded fixture should use:

- broadcast: one source core owns the full logical source cell; several consumer cores request that same cell;
- multicast: one source core per group owns that group's logical source cell; multiple consumer cores in the group request that same cell;
- matching `source_lx_tensor.numWkSlicesPerDim_`, `source_lx_tensor.coreIdToWkSlice_`, `target_kernel_tensor.computeCoreIdToWkSlice_`, and real LX start-address metadata.

## Branch Hygiene

The speculative broadcast/multicast code was not pushed.

After archiving the diff and logs, the CDX Deeptools checkout was reset to the green pushed head:

```text
## ah/comms-collectives...origin/ah/comms-collectives
```

