# Bounded Broadcast Artifact Audit - 2026-07-07

This note records a narrow artifact issue found during the nightly handoff.

## Summary

The current Deeptools `ah/comms-collectives` source appears to contain the
intended bounded broadcast/multicast gather-restickify path, but the archived
bounded broadcast JSON in this artifact branch is not a clean proof of that
path.

The original safest interpretation was:

- current code/test intent: bounded broadcast and multicast can use
  `gather_then_restickify`;
- archived multicast JSON: clean proof;
- archived broadcast JSON: stale or captured from the older loop-scoped path,
  so regenerate before sharing it as evidence.
- Deeptools `2ccd5cefbf638e4d7fb04c88ed56a26c93a4459c` fixes one
  diagnostic-only issue where emitted plan JSON could report the selected
  `realization_strategy` correctly but retain a stale `stages` entry from the
  pre-lowering plan.
- Deeptools `320630da56beb2bb12e6c96ae5b016127962353c` adds a focused test
  guard for that diagnostic behavior.

Post-auth update: the bounded broadcast artifact has now been regenerated at a
`320630da`-equivalent local CDX head and is clean. Use the new artifact below
instead of the older `bounded_broadcast_plan_071e293cf.json`.

## Archived Broadcast JSON

File:

```text
docs/results/granite_e2e/comms_collectives_20260707/bounded_broadcast_gather_restickify_20260707/bounded_broadcast_plan_071e293cf.json
```

Key fields currently read:

```text
communication_pattern      = multicast
source_core_count          = 4
consumer_core_count        = 32
group_count                = 4
consumer_replicas_per_group = 8
logical_transfer_count     = 32
realization_strategy       = loop_scoped_input_fetch
physical_lowering_status   = lowered_loop_scoped_kernel_neighbor
stages                     = [..., grouped_multicast, ..., loop_scoped_input_fetch, ...]
```

That is not a bounded broadcast gather-restickify artifact. It is a grouped
multicast/kernel-neighbor artifact. It should not be used as proof that
broadcast lowers through the final gather-restickify carrier.

## Regenerated Broadcast JSON

File:

```text
docs/results/granite_e2e/comms_collectives_20260707/latest_head_validation_20260707/bounded_broadcast_plan_320630da_equiv.json
```

Key fields read:

```text
communication_pattern      = broadcast
source_core_count          = 1
consumer_core_count        = 2
group_count                = 1
consumer_replicas_per_group = 2
logical_transfer_count     = 2
realization_strategy       = gather_then_restickify
physical_lowering_status   = lowered_gather_then_restickify
stages                     = [..., grouped_broadcast, ..., gather_then_restickify, ...]
```

This is the clean bounded broadcast gather-restickify proof.

## Archived Multicast JSON

File:

```text
docs/results/granite_e2e/comms_collectives_20260707/bounded_multicast_gather_restickify_20260707/bounded_multicast_plan_3a4349e62.json
```

Key fields read:

```text
communication_pattern      = multicast
source_core_count          = 2
consumer_core_count        = 4
group_count                = 2
consumer_replicas_per_group = 2
logical_transfer_count     = 4
realization_strategy       = gather_then_restickify
physical_lowering_status   = lowered_gather_then_restickify
stages                     = [..., grouped_multicast, ..., gather_then_restickify, ...]
```

This is a clean bounded multicast gather-restickify proof.

## Source Check

The current Deeptools branch inspected locally was:

```text
Adnan-Hoque1/deeptools:ah/comms-collectives
3a4349e62baff978faa21b8cbad376a524658398
```

Relevant source behavior:

- `dxp/SdscRelayoutInsertion.cpp` enables the gather-restickify path from the
  public `SPYRE_LX_PLANNER_RELAYOUT` flag.
- `shouldUseMatmulOperandGatherRestickify(...)` accepts:
  - `all_gather_replicate`;
  - `broadcast` and `multicast` only when the plan requests
    `realization_strategy == gather_then_restickify`.
- `dxp/test/dxp_unittest.cpp` has
  `rewriteMatmulOperandBroadcastGatherRestickifyFixture(...)`, which sets:
  - `communication_pattern = pattern`;
  - `realization_strategy = gather_then_restickify`;
  - bounded producer/consumer core counts.
- The focused test log records
  `DxpTestFixture.MatmulOperandBroadcastPatternBroadcastGatherRestickifyCompiles`
  as passing.

This points to an artifact-capture problem rather than immediate evidence of a
broken code path.

## Diagnostic Fix

Fix committed on:

```text
Adnan-Hoque1/deeptools:ah/comms-collectives
2ccd5cefbf638e4d7fb04c88ed56a26c93a4459c
```

Patch archive:

```text
docs/results/granite_e2e/comms_collectives_20260707/patches/deeptools_relayout_plan_artifact_stages_20260707.patch
```

The change refreshes `matmul_operand_broadcast` plan artifact `stages` after
`emitMatmulOperandBroadcastPlanArtifact(...)` marks a plan as lowered through
either `gather_then_restickify` or `loop_scoped_input_fetch`.

Test guard committed on:

```text
Adnan-Hoque1/deeptools:ah/comms-collectives
320630da56beb2bb12e6c96ae5b016127962353c
```

Patch archive:

```text
docs/results/granite_e2e/comms_collectives_20260707/patches/deeptools_relayout_plan_artifact_stage_test_20260707.patch
```

The test asserts that the bounded broadcast/multicast gather-restickify compile
path emits a plan artifact containing `gather_then_restickify` and not
`loop_scoped_input_fetch`.

This does not change physical lowering. It only keeps the JSON artifact
internally consistent.

## Remaining Follow-Up

The bounded backend artifact is now regenerated. The remaining question is
whether Granite or flash naturally produces a bounded broadcast/multicast edge
in the current useful workloads. If not, record broadcast/multicast as
backend-supported communication classes that are not exercised by the current
Granite/flash path.
