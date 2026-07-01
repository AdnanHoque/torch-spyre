# Layout AllGather Restickify Backend Hook Report

## Scope

Inspected the DLDSC/PerfDSC relayout and movement paths in the Deeptools fork at commit `4afc4d9f5` on `ah/comms-collectives`, with `origin/master` available locally at `0a9da5eb19d08712383312bb7dec18fbd7caf711`. For the selected backend hook files, `origin/master...HEAD` showed no changes in `dsm/workOptimizer/baseOptimizer/lxopt.cpp` or `dsm/workOptimizer/multiAIUOptimizer/*`; the branch adds only the util checker/test files among the selected paths. The validated contract is currently standalone in `util/LayoutAllgatherRestickify.*`; no existing caller or metadata carrier for `layout_allgather_restickify` was found in DSM/DSC backend code.

## Existing Code Paths

- `dsm/workOptimizer/baseOptimizer/lxopt.cpp:601` `BaseOptimizer::attemptLxOptForBundle(...)` decides whether a normal LX relayout is needed. Its `insRelayoutPsId` lambda starts at `lxopt.cpp:801` and records relayout work in `LxOptMetaData::consCompQinBundleToRelayoutPsIdMap`. The normal trigger is unique-form mismatch between producer and consumer, checked around `lxopt.cpp:1044-1079`.
- `dsm/workOptimizer/baseOptimizer/lxopt.cpp:1645` `BaseOptimizer::modifyPerfDsc(...)` consumes that metadata and mutates the PerfDSC graph. The key relayout helpers are `fillMniQLxQForRelayout` at `lxopt.cpp:2073`, `updateNewRelayoutStcdp` at `lxopt.cpp:2514`, and `consumerSideUpdates` at `lxopt.cpp:3454`.
- `dsm/workOptimizer/baseOptimizer/lxopt.cpp:3732` `checkAndUseLxOpFunc` can convert existing `ReStickifyOpHBM` to `ReStickifyOpLx` only when both input and output bundles are already LX-optimized. It does not insert a grouped all-gather or bind a BMM KERNEL operand.
- `dsm/workOptimizer/multiAIUOptimizer/multiAIUopt.cpp:1291` `multiAIUOpt::insertAllGatherOp(...)` inserts an `OpFuncs::AllGather` quanta around memory quanta for multi-AIU sharding. `identifyDynamicAllGatherQuantas` starts at `multiAIUopt.cpp:5037`, and `moveAllGatherDown` starts at `multiAIUopt.cpp:5080`. This is not a layout/stick restickification path and does not target a BMM KERNEL operand.
- Front-end collective handling exists for real graph `AllGather` nodes in `dsm/commCollectiveFission.cpp` and DSM mapping checks, but that path is for explicit collective ops, not the flash `mul -> ReStickifyOpHBM -> batchmatmul` edge contract.

## Smallest Backend Hook

The smallest backend hook should be in the PerfDSC/LX mutation lane, not in the standalone checker and not in the multi-AIU all-gather optimizer. The most practical insertion point is inside `BaseOptimizer::modifyPerfDsc(...)`, adjacent to `consumerSideUpdates` once the consumer input bundle and producer form are known, but before the normal relayout helper hardcodes `STCDPOpLx`.

At that point the backend has the objects needed to synthesize on-chip movement: `QuantaC` producer/consumer, `QuantaM` memory operands, `PerfLdsInfo` layout/stick metadata, `ComputeOpProperties`, and QC edge APIs. A contract-aware hook there can fail closed unless the edge is exactly `mul -> ReStickifyOpHBM -> batchmatmul`, then choose `ReStickifyOpLx`, validate dimension rename into the consumer layout, and create grouped all-gather replication into the consumer KERNEL operand.

`checkAndUseLxOpFunc` is too late and too narrow because it only changes an existing op enum. `multiAIUOpt::insertAllGatherOp` is the wrong abstraction because it inserts a standalone all-gather op for sharding, not a fused layout/stick transform plus BMM operand movement.

## Prototype Added

Added a fail-closed movement-plan prototype in `util/LayoutAllgatherRestickify.*`:

- Consumes the existing checker result.
- Requires `kind=layout_allgather_restickify`, `communication_class=all_gather`, and optional `communication_pattern=layout_allgather_restickify`.
- Requires the flash edge shape `producer_op=mul`, `restickify_op=ReStickifyOpHBM`, `consumer_op=batchmatmul`.
- Requires producer and restickify core counts to match.
- Derives the flash all-gather from dimension splits, not total core-count ratio: `groupCount=restickify.mb`, `producerChunksPerGroup=restickify.x`, `consumerCoresPerGroup=batchmatmul.mb`, and `logicalTransferCount=groupCount*producerChunksPerGroup*consumerCoresPerGroup`.
- Requires every `restickify_kernel_layout.layoutDimOrder_` dim to rename to a `batchmatmul.*` dim that exists in `consumer_kernel_layout.layoutDimOrder_`.
- Emits a backend-facing plan with stages: `restickify_layout_on_chip`, `grouped_all_gather`, `bind_bmm_kernel_operand`, with `restickifyOp=ReStickifyOpLx` and `consumerOperandDsType=KERNEL`.

This advances beyond structural checking without pretending the PerfDSC graph rewrite exists.

## Remaining Backend Gap

A full backend implementation was not done in this slice because the inspected code does not yet carry the contract metadata into PerfDSC/QC edges, and no existing helper creates the combined primitive needed here: restickify layout/stick transform plus dimension rename plus grouped all-gather into the BMM KERNEL operand. The next implementation step is to add a metadata carrier from DLDSC into the consumer input bundle/edge and call the new plan builder from the `BaseOptimizer::modifyPerfDsc(...)` consumer-side mutation path.

## Validation

Configured a pod-local minimal build with DCC/DR5 disabled and ran:

`./util/util_unit_test --gtest_filter=LayoutAllgatherRestickify.*`

Result: 8 tests passed.
