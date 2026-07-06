# Backend Fail-Closed Checkpoint - 2026-07-06

This checkpoint records the current DLDSC collectives boundary after separating flash attention value correctness from on-chip communication correctness.

## Current Source State

Torch artifact branch:

```text
AdnanHoque/torch-spyre ah/comms-collectives
checkpoint before this note: 819b62b
```

Deeptools prototype branch:

```text
Adnan-Hoque1/deeptools ah/comms-collectives
new checkpoint: f23ab8b85 dcg: fail closed for staged matmul operand conversion
previous checkpoint: 16e9c4f4e ddc: preserve relayout core maps during fold propagation
```

The Deeptools commit is intentionally small: 34 inserted lines across `dsc/dsc2.h`, `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp`, and `dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp`.

## What Changed In Deeptools

The backend now records when a `matmul_operand_broadcast` plan requires layout conversion:

```text
TransferNode::stagedLayoutConversion_.enabled_ = true
```

That metadata is populated from the existing relayout plan fields:

```text
kind
layoutTransformJson
sourceLxTensorJson
targetKernelTensorJson
stagedDestinationJson
```

The PCFG lowering path then refuses to emit direct LX-neighbor ring movement into the final KERNEL operand when this staged conversion marker is present, unless the diagnostic bypass is explicitly set:

```text
DEEPTOOLS_ALLOW_DIRECT_KERNEL_NEIGHBOR_LAYOUT_BYPASS=1
```

This is a defensive correctness boundary. It does not complete the production staged lowering. It prevents the known value-unsafe shortcut from being mistaken for the production solution.

## Why This Boundary Is Needed

The matmul RHS all-gather case is not just a byte copy.

The source bytes are in the producer LX/source layout. The matmul consumes a PT/KERNEL operand layout. A direct ring write into the final KERNEL operand skips the local layout conversion step and was shown value-wrong in synthetic row-pattern tests.

The value-correct shape is:

```text
producer LX/source-layout shards
  -> ring/local gather into source-layout LX staging
  -> local ReStickifyOpLx or equivalent layout conversion
  -> consumer matmul KERNEL RHS tile
```

So the direct KERNEL-neighbor path remains useful only as a diagnostic replay path.

## Validation On DEV

Pod:

```text
adnan-spyre-dev-pf
workroot: /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404
```

Build gate:

```text
cmake --build build-deeptools --target dxp_standalone -j8
PASS
```

Deeptools focused gates:

```text
./build-deeptools/util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
PASS: 25 tests

./build-deeptools/dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
PASS: 2 tests
```

Whitespace gate:

```text
