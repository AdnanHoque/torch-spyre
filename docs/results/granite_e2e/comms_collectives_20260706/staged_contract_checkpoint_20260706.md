# Staged Matmul Operand Contract Checkpoint - 2026-07-06

This checkpoint records the first small code step after separating flash baseline value correctness from LX communication correctness.

## Deeptools Branch

```text
repo: Adnan-Hoque1/deeptools
branch: ah/comms-collectives
commit: eb68de6f7
commit title: util: accept staged matmul operand realization
```

## What Changed

`matmul_operand_broadcast` can now explicitly name the value-correct staged realization strategy:

```text
gather_then_restickify
```

Before this checkpoint, `synthesizeMatmulOperandBroadcastMovementPlan()` accepted only:

```text
loop_scoped_input_fetch
```

That strategy was tied to the direct KERNEL-neighbor path. The synthetic row-pattern evidence showed that direct KERNEL-neighbor movement is not value-correct when the all-gather also needs a local layout conversion.

The accepted strategies are now:

```text
loop_scoped_input_fetch
gather_then_restickify
```

The staged strategy records this stage sequence:

```text
source_operand_shards
grouped_all_gather_replicate
local_layout_conversion
gather_then_restickify
bind_matmul_kernel_operand
```

This is still a contract-level change. It does not claim that production physical lowering is complete.

## Validation

Ran on `adnan-spyre-dev-pf`:

```text
cmake --build build-deeptools --target util_unit_test -j8

./build-deeptools/util/util_unit_test \
  --gtest_filter="LayoutAllgatherRestickify.*"
```

Result:

```text
26 tests passed
```

The new test is:

```text
LayoutAllgatherRestickify.matmulOperandBroadcastAcceptsGatherThenRestickifyRealization
```

## Why This Matters

The contract now names the implementation we actually want:

```text
producer LX/source-layout shards
  -> STCDPOpLx gather into source-layout LX staging
  -> ReStickifyOpLx local layout conversion
  -> final matmul operand layout
```

This keeps the metadata aligned with the value-correct synthetic evidence:

- direct KERNEL-neighbor path: not value-correct for layout conversion;
- staged gather + `ReStickifyOpLx`: value-correct for M16, M32, M64 synthetic row-pattern probes.

## Next Backend Step

Implement `attachMatmulOperandBroadcastGatherThenRestickify(...)` in `dxp/SdscRelayoutInsertion.cpp` as a production-shaped scheduled data-op path:

1. parse the `source_lx_tensor` and `target_kernel_tensor` metadata;
2. allocate temporary source-layout LX staging per destination core;
3. emit an `STCDPOpLx` gather/all-gather data-op into that staging;
4. emit a local `ReStickifyOpLx` data-op into the final consumer operand layout;
5. schedule both data-ops before the consumer `batchmatmul`;
6. reject/fail closed if capacity or metadata is incomplete.

Keep the existing direct KERNEL-neighbor path behind diagnostic guards only.
