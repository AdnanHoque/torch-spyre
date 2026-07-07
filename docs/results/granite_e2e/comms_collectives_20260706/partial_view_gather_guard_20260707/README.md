# Partial-View Gather Guard - 2026-07-07

This snapshot records the next communication-substrate gap after bounded
all-gather/multicast worked structurally for flash attention.

## Problem

Several remaining Granite and flash HBM handoffs are not whole-tensor relayouts.
They are partial views of a larger producer allocation. Examples seen in prior
QD/artifact analysis:

```text
QKV projection output buf6:
  q view offset 0
  k view offset 4096
  v view offset 5120

Fused MLP front projection output buf33:
  silu/gate half offset 0
  mul/up half offset 12800
```

The communication class is gather-like, but it is not safe to lower as a
generic offset-zero LX relayout. The consumer must read from the producer base
plus a constant source offset.

## Torch Change

Torch now emits `partial_view_gather` metadata from `TensorArg` provenance even
when there is no full relayout plan attached to the edge:

```text
repo: AdnanHoque/torch-spyre
branch: gather-restickify
commit: 6bc8b00d inductor: emit partial-view gather relayout metadata
```

The metadata includes:

```text
kind=partial_view_gather
communication_pattern=partial_view_gather
source_name=<producer buffer>
source_offset_elems=<constant element offset>
producer_core_id_to_device_slice=<producer LX residency map>
requires_staged_realization=True
materialization_pattern=partial_view_gather_to_lx
layout_transform.carrier_hint=lx_partial_view_gather
```

Torch intentionally avoids duplicating an already-staged matmul operand
contract. If an existing plan has `requires_staged_realization=True`, the helper
does not emit a second partial-view record for the same input.

## Deeptools Change

Deeptools now detects the named `partial_view_gather` class before generic
relayout insertion:

```text
repo: Adnan-Hoque1/deeptools
branch: ah/comms-collectives
commit: faa78233e [DXP] fail closed for partial-view gather relayout
```

Current behavior is intentionally fail-closed:

```text
partial_view_gather with source_offset_elems=<N> cannot be lowered by generic LX relayout
```

That guard prevents a value-wrong lowering where Deeptools silently ignores the
source offset and materializes the wrong subview.

## Validation

Torch validation on CDX:

```text
python3 -m py_compile torch_spyre/_inductor/spyre_kernel.py tests/inductor/test_lx_relayout_dldsc.py
partial_view_gather helper smoke with _C stub: passed
```

The focused pytest could not be run in this exact CDX worktree because `_C.so`
was linked against a different Flex ABI. The helper smoke directly exercised
the new classification path.

Deeptools validation on CDX:

```text
cmake --build build-deeptools --target dxp_unit_test -j$(nproc)
./build-deeptools/dxp/dxp_unit_test --gtest_filter="DxpTestFixture.PartialViewGatherFailsClosedBeforeGenericLxRelayout"
./build-deeptools/dxp/dxp_unit_test --gtest_filter="DxpTestFixture.MatmulOperandBroadcastPattern*:DxpTestFixture.MatmulOperandBroadcastChunkCapFailsClosed:DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
./build-deeptools/util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
```

Results:

```text
PartialViewGatherFailsClosedBeforeGenericLxRelayout: 1/1 passed
MatmulOperandBroadcastPattern* / chunk-cap / core-work-div relayout: 5/5 passed
LayoutAllgatherRestickify.*: 32/32 passed
```

## Current Status

This is a correctness guard and metadata proof, not the final physical lowering.

Implemented:

- classify partial-view gather from Torch provenance;
- carry source offset and producer residency into DLDSC metadata;
- prevent Deeptools from lowering the class through generic offset-zero relayout;
- keep existing bounded all-gather/broadcast/multicast tests green.

Still missing:

- bounded offset-aware physical realization for `partial_view_gather`;
- value proof that a non-zero producer subview is copied into the expected LX
  destination;
- Granite/flash structural probe showing the corresponding HBM spill removed.

This remains communication-substrate work. If the partial view is too large to
materialize as one resident tile, WSR must tile the region first.

