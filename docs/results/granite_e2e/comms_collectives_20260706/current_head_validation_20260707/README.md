# Current-head bounded DLDSC validation - 2026-07-07

This archive records current-head validation for the Granite/flash DLDSC LX communication substrate.

## Code heads

- Torch branch: `gather-restickify`
- Torch SHA: `bced14b4 inductor: enrich partial-view gather contracts`
- Deeptools branch: `ah/comms-collectives`
- Deeptools SHA: `9cd9c79c3 [DXP] test partial-view gather offset validation`

## Passing evidence

### DXP relayout insertion

Directory: `dxp_focused_relayout/`

Command filter:

```bash
./build-deeptools/dxp/dxp_unit_test \
  --gtest_filter="*MatmulOperandBroadcastPattern*:*MatmulOperandBroadcastChunkCapFailsClosed:*CoreWorkDivIncomptLxRelayout*:*PartialViewGather*"
```

Result: 8/8 passed.

Coverage:

- bounded broadcast pattern compiles
- bounded multicast pattern compiles
- broadcast chunk cap fails closed
- core-work-division incompatible LX relayout compiles and has expected cardinality
- partial-view gather compiles with source-offset-aware LX address
- missing/invalid partial-view gather source offset fails closed

### Layout/all-gather movement planner

Directory: `layout_allgather_restickify/`

Command filter:

```bash
./build-deeptools/util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
```

Result: 32/32 passed.

Coverage includes flash-style all-gather/restickify planning, grouped matmul operand broadcast/multicast, coordinate-overlap expansion, deterministic plan artifacts, and rejection of unsafe resident replication cases.

## Current Python value-probe status

Directory: `current_head_value_probe_blocked/`

The current-head synthetic Python value probe did not reach SDSC or Deeptools relayout insertion. It fails during PyTorch/Spyre fake-tensor setup before backend plans are emitted:

- initial probe hit current `_C` missing `_get_default_generator` through `torch.manual_seed` after Spyre autoload
- no-seed probe hit fake tensor failure for `torch.tensor(..., device="spyre")`
- scalar-literal and pattern-matcher-off probe still failed before SDSC emission through Spyre copy/fake-tensor interaction in Inductor joint graph setup

This is not evidence against the DLDSC communication path, because no backend plan was emitted and the failing stack is upstream of SDSC generation. The bounded substrate evidence for this checkpoint is the DXP/util host-side validation above.

## Communication-substrate interpretation

Current supported bounded classes:

- scatter/permutation: covered by `CoreWorkDivIncomptLxRelayout*`
- broadcast/multicast: covered by `MatmulOperandBroadcastPattern*`
- all-gather/gather-then-restickify: covered by `LayoutAllgatherRestickify.*`
- partial-view gather: covered by `PartialViewGather*` including offset-aware source address and fail-closed malformed metadata

Remaining gap:

- full Granite activations that exceed bounded LX residency remain WSR-owned. This branch should not implement bespoke full-tensor streaming; it should preserve fallback or fail closed until WSR/tile scoping makes the activation fit.
