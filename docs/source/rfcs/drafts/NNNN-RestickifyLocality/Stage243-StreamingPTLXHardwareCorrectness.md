# Stage 243: Streaming PT-LX Hardware Correctness Check

## Goal

Check whether the production-shaped streaming PT-LX bridge is value-correct on
hardware, not only compile-clean.

The target path is the cross-bundle producer-mixed bridge:

```text
producer DLDsc + streaming PT-LX bridge data ops
  -> consumer DLDsc reads bridge output from LX
```

## Device Smoke

A tiny stock Torch-Spyre run succeeded:

```text
device spyre:0
value 2.0
```

The stock HBM restickify baseline for `adds_then_matmul`, size `512`, also
passed:

```text
ok size=512 case=adds_then_matmul restickifies=2 bytes=1048576 byte_hops=0
```

So the device and the probe are healthy.

## Streaming PT-LX Results

### `adds_then_matmul`, size 512

The cross-bundle PT-LX path compiled and launched, but failed correctness:

```text
Mismatched elements: 210465 / 262144 (80.3%)
Greatest absolute difference: 2.17578125
```

The audit row showed an LX-only, contract-valid bridge:

```text
kind: ptlx-streaming-cross-bundle-handoff
tile_size: 64
total_tiles: 64
datadsc_count: 192
max_fan_in: 4
max_fan_out: 1
has_hbm_restickify: false
hbm_placements: 0
value_flow_contract.valid: true
```

### `adds_then_matmul`, size 2048

The row-stripe coalesced 2048 path also compiled and launched, but failed
correctness:

```text
Mismatched elements: 3640383 / 4194304 (86.8%)
Greatest absolute difference: 6.0
```

The audit row:

```text
kind: ptlx-streaming-cross-bundle-handoff
tile_size: 64
total_tiles: 1024
coalescing: row-stripe-direct-output
datadsc_count: 64
max_fan_in: 1
max_fan_out: 1
has_hbm_restickify: false
hbm_placements: 0
value_flow_contract.valid: true
```

So the Python-side value-flow verifier proves only endpoint identity and
absence of HBM placements. It does not prove that the generated data-op bridge
implements the same logical coordinate transform as stock restickification.

## Compact Workspace Diagnostic

The generated data-op bridge originally used global tensor coordinates for
intermediate tile workspace pieces. That compiles, but it is suspicious because
the workspace is a compact per-tile LX buffer.

This stage added a diagnostic-only knob:

```text
SPYRE_RESTICKIFY_PTLX_COMPACT_TILE_WORKSPACE=1
```

When enabled, internal gather/restickify workspace fragments use compact
zero-based tile coordinates while external producer/consumer fragments remain
global.

Result for size `512`:

```text
DtException: 0
file dcg_fe/pcfg_gen/stcdpOp.cpp line 440
```

The Deeptools check at that line is `checkSubPieceCoverage(STCDPOpLx)`, which
requires output subpieces to cover the output piece coordinate range. This
suggests plain `STCDPOpLx` is not a coordinate-remapping gather/scatter. It can
copy between pieces with compatible logical coordinates, but it cannot by
itself compact global source coordinates into a local tile workspace.

## Interpretation

The streaming data-op route is not production-ready.

What works today:

- Stock HBM restickify is value-correct.
- The older Stage203 same-bundle, non-streaming mixed PT-LX bridge was
  value-correct for the high-signal 2048 tuple case.
- Current streaming bridge generation can produce LX-only artifacts and can be
  compile-clean in some bundle shapes.

What does not work today:

- The current streaming data-op bridge is not value-correct at 512.
- The row-stripe coalesced streaming bridge is not value-correct at 2048.
- Compact tile workspace coordinates expose an STCDP coverage limitation.

## Next Step

The next production-shaped fix should stop treating `STCDPOpLx` as a general
coordinate-remapping gather/scatter.

Promising directions:

1. Use the Stage203 non-streaming mixed bridge for the one shape where
   producer/restickify pieces are already compatible, and fail closed to HBM
   for streaming cases.
2. Find or add a Deeptools data movement primitive that can gather global
   source fragments into compact tile-local workspace coordinates.
3. Revisit the value-correct interslice DDL path, because it proved the
   semantic transform but still has high-size allocation limits.

