# Stage 127: Tile Ownership And Inter-Slice Probe

## Goal

Move the LX-to-LX restickify prototype from "launches but computes wrong values"
toward a value-correct internal edge.  Stage 126 showed that the raw boundary
patch can remove the consumer HBM reload, but the bridge corrupts the transposed
intermediate.

## New Diagnostic

Added `tools/restickify_tile_ownership_probe.py`, a hardware-free diagnostic for
64x64 tile ownership.  For a 2048x2048 transpose-shaped edge it reports:

- `row -> col` ownership requires 992 of 1024 tiles to move between cores.
- `col -> col` ownership requires 0 tile moves.
- deterministic value fingerprints distinguish:
  - no tile exchange but local transpose,
  - tile exchange but no local transpose,
  - tile exchange plus local transpose.

This makes the current blocker easier to identify from a few sample elements.

## Deterministic Hardware Probe

Fixture:

```python
def fn(a, b, c, d):
    u = a + (b + c).t()
    return u, u @ d
```

Inputs:

- `b[i, j] = i`
- `a = c = d = 0`
- expected `u[row, col] = col`

With the DDL bridge, boundary patch, add-corelet skip, and core-division/core-map
continuity enabled:

```text
total_bad 3743744 / 4194304
max_abs 63.0
sample {
  "0,64": 64.0,
  "0,1024": 1024.0,
  "64,0": 0.00009155,
  "127,0": 63.0,
  "128,0": 0.00009155,
  "128,64": 64.0
}
```

Interpretation: tile ownership is mostly being exchanged correctly, but the
value pattern matches "right 64x64 tile, wrong in-tile coordinate."  The
remaining failure is therefore not primarily the producer/restickify/consumer
core mapping.  It is the local transformation semantics inside the fetched tile.

## Inter-Slice Probe

Deeptools contains an `interslicetranspose_fp16` op and an
`inter_slice_transpose.ddl` template in source.  The installed pod image only
ships `restickify.ddl`, so the first inter-slice probe needed a temporary
`DEEPTOOLS_PATH=/tmp/deeptools-share` overlay containing the source templates.

Results:

- Renaming the bridge opfunc to `interslicetranspose_fp16` gets past the old
  corelet split abort only if the pre-DDC shim explicitly skips the renamed
  bridge.
- With the stock source `inter_slice_transpose.ddl`, DXP aborts in DDL
  conversion:

  ```text
  DtException: Could not find any suitable dimension mapping
  ddc/ddl/ddl_conversion.cpp line 2493
  ```

- A narrower temporary 2D version of the template still fails the same dimension
  mapping step.
- Forcing the bridge input/output primary layouts to share the input or output
  global layout also fails the same mapping step.

Interpretation: `interslicetranspose_fp16` is not a drop-in opfunc replacement
for the synthetic ReStickify-shaped SDSC.  It appears to need a different
DSC/DDL contract, not just a different op name.

## Current Blocker

The current DDL bridge gives us an LX-only internal edge, but not value-correct
transpose-shaped restickification.  The observed failure decomposes into two
separate requirements:

1. Move/fetch the correct producer-owned tile to the destination core.
2. Apply the correct in-tile coordinate transform before the consumer reads it.

Core continuity helps with requirement 1.  The current `restickify.ddl` bridge
does not satisfy requirement 2 for this Torch-Spyre logical view edge.

## Next Step

Stop treating `restickify.ddl` as the final lowering for transpose-shaped
internal edges.  The next prototype should generate a first-class internal-edge
descriptor with both:

- source/destination tile ownership,
- source-to-destination coordinate mapping inside each tile.

Then lower that descriptor either to:

- a Deeptools-native inter-slice transpose DSC that satisfies the real
  `interslicetranspose_fp16` DDL contract, or
- a purpose-built Torch-Spyre temporary DDL template for 2D internal-edge
  transpose, loaded through a template overlay only in prototype mode.

The acceptance criterion remains unchanged: the tuple fixture must retire on
hardware with no `ReStickifyOpHBM`, no consumer `hbm_dst:lx` reload for `u`, and
`u` must be value-correct.
