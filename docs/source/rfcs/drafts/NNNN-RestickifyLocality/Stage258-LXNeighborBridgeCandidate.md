# Stage 258: LX Neighbor Bridge Candidate Sidecar

## Goal

Move the Stage257 descriptor one step closer to lowering.  Stage257 proved that
the compiler can derive the real producer-owned and restickify-owned LX tile
movement plan.  This stage consumes that plan and emits a Deeptools-shaped
streaming PT-LX bridge candidate as a sidecar JSON file.

The sidecar is intentionally not inserted into `bundle.mlir` yet.  The stock
`ReStickifyOpHBM` path remains the executable fallback.

## Code Changes

Added a new default-off flag:

```text
SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE=1
```

When this flag is enabled together with:

```text
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1
```

bundle generation now writes:

```text
restickify_lx_neighbor_streaming_bridge_edge_<idx>.json
```

for each descriptor edge with an available streaming PT-LX materialization
plan.

The emitted bridge candidate:

- materializes every logical 64x64 tile record;
- uses `STCDPOpLx` gather only when source fragments are smaller than a tile;
- uses `ReStickifyOpWithPTLx` for the local PT-LX restickify step;
- contains no `ReStickifyOpHBM` data op;
- records `fallback = ReStickifyOpHBM`;
- records `executable_in_bundle = false` in the descriptor.

## Why Direct 64x64 Tiles

The existing generic full-bridge helper can coalesce simple cases into larger
stripe-shaped data ops.  That is useful as a compact diagnostic, but the active
goal is a bounded tiled PT-LX path.  This stage therefore emits the direct
64x64 bridge form:

```text
coalescing: direct-64x64-tiles
```

That keeps the candidate aligned with the production target: bounded per-core
LX workspace and explicit tile-level movement.

## Validation

Unit/import validation in the pod:

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_tile_ownership_probe.py \
  tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
59 passed in 6.72s
```

Focused descriptor tests:

```text
10 passed in 5.86s
```

Local static validation:

```text
python3 -m py_compile \
  torch_spyre/_inductor/codegen/lx_neighbor_descriptor.py \
  torch_spyre/_inductor/config.py
```

passed.

## Real Compiler Artifact Probes

Probe shape:

```text
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size <size> \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code \
  --output-dir /tmp/stage258-lx-neighbor-direct-bridge-candidate-<size> \
  --fail-on-error
```

### Size 512

```text
restickifies:              1
restickify bytes:          524,288
total tiles:               64
tile records materialized: 64
bridge data ops:           128
bridge coalescing:         direct-64x64-tiles
ops used:                  STCDPOpLx, ReStickifyOpWithPTLx
contains ReStickifyOpHBM:  no
fallback:                  ReStickifyOpHBM
```

This case needs a gather before the PT-LX restickify because each 64x64
destination tile spans four producer row fragments:

```text
producer slices:     mb:32,out:1
destination slices:  mb:4,out:8
max fan-in:          4
max fan-out:         1
```

### Size 2048

```text
restickifies:              1
restickify bytes:          8,388,608
total tiles:               1024
tile records materialized: 1024
bridge data ops:           1024
bridge coalescing:         direct-64x64-tiles
ops used:                  ReStickifyOpWithPTLx
contains ReStickifyOpHBM:  no
fallback:                  ReStickifyOpHBM
```

This is the clean high-signal case:

```text
producer slices:     mb:32,out:1
destination slices:  mb:1,out:32
max fan-in:          1
max fan-out:         1
```

Each 64x64 tile can be represented as one direct `ReStickifyOpWithPTLx` tile
write into the destination-owned LX region.

## Interpretation

We have not completed the production fix yet.  The important change is that
the compiler now produces a concrete Deeptools-shaped bridge candidate from
real generated SDSC metadata, rather than only describing the movement in an
abstract JSON contract.

What this proves:

- the descriptor contains enough information to lower real generated edges;
- non-2048 shapes can produce bounded tile bridge candidates;
- the emitted sidecar can avoid `ReStickifyOpHBM` internally;
- the HBM fallback remains preserved because the sidecar is not executable.

What remains:

- insert the bridge into the normal bundle schedule;
- patch producer and consumer LX endpoints around that inserted bridge;
- prove hardware value correctness;
- add legality gates before any executable replacement of `ReStickifyOpHBM`.

## Artifacts

```text
artifacts/stage258_lx_neighbor_bridge_candidate/
```

