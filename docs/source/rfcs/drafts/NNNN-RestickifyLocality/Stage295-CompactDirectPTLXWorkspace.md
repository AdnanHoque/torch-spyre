# Stage 295: Compact Direct PT-LX Workspace

## Summary

This stage moved the direct PT-LX sidecar from a Deeptools
`ReStickifyOpWithPTLx` valid-gap failure to a late export/runtime-packaging
failure. The important compiler-side change is that the direct bridge now
describes its temporary gather buffer as a compact tile-local 64x64 workspace,
and its gather input fragments use compact tile coordinates plus per-fragment
LX start-address offsets back into the producer allocation.

The stock `ReStickifyOpHBM` path remains the fallback. The PT-LX path is still
uncertified and default-off.

## Why This Was Needed

Stage 293 selected the correct direct bridge direction for
`computed_transpose_adds_then_matmul_tuple` at size 512:

```text
direction = output-to-kernel
bridge_kind = direct-ptlx-layout-transform
```

But Deeptools rejected later tiles because the local PT restickify input kept
global tile coordinates. For tile 1, the gather output/restickify input had an
`out_` layout size of 128 while the local output tile had `out_ = 64`.

That mixed two coordinate systems:

- source tensor coordinates, used to locate the producer data
- tile-local workspace coordinates, needed by the local 64x64 PT transform

The new lowering separates them:

- gather input: compact tile coordinates, with LX `startAddr` adjusted to the
  source fragment offset
- gather output: compact 64x64 tile workspace
- PT restickify input/output: compact 64x64 tile-local descriptors

## Code Changes

- `generate_streaming_ptlx_direct_tile_bridge_sdsc` now compacts direct gather
  inputs and outputs before invoking `ReStickifyOpWithPTLx`.
- `_tile_piece_info` accepts a per-fragment `start_addr` override for LX
  fragments.
- `_compact_lx_read_fragments` computes compact tile coordinates while
  preserving the producer LX source location with `_lx_fragment_offset_bytes`.
- `_streaming_value_flow_contract` now treats direct PT-LX producer starts as a
  producer allocation range rather than a single base address, because compact
  reads legitimately start at offsets inside the producer allocation.

## Validation

Focused unit/import checks in the pod:

```sh
python -m py_compile \
  torch_spyre/_inductor/codegen/restickify_lx_dataop.py \
  torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py \
  tests/inductor/test_restickify_lx_dataop.py

python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  -q
```

Result:

```text
55 passed in 8.97s
```

The new focused unit test verifies that tile 1 is described as a compact 64x64
workspace instead of carrying global valid-gap state into
`ReStickifyOpWithPTLx`.

## Deeptools Export Probe

Probe:

```sh
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code \
  --output-dir /tmp/stage295-descriptor-512 \
  --fail-on-error
```

The compiler-side descriptor still reports:

```text
restickifies=1 bytes=524288 byte_hops=0 device_events=0
```

Exporting the generated sidecar with `/tmp/stage65-deeprt-dataop-probe` reached
DCG, DCC, ProgIR verification, DIP, frame-pointer fill, and export. It produced:

```text
/tmp/stage295-direct-sidecar-export-512/execute/1_LXNeighborStreamingReStickifyOpWithPTLx/senprog.txt
```

A simple opcode-prefix count over that `senprog.txt` showed:

```text
HBM=0
SFP=896
PT=256
```

The exporter then returned `SIGSEGV` late in the export path after writing the
program files. This is progress relative to the previous blocker:

- before: Deeptools rejected `ReStickifyOpWithPTLx` valid-gap metadata
- now: the full direct PT-LX sidecar compiles far enough to emit an HBM-free
  program, then fails during late export/packaging

## Non-Signal Probe

A manually truncated 2-tile sidecar failed earlier in DCC with:

```text
DtException: dsc_schedule.size() > 0
```

That result is not treated as evidence against the descriptor shape because the
manual pruning also invalidated schedule structure.

## Current Blocker

The next blocker is not the PT local valid-gap contract anymore. It is the late
Deeptools export/packaging crash for the full 64-tile direct PT-LX sidecar.

The next useful step is to make the full sidecar export cleanly, or to package
the generated HBM-free program through the same runtime path used by the normal
Torch-Spyre bundle and run a value-correct 512 hardware test.

