# Stage 037: Flash Overlap Row Execution Still Mismatches

Date: 2026-05-26

## Purpose

Stage 035 proved that an overlap-prefix `STCDPOpLx` sidecar can be routed
through corelet-1 units, but the generated SDPA result mismatched PyTorch. This
stage split the failure into smaller checks:

- whether synthetic LX buffer placement was colliding with compute scratch;
- whether corelet-1 sidecars are inherently unsafe;
- whether the old same-corelet DCC stitcher failure can be removed; and
- whether same-corelet overlap rows become value-correct once stitched.

## Agent Findings Integrated

Parallel agents found three useful constraints:

- The Torch overlap-prefix sidecar is synthetic. Its buffers are not consumed by
  the copied batchmatmul compute DSC.
- `InputFetchNeighbor` remains the likely value-correct path because it owns the
  "populate this DL input's LX" dataflow and synchronization semantics.
- Stage 033 failed in DCC because `ModuleStitcher` allowed only one
  `ProgramUnitOp` per physical `(core, corelet, component)` schedule slot.

## Experiments

All device probes used:

```sh
tools/onchip_sdpa_sweep.py \
  --lengths 128 \
  --variants warp_overlap_probe \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --warmup 1 --iters 2
```

### High LX Isolation

Temporary Torch patch: allocate the synthetic overlap-prefix sidecar at the top
of the 2 MiB LX range.

Result:

```text
/tmp/sdpa-stage036-high-lx.json
status=failed returncode=255
stderr contains RAS::PCI::BusFence
source_dst_start_addrs: [2094080], [2095104], [2094592], [2095616]
```

Conclusion: near-ceiling LX addresses are not a safe allocator fix.

### Mid LX Isolation

Temporary Torch patch: allocate the synthetic overlap-prefix sidecar at
`MIN_BRIDGE_REGION_BYTES` (256 KiB), away from the low LX addresses used by the
Stage 035 probe.

Result:

```text
/tmp/sdpa-stage036-mid-lx.json
status=failed returncode=1
Mismatched elements: 12644 / 16384 (77.2%)
Greatest absolute difference: 8.59375
source_dst_start_addrs: [263168], [264192], [263680], [264704]
```

Conclusion: low-address collision is not the sole cause. Moving the synthetic
sidecar away from the low scratch area produced the same numeric mismatch.

### Non-Overlap Control

Custom child run: execute tile 0 with mixed-pipeline tile replacement enabled,
but with overlap-prefix disabled.

Result:

```text
/tmp/sdpa-stage036-no-overlap-tile0-child
status=ok
max_abs_error=0.00341796875
```

Conclusion: the copied HBM-backed batchmatmul tile is value-correct when the
synthetic sidecar is not in the same schedule row.

### Corelet-1 Serial Control

Temporary Torch patch: keep four corelet-1 synthetic prefetch dataops, but run
them serially before the copied compute DSC instead of using the overlapped row.

Result:

```text
/tmp/sdpa-stage036-corelet1-serial.json
status=ok
max_abs_error=0.00341796875
debug_components: lxlu1/lxsu1/pe1
```

Conclusion: a corelet-1 sidecar by itself is not enough to corrupt values. The
bad behavior requires the independent sidecar to share the compute schedule row.

### Same-Corelet DCC Stitcher

Temporary Deeptools patch: change `ModuleStitcher` from one
`ProgramUnitOp` per schedule slot to a small ordered list per slot, gated to
`mixed_flash_pipeline_tile` rows where a `prefetch_` data-op and DLDSC share the
same row. This removes the Stage 033 duplicate-unit stitch failure.

Build:

```text
make dxp_standalone -j8
result: success
```

Temporary Torch patch: set overlap-prefix `STCDPOpLx.coreletId` and
`prefetch_corelet_id` to `0`, so the overlapped prefetch uses the same corelet
as the copied compute.

Result:

```text
/tmp/sdpa-stage036-same-corelet-stitch.json
status=failed returncode=1
Mismatched elements: 12644 / 16384 (77.2%)
debug_components: lxlu0/lxsu0/pe0
```

A follow-up prefetch-first ordering variant also failed with the same mismatch:

```text
/tmp/sdpa-stage036-same-corelet-prefetch-first.json
status=failed returncode=1
Mismatched elements: 12644 / 16384 (77.2%)
```

Conclusion: the DCC representation blocker can be removed, and the same-corelet
overlap row now reaches device execution, but the independent synthetic sidecar
still corrupts SDPA values when scheduled in the same row as compute.

## Current Interpretation

The failure is no longer best explained as a simple LX allocation collision or a
pure corelet-1 visibility problem. The evidence points at the row shape itself:
an independent synthetic `STCDPOpLx` sidecar sharing a schedule row with the
batchmatmul compute path is not value-safe, even when the sidecar addresses are
isolated and even when it is routed through the same corelet.

The next value-correct path should switch from independent synthetic sidecars to
`InputFetchNeighbor`-shaped descriptors, so the prefetched movement is attached
to the DL input it feeds and receives the scheduler's existing `NO_COMPONENT ->
LX` transfer and L3LU/LXLU synchronization semantics.

## Pod State

The pod Deeptools tree contains experimental dirty changes in:

```text
dcc/src/Stitcher/ModuleStitcher.hpp
dcc/src/Stitcher/ModuleStitcher.cpp
dcg/dcg_fe/pcfg_gen/stcdpOp.cpp
dsc/dataOpDsc.h
dsc/dataOpDsc.cpp
```

The local Torch tree does not keep the temporary same-corelet or mid-LX
Torch patch because those variants are not value-correct.
