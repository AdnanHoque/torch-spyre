# Stage 035: Flash Prefetch Corelet Execution Mismatch

Date: 2026-05-26

## Purpose

Stage 034 changed Torch-Spyre to emit overlap-prefix flash prefetch
`STCDPOpLx` data-ops with:

```json
{"name": "STCDPOpLx", "coreletId": 1}
```

This stage patched the pod Deeptools tree to honor that field and reran the
smallest `warp_overlap_probe` case.  The immediate question was whether routing
prefetch through `LXLU1/LXSU1/PE1` avoids the Stage 033 stitch conflict:

```text
component=lxlu
corelet=0
schedule_idx=2
```

## Pod State

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
deeptools=/home/adnan-cdx/dt-inductor-mixed/deeptools-onchip-foundation-clean
```

Torch-Spyre was fast-forwarded to:

```text
d1f170a Target flash overlap prefetch corelet
```

The Deeptools tree already had the Stage 033 experimental changes plus other
dirty files.  After this stage, the pod Deeptools diff was:

```text
 dcc/src/Stitcher/ModuleStitcher.cpp       |  12 +++
 dcg/dcg_fe/pcfg_gen/stcdpOp.cpp           | 140 ++++++++++++++++--------------
 dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp |   6 +-
 dcg/dcg_manager/dcg_manager.cpp           |  28 +++++-
 dpc/dpc.cpp                               |   4 +-
 dsc/dataOpDsc.cpp                         |   5 +-
 dsc/dataOpDsc.h                           |   1 +
 dxp/SdscTree.cpp                          |   6 +-
 dxp/dxp.cpp                               |  29 ++++++-
 9 files changed, 157 insertions(+), 74 deletions(-)
```

Only these files were newly changed for Stage 035:

```text
dsc/dataOpDsc.h
dsc/dataOpDsc.cpp
dcg/dcg_fe/pcfg_gen/stcdpOp.cpp
```

The Stage 033 `ModuleStitcher.cpp` duplicate-unit diagnostic remains a temporary
pod-local experiment.

## Deeptools Patch

`STCDPOpLx` now has:

```text
int coreletId = -1
```

`dsc/dataOpDsc.cpp` now parses and prints `coreletId` for `STCDPOpLx`.

`dcg/dcg_fe/pcfg_gen/stcdpOp.cpp` now routes ordinary `STCDPOpLx` LX PCFG
generation as follows:

```text
coreletId == 1  -> LXLU1 / LXSU1 / PE1
default/-1/0    -> LXLU0 / LXSU0 / PE0
```

The common `transformToPcfg(...)` LX path was generalized from exact
`LXLU0/LXSU0` checks to LU/SU helper predicates that also accept
`LXLU1/LXSU1`.  The SFP helper now sets the generated
`PTSFPDATATRANSFER` node `coreletId` to `1` when called for `PE1`.

## Build

Command:

```sh
cd "$DTI_PROJECT_ROOT/build/deeptools-onchip-foundation-clean"
make dxp_standalone -j8
```

Result:

```text
build succeeded
```

The modified `dsc/dataOpDsc.cpp` and `dcg/dcg_fe/pcfg_gen/stcdpOp.cpp` compiled
and `dxp_standalone` linked.

## Device-Facing Probe

Command:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128 \
  --variants warp_overlap_probe \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --warmup 1 --iters 2 \
  --timeout-s 240 \
  --cache-prefix /tmp/sdpa-stage035-prefetch-corelet \
  --output-json /tmp/sdpa-stage035-prefetch-corelet.json
```

Result:

```text
L=128 warp_overlap_probe status=failed rc=1
cache=/tmp/sdpa-stage035-prefetch-corelet-warp_overlap_probe-B1-H2-L128-D64-535169-828593
```

This is a different failure mode from Stage 033.  DXP/DCC completed and the
program executed, then PyTorch output validation failed:

```text
AssertionError: Tensor-likes are not close!
Mismatched elements: 12644 / 16384 (77.2%)
Greatest absolute difference: 8.59375 at index (0, 0, 56, 61) (up to 0.1 allowed)
Greatest relative difference: 17040.0 at index (0, 0, 76, 32) (up to 0.1 allowed)
```

Generated artifacts confirmed that the prefetch descriptors and final PCFGs
honored the requested corelet split:

```text
sdsc_mixed_flash_pipeline_tile_0.json
  STCDPOpLx prefetch entries include "coreletId": 1
  flashAttentionPipeline_.prefetch_corelet_id = 1

debug/.../sdsc_mixed_flash_pipeline_tile_0.out.out.out.json
  c0-lxsu1-ringDT-pe1-lx-0-0
  c0-pe1-FIFO-lx-lx-0 with "coreletId": 1

senprog.txt
  LXLU:<core>:1
  LXSU:<core>:1
```

## Interpretation

Stage 035 proves the narrow Deeptools routing patch is sufficient to get past
the Stage 033 duplicate-unit stitch conflict.  The overlap-prefix bundle now
builds and reaches execution.

The new correctness failure is consistent with a corelet-local value-flow
problem.  The prefetch sidecar writes K/V tile data through corelet-1 LX units,
while the generated batchmatmul consumer remains a corelet-0 program.  If the
LX storage addressed by those units is corelet-local, the prefetch no longer
feeds the consumer even though it no longer conflicts with the consumer's
`LXLU0/LXSU0/PE0` resource claims.

So the current "move prefetch to corelet 1" route is not sufficient by itself.
It trades the stitcher resource conflict for a runtime data-visibility problem.

## Next Target

The next experiment should keep the useful part of Stage 035, namely that DCC
can represent the independent prefetch row on distinct units, but solve value
visibility explicitly.  Plausible routes:

1. find or add a legal cross-corelet transfer/sync path from the corelet-1
   prefetch destination into the corelet-0 LX location consumed by batchmatmul;
2. split generated batchmatmul so the overlapped row uses corelet 1 only for
   true prefetch work and does not expect those values to be consumed by the
   same row's corelet-0 compute without an explicit handoff;
3. revisit the Stage 033 same-corelet route with a deeper DCC representation
   that can serialize or split the `LXLU0/LXSU0` claims inside one high-level
   overlap row, rather than assigning both programs to the same stitch slot.

The observed failure rules out a simple descriptor-only corelet move as the
final implementation shape.
