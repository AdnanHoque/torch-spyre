# Stage 033: Flash Independent-Overlap DCC Conflict

Date: 2026-05-26

## Purpose

Stage 032 showed that real flash batchmatmul value-flow is blocked by generated
graph/layout facts, not only by Foundation `InputFetchNeighbor` constraints.
This stage tried the narrower warp-overlap scheduler route: treat the generated
flash prefetch data-op as independent overlap work, not as an
`InputFetchNeighbor` producer for the current DL compute input.

The intended schedule row remains:

```text
current tile compute + next tile prefetch
```

but the current prefetch is still a scheduler/descriptor proof built from
`STCDPOpLx` LX-to-LX moves, not true HBM K/V loading.

## Torch-Spyre Change

Commit:

```text
c2239a2 Allow independent flash overlap sidecars
```

Changed:

```text
torch_spyre/_inductor/onchip_realize.py
tests/_inductor/test_onchip_realize_logic.py
```

The overlap-prefix sidecar builder no longer rejects generated flash
batchmatmul descriptors for the current Foundation `InputFetchNeighbor`
requirements:

```text
compute_dsc pinned HBM
compute input not LX-pinned
missing NO_COMPONENT -> LX transfer node
missing i/j layout coordinates
```

That is deliberate for this experiment.  The sidecar is now modeled as an
ordinary independent prefetch row paired with ordinary DL compute, so the Python
guard only validates:

- two compatible adjacent flash batchmatmul tiles exist;
- the two tile outputs share layout, stick dim, split dim, and iter sizes;
- the two-tile LX allocation fits.

Local validation:

```text
python3 tests/_inductor/test_onchip_realize_logic.py
  31/31 passed

python3 tests/_inductor/test_onchip_flash_pipeline_logic.py
  10/10 passed

python3 -m py_compile \
  torch_spyre/_inductor/onchip_realize.py \
  torch_spyre/_inductor/codegen/bundle.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_onchip_realize_logic.py

git diff --check
  passed
```

## Deeptools Patch

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
deeptools=/home/adnan-cdx/dt-inductor-mixed/deeptools-onchip-foundation-clean
```

Patched:

```text
dcg/dcg_manager/dcg_manager.cpp
```

`DcgManager::runDcgForDataOpsDlOps(SuperDsc& mySDsc)` was changed so paired
`[dataop, dldsc]` rows whose SuperDSC name contains
`mixed_flash_pipeline_tile` and whose data-op name contains `prefetch_` are
treated as independent overlap rows, not as `InputFetchNeighbor` rows.

The same predicate was added to `DcgManager::mergePcfgInSuperDSC(...)` so the
ordinary DL PCFG still merges for those paired rows.

Temporary diagnostic patch:

```text
dcc/src/Stitcher/ModuleStitcher.cpp
```

The diagnostic prints the first duplicate physical unit before the stitcher
asserts.  It is intentionally experimental and should not be carried into a
final Deeptools patch without cleanup.

Rebuild:

```sh
cd "$DTI_PROJECT_ROOT/build/deeptools-onchip-foundation-clean"
make dxp_standalone -j8
```

Result:

```text
build succeeded
```

## Device-Facing Probe

Command:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128 \
  --variants warp_overlap_probe \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --warmup 1 --iters 2 \
  --timeout-s 240 \
  --cache-prefix /tmp/sdpa-stage033-independent-overlap \
  --output-json /tmp/sdpa-stage033-independent-overlap.json
```

Result:

```text
L=128 warp_overlap_probe status=failed rc=1
cache=/tmp/sdpa-stage033-independent-overlap-warp_overlap_probe-B1-H2-L128-D64-530359-930793
```

Manual DXP repro:

```sh
cd /tmp/sdpa-stage033-independent-overlap-warp_overlap_probe-B1-H2-L128-D64-530359-930793/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_nz631izw
dxp_standalone --bundle -d .
```

Diagnostic output:

```text
ModuleStitcher duplicate unit:
  sdsc=mixed_flash_pipeline_tile_0
  core=0
  corelet=0
  component=lxlu
  schedule_idx=2
  module_idx=3

DtException: unit already set for associated schedule step
dcc/src/Stitcher/ModuleStitcher.cpp line 272
```

The generated overlap-prefix tile had the intended row:

```text
[0, -1, 0, 1]
[1, -1, 1, 1]
[2,  0, 1, 1]
[3, -1, 1, 0]
```

Row `[2, 0]` is the attempted independent overlap row:

```text
data-op 2: next-tile prefetch
DL DSC 0: current tile batchmatmul compute
```

## Interpretation

The independent-overlap route now gets past the earlier Python guard and past
the Foundation path that previously forced every paired row through
`InputFetchNeighbor`.  The next blocker is lower in DCC stitching:

```text
STCDPOpLx prefetch and generated batchmatmul compute both claim lxlu in the
same schedule slot.
```

That is a real scheduling/resource conflict for the current descriptor shape.
It means a naive row-level pairing of ordinary `STCDPOpLx` prefetch work with an
ordinary generated DL batchmatmul is not enough for warp-specialized flash.

The next viable implementation target is therefore not another Python flag flip.
It needs one of:

1. a Deeptools/DCC representation that can split a generated batchmatmul into
   loader and compute subprograms so the overlap row pairs prefetch with a
   compute-only phase;
2. a generated flash tile whose current operands are already LX-resident before
   the overlap row, with no LX load unit claim during the paired compute slot;
3. a deeper DCC scheduler/stitcher model that can serialize or otherwise
   represent multiple `lxlu` users inside one high-level row.

Option 1 or 2 is the cleaner compiler direction.  The existing value-flow
diagnostics from Stage 032 show why option 2 requires real graph/layout work:
the current generated flash batchmatmul inputs are often external HBM operands,
fanout values, or layout-mismatched producer outputs.

## Continuation Note

After recording the failure, local Torch-Spyre was clean at `c2239a2`.  A later
attempt to re-enter the pod for more DCC inspection timed out against the
cluster API:

```text
Unable to connect to the server:
net/http: request canceled while waiting for connection
```

The pod-local Deeptools state should be rechecked before the next Deeptools
edit, especially because the temporary `ModuleStitcher.cpp` diagnostic is still
expected to be present there.
