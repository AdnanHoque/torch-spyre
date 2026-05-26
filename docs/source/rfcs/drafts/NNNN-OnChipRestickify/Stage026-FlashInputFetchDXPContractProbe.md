# Stage 026: Flash InputFetch DXP Contract Probe

Date: 2026-05-26

## Purpose

Stage 025 showed that a generated flash `batchmatmul` cannot use the paired
`STCDPOpLx + DL compute` schedule row by simply flipping the compute descriptors
to LX.  This stage pushed the same question one layer deeper: if we locally fix
the obvious Foundation transfer lookup issue and provide cleaner LX allocation
metadata, does the InputFetchNeighbor path become usable for flash attention?

The answer is still no for current flash `batchmatmul` descriptors.  The path
progresses further, but `inputNeighFetchOp.cpp` assumes `i/j` coordinates and
does not currently support the `mb/x/in/out` geometry used by generated SDPA
tiles.

## Local Deeptools Patch

The current Foundation-clean scheduler has this lookup:

```cpp
// Return empty result if the tensor is LX pinned because of no L3 transfers.
if (lds.isLxPinned()) return std::vector<const dsc2::TransferNode *>();
```

That is incompatible with the same file's LX-neighbor handling because
`isLabeledDsLXNeighbor(...)` first requires `lds.isLxPinned()`.  I patched the
pod source locally to let LX-neighbor tensors continue through transfer lookup:

```cpp
if (lds.isLxPinned() && !isLabeledDsLXNeighbor(mySDsc, dscIdx, lds))
  return std::vector<const dsc2::TransferNode *>();
```

Patched file:

```text
/home/adnan-cdx/dt-inductor-mixed/deeptools-onchip-foundation-clean/
  dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
```

Rebuilt:

```sh
cd /home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean
source /home/adnan-cdx/dt-inductor-mixed/torch-spyre-docs/scripts/dev-env.sh
make dxp_standalone -j8
```

This patch is not part of the Torch-Spyre branch.  It is a local DXP probe to
test whether the compiler target shape is viable.

## Manual Descriptor Probes

### Probe A: Let Foundation Create The Neighbor Allocate

Descriptor:

```text
/tmp/stage025-force-lx-neighbor-owned-input
```

Shape:

- Stage 023 overlap-prefix mixed tile;
- input tensor `ldsIdx_ == 0` made LX-neighbor-like but with no allocate node;
- kernel and output tensors LX-local.

With the patched transfer lookup, DXP passed the previous
`Expect valid transfer nodes` failure and then segfaulted later:

```text
SIGSEGV
L3DlOpsScheduler::fillFinalStartAddressAndOffset
L3DlOpsScheduler.cpp:4305
```

Interpretation: the scheduler-created neighbor allocate node did not have the
start-address coordinate metadata expected by the later allocation finalization
pass.

### Probe B: Predeclare Clean LX Allocate Nodes

Descriptor:

```text
/tmp/stage026-minimal-lx-alloc-overlap
```

Shape:

- Stage 023 overlap-prefix mixed tile;
- all compute tensors LX-only;
- schedule tree replaced with minimal LX allocate nodes that have per-core start
  addresses;
- paired overlap schedule preserved.

With the patched transfer lookup, DXP reached `inputNeighFetchOp.cpp` and then
failed a geometry invariant:

```text
DtException:
currdsc.CoreD_.paramNameToVal(dimName) ==
DB_ratio * myIFNInfo_.dscIdxToB_.at(dsc_idx).paramNameToVal(dimName)
inputNeighFetchOp.cpp line 1926
```

The generated DSC2 flash tile has valid `dataStageParam_`, but legacy
`CoreD_`/`B_` fields remain unset (`-1`).  The input-neighbor code still reads
the legacy fields in this path.

### Probe C: Predeclare Legacy CoreD/B

Descriptor:

```text
/tmp/stage026-minimal-lx-alloc-overlap-legacydims
```

Additional patch:

- set `CoreD_`, `CoreletD_`, and `B_` to match the generated DSC2 core/chunk
  shape:

```text
in=64, out=64, mb=4, x=2
```

With the patched transfer lookup and legacy dimension fields, DXP reached the
input-neighbor subpiece ordering code and failed here:

```text
DtException:
op->outSP_.at(mainOutSPIdx).dimToStartCordinate.count("i")
inputNeighFetchOp.cpp line 1644
```

The code path explicitly expects `i` and `j` coordinates:

```cpp
DT_CHECK(op->outSP_.at(mainOutSPIdx).dimToStartCordinate.count("i"));
DT_CHECK(op->outSP_.at(mainOutSPIdx).dimToStartCordinate.count("j"));
```

Generated flash `batchmatmul` uses `mb/x/in/out` coordinates, not `i/j`.

## Compiler Change

The overlap-prefix eligibility check is now stricter:

```text
torch_spyre/_inductor/onchip_realize.py
```

`_input_fetch_neighbor_compute_eligible(...)` now also requires that the first
input layout contains `i_` and `j_`, matching the current Foundation
InputFetchNeighbor ordering implementation.  This keeps the future overlap
builder available for IJ-shaped Foundation descriptors, but fails closed for
generated SDPA `batchmatmul` tiles.

Unit coverage:

```text
tests/_inductor/test_onchip_realize_logic.py
  test_flash_pipeline_overlap_prefix_tile_artifacts_overlap_one_compute
  test_flash_pipeline_overlap_prefix_rejects_hbm_backed_compute
  test_flash_pipeline_overlap_prefix_rejects_lx_compute_without_transfer
  test_flash_pipeline_overlap_prefix_rejects_non_ij_input_neighbor_shape
```

The positive overlap-prefix unit now uses an IJ-shaped fake descriptor.  The
flash-shaped fake descriptor with `mb/x/in/out` and a transfer node is rejected.

## Local Validation

```text
tests/_inductor/test_onchip_realize_logic.py          30/30 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   10/10 passed
py_compile(onchip_realize.py, test_onchip_realize_logic.py) passed
git diff --check passed
```

## Pod Validation

```text
pod: adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed

tests/_inductor/test_onchip_realize_logic.py          30/30 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   10/10 passed
py_compile(onchip_realize.py, test_onchip_realize_logic.py) passed
git diff --check passed
```

Device smoke with the overlap flag and the locally patched DXP build still
failed closed to the serial mixed tile:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=0
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1
export SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=0
export SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF=0
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-stage026-overlap-ij-guard-1779828304
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 7 deselected in 20.52s
```

Both emitted SDPA bundles stayed serial:

```text
source=generated-flash-prefill-batchmatmul-tiles
overlap_prefix=false
overlap_candidate=false
dataop_count=2
tile_count=1
```

Mixed-tile `senprog.txt` counts:

```text
bundle 0 sdsc_mixed_flash_pipeline_tile_0: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=192
bundle 1 sdsc_mixed_flash_pipeline_tile_0: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=160
```

## Interpretation

Current Foundation has three separate gaps for using InputFetchNeighbor as the
flash-attention load/compute overlap primitive:

1. `getLdsL3TransferNodes(...)` returns early for LX-pinned tensors, which also
   includes LX-neighbor tensors.
2. The input-neighbor path still relies on legacy `CoreD_`/`B_` in places even
   for generated DSC2 descriptors.
3. The subpiece ordering assumes `i/j` coordinates and does not handle
   batchmatmul-style `mb/x/in/out` tensors.

That means the production on-chip SDPA path should continue to use the serial
mixed tile plus same-stick pointwise/value-flow handoffs.  True flash prefetch
overlap needs either a generalized Foundation InputFetchNeighbor path for
batchmatmul/SDPA geometry or a non-InputFetch scheduler contract for overlapping
ordinary `STCDPOpLx` rows with DL compute rows.
