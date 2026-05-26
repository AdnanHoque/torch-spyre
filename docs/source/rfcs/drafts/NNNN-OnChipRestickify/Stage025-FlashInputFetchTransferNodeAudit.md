# Stage 025: Flash InputFetch Transfer-Node Audit

Date: 2026-05-26

## Purpose

Stage 024 made the overlap-prefix path fail closed for ordinary HBM-backed
generated flash tiles.  This stage tested the next obvious question: if the
generated flash compute DSC is forced to be LX-pinned, does DXP accept the
paired `STCDPOpLx + batchmatmul` schedule row as a legal InputFetchNeighbor
execution shape?

The answer is still no.  The pin guard is necessary, but not sufficient.  The
DL scheduler also expects an input-neighbor transfer-node shape in the compute
DSC schedule tree.

## Probe 1: Force The Overlap Tile To LX

Starting point:

```text
/tmp/sdpa-stage023-exec-tile0-overlap-prefix-final-1779826149/inductor-spyre/
  sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_yq4nauhw/
    sdsc_mixed_flash_pipeline_tile_0.json
```

The descriptor already had the Stage 023 overlap-prefix schedule:

```text
[
  [0, -1, 0, 1],
  [1, -1, 1, 1],
  [2,  0, 1, 1],
  [3, -1, 1, 0],
]
```

First patch attempt:

- set all compute `labeledDs_` `memOrg_` to LX-only;
- set HBM sizes to zero;
- add `coreStateInit_`.

DXP aborted while importing because the schedule-tree allocate nodes still
pointed at HBM:

```text
std::out_of_range: map::at
dsc/dsc2.cpp:1786
```

Second patch attempt:

- keep a non-present HBM memOrg entry;
- switch all schedule-tree allocate nodes to LX.

DXP imported the descriptor, then rejected the mixed residency:

```text
DtException:
Do not support both HBM pinned tensor and input-neighbor fetch tensor existing
in the same DSC.
dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 3013
```

Third patch attempt:

- use LX-only `memOrg_`;
- switch all compute allocate nodes to LX.

DXP got past the mixed-residency check, then failed later:

```text
DtException: Expect valid transfer nodes.
dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 1589
```

Fourth patch attempt:

- leave the input tensor (`ldsIdx_ == 0`) as an LX neighbor without a declared
  allocate node, hoping the scheduler would own the `NO_COMPONENT -> LX`
  transfer;
- keep the kernel and output tensors LX-local.

DXP still failed at the same transfer-node check:

```text
DtException: Expect valid transfer nodes.
dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 1589
```

## Probe 2: Two-SDSC InputFetch Standalone

The Foundation-clean build has the `dcg_inpfetch_standalone.cpp` source, but
does not currently have a built standalone binary.  The older build does:

```text
/home/adnan-cdx/dt-inductor-mixed/build/deeptools/dcg/tools/dcg_inpfetch_standalone
```

Using the generated SDPA `batchmatmul` as both:

- main consumer DSC, patched all-LX;
- pre producer DSC, patched output-LX;

failed because the raw generated Torch-Spyre DSC has no chunk loop order for
this standalone path:

```text
DtException: Do not expect empty loop order.
dcg/dcg_fe/pcfg_gen/pcfg_gen_utils.cpp line 329
```

Adding a lowercase Deeptools-style 15-loop `loopOrder_` did not fix it because
the generated DSC imports as DSC2, so the standalone path tries to derive DB
loop order from the schedule tree instead of the legacy `loopOrder_` field.
That schedule tree does not contain the loop nodes the standalone path expects.

## Implementation

The compiler guard now mirrors both observed Foundation requirements:

```text
torch_spyre/_inductor/onchip_realize.py
```

`_input_fetch_neighbor_compute_eligible(...)` now requires:

- every compute `labeledDs_` to pin to something other than HBM or no component;
- the first compute input to be `ldsIdx_ == 0`;
- that first input to be LX-pinned;
- a schedule-tree transfer node with:

```text
src.storage_ == "no_component"
dstVias[*].loc_.storage_ == "lx"
dstLdsAndLoopOffsets_[*].myLdsIdx_ == 0
```

This keeps the overlap-prefix builder available for a future Foundation-owned
InputFetchNeighbor compute DSC, but prevents synthetic all-LX descriptors from
passing Python eligibility and failing later inside DXP.

Unit coverage:

```text
tests/_inductor/test_onchip_realize_logic.py
  test_flash_pipeline_overlap_prefix_tile_artifacts_overlap_one_compute
  test_flash_pipeline_overlap_prefix_rejects_hbm_backed_compute
  test_flash_pipeline_overlap_prefix_rejects_lx_compute_without_transfer
```

## Local Validation

```text
tests/_inductor/test_onchip_realize_logic.py          29/29 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   10/10 passed
py_compile(onchip_realize.py, test_onchip_realize_logic.py) passed
git diff --check passed
```

## Interpretation

The current mixed-SDSC overlap route is not simply missing an LX flip.  It needs
Foundation to materialize or accept the input-neighbor transfer-node contract in
the compute DSC.  The serial mixed tile and the existing same-stick pointwise
handoffs remain the production-safe SDPA path today.

The next real implementation path is one of:

1. emit a Foundation-owned InputFetchNeighbor compute DSC that already contains
   the `NO_COMPONENT -> LX` transfer node DXP expects;
2. expose the two-SDSC `dcg_inpfetch_standalone` contract through DXP/bundle
   lowering, with valid DSC2 loop-node context;
3. get Foundation support for ordinary `STCDPOpLx` data-op rows overlapped with
   DL compute rows without routing through InputFetchNeighbor.

