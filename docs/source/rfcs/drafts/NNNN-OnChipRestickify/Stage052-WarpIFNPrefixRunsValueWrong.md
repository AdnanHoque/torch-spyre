# Stage 052: Warp IFN-Prefix Reaches Runtime, Values Wrong

Date: 2026-05-27

## Purpose

Stage051 forced the InputFetchNeighbor-shaped overlap-prefix artifact and proved
that Torch selected the intended sidecar, but DXP rejected the same-row data-op
+ DL compute schedule with:

```text
DtException: unit already set for associated schedule step
```

The pod Deeptools diagnostic stitcher patch already allows duplicate physical
unit slots for mixed flash pipeline rows when the scheduled data-op is visibly a
prefetch.  Stage052 aligns the Torch-authored IFN-prefix data-op name with that
diagnostic contract.

## Change

The generated IFN-prefix data-op was renamed from:

```text
0_STCDPOpLx_ifn_Tensor0_idx0_tile0
```

to:

```text
0_STCDPOpLx_prefetch_ifn_Tensor0_idx0_tile0
```

No runtime-safety gate changed.  The artifact remains default-off and still
requires:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PREFIX_FORCE=1
```

## DXP Result

To isolate the DXP schedule blocker, the Stage051 failing bundle was copied and
only the data-op key was renamed to the new `prefetch_ifn` form:

```text
/tmp/sdpa-stage052-ifn-prefix-renamed-direct
```

Running the patched local DXP binary succeeded:

```sh
PATH=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp:$PATH
LD_LIBRARY_PATH=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/lib:\
/home/adnan-cdx/dt-inductor-mixed/sentient/runtime/lib:$LD_LIBRARY_PATH
dxp_standalone --bundle -d /tmp/sdpa-stage052-ifn-prefix-renamed-direct
```

Result:

```text
rc=0
```

This confirms that the IFN-prefix same-row descriptor can get past the previous
`ModuleStitcher` duplicate-unit failure under the diagnostic DXP patch.

## Device Result

A source-tree direct run used the patched Torch checkout, explicit
`torch_spyre._autoload()`, and the same local DXP binary:

```text
/tmp/sdpa-stage052-prefetch-ifn-prefix-localdxp-warp_ifn_prefix_probe-B1-H2-L128-D64-C0-634052-1779863486
```

The forced artifact was selected:

```text
bundle.mlir -> sdsc_mixed_flash_pipeline_tile_0.json
source=generated-flash-prefill-overlap-prefix-ifn-tile
datadsc=0_STCDPOpLx_prefetch_ifn_Tensor0_idx0_tile0
schedule=[[0, 0, 0, 0]]
replaces_sdsc=0_batchmatmul
ifn_attached_input_idx=0
ifn_runtime_safe=false
ifn_runtime_forced=true
```

The program compiled and ran, then failed the value check:

```text
Mismatched elements: 16253 / 16384 (99.2%)
Greatest absolute difference: nan at index (0, 1, 1, 17)
Greatest relative difference: nan at index (0, 1, 1, 17)
```

## Current Status

The active blocker moved from schedule representation to runtime/dataflow
correctness.  The same-row IFN-prefix overlap descriptor is now stitchable and
launchable under the diagnostic DXP stack, but it is not value-correct.

This is consistent with the Stage038 warning: a single-SDSC IFN-shaped row has
no real predecessor producer, so the input-neighbor movement can compile while
still failing to deliver valid data to the batchmatmul input.  The next
value-correct direction is to combine the Stage039 predecessor-backed LX copy
contract with an overlapped schedule row, rather than relying on a same-LX
single-SDSC diagnostic data-op as the actual producer.
