# Stage 024: Flash Overlap InputFetch Guard

Date: 2026-05-26

## Purpose

Stage 023 proved that the compiler could emit a one-compute overlap-prefix flash
sidecar, but the device run aborted inside DXP before runtime execution.  This
stage separates the architectural idea from the current Foundation contract:
keep the warp-specialized schedule builder available for eligible compute DSCs,
but fail closed for generated HBM-backed flash tiles so the production path
continues to execute the known-good serial mixed SDSC.

## DXP Contract Audit

The failing schedule row had both a data-op index and a DL compute index:

```text
[2, 0, 1, 1]
```

In `dcg_manager.cpp`, current DXP treats that paired row as
`InputFetchNeighbor` unless the SuperDSC target is `SENPCFG`.  The path then
calls `generatePcfgIRForDataOpInpFetch(...)`.

For the one-SuperDSC form used by the mixed sidecar, `inputNeighFetchOp.cpp`
checks every labeled DS in the main DL compute DSC:

```text
!is_any_of(lds.pinnedComponent(), HBM, NO_COMPONENT)
```

`pinnedComponent()` checks memory components in this order:

```text
HBM, RING, SFPRING, LX, PT, PTXRF, PTARF, SFPLRF, PELRF, L0, PTIRF
```

That means a generated flash compute DSC with both HBM and LX present is still
classified as HBM-pinned.  It is not eligible for the paired-row input-neighbor
path.  The Stage 023 overlap-prefix artifact was therefore structurally useful
as a probe, but not a legal executable descriptor for current generated SDPA
tiles.

## Implementation

Added a compiler-side mirror of the DXP pin guard in:

```text
torch_spyre/_inductor/onchip_realize.py
```

The new helper computes the effective pinned component from `memOrg_` using the
same component precedence as Foundation, then allows overlap-prefix emission
only when every labeled DS in the copied compute DSC is pinned to something
other than HBM or no component.

When `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1` is requested:

- eligible LX/ring/SFP-ring compute DSCs may still use the overlap-prefix
  schedule;
- ordinary HBM-backed generated flash tiles return `None` from the
  overlap-prefix builder;
- `build_flash_attention_pipeline_tile_artifacts(...)` falls back to the serial
  one-compute sidecar for that tile.

Tests now cover both sides:

```text
tests/_inductor/test_onchip_realize_logic.py
  test_flash_pipeline_overlap_prefix_tile_artifacts_overlap_one_compute
  test_flash_pipeline_overlap_prefix_rejects_hbm_backed_compute
  test_flash_pipeline_overlap_prefix_rejects_mismatched_next_tile
```

The fake SDSC helper can now produce either HBM-backed or LX-only descriptors,
so the positive overlap-prefix unit test remains meaningful without letting the
real generated-HBM case crash DXP.

## Validation

Local:

```text
tests/_inductor/test_onchip_realize_logic.py          28/28 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   10/10 passed
py_compile(onchip_realize.py, onchip_bridge.py, test_onchip_realize_logic.py) passed
git diff --check passed
```

Pod static:

```text
pod: adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed

tests/_inductor/test_onchip_realize_logic.py          28/28 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   10/10 passed
py_compile(onchip_realize.py, onchip_bridge.py, test_onchip_realize_logic.py) passed
git diff --check passed
```

Device guard check:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=0
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1
export SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=0
export SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF=0
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-stage024-overlap-guard-1779826672
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 7 deselected in 24.36s
```

Generated tile metadata for both emitted SDPA bundles confirmed the requested
overlap path failed closed to serial:

```text
source=generated-flash-prefill-batchmatmul-tiles
overlap_prefix=false
overlap_candidate=false
dataop_count=2
tile_count=1

core 0 schedule:
[
  [0, -1, 0, 1],
  [1, -1, 1, 1],
  [-1, 0, 1, 0],
]
```

Mixed-tile `senprog.txt` counts stayed on-chip and matched the Stage 023 serial
control:

```text
bundle 0 sdsc_mixed_flash_pipeline_tile_0: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=192
bundle 1 sdsc_mixed_flash_pipeline_tile_0: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=160
```

## Interpretation

This is a production guard, not a performance win by itself.  The overlap
algorithm remains the intended direction for warp-specialized flash attention:

```text
prefetch K/V tile N+1 while computing tile N
```

But current generated SDPA tiles are HBM-backed DL compute descriptors, and DXP
does not accept those in the paired-row `InputFetchNeighbor` path.  Until we can
legally produce all-LX/ring/SFP-ring compute descriptors, use a two-SDSC
input-fetch contract, or get Foundation support for regular data-op plus DL
overlap rows, production SDPA must stay on the serial mixed-tile path.
