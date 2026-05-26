# Stage 023: Flash Overlap-Prefix Tile Probe

Date: 2026-05-26

## Purpose

Stage 022 left the current on-chip SDPA prototype in a conservative shape:
serial mixed SDSCs work, selected same-stick flash handoffs work, and wide
score-scale blocks fail closed.  The next architectural question was whether the
mixed SDSC scheduler can execute a warp-specialized flash pattern:

```text
prefetch next tile while computing current tile
```

The full multi-compute pipeline is still blocked by the current Foundation/DXP
contract (`dscs_.size() == 1`).  This stage therefore builds the smallest
possible executable probe inside that limit: one generated flash `batchmatmul`
compute DSC, four `STCDPOpLx` prefetch data-ops, and one schedule row containing
both a data-op index and compute index.

## Implementation

Added:

```text
torch_spyre/_inductor/codegen/onchip_bridge.py
  flash_pipeline_overlap_prefix_schedule

torch_spyre/_inductor/onchip_realize.py
  build_flash_attention_pipeline_overlap_prefix_tile_artifact
  build_flash_attention_pipeline_tile_artifacts(..., overlap_prefix=True)
```

The overlap-prefix schedule is:

```text
[0, -1]  prefetch K tile 0
[1, -1]  prefetch V tile 0
[2,  0]  prefetch K tile 1 while computing tile 0
[3, -1]  prefetch V tile 1
```

After schedule normalization, core 0 has:

```text
[
  [0, -1, 0, 1],
  [1, -1, 1, 1],
  [2,  0, 1, 1],
  [3, -1, 1, 0],
]
```

The builder remains fail-closed:

- it needs at least two generated `batchmatmul` tiles;
- the next tile must have the same `numCoresUsed_`, output layout, stick dim,
  split dim, and iteration sizes;
- the last tile falls back to the previous serial one-compute mixed sidecar;
- if allocation or descriptor construction fails, it falls back to serial.

`bundle.py` now asks for these overlap-prefix tile sidecars when both flags are
set:

```sh
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=<n>
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1
```

The normal production path keeps `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=0`.

## Validation

Local:

```text
tests/_inductor/test_onchip_flash_pipeline_logic.py   10/10 passed
tests/_inductor/test_onchip_realize_logic.py          27/27 passed
py_compile(onchip_bridge.py, onchip_realize.py, bundle.py, config.py) passed
git diff --check passed
```

Pod:

```text
pod: adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
```

Pod standalone/static:

```text
tests/_inductor/test_onchip_flash_pipeline_logic.py   10/10 passed
tests/_inductor/test_onchip_realize_logic.py          27/27 passed
py_compile(onchip_bridge.py, onchip_realize.py, bundle.py, config.py) passed
```

Serial executed-tile control:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=0
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=0
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-stage023-exec-tile0-serial-1779825831
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 7 deselected in 26.83s
```

Serial mixed tile `senprog.txt` evidence:

```text
bundle 0 sdsc_mixed_flash_pipeline_tile_0: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=192
bundle 1 sdsc_mixed_flash_pipeline_tile_0: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=160
```

Overlap-prefix executed-tile probe:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=0
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-stage023-exec-tile0-overlap-prefix-final-1779826149
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

The generated sidecar had the intended overlap row and metadata:

```text
file:
/tmp/sdpa-stage023-exec-tile0-overlap-prefix-final-1779826149/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_yq4nauhw/sdsc_mixed_flash_pipeline_tile_0.json

schedule row:
[2, 0, 1, 1]

metadata:
source=generated-flash-prefill-overlap-prefix-tile
tile_count=1
dataop_count=4
prefetch_tile_count=2
compute_tile_count=1
tile_bytes=512
overlap_prefix=true
overlap_candidate=true
```

DXP rejected the bundle before runtime execution:

```text
terminate called after throwing an instance of 'DtException'
what(): DtException: !is_any_of(lds.pinnedComponent(), HBM, NO_COMPONENT),
file .../dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp line 82

CalledProcessError:
dxp_standalone --bundle -d .../sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_yq4nauhw
died with SIGABRT
```

## Interpretation

The compiler can now emit an executable one-compute overlap-prefix mixed tile
shape, and the emitted descriptor is exactly the scheduler question we needed to
ask.  The answer from current Foundation/DXP is no for this STCDP/LX-pinned
form: a schedule row that pairs `STCDPOpLx` with a DL compute DSC trips
`inputNeighFetchOp.cpp` because the data-op labeled DS is LX-pinned, while that
path expects HBM or no pinned component.

This is a useful negative result.  It narrows the next production work:

```text
certified today:
  serial one-compute mixed flash tile
  selected same-stick flash pointwise/score-scale handoffs

not certified:
  STCDPOpLx prefetch row overlapped with DL compute
  full multi-compute flash pipeline
```

The design should keep overlap off by default.  To get true load/compute overlap
for flash attention, the next step is one of:

1. add or expose a Foundation-supported `InputFetchNeighbor`/HBM-to-LX prefetch
   data-op contract that can be scheduled with a DL row; or
2. get Foundation/DXP support for LX-pinned `STCDPOpLx` rows paired with DL
   compute rows.

Until one of those contracts exists, production on-chip SDPA should continue
through the value-correct serial/mixed handoff path and should not claim
warp-specialized overlap.
