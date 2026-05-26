# Stage 031: Flash Overlap Probe Diagnostics

Date: 2026-05-26

## Purpose

Stage 030 closed the natural-score-layout path because the required
singleton-stick gather does not compile through current Foundation/DXP.  The
remaining "warp-specialized flash" path is the Stage 023 overlap-prefix design:

```text
prefetch K/V tile N+1 while computing tile N
```

Stages 024-026 proved that current generated flash `batchmatmul` descriptors
are not legal for DXP's paired-row `InputFetchNeighbor` path.  This stage keeps
that design visible as an executable probe, but makes the fail-closed result
actionable: when overlap is requested and the compiler falls back to the serial
mixed tile, the generated sidecar now records why.

## Implementation

Added overlap rejection diagnostics in:

```text
torch_spyre/_inductor/onchip_realize.py
```

New helpers:

```text
_input_fetch_neighbor_rejection_reasons(...)
flash_attention_overlap_prefix_rejection_reasons(...)
```

`build_flash_attention_pipeline_tile_artifacts(..., overlap_prefix=True)` now
annotates serial fallback sidecars with:

```text
overlap_prefix_requested: true
overlap_prefix_rejection_reasons: [...]
```

The reasons mirror the current DXP contract checks:

- effective pinned component uses Foundation precedence, so `hbm+lx` is still
  treated as HBM-pinned;
- first compute input must be `ldsIdx_ == 0`;
- first compute input must be LX-pinned;
- the compute schedule tree must contain a `NO_COMPONENT -> LX` transfer for
  that input;
- the first input layout must contain `i_` and `j_`.

Updated:

```text
tests/_inductor/test_onchip_realize_logic.py
tools/onchip_sdpa_sweep.py
```

The sweep harness now includes a diagnostic variant:

```text
warp_overlap_probe:
  SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
  SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=0
  SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1
```

and reports `flashAttentionPipeline_` metadata in `mixed_sdscs`.

## Local Validation

```text
python3 tests/_inductor/test_onchip_realize_logic.py
  30/30 passed

python3 tests/_inductor/test_onchip_flash_pipeline_logic.py
  10/10 passed

python3 -m py_compile \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_onchip_realize_logic.py

git diff --check
  passed
```

## Pod Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
commit=7d40761
```

Static tests:

```text
tests/_inductor/test_onchip_realize_logic.py
tests/_inductor/test_onchip_flash_pipeline_logic.py

40 passed in 0.24s
```

Device-facing sweep:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128 \
  --variants warp_overlap_probe,onchip_master \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --warmup 1 --iters 2 \
  --timeout-s 240 \
  --cache-prefix /tmp/sdpa-stage031-overlap-sweep \
  --output-json /tmp/sdpa-stage031-overlap-sweep.json
```

Result:

```text
L=128 warp_overlap_probe status=ok median=0.260915ms max_err=0.00341797 mixed=4
L=128 onchip_master      status=ok median=0.315447ms max_err=0.00341797 mixed=4
```

The `warp_overlap_probe` mixed sidecars stayed serial and value-correct.  Their
metadata captured the two real generated-SDPA blockers:

```text
overlap=false requested=true
reasons=[
  compute_dsc:lds0_pinned_hbm,
  compute_dsc:lds1_pinned_hbm,
  compute_dsc:lds2_pinned_hbm,
  compute_dsc:input_lds0_pinned_hbm,
  compute_dsc:input_layout_missing_i_j,
  compute_dsc:missing_no_component_to_lx_transfer_lds0
]
```

The last tile in each bundle reports:

```text
not_enough_following_tiles
```

which is expected because overlap-prefix needs a next tile to prefetch.

`senprog.txt` evidence for the executed mixed tile remained on chip:

```text
sdsc_mixed_flash_pipeline_tile_0: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=192
sdsc_mixed_flash_pipeline_tile_0: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=160
```

## Interpretation

This is not a new performance win and it does not claim warp-specialized flash
execution.  It turns the overlap design into a stable production diagnostic:
request the overlap probe, execute the safe serial mixed tile if Foundation
cannot accept the paired row, and record exactly which contract checks blocked
overlap.

The current generated flash descriptors still fail the two important
InputFetchNeighbor requirements:

```text
Foundation sees hbm+lx tensors as HBM-pinned.
Generated SDPA batchmatmul uses mb/x/in/out, while InputFetchNeighbor still
expects i/j input-neighbor ordering.
```

The next implementation path for true load/compute overlap is still Foundation
work, not an Inductor-only flag flip: either generalize InputFetchNeighbor to
batchmatmul/SDPA geometry or add a scheduler contract that permits ordinary
`STCDPOpLx` rows to overlap DL compute rows without rerouting through the
current `i/j`-only input-neighbor path.
