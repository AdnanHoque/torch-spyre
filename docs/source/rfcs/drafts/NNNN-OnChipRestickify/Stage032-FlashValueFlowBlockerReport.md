# Stage 032: Flash Value-Flow Blocker Report

Date: 2026-05-26

## Purpose

Stage 031 made the warp-overlap probe fail closed with explicit reasons.  This
stage does the same for the stricter real value-flow path:

```text
real producer output -> STCDPOpLx -> flash batchmatmul input
```

This path is stronger than the generic mixed tile sidecar because the compute
DSC would consume the transferred LX value.  Before changing Deeptools, we need
to distinguish value-flow blockers that are compiler graph/layout facts from
overlap blockers that are Foundation `InputFetchNeighbor` limitations.

## Implementation

Added:

```text
torch_spyre/_inductor/onchip_realize.py
  flash_attention_value_flow_tile_rejection_reasons(...)
```

When `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE=<n>` is requested and
the real value-flow tile cannot be built, `bundle.py` now annotates the matching
serial tile sidecar:

```text
value_flow_requested: true
value_flow_rejection_reasons: [...]
```

Updated:

```text
torch_spyre/_inductor/codegen/bundle.py
tests/_inductor/test_onchip_realize_logic.py
tools/onchip_sdpa_sweep.py
```

The sweep harness now includes:

```text
value_flow_tile0
value_flow_tile1
value_flow_tile2
```

Each variant sets `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE` to the
corresponding tile index and leaves tile replacement/overlap disabled.

## Local Validation

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

## Contiguous PV Layout Probe

I also tried a contained compiler probe:

```text
pv_scores = exp_scores.transpose(-1, -2).contiguous()
torch.bmm(pv_scores.flatten(0, 1), value_block.flatten(0, 1))
```

Commit:

```text
894ac57 Probe contiguous flash PV score layout
57c0302 Revert contiguous flash PV layout probe
```

The probe remained value-correct on device, but it did not make
`value_flow_tile0`, `value_flow_tile1`, or `value_flow_tile2` realizable.  The
extra contiguous conversion was reverted because it did not move the value-flow
contract forward.

## Pod Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
commit=22a0ad7
```

Static tests:

```text
tests/_inductor/test_onchip_realize_logic.py
tests/_inductor/test_onchip_flash_pipeline_logic.py

41 passed in 0.25s
```

Device-facing sweep:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128 \
  --variants value_flow_tile0,value_flow_tile1,value_flow_tile2 \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --warmup 1 --iters 2 \
  --timeout-s 240 \
  --cache-prefix /tmp/sdpa-stage032-value-flow-report \
  --output-json /tmp/sdpa-stage032-value-flow-report.json
```

Result:

```text
L=128 value_flow_tile0 status=ok median=0.261448ms max_err=0.00341797 mixed=4
L=128 value_flow_tile1 status=ok median=0.272603ms max_err=0.00341797 mixed=4
L=128 value_flow_tile2 status=ok median=0.259710ms max_err=0.00341797 mixed=4
```

Real generated-SDPA rejection metadata:

```text
value_flow_tile0:
  mixed_flash_pipeline_tile_0:
    input0:no_latest_producer
    input1:not_single_consumer:3_batchmatmul:input1,6_maxnonstick:input1
  mixed_flash_pipeline_tile_0:
    input0:no_latest_producer
    input1:no_latest_producer

value_flow_tile1:
  mixed_flash_pipeline_tile_1:
    input0:no_latest_producer
    input1:physical_layout_mismatch:
      producer=['x_', 'mb_', 'out_']/out_
      consumer=['in_', 'x_', 'out_']/out_

value_flow_tile2:
  mixed_flash_pipeline_tile_2:
    input0:physical_layout_mismatch:
      producer=['mb_', 'x_', 'out_']/out_
      consumer=['x_', 'mb_', 'in_']/in_
    input1:no_latest_producer
```

## Interpretation

The value-flow path is failing for graph/layout reasons, not the same reason as
warp overlap:

- several flash BMM inputs are true external operands with no latest in-bundle
  producer;
- one candidate producer fans out to both the BMM and `maxnonstick`;
- remaining candidates require a layout transform, not a pure Tier 1
  same-physical-stick copy.

That means Deeptools work should target the warp-overlap path first.  The
overlap path is still desirable because it can prefetch K/V tiles while the
compute DSC reads its normal operands, whereas real BMM value-flow requires the
generated graph to expose a same-stick single-consumer producer edge.

With Deeptools changes now allowed, the next implementation step should attack
the Foundation `InputFetchNeighbor` blockers from Stage 031:

```text
HBM+LX effective pinning is treated as HBM.
Generated flash batchmatmul uses mb/x/in/out, but InputFetchNeighbor expects i/j.
```
