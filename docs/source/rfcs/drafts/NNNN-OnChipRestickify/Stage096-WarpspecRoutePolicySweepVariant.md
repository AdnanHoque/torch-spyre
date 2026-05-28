# Stage096 - Warpspec Route Policy Sweep Variant

## Question

Stage095 created an offline route-policy generator from perf-compare JSON. The
next step is to make that policy executable in the benchmark harness so a single
variant can represent "use decoupled warpspec where it is performance-preferred,
otherwise use `onchip_master`."

## Change

Add a shape-aware sweep variant:

```text
onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled_route_policy
```

The variant uses the Stage234 `min_speedup=1.0` policy table:

```text
target route:
  onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled

fallback route:
  onchip_master
```

Target rows:

```text
B1 H4 D64  block64 non-causal L768,L1024
B2 H4 D128 block64 non-causal L768,L1024
```

Every other shape currently falls back to `onchip_master`.

## Implementation

`tools/onchip_sdpa_sweep.py` now resolves the child environment dynamically for
the route-policy variant. This is deliberately implemented in the sweep layer,
not in core codegen, because the policy is still based on a small Stage234 perf
table and should remain easy to benchmark and revise.

For a selected target row, the child env is exactly the existing certified
decoupled env:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM=0
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE=-1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_HOIST_TILE=-2
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_CORE=31
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIALIZE_LOADER_CORE=1
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT=0
```

For fallback rows, the child env is the existing `onchip_master` env.

The sweep payload also records:

```text
route_policy
route_selected_variant
```

This allows downstream perf JSON to distinguish the logical policy variant from
the concrete route chosen for each row.

## Why This Helps

Before this stage, we had two separate artifacts:

```text
promotion gate:
  proves that decoupled warpspec is value-correct and emitted the certified
  loader-core artifact on selected shapes

route-policy generator:
  converts perf JSON into a table saying where decoupled warpspec beats
  onchip_master
```

The new sweep variant lets us benchmark the combined policy as a single
candidate:

```text
tools/onchip_sdpa_sweep.py \
  --variants onchip_master,onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled_route_policy \
  --batch 1 \
  --heads 4 \
  --dim 64 \
  --block-size 64 \
  --lengths 768,1024
```

For the Stage234 table, that policy should select the decoupled route on the
long B1/H4/D64 and B2/H4/D128 rows, and fall back on the shorter or less stable
rows.

## Interpretation

This is still not the final production dispatcher. It is the first executable
shape-aware route variant. The useful next checks are:

1. Run the route-policy variant across the full eight-row gate island.
2. Compare it to `onchip_master`, `flash_hbm`, and raw decoupled warpspec.
3. Repeat the policy benchmark enough times to decide whether `min_speedup=1.0`
   is too weak for the B2/H4/D128 long rows.
4. Once stable, move the table from sweep-only policy into the compile/runtime
   routing path.
