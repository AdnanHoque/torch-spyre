# Granite S512 Current-Head Structural Run - Pinned FMS

This directory archives the latest current-head Granite S512 structural run that
got far enough to emit SDSCs. It uses the current Torch and Deeptools
communication branches, but pins FMS to the older known-good Granite probe
checkout because the newer clean FMS path currently fails before SDSC generation.

## Heads

| Component | Value |
| --- | --- |
| Torch branch | `AdnanHoque/torch-spyre:gather-restickify` |
| Torch SHA | `102520820da890d6a62f781e86573f38dcc6f244` |
| Deeptools branch | `Adnan-Hoque1/deeptools:ah/comms-collectives` |
| Deeptools SHA | `a5ff55eee627c5c2bd4b7b0518bb0cbaad385952` |
| Granite bench SHA | `76cd51426ba1de6e99dd8fbf613cb0f32b71e87f` |
| FMS path | `/home/adnan-cdx/dt-inductor/foundation-model-stack` |
| FMS branch/SHA | `eager_spyre`, `b4f36b5af526b938db506a17dcd32d468a7a91d8` |

## Environment

The run used the usual split-LX setup:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
export PATH="$RUNROOT/tools/dxp-split-wrapper:$ROOT/deeptools/build-deeptools/dxp:$PATH"
```

The wrapper rewrites `DXP_LX_FRAC_AVAIL` only for the `dxp_standalone`
subprocess. This lets Torch plan with full LX while Deeptools still has backend
chunk space.

## Result

The run did not complete execution:

```text
RuntimeError: convert_address not yet implemented - waiting for flex support
```

It did generate SDSC artifacts before the runtime failure, so this directory is
useful for classifying the current-head HBM handoffs. No Kineto timing should be
quoted from this run.

## SDSC Counts

Generated reports:

- `sdsc_artifacts/jamie/sdsc_jamie_summary.md`
- `sdsc_artifacts/jamie/sdsc_jamie_table.md`
- `sdsc_artifacts/jamie/sdsc_jamie_table.csv`
- `sdsc_artifacts/summary/sdsc_table.md`
- `sdsc_artifacts/summary/sdsc_diff.md`

The structural summary reports:

```text
sdsc_count=44
row_count=114
sdsc_with_dataops=0
remap_chunks=0
ReStickifyOpHBM rows=10
```

The Jamie-style report groups by JSON file and reports 18 SDSC JSON files across
the generated kernel directories.

## ReStickifyOpHBM Classification

There are five explicit `ReStickifyOpHBM` ops, represented as ten tensor rows in
the table.

| Kernel / file | Role in Granite block | Classification | Status |
| --- | --- | --- | --- |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1.../sdsc_7.json` | Attention value-side activation handoff into the value-side BMM operand. Input is LX, output is `hbm+lx` kernel operand. | Non-weight activation handoff. Communication class is all-gather/replicate plus layout restickify for a matmul operand. | Still present at this current bounded-substrate head. This is the WSR/loop-scoped collectives boundary for full Granite, not a missing bounded-tile primitive. |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2.../sdsc_0.json` | Attention output-projection kernel/weight prelayout. | Weight/kernel prelayout. | Out of scope for comms. Expected to be handled by offline/preloaded weight layout work. |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3.../sdsc_0.json` | SwiGLU first projection kernel/weight prelayout. | Weight/kernel prelayout. | Out of scope for comms. |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3.../sdsc_4.json` | SwiGLU down-projection kernel/weight prelayout. | Weight/kernel prelayout. | Out of scope for comms. |
| `sdsc_fused_linear_rms_norm_0.../sdsc_6.json` | QKV / linear projection kernel/weight prelayout. | Weight/kernel prelayout. | Out of scope for comms. |

The important current-head conclusion is therefore narrow but useful:

```text
At current bounded-substrate heads, bounded scatter/broadcast/multicast/gather
tests are green, but full Granite S512 still has one non-weight activation HBM
handoff. That remaining handoff is the large attention matmul-operand
all-gather/restickify case, which needs WSR or loop-scoped tile execution before
it should be expected to disappear in full Granite.
```

Earlier loop-scoped prototype artifacts removed this class for S512, but that
older path is not the current bounded-substrate head.

