# Granite One-Gate Loop-Scoped LX Collective Update - 2026-07-07

## Summary

This update records the first CDX run where the Granite block prefill path compiles and runs end to end with the collectives prototype using the public gate only:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=1
```

The key backend gap fixed in this checkpoint was over-reservation for loop-scoped matmul operand movement. Deeptools was selecting the right communication class (`matmul_operand_broadcast` / `all_gather_replicate`) but allocated the full replicated operand in LX. For the attention value-side operand this requested 2,097,152 bytes per core and failed. The fix makes the loop-scoped neighbor destination reserve one producer chunk instead of the whole replicated operand.

## Code Pointers

Torch artifact branch:

- `AdnanHoque/torch-spyre:ah/comms-collectives`
- artifact commit at time of this note: `b12a8eac`

Deeptools prototype branch:

- `Adnan-Hoque1/deeptools:ah/comms-collectives`
- commit: `4ec7b9ae9`
- message: `[DXP] prototype loop-scoped LX collectives for Granite`

Archived patches:

- Full branch patch against Deeptools master: `deeptools_patch_ah_comms_collectives_20260707/deeptools_ah_comms_collectives_loop_scoped_chunk_alloc.patch`
- Incremental patch from previous prototype head `5faca110`: `deeptools_patch_ah_comms_collectives_20260707/deeptools_incremental_loop_scoped_chunk_alloc_from_5faca110.patch`

## CDX Reproduction Environment

Root:

```bash
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236
```

DXP wrapper behavior:

```bash
if [[ -n "${DXP_BACKEND_LX_FRAC_AVAIL:-}" ]]; then
  export DXP_LX_FRAC_AVAIL="${DXP_BACKEND_LX_FRAC_AVAIL}"
fi
exec /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools/build-deeptools/dxp/dxp_standalone "$@"
```

This split matters because Torch sees `DXP_LX_FRAC_AVAIL=0` as full frontend LX planning, while the DXP subprocess needs `DXP_LX_FRAC_AVAIL=1` to have backend chunk space for inserted relayout movement.

## Targeted DXP Replay

Previously failing attention bundle:

```bash
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_kernel_neighbor_embedded_marker_20260707_014735/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_v6or8xyq
```

Passing replay:

```bash
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/dxp_replay_attention_chunk_alloc_clean_20260707_022424
```

Result:

- return code: `0`
- plans: `3`
- all plans lowered as `loop_scoped_input_fetch`

Plans:

| plan | logical transfers | lowering |
| --- | ---: | --- |
| `16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json` | 1024 | `lowered_loop_scoped_kernel_neighbor` |
| `8_batchmatmul_Tensor0_0_matmul_operand_broadcast_plan.json` | 64 | `lowered_loop_scoped_kernel_neighbor` |
| `8_batchmatmul_Tensor1_1_matmul_operand_broadcast_plan.json` | 1024 | `lowered_loop_scoped_kernel_neighbor` |

## Granite Prefill E2E Run

Compile/run archive:

```bash
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_onegate_chunk_alloc_20260707_021624
```

Result:

- return code: `0`
- shape: `B=1, S=512, E=4096`
- public feature gate: `SPYRE_LX_PLANNER_RELAYOUT=1`
- legacy force flags unset
- backend plans: `5`
- remaining `ReStickifyOpHBM` files: `4`, all directly precede `batchmatmul` and use a `KERNEL` tensor, so they are weight/kernel prelayout restickifies rather than activation handoff spills.

Loop-scoped plans:

| plan | logical transfers | lowering |
| --- | ---: | --- |
| `16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json` | 1024 | `loop_scoped_input_fetch` |
| `5_batchmatmul_Tensor0_0_matmul_operand_broadcast_plan.json` | 128 | `loop_scoped_input_fetch` |
| `7_batchmatmul_Tensor0_0_matmul_operand_broadcast_plan.json` | 256 | `loop_scoped_input_fetch` |
| `8_batchmatmul_Tensor0_0_matmul_operand_broadcast_plan.json` | 64 | `loop_scoped_input_fetch` |
| `8_batchmatmul_Tensor1_1_matmul_operand_broadcast_plan.json` | 1024 | `loop_scoped_input_fetch` |

Remaining HBM restickify rows:

| kernel directory | row | interpretation |
| --- | --- | --- |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_*` | `sdsc_0: ReStickifyOpHBM -> batchmatmul` | attention/output-projection kernel weight prelayout |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_*` | `sdsc_0: ReStickifyOpHBM -> batchmatmul` | SwiGLU first projection kernel weight prelayout |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_*` | `sdsc_4: ReStickifyOpHBM -> batchmatmul` | SwiGLU down projection kernel weight prelayout |
| `sdsc_fused_linear_rms_norm_0_*` | `sdsc_6: ReStickifyOpHBM -> batchmatmul` | linear kernel weight prelayout |

These are out of scope for this pass because weight preloading/prelayout is owned separately. The activation matmul operand handoffs covered by the five backend plans are now represented as on-chip loop-scoped movement.

## Profiled Run

Profile archive:

```bash
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_onegate_chunk_alloc_profile_20260707_022114
```

Result:

- return code: `0`
- `all_ms`: `[29.782, 28.985, 30.06]`
- trace: `block_prefill/trace/adnan-cdx-spyre-dev-pf_280358.1783390928934428690.pt.trace.json`
- trace summary exists, but `kernel_ms_per_iter` is `0.0` because this trace did not classify Spyre kernel events under the current summarizer.

The run is therefore profiler-enabled and stable, but the Kineto summarizer still needs adjustment before using this run for kernel-time speedup claims.

## Communication Class Status

| class | current status |
| --- | --- |
| scatter / permutation | covered by PR1-style DLDSC relayout path |
| broadcast / multicast | classified for matmul operands; loop-scoped path is the viable realization direction |
| all-gather / replicate | working for Granite matmul operands through `matmul_operand_broadcast` and loop-scoped input fetch |
| gather | partially represented, but generic non-matmul gather still needs a production realization path |
| form-changing restickify | `ReStickifyOpLx` exists; dense gather-then-restickify can hit IBUFF and should not be the default for large Granite operands |
| reduce / all-reduce | not implemented in this copy-only relayout lane; requires arithmetic collective support |

## Next Steps

1. Clean the loop-scoped chunk allocation patch into a smaller Deeptools branch if this is to become a reviewable backend PR.
2. Fix trace summarization for Spyre kernel events in this Granite probe so speedup can be reported as kernel time, not just wall time.
3. Run the standalone flash script with the same Deeptools commit and one-gate setup.
4. For remaining non-weight HBM rows, verify with SDSC tables that they are only weight/kernel prelayout. If a non-weight activation row appears in another Granite shape, classify it into the next communication class before adding backend support.
5. Add generic gather and reduce/all-reduce only after the matmul operand broadcast/all-gather lane is clean, because reduce/all-reduce is arithmetic, not a pure copy movement.
