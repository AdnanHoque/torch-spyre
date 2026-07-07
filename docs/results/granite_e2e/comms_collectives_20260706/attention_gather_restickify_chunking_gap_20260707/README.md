# Attention gather-restickify chunking gap

Date: 2026-07-07

This artifact records a direct DXP replay experiment for the Granite prefill attention bundle after the one-gate gather/restickify patches.

## Source

- Torch branch: `gather-restickify`
- Torch SHA: `cc29ba259ec190e62f760f032ffd3355581a57df`
- Deeptools branch under test: local CDX checkout based on `ah/comms-collectives`
- Public gate: `SPYRE_LX_PLANNER_RELAYOUT=1`
- Backend LX capacity for replay: `DXP_LX_FRAC_AVAIL=1`
- Failing bundle:
  `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_onegate_gather_restickify_chunked_20260707_000644/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_1duninmz`

## What was tested

The backend staged gather/restickify path can split a matmul-operand all-gather into smaller physical transfer chunks. This replay varied the debug chunk cap:

| max pieces per chunk | DXP rc | result |
| --- | ---: | --- |
| 16 | 134 | fails in DCC lowering |
| 8 | 134 | fails in DCC lowering |
| 4 | 134 | fails in DCC lowering |
| 2 | 134 | fails in DCC lowering |
| 1 | 134 | fails in DCC lowering |

Raw logs are in `raw/`.

## Backend plans

The chunk=1 replay still emitted the expected backend plans:

| plan | operand | pattern | strategy | logical transfers | groups |
| --- | ---: | --- | --- | ---: | ---: |
| `8_batchmatmul_Tensor0_0_matmul_operand_broadcast_plan.json` | 0 | `all_gather_replicate` | `gather_then_restickify` | 64 | 16 |
| `8_batchmatmul_Tensor1_1_matmul_operand_broadcast_plan.json` | 1 | `all_gather_replicate` | `gather_then_restickify` | 1024 | 2 |
| `16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json` | 1 | `all_gather_replicate` | `gather_then_restickify` | 1024 | 1 |

The plan JSONs are archived in `chunk_1_backend_plans/`.

## Interpretation

This is not a frontend classification failure. Torch emits the relayout contract, and Deeptools recognizes the attention matmul operands as `all_gather_replicate` with `gather_then_restickify` lowering.

The failure is in the physical lowering/scheduling shape. Even with one gathered piece per chunk, DCC still lowers the scheduled receive/store work for `8_batchmatmul` into a region that exceeds LXSU instruction-buffer capacity:

```text
Require larger IBUFF
Max IBUFF(128) Current IBUFF(202) for unit:
... name = "lxsu-CL0" ...
error: Unable to lower successfully the module for sdsc: 8_batchmatmul
```

That means the next backend fix should not be a smaller transfer-list chunk. The attention all-gather path needs a more staged/loop-scoped realization so DCC does not see the entire gather/restickify schedule as one dense pre-matmul block.

## Current status

- One public feature flag is present in both portable patches: `SPYRE_LX_PLANNER_RELAYOUT=1`.
- The flash structural probe still reaches zero `ReStickifyOpHBM` with this gate on the known CDX run.
- Granite S=512 prefill now advances beyond the fused SwiGLU all-gather handoff, then fails in attention `8_batchmatmul` backend lowering.
- Next implementation target: backend staged scheduling for attention all-gather-replicate matmul operands.
