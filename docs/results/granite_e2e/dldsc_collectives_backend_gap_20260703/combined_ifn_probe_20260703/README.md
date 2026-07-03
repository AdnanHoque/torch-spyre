# DLDSC collectives IFN/STCDP probe, 2026-07-03

This artifact records the backend boundary found while trying to realize Granite/attention `matmul_operand_broadcast` / `layout_allgather_restickify` metadata through DLDSC + `STCDPOpLx` input-neighbor fetch.

## Source bundle

Replay bundle:

```text
/home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108/runs/granite_relayout_s512_failclosed_20260703_173131/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_e0v78nrw
```

The relevant SDSCs are `sdsc_8.json` and `sdsc_16.json`; both carry `matmul_operand_broadcast` classifications with `communication_class=all_gather`, `communication_pattern=all_gather_replicate`, and `transfer_count=1024`.

## Probe sequence

| probe | env/change | result |
| --- | --- | --- |
| `dxp_replay_combined_ifn_20260703_175949` | `DEEPTOOLS_ENABLE_UNSAFE_MATMUL_OPERAND_BROADCAST=1`, `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_COMBINED_IFN=1` | Fails in `L3DlOpsScheduler.cpp`: double buffering and input-neighbor fetch cannot coexist in one DSC. |
| `dxp_replay_mixed_hbm_ifn_probe_20260703_192341` | Also bypassed the mixed HBM/IFN scheduler guard behind `DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1` | Fails in `dcg_manager.cpp`: DCG allows only one combined input-fetch data-op mapping; many data-op rows violate the single-IFN assumption. |
| `dxp_replay_one_ifn_probe_20260703_192656` | Also limited materialization to one IFN data-op with `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_ONE_IFN_PROBE=1` | Fails in `dcg_manager.cpp`: the inserted combined IFN row plus the original DL row makes DCG see both input-fetch and separate DL op. |
| `dxp_replay_replace_dl_step_probe_20260703_192950` | Replaced the original DL schedule row with the combined IFN+DL row using `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_REPLACE_DL_STEP=1` | Gets further, then fails in `inputNeighFetchOp.cpp`: IFN requires every labeled tensor in the DL DSC to be LX-only. |
| `dxp_replay_ifn_allow_output_hbm_probe_20260703_193403` | Also bypassed the all-LX IFN check under the same diagnostic env | Gets into `STCDPOpLx` physicalization, then fails in `stcdpOp.cpp`: generated subpiece size is smaller than the stick dimension. |

## Interpretation

The current backend can classify the edge and expand it to the expected logical movement plan. The saved plan has `logical_transfer_count=1024`, matching 32 producer shards fanned out to 32 consumer cores.

The missing production support is the realization layer:

1. DCG/DCC currently assumes at most one input-neighbor fetch data-op associated with a DL op.
2. Input-neighbor fetch currently assumes the DL op is otherwise all-LX; mixed HBM writeback or double buffering is rejected.
3. The generated `STCDPOpLx` slices for this matmul RHS all-gather are not stick-aligned; the physicalizer rejects sub-stick pieces.

That means this class is not just PR1 scatter. It is an all-gather/broadcast-into-matmul-operand class requiring a staged matmul operand realization, or a stick-aware Lx restickify/gather representation that avoids sub-stick `STCDPOpLx` pieces.

## What this implies

For Granite and flash attention, the next non-weight HBM spill class is `matmul_operand_broadcast` / `layout_allgather_restickify`, not basic scatter. Frontend metadata is now good enough to expose the class. Backend support still needs either:

- a true multi-piece/multi-row IFN path that can coexist with HBM output/writeback and matmul DL execution, plus stick-aligned movement, or
- a higher-level backend relayout op that materializes the matmul RHS in an LX layout the existing matmul lowering can consume.

The current diagnostic patches are not production patches. They are only boundary probes.
