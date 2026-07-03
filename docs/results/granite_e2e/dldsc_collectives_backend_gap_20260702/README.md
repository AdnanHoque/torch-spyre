# DLDSC all-gather/broadcast backend gap checkpoint - 2026-07-02

This checkpoint records the current state of the DLDSC communication-class path for Granite/flash attention after PR1-style scatter support.

## Source workload

The replayed bundle is the optimized Granite flash/prefill attention SDSC bundle from dev-pf:

/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/runs/granite_prefill_optimized_split_graphfix_20260702_114453/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_cypdouis

The relevant SDSCs are:

| SDSC | Edge | Communication class | Status |
|---|---|---|---|
| sdsc_10.json / 10_batchmatmul Tensor1 | mul -> ReStickifyOpLx -> batchmatmul | all_gather, pattern layout_allgather_restickify | Classified by Torch, not physically realized yet. |
| sdsc_18.json / 18_batchmatmul Tensor1 | activation/value operand into matmul | all_gather, pattern all_gather_replicate / matmul_operand_broadcast | Classified by Torch, needs staged backend realization. |

These are not scatter cases. They require source shards to be fanned out to multiple consumer cores.

## Backend experiments

Two DXP replays were run against the same bundle:

| Experiment | Run directory | Result |
|---|---|---|
| Single data row fanout | /home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/runs/dxp_replay_attention_kernel_diag_operand1_allgathers_stcdp_20260702_140806 | Fails in 10_batchmatmul after Running DCG for DL+Data; DCC verifier rejects LX_MODLRFIMM :: lrfimm:-2101120. |
| Staged standalone rows, one source shard per data row | /home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/runs/dxp_replay_attention_kernel_staged_standalone_allgather_rows_20260702_142603 | Also fails in 10_batchmatmul with the same LX_MODLRFIMM :: lrfimm:-2101120. |
| Staged standalone rows plus generic DCC LX LRF immediate split | /home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/runs/dxp_replay_attention_kernel_staged_standalone_allgather_rows_lrfimm_split_20260702_144019 | Passes DXP replay, exit_code=0. Both 10_batchmatmul and 18_batchmatmul compile with 32 staged DataDsc PCFG sections. |
| Full one-layer Granite prefill with the same DCC immediate split | /home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/runs/granite_prefill_optimized_lrfimm_split_20260702_144212 | Reaches runtime, then fails with RAS::PCI::BusFence. This is no longer the previous DXP/DCC compile failure. |
| Standalone latest flash attention on CDX with same Deeptools prototype | /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_lrfimm_split_autoload_20260702_151321 | Compiles and runs three kernels, then fails correctness: 96.7% mismatched, greatest absolute difference inf. No ReStickifyOpHBM rows; 32 ReStickifyOpLx rows; 32 backend layout-allgather plans. |
| Standalone latest flash attention with single-row translated all-gather | /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_single_dataop_20260703_002453 | Compiles and runs three kernels, then fails correctness: 99.2% mismatched, greatest absolute difference inf. This rules out row-count alone as the cause. |
| Standalone latest flash attention with separate relayout SuperDSC carrier | /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_standalone_relayout_20260703_003708 | Compiles and runs three kernels, then fails correctness: 99.2% mismatched, greatest absolute difference inf. This weakens mixed-SDSC scheduling as the primary explanation. |
| Standalone latest flash attention with layout-allgather disabled and matmul operand collective enabled | /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_matmul_operand_only_20260703_005239 | DXP aborts before runtime with `query fold dimension with higher fold factor`; emits 32 `matmul_operand_broadcast` plans. This is a separate backend compiler gap from the layout-transform value-wrong path. |
| Standalone latest flash attention, only `105_batchmatmul`, fissioned layout-allgather path | /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_allgather_only105_fission4_sourceonly_20260703_065227 | DXP/runtime completes, then fails correctness: 92.6% mismatched, greatest absolute difference inf. The emitted plan has 256 logical transfers. |
| Standalone latest flash attention, only `105_batchmatmul`, generic matmul operand path | /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_matmul_operand_only105_filtered_20260703_070937 | Emits the expected 1024-transfer `matmul_operand_broadcast` plan, then DXP aborts with `query fold dimension with higher fold factor`. |

The staged replay emitted 32 Creating PCFG for DataDsc sections for 10_batchmatmul, so splitting the monolithic row reduces row size but does not by itself fix the address-encoding issue.

After adding the generic DCC immediate split, the same staged replay completes. The next blocker is no longer descriptor import or immediate encoding. It is hardware-safe realization/scheduling of the generated movement in the full Granite runtime path.

## What this proves

Torch-side classification is working for this attention spill: the SDSC carries explicit DLDSC metadata that distinguishes layout all-gather/restickify and matmul operand all-gather/broadcast from PR1 scatter.

The first backend blocker was immediate encoding: staged source-shard fanout asked the generated STCDP/DataDsc path to encode a large negative LX offset. The observed immediate, -2101120, was just past a likely signed-immediate boundary. The generic DCC LX LRF immediate split clears that DXP/DCC replay blocker.

The current backend blocker is hardware-safe physical realization. Full Granite now reaches runtime execution, then bus-fences. That means the staged movement is no longer merely a descriptor/import problem; the remaining work is to validate the generated schedule and addresses on hardware.

Separately, full resident materialization is not the production target for this class. It has poor scaling: 32 producer shards x 32 consumers creates 1024 logical transfers. The production shape should be staged or loop-scoped all-gather/broadcast that feeds the consumer operand without pretending the entire replicated operand is permanently resident.

## Current split of responsibility

Frontend/Torch should:

- classify the edge: scatter, broadcast, multicast, gather, all_gather, reduction classes;
- attach the logical tensor distribution and consumer compute distribution to DLDSC metadata;
- expose enough metadata to cost and reject obviously non-viable relayouts.

Backend/Deeptools should:

- synthesize physical STCDP/InputFetch/L3-ring movement from that contract;
- stage all-gather/broadcast where a full materialized operand is too large;
- legalize LX addresses for STCDP/DataDsc PCFGs, including large base deltas that cannot fit in immediate fields;
- validate that the staged movement schedule is hardware-safe, not only DXP/DCC-compilable.

## Current next step

The standalone flash run has now isolated two backend gaps. First, the layout-changing attention edge is executable but value-wrong: destination-grouped transfer-coordinate (`runs/test_flash_grouped_materializer_transfercoords_20260703_000721`), single-row transfer-coordinate (`runs/test_flash_single_dataop_20260703_002453`), standalone relayout SuperDSC (`runs/test_flash_standalone_relayout_20260703_003708`), and the isolated fission-4 `105_batchmatmul` run (`runs/test_flash_direct_allgather_only105_fission4_sourceonly_20260703_065227`) all pass DXP/runtime and fail correctness. The diagnosis is that this edge is `layout_allgather_restickify`, not pure all-gather: the producer/restickify stick layout differs from the consumer KERNEL layout, and the 256-transfer summary under-materializes the operand. Second, with that edge disabled, the generic matmul operand path emits the expected 1024-transfer plan but hits a DXP fold-dimension abort before runtime. The next backend tasks are therefore transform-aware many-source LX->LX realization for `layout_allgather_restickify`, plus a separate DXP/codegen fix for staged matmul operand broadcast/all-gather.

## Files in this directory

- sdsc10_layout_allgather_restickify_plan_single_row.json
- sdsc18_matmul_operand_broadcast_plan_single_row.json
- sdsc10_layout_allgather_restickify_plan_staged_rows.json
- sdsc18_matmul_operand_broadcast_plan_staged_rows.json
- single_replay_summary.txt
- staged_replay_summary.txt
- staged_replay_lrfimm_split_summary.txt
- granite_runtime_lrfimm_split_summary.txt
- flash_runtime_value_wrong_summary.txt
- cdx_collective_investigation_update_20260703.md
- deeptools_experiment_diff_stat.txt
- attention_105_contract_isolation_20260703.md
- 105_batchmatmul_Tensor1_layout_allgather_restickify_fission4_sourceonly_plan_20260703.json
- 105_batchmatmul_Tensor1_matmul_operand_broadcast_filtered_plan_20260703.json
- deeptools_experiment.patch
- deeptools_experiment_lrfimm_split.patch
