# Granite S512 full AIU timeout: staged gather/restickify

This artifact records the first full Granite prefill AIU attempt after the staged gather_then_restickify backend lowering passed DXP replay and focused unit tests.

## Result

- Run directory: /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_full_aiu_gather_restickify_20260706_072858
- Return code: 124 (timeout killed the process)
- Backend plans emitted: 2
- Logical transfers planned: 1536
- Kineto/profile output: none produced before timeout
- Execution reached runtime kernel launch after SDSC generation.

## Backend plans

| SDSC | kind | pattern | strategy | status | logical transfers |
|---|---|---|---|---|---:|
| 16_batchmatmul | matmul_operand_broadcast | all_gather_replicate | gather_then_restickify | lowered_gather_then_restickify | 1024 |
| 8_batchmatmul | matmul_operand_broadcast | all_gather_replicate | gather_then_restickify | lowered_gather_then_restickify | 512 |

## Interpretation

The frontend/backend contract and DXP lowering are working well enough to produce two realized matmul_operand_broadcast plans for the Granite attention block. The failure is later: full AIU execution did not complete the first profiled block iteration before the 30 minute timeout.

The stderr tail shows runtime execution had reached these kernels before termination:

1. sdsc_fused_linear_rms_norm_0
2. sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1
3. sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2

The timeout is therefore a runtime/scheduling gap for the staged data movement path, not a DXP import/replay failure. The next diagnosis should isolate the exact generated bundle that hangs and compare the staged ReStickifyOpLx/STCDPOpLx schedule against the previously passing kernel-neighbor backend path.

## Files

- run_summary.json: machine-readable summary with return code and backend plans
- backend_plans/*.json: DXP-produced communication plans
- command.txt, env.txt: exact run command and environment
- logs/stdout_tail.txt, logs/stderr_tail.txt: terminal log tails

## Runbook comparison

After this timeout, we re-read the maintained Granite block runbook in spyre-granite-e2e-bench/runbooks/granite_block_e2e.md and profiler runbook in runbooks/aiu_kernel_profiler.md. The runbook calls out two details that should be treated as required for the next e2e attempt:

- Use the known Granite block probe stack/environment rather than an ad hoc active checkout environment.
- Be careful with profiler/runtime library ordering; Deeptools library directories should not outrank the runtime/Flex libraries needed by torch_spyre/_C.so.

This timeout run is still useful because it proves the staged backend generated two realized communication plans and reached runtime kernel launch. It should not be treated as a final performance or correctness result until rerun using the maintained Granite block runbook.
