# Granite S512 lost-completion repro for staged gather/restickify

This artifact narrows the full Granite timeout to runtime completion after the relayout-bearing attention bundle.

## Source timeout

Original full-block timeout artifact:

- docs/results/granite_e2e/comms_collectives_20260706/granite_s512_full_aiu_timeout_gather_restickify_20260706

The full run emitted two backend plans in the first attention bundle:

- 8_batchmatmul: matmul_operand_broadcast, all_gather_replicate, gather_then_restickify, 512 logical transfers
- 16_batchmatmul: matmul_operand_broadcast, all_gather_replicate, gather_then_restickify, 1024 logical transfers

## Autoload wrapper repro

Directory: autoload_repro

The generated wrapper printed before/after logs around each compiled kernel. It completed:

1. sdsc_fused_linear_rms_norm_0
2. sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1

It then timed out after printing BEFORE for the next kernel:

- sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2

Because kernel launches are asynchronous, this does not prove that the second kernel fully completed on device. It proves the Python call returned and the next launch/barrier exposed the problem.

## Sync-after-attention repro

Directory: sync_after_attention_repro

This wrapper attempted to force synchronization immediately after the relayout-bearing attention bundle. The inserted Python call used torch_spyre.synchronize, which is not available in this runtime, so the wrapper also raised AttributeError. However, the runtime cleanup then emitted the useful Flex message:

RuntimeStream::synchronize() still waiting after 60000ms: in_flight_=1 device=0 - possible lost completion

That points to a device completion issue after the staged gather/restickify attention bundle. Current evidence favors an invalid staged schedule/barrier or unsupported completion semantics in the lowered STCDPOpLx/ReStickifyOpLx sequence over profiler overhead or library ordering.

## Next debug slice

Add a backend debug mode for the staged gather/restickify lowering that can selectively reduce the emitted plan:

1. emit only one of the two relayout sites, 8_batchmatmul or 16_batchmatmul;
2. emit only a small destination-core group;
3. emit one staged chunk at a time;
4. compare completion behavior after each bundle.

The aim is to find whether lost completion appears after a specific STCDP stage, a ReStickifyOpLx stage, or their schedule ordering.
