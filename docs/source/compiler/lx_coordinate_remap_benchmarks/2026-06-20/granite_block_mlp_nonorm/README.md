# Granite Block FF Coordinate-Remap Probe

This run checks whether the PR 1 coordinate-remap pass gives an end-to-end
kernel-time win inside a Granite-block-derived module, not only in the isolated
`fms_granite_micro.swiglu` benchmark.

The benchmark uses `tools/perf_suite_fms_granite_block_empty_params_op.py` with
`SPYRE_FMS_GRANITE_BLOCK_SCOPE=mlp`.  That scope runs the FMS Granite block's
`ff_sub_layer` directly, with fake/empty Spyre-resident weights and no
normalization prefix.  This isolates the SwiGLU feed-forward path because the
full block and norm-prefixed MLP currently hit an unrelated backend element
arrangement limitation before SDSC generation.

## Result

Primary metric: Kineto trace-derived `kernel_ms_per_iter`.

| Variant | kernel_ms_per_iter | memory_ms_per_iter | Planned exact-reshard bytes | Remap chunks | Result |
|---|---:|---:|---:|---:|---:|
| branch-baseline | 16.2672535 | 0.200217 | 0 | 0 | reference |
| coordinate-remap | 13.0504460 | 0.3063223 | 39,321,600 logical bytes | 10 | 19.78% faster |

The SDSC summary reports `27,033,600` realized remap bytes.  The edge report
reports `39,321,600` logical planned bytes because it counts the three
producer-consumer exact-reshard edges before local-relay/range realization.

## Communication Classes

Coordinate-remap emitted three exact-reshard edges from the fused projection
into the pointwise chain:

- `op0 -> op1`
- `op0 -> op4`
- `op0 -> op5`

Remaining classes:

- `same-view-lx-planner`: four pointwise-chain edges are left to main's
  `LX_PLANNER`.
- `fanout-multicast-unsupported`: final `mul -> down_projection` activation
  remains a follow-up.
- `layout-or-stick-unsupported`: weight restickifies remain out of scope.

See:

- [branch-baseline edge report](branch-baseline/onchip_move_edge_report.md)
- [coordinate-remap edge report](coordinate-remap/onchip_move_edge_report.md)
- [coordinate-remap Jamie-style SDSC](coordinate-remap/sdsc_jamie_summary.md)
- [HBM round-trip comparison](coordinate-remap/sdsc_hbm_roundtrip_comparison.md)

## Full-Block And Attention Attempts

Update: full one-layer FMS `GraniteBlock` prefill now runs end to end after the
split-multi trailing-unflattening fix and with the eager-spyre FMS norm patch.
See [granite_block_layer_e2e](../granite_block_layer_e2e/README.md).  The notes
below are retained as historical failure isolation from before that fix.

Full block attempt:

- Run directory:
  `/tmp/granite_block_coordinate_remap_profile_20260620_full_frozen`
- Status: failed before SDSC generation.
- Blocker: `Unsupported: All inputs to an op must have same element
  arrangement, op: mul, args: DL16_TO_FP32 and STANDARD`.
- Interpretation: this is a norm/residual scalar path blocker, not a
  coordinate-remap fallback.

Norm-prefixed MLP attempt:

- Run directory: `/tmp/granite_block_coordinate_remap_profile_20260620_mlp`
- Status: same element-arrangement failure as full block.

Attention-only attempt:

- Run directory:
  `/tmp/granite_block_coordinate_remap_profile_20260620_attention_nonorm`
- Status: failed before complete benchmark.
- Blocker: RoPE path calls `q_rope.float()` and reaches
  `SpyreOpFuncs.to_dtype(..., use_compute_types=True)`, but the Spyre op
  handler does not accept `use_compute_types`.
- Interpretation: attention applicability for PR 1 is not measurable until the
  RoPE/to-dtype lowering issue is fixed or bypassed.
