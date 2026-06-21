# Granite Block Layer E2E Probe

This snapshot records the first value-independent one-layer FMS Granite block
prefill run that completed end to end on the AIU from the
`swiglu-ws-co-remap` branch.

The enabling fix is in `torch_spyre/_inductor/split_multi_ops.py`: split-multi
reconstruction now handles a trailing flattened input view, specifically the
Granite RoPE case where a consumer indexed as `[B, S, H, 128]` loads an
intermediate shaped `[B, S, H, 2, 1, 64]`.

## Results

Metric here is the probe's wall-sync measured iteration after one warmup.  Use
Kineto trace-derived `kernel_ms` for publication-quality performance.

| Run | Attention | Coordinate remap | returncode | median_ms | Output | KV cache |
|---|---|---:|---:|---:|---|---|
| [baseline-causal](baseline-causal/result.json) | `sdpa_causal` | off | 0 | 23.835897 | `[1,512,4096]` | `2 x [1,8,512,128]` |
| [baseline-bidirectional](baseline-bidirectional/result.json) | `sdpa_bidirectional` | off | 0 | 24.121284 | `[1,512,4096]` | `2 x [1,8,512,128]` |
| [coordinate-remap-bidirectional](coordinate-remap-bidirectional/result.json) | `sdpa_bidirectional` | on | 0 | 25.817156 | `[1,512,4096]` | `2 x [1,8,512,128]` |

## Coordinate-Remap Coverage

The coordinate-remap full-block run completed and emitted
`OnChipMoveCoordinateRemap` SDSC rows in attention and MLP kernels.  The
planner JSONL reported:

- `10` planned coordinate-remap edges
- `44` skipped edges
- `87,556,096` planned bytes
- `684,032` planned cells

Skipped edges:

- `31` same-view edges owned by main's `LX_PLANNER`
- `7` duplicate consumer-owner edges
- `5` unsupported non-128-byte-stick-dim edges
- `1` ambiguous stick outer-dim edge

The coordinate-remap run is slower in this single wall-sync smoke.  That is not
surprising: full-block movement coverage now includes attention and residual
edges that were not tuned for PR 1.  Treat this as an e2e correctness/probing
artifact, not as a speedup result.

## Artifacts

- [baseline causal result](baseline-causal/result.json)
- [baseline causal summary](baseline-causal/summary.md)
- [baseline bidirectional result](baseline-bidirectional/result.json)
- [baseline bidirectional summary](baseline-bidirectional/summary.md)
- [coordinate-remap bidirectional result](coordinate-remap-bidirectional/result.json)
- [coordinate-remap bidirectional summary](coordinate-remap-bidirectional/summary.md)
- [coordinate-remap planner JSONL](coordinate-remap-bidirectional/onchip_move.jsonl)
