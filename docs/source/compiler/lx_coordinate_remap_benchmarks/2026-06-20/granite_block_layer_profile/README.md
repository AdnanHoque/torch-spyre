# Granite Block Layer Profile

This directory captures the June 21, 2026 profiled one-layer FMS Granite block
prefill run on pod `adnan-cdx-spyre-dev-pf`.

The probe uses empty Spyre-resident weights and the branch-owned
`benchmarks/granite_block_layer_probe.py` wrapper, so this is a compiler/runtime
measurement of the real FMS `GraniteBlock` path without checkpoint transfer
noise.

## Stack

- Torch checkout: `/tmp/torch-spyre-co-remap-native`, branch
  `swiglu-ws-co-remap`
- Deeptools checkout: `/tmp/deeptools-coordinate-remap-mainport-lean`
- FMS checkout:
  `/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/decode_regression_rev_ab_20260610_163300/foundation-model-stack-eager_spyre`
- Profiler overlay:
  `/home/adnan-cdx/dt-inductor-codex-clean/torch-spyre/torch_spyre/_C.so`
- Pod run root:
  `/tmp/granite_block_layer_profile_20260621_005122`

## Shape

- case: prefill
- input: `[1, 512, 4096]`
- weights: fused, empty fp16 Spyre tensors
- output: `[1, 512, 4096]`
- returned KV cache: `[[1, 8, 512, 128], [1, 8, 512, 128]]`
- warmups: `1`
- profiled iterations: `5`

## Results

Primary metric is Kineto trace-derived `kernel_ms_per_iter`.  The canonical
production comparison for this work is causal prefill, because that matches the
normal Granite prefill path.  Bidirectional prefill was collected as a
diagnostic cross-check only and should not be mixed into headline claims.

| variant | attention | coord remap | kernel ms/iter | memory ms/iter | wall median ms | delta vs baseline | kernel speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline_causal` | `sdpa_causal` | no | `16.385016` | `0.173744` | `23.332596` | - | - |
| `coord_causal` | `sdpa_causal` | yes | `13.819416` | `0.351185` | `21.236420` | `-2.565600 ms` | `15.66%` |

Diagnostic bidirectional cross-check:

| variant | attention | coord remap | kernel ms/iter | memory ms/iter | wall median ms | delta vs baseline | kernel speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline_bidirectional` | `sdpa_bidirectional` | no | `16.295306` | `0.288433` | `23.861885` | - | - |
| `coord_bidirectional` | `sdpa_bidirectional` | yes | `13.796416` | `0.473019` | `21.608829` | `-2.498889 ms` | `15.34%` |

The coordinate-remap win survives in the full one-layer block, not just the
isolated MLP/SwiGLU kernel. Wall time also improves, but the speedup claim above
is based on trace kernel time.

## Coordinate-Remap Coverage

Both coordinate variants planned the same movement set:

| variant | planned edges | skipped edges | planned bytes | planned cells |
|---|---:|---:|---:|---:|
| `coord_causal` | `10` | `44` | `87,556,096` | `684,032` |
| `coord_bidirectional` | `10` | `44` | `87,556,096` | `684,032` |

Skipped-edge reasons:

| reason | count |
|---|---:|
| `same-per-core-view-owned-by-lx-planner` | `31` |
| `consumer-duplicate-owner` | `7` |
| `coordinate-remap-v1-requires-128-byte-stick-dim` | `5` |
| `coordinate-remap-v1-ambiguous-stick-outer-dim` | `1` |

## Artifacts

Each variant directory contains:

- `result.json`: full probe result with trace summary embedded.
- `summary.md`: human-readable probe summary and generated SDSC inventory.
- `trace_summary.json`: compact Kineto trace summary.
- `probe.log`: stdout/stderr from the run.
- `onchip_move.jsonl`: raw planner decisions for coordinate variants only.
- `onchip_move_summary.json`: compact planner decision summary for coordinate
  variants only.

Raw `*.pt.trace.json` files were intentionally left under the pod run root
rather than checked into the branch.
