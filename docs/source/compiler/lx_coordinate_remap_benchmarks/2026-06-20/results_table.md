# LX Coordinate Remap Benchmark Results

| case | variant | shape | kernel_ms_per_iter | vs_upstream_main | vs_branch_baseline | memory_ms_per_iter | sdsc_count | row_count | remap_chunks | remap_bytes | planned_edges | planned_bytes | benchmark_rc | artifact_rc | artifact_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_bmm | upstream-main | Custom BMM SwiGLU probe, B=1 S=256 E=128 H=512 | 0.042227 | 0.00% | 0.64% | 0.038941 | 8 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | small_bmm/upstream-main |
| small_bmm | branch-baseline | Custom BMM SwiGLU probe, B=1 S=256 E=128 H=512 | 0.042498 | -0.64% | 0.00% | 0.040955 | 8 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | small_bmm/branch-baseline |
| small_bmm | planned-only | Custom BMM SwiGLU probe, B=1 S=256 E=128 H=512 | 0.041898 | 0.78% | 1.41% | 0.046052 | 8 | 22 | 0 | 0 | 3 | 786432 | 0 | 0 | small_bmm/planned-only |
| small_bmm | coordinate-remap | Custom BMM SwiGLU probe, B=1 S=256 E=128 H=512 | 0.038987 | 7.67% | 8.26% | 0.048497 | 8 | 25 | 3 | 270336 | 3 | 786432 | 0 | 0 | small_bmm/coordinate-remap |
| prefill_bmm | upstream-main | Custom BMM prefill SwiGLU probe, B=1 S=512 E=4096 H=12800 | 5.717748 | 0.00% | -0.20% | 5.421842 | 8 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | prefill_bmm/upstream-main |
| prefill_bmm | branch-baseline | Custom BMM prefill SwiGLU probe, B=1 S=512 E=4096 H=12800 | 5.706476 | 0.20% | 0.00% | 5.454924 | 8 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | prefill_bmm/branch-baseline |
| prefill_bmm | coordinate-remap | Custom BMM prefill SwiGLU probe, B=1 S=512 E=4096 H=12800 | 5.445835 | 4.76% | 4.57% | 5.939964 | 8 | 25 | 3 | 13516800 | 3 | 39321600 | 0 | 0 | prefill_bmm/coordinate-remap |
| decode_bmm | upstream-main | Custom BMM decode-shaped SwiGLU probe, B=1 S=1 E=4096 H=12800 | 4.031822 | 0.00% | 0.34% | 5.297144 | 8 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | decode_bmm/upstream-main |
| decode_bmm | branch-baseline | Custom BMM decode-shaped SwiGLU probe, B=1 S=1 E=4096 H=12800 | 4.045427 | -0.34% | 0.00% | 6.014840 | 8 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | decode_bmm/branch-baseline |
| decode_bmm | coordinate-remap | Custom BMM decode-shaped SwiGLU probe, B=1 S=1 E=4096 H=12800 | 4.117234 | -2.12% | -1.77% | 5.389968 | 8 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | decode_bmm/coordinate-remap |
| jamie_mlp | upstream-main | spyre-perf-suite jamie/dev built-in mlp, shape 1x512x4096 | 5.698974 | 0.00% | -0.10% | 5.397703 | 8 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | jamie_mlp/upstream-main |
| jamie_mlp | branch-baseline | spyre-perf-suite jamie/dev built-in mlp, shape 1x512x4096 | 5.693087 | 0.10% | 0.00% | 5.633852 | 8 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | jamie_mlp/branch-baseline |
| jamie_mlp | coordinate-remap | spyre-perf-suite jamie/dev built-in mlp, shape 1x512x4096 | 5.441576 | 4.52% | 4.42% | 6.562700 | 8 | 25 | 3 | 13516800 | 3 | 39321600 | 0 | 0 | jamie_mlp/coordinate-remap |
