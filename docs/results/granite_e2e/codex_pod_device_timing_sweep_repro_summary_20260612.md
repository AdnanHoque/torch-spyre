# Device Timing Sweep Reproduction

- rows: `758`
- ok rows: `758`
- raw: `profiler_runs/device_timing_repro_20260612_040740/all_splits/device_timing_sweep_raw_repro.txt`
- csv: `profiler_runs/device_timing_repro_20260612_040740/all_splits/device_best_vs_picks_repro.csv`

## Best Vs Picks

| op | phase | shape | repro best | best us | expected best | main pick | fix pick |
|---|---|---|---|---:|---|---|---|
| QK^T | prefill | 512x32x512x128 | `4_1_8_1` | 731.06 | `1_2_8_2` | `32_1_1_1` | `1_4_8_1` |
| attn@V | prefill | 32x512x128x512 | `1_16_2_1` | 197.72 | `4_4_2_1` | `1_32_1_1` | `2_8_2_1` |
| Q/O proj | prefill | 1x512x4096x4096 | `1_8_4_1` | 331.07 | `1_4_8_1` | `1_4_8_1` | `1_4_8_1` |
| K/V proj | prefill | 1x512x1024x4096 | `1_8_4_1` | 117.55 | `1_4_8_1` | `1_4_8_1` | `1_4_8_1` |
| MLP up | prefill | 1x512x12800x4096 | `1_4_8_1` | 1037.61 | `1_4_8_1` | `1_4_8_1` | `1_4_8_1` |
| MLP down | prefill | 1x512x4096x12800 | `1_4_8_1` | 926.95 | `1_4_8_1` | `1_4_8_1` | `1_4_8_1` |
| QK^T | decode | 64x32x576x128 | `8_2_1_2` | 89.93 | `4_4_1_2` | `32_1_1_1` | `1_32_1_1` |
| attn@V | decode | 32x64x128x576 | `1_4_2_3` | 55.04 | `2_8_2_1` | `1_32_1_1` | `1_16_2_1` |
| Q/O proj | decode | 1x64x4096x4096 | `1_4_8_1` | 231.84 | `1_4_8_1` | `1_32_1_1` | `1_4_8_1` |
| K/V proj | decode | 1x64x1024x4096 | `1_8_4_1` | 66.78 | `1_8_4_1` | `1_32_1_1` | `1_4_8_1` |
| MLP up | decode | 1x64x12800x4096 | `1_4_8_1` | 673.44 | `1_4_8_1` | `1_4_8_1` | `1_4_8_1` |
| MLP down | decode | 1x64x4096x12800 | `1_4_4_1` | 689.20 | `1_4_8_1` | `1_32_1_1` | `1_4_8_1` |
