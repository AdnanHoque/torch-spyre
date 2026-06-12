# Granite Device-Timing Cost-Model Tune

Run root: `profiler_runs/device_timing_repro_20260612_040740`

## Validation

- Exhaustive forced-split sweep: `758/758` rows, `0` errors.
- Pure tests: `pytest -q torch-spyre/tests/inductor/test_work_division.py` -> `11 passed`.
- Post-patch selected-split probe: 12 Granite shapes, 20 profiler reps each.

## Selected Splits

| phase | op | best split | best us | before split | before us | after split | after us | after vs before | after vs best |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill | QK^T | `4_1_8_1` | 731.06 | `1_4_8_1` | 737.75 | `1_4_8_1` | 733.83 | 1.01x | 1.00x |
| prefill | attn@V | `1_16_2_1` | 197.72 | `1_32_1_1` | 326.89 | `1_16_2_1` | 195.89 | 1.67x | 0.99x |
| prefill | Q/O proj | `1_8_4_1` | 331.07 | `1_8_4_1` | 330.72 | `1_8_4_1` | 331.10 | 1.00x | 1.00x |
| prefill | K/V proj | `1_8_4_1` | 117.55 | `1_8_4_1` | 117.69 | `1_8_4_1` | 116.46 | 1.01x | 0.99x |
| prefill | MLP up | `1_4_8_1` | 1037.61 | `1_4_8_1` | 1028.27 | `1_4_8_1` | 1027.53 | 1.00x | 0.99x |
| prefill | MLP down | `1_4_8_1` | 926.95 | `1_8_4_1` | 1006.37 | `1_8_4_1` | 1008.07 | 1.00x | 1.09x |
| decode | QK^T | `8_2_1_2` | 89.93 | `32_1_1_1` | 202.65 | `1_4_3_2` | 105.09 | 1.93x | 1.17x |
| decode | attn@V | `1_4_2_3` | 55.04 | `1_32_1_1` | 94.06 | `1_4_2_3` | 55.14 | 1.71x | 1.00x |
| decode | Q/O proj | `1_4_8_1` | 231.84 | `1_4_8_1` | 231.90 | `1_4_8_1` | 226.43 | 1.02x | 0.98x |
| decode | K/V proj | `1_8_4_1` | 66.78 | `1_4_8_1` | 67.64 | `1_4_8_1` | 68.61 | 0.99x | 1.03x |
| decode | MLP up | `1_4_8_1` | 673.44 | `1_4_8_1` | 672.29 | `1_4_8_1` | 669.78 | 1.00x | 0.99x |
| decode | MLP down | `1_4_4_1` | 689.20 | `1_4_8_1` | 672.97 | `1_4_8_1` | 706.11 | 0.95x | 1.02x |

## Totals

- Before selected total: `5489.20 us`.
- After selected total: `5244.05 us` (`1.05x` faster than before).
- Device-best total from exhaustive sweep: `5148.19 us`; after is `1.02x` device-best.

## Read

The tune primarily fixes true-BMM attention splits. Prefill `attn@V` moves from `1_32_1_1` to `1_16_2_1`, decode `attn@V` moves from `1_32_1_1` to `1_4_2_3`, and decode `QK^T` moves from `32_1_1_1` to `1_4_3_2`. Shared-weight projection and MLP split choices are otherwise stable.
