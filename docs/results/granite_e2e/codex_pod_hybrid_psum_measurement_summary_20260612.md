# Hybrid PSUM Measurement

Hybrid = current soft-core/additive-batch tune plus Claude-style per-core PSUM accounting.

| phase | op | best split | best us | previous split | previous us | hybrid split | hybrid us | hybrid vs previous | hybrid vs best |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill | QK^T | `4_1_8_1` | 731.06 | `1_4_8_1` | 733.83 | `1_4_8_1` | 731.46 | 0.997x | 1.001x |
| prefill | attn@V | `1_16_2_1` | 197.72 | `1_16_2_1` | 195.89 | `1_8_2_2` | 220.07 | 1.123x | 1.113x |
| prefill | Q/O proj | `1_8_4_1` | 331.07 | `1_8_4_1` | 331.10 | `1_8_4_1` | 330.90 | 0.999x | 1.000x |
| prefill | K/V proj | `1_8_4_1` | 117.55 | `1_8_4_1` | 116.46 | `1_8_4_1` | 116.12 | 0.997x | 0.988x |
| prefill | MLP up | `1_4_8_1` | 1037.61 | `1_4_8_1` | 1027.53 | `1_4_8_1` | 1031.10 | 1.003x | 0.994x |
| prefill | MLP down | `1_4_8_1` | 926.95 | `1_8_4_1` | 1008.07 | `1_8_4_1` | 1004.26 | 0.996x | 1.083x |
| decode | QK^T | `8_2_1_2` | 89.93 | `1_4_3_2` | 105.09 | `1_4_3_2` | 104.77 | 0.997x | 1.165x |
| decode | attn@V | `1_4_2_3` | 55.04 | `1_4_2_3` | 55.14 | `1_4_2_3` | 54.60 | 0.990x | 0.992x |
| decode | Q/O proj | `1_4_8_1` | 231.84 | `1_4_8_1` | 226.43 | `1_4_8_1` | 231.59 | 1.023x | 0.999x |
| decode | K/V proj | `1_8_4_1` | 66.78 | `1_4_8_1` | 68.61 | `1_4_8_1` | 67.52 | 0.984x | 1.011x |
| decode | MLP up | `1_4_8_1` | 673.44 | `1_4_8_1` | 669.78 | `1_4_8_1` | 672.57 | 1.004x | 0.999x |
| decode | MLP down | `1_4_4_1` | 689.20 | `1_4_8_1` | 706.11 | `1_4_8_1` | 686.15 | 0.972x | 0.996x |

## Totals

- Previous tuned total: `5244.05 us`
- Hybrid total: `5251.11 us`
- Device-best total: `5148.19 us`
- Hybrid vs previous tuned: `1.001x`
- Hybrid vs device-best: `1.020x`

## Read

The hybrid per-core PSUM term is theoretically cleaner, but with the existing soft-core/additive-batch terms it over-discounts K-splits for prefill `attn@V`, moving from the measured-best `1_16_2_1` to `1_8_2_2`. Decode `attn@V` and decode `QK^T` stay in the improved split family. Net result is essentially flat/slightly worse than the previous tuned model on this Codex-pod table.
