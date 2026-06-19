# Granite Isolated-Matmul Peak Table

This table uses the Codex-pod standalone split sweep, not the fused GraniteBlock timing window. It is the cleaner view for the question: after forcing the best measured work division for a matmul shape, how close does that isolated matmul get to nominal DL16 peak?

Nominal DL16 peak used here: `98.304 TFLOP/s` (`49.152 TMAC/s`).

| phase | op | shape `B x M x N x K` | device-best split | best us | best TFLOP/s | peak % |
|---|---|---|---:|---:|---:|---:|
| prefill | QK^T | `512x32x512x128` | `4_1_8_1` | 731.06 | 2.94 | 3.0% |
| prefill | attn@V | `32x512x128x512` | `1_16_2_1` | 197.72 | 10.86 | 11.0% |
| prefill | Q/O proj | `1x512x4096x4096` | `1_8_4_1` | 331.07 | 51.89 | 52.8% |
| prefill | K/V proj | `1x512x1024x4096` | `1_8_4_1` | 117.55 | 36.54 | 37.2% |
| prefill | MLP up | `1x512x12800x4096` | `1_4_8_1` | 1037.61 | 51.74 | 52.6% |
| prefill | MLP down | `1x512x4096x12800` | `1_4_8_1` | 926.95 | 57.92 | 58.9% |
| decode | QK^T | `64x32x576x128` | `8_2_1_2` | 89.93 | 3.36 | 3.4% |
| decode | attn@V | `32x64x128x576` | `1_4_2_3` | 55.04 | 5.49 | 5.6% |
| decode | Q/O proj | `1x64x4096x4096` | `1_4_8_1` | 231.84 | 9.26 | 9.4% |
| decode | K/V proj | `1x64x1024x4096` | `1_8_4_1` | 66.78 | 8.04 | 8.2% |
| decode | MLP up | `1x64x12800x4096` | `1_4_8_1` | 673.44 | 9.97 | 10.1% |
| decode | MLP down | `1x64x4096x12800` | `1_4_4_1` | 689.20 | 9.74 | 9.9% |

## Readout

- Prefill isolated matmuls range from `3.0%` to `58.9%` of nominal peak on their best measured split.
- Decode isolated matmuls range from `3.4%` to `10.1%` of nominal peak on their best measured split.
- Small attention matmuls are structurally far from peak because the total work is small and launch/pipeline overhead dominates.
- Wide projection and MLP matmuls are much closer to peak; those are the cases where work-division tuning has the clearest payoff.

## Source Artifact

- `/home/adnan/dt-inductor/granite-e2e/docs/results/granite_e2e/codex_pod_device_best_vs_picks_repro_20260612.csv`
