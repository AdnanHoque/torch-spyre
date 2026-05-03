# Core-emission sweep results
# PyTorch 2.10.0+cpu, SENCORES=32 (default), warmup=3, iters=15

## Phase A — natural picks (element_priority=True)

| shape | forced split | default split | default ms | reversed split | reversed ms | speedup |
|---|---|---|---:|---|---:|---:|
| L3-8B q_proj prefill | — | `[128x1c, 4096x32c, 4096x1c]` | 3.14 | `[128x1c, 4096x32c, 4096x1c]` | 3.14 | 0.999x |
| L3-8B GQA kv_proj prefill | — | `[128x2c, 1024x16c, 4096x1c]` | 2.97 | `[128x2c, 1024x16c, 4096x1c]` | 2.99 | 0.995x |
| L3-8B MLP gate/up prefill | — | `[128x1c, 14336x32c, 4096x1c]` | 3.68 | `[128x1c, 14336x32c, 4096x1c]` | 3.69 | 0.999x |
| L3-8B MLP down prefill | — | `[128x1c, 4096x32c, 14336x1c]` | 4.57 | `[128x1c, 4096x32c, 14336x1c]` | 4.61 | 0.993x |
| L3-70B q_proj prefill | — | `[128x1c, 8192x32c, 8192x1c]` | 3.96 | `[128x1c, 8192x32c, 8192x1c]` | 3.96 | 1.000x |
| L3-70B GQA kv_proj prefill | — | `[128x2c, 1024x16c, 8192x1c]` | 3.04 | `[128x2c, 1024x16c, 8192x1c]` | 3.04 | 1.002x |
| L3-70B GQA TP=8 kv prefill | — | `[128x32c, 128x1c, 8192x1c]` | 2.90 | `[128x32c, 128x1c, 8192x1c]` | 2.90 | 0.997x |
| L3-70B MLP down prefill | — | `[128x16c, 8192x2c, 28672x1c]` | 7.91 | `[128x16c, 8192x2c, 28672x1c]` | 7.82 | 1.012x |
| Mixtral down per-expert | — | `[128x1c, 4096x32c, 14336x1c]` | 4.54 | `[128x1c, 4096x32c, 14336x1c]` | 4.59 | 0.988x |
| Qwen3-MoE gate per-expert | — | `[128x1c, 1536x24c, 2048x1c]` | 2.95 | `[128x1c, 1536x24c, 2048x1c]` | 2.93 | 1.006x |
| DeepSeek-MoE gate (M=192) | — | `[192x1c, 1408x22c, 2048x1c]` | 2.90 | `[192x1c, 1408x22c, 2048x1c]` | 2.90 | 0.999x |
| L3-8B q_proj decode | — | `[4096x32c, 4096x1c]` | 3.09 | `[4096x32c, 4096x1c]` | 3.06 | 1.010x |
| L3-70B GQA TP=8 kv decode | — | `[128x2c, 8192x16c]` | 2.94 | `[128x2c, 8192x16c]` | 2.93 | 1.002x |

**Geomean**: 1.000x   **Best**: 1.012x   **Worst**: 0.988x
**>=5% wins**: 0/13   **>=5% regressions**: 0/13

## Phase B — forced mixed splits on hot shapes

| shape | forced split | default split | default ms | reversed split | reversed ms | speedup |
|---|---|---|---:|---|---:|---:|
| L3-8B q_proj prefill | (2, 16, 1) | `[128x2c, 4096x16c, 4096x1c]` | 3.24 | `[128x2c, 4096x16c, 4096x1c]` | 3.25 | 0.999x |
| L3-8B q_proj prefill | (4, 8, 1) | `[128x4c, 4096x8c, 4096x1c]` | 3.19 | `[128x4c, 4096x8c, 4096x1c]` | 3.20 | 0.998x |
| L3-8B q_proj prefill | (8, 4, 1) | `[128x8c, 4096x4c, 4096x1c]` | 3.24 | `[128x8c, 4096x4c, 4096x1c]` | 3.23 | 1.002x |
| L3-8B q_proj prefill | (16, 2, 1) | `[128x16c, 4096x2c, 4096x1c]` | 3.35 | `[128x16c, 4096x2c, 4096x1c]` | 3.35 | 1.000x |
| L3-70B q_proj prefill | (2, 16, 1) | `[128x2c, 8192x16c, 8192x1c]` | 3.92 | `[128x2c, 8192x16c, 8192x1c]` | 3.90 | 1.006x |
| L3-70B q_proj prefill | (4, 8, 1) | `[128x4c, 8192x8c, 8192x1c]` | 4.03 | `[128x4c, 8192x8c, 8192x1c]` | 4.07 | 0.991x |
| L3-70B q_proj prefill | (8, 4, 1) | `[128x8c, 8192x4c, 8192x1c]` | 3.98 | `[128x8c, 8192x4c, 8192x1c]` | 3.96 | 1.005x |
| L3-70B q_proj prefill | (16, 2, 1) | `[128x16c, 8192x2c, 8192x1c]` | 4.52 | `[128x16c, 8192x2c, 8192x1c]` | 4.49 | 1.007x |
| L3-70B MLP down prefill | (2, 16, 1) | `[128x2c, 8192x16c, 28672x1c]` | 9.24 | `[128x2c, 8192x16c, 28672x1c]` | 9.12 | 1.013x |
| L3-70B MLP down prefill | (4, 8, 1) | `[128x4c, 8192x8c, 28672x1c]` | 10.32 | `[128x4c, 8192x8c, 28672x1c]` | 10.32 | 1.001x |
| L3-70B MLP down prefill | (8, 4, 1) | `[128x8c, 8192x4c, 28672x1c]` | 8.33 | `[128x8c, 8192x4c, 28672x1c]` | 8.32 | 1.001x |
| L3-70B MLP down prefill | (16, 2, 1) | `[128x16c, 8192x2c, 28672x1c]` | 7.94 | `[128x16c, 8192x2c, 28672x1c]` | 7.81 | 1.016x |

**Geomean**: 1.003x   **Best**: 1.016x   **Worst**: 0.991x
**>=5% wins**: 0/12   **>=5% regressions**: 0/12
