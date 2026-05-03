# output_element_priority comparison

PyTorch 2.10.0+cpu, SENCORES=32 (default), warmup=3, iters=15

| shape | default split | default ms | heuristic split | heuristic ms | speedup |
|---|---|---:|---|---:|---:|
| L3-8B q_proj prefill | `(32, 1, 1)` | 3.85 | `(1, 32, 1)` | 3.24 | 1.19x ✓ |
| L3-8B GQA kv_proj prefill | `(32, 1, 1)` | 3.21 | `(2, 16, 1)` | 3.04 | 1.06x ✓ |
| L3-8B MLP gate/up prefill | `(1, 32, 1)` | 3.78 | `(1, 32, 1)` | 3.78 | 1.00x |
| L3-8B MLP down prefill | `(32, 1, 1)` | 6.04 | `(1, 32, 1)` | 4.64 | 1.30x ✓ |
| L3-70B q_proj prefill | `(32, 1, 1)` | 6.54 | `(1, 32, 1)` | 4.05 | 1.61x ✓ |
| L3-70B GQA kv_proj prefill | `(32, 1, 1)` | 3.44 | `(2, 16, 1)` | 3.13 | 1.10x ✓ |
| L3-70B GQA TP=8 kv prefill | `(32, 1, 1)` | 2.99 | `(32, 1, 1)` | 3.00 | 1.00x |
| L3-70B MLP down prefill | `(16, 2, 1)` | 8.04 | `(16, 2, 1)` | 8.03 | 1.00x |
| Mixtral down per-expert | `(32, 1, 1)` | 6.05 | `(1, 32, 1)` | 4.65 | 1.30x ✓ |
| Qwen3-MoE gate per-expert | `(32, 1, 1)` | 3.17 | `(1, 24, 1)` | 3.05 | 1.04x |
| DeepSeek-MoE gate (M=192) | `(32, 1, 1)` | 3.20 | `(1, 22, 1)` | 3.00 | 1.07x ✓ |
| L3-8B q_proj decode | `(32, 1)` | 3.17 | `(32, 1)` | 3.15 | 1.01x |
| L3-70B GQA TP=8 kv decode | `(2, 16)` | 3.01 | `(2, 16)` | 3.00 | 1.00x |

**Geometric mean speedup**: 1.117x
**Best**: 1.61x   **Worst**: 1.00x
**>= 5% faster**: 7/13 shapes   **>= 5% regression**: 0/13
