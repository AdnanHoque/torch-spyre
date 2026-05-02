# Wave-quantization diagnostic — Stream-K Phase 0

PyTorch:        2.10.0+cpu
torch_spyre:    (editable)
SENCORES:       32 (default)
warmup iters:   3
measure iters:  15

**Method**: at each shape, default planner picks an `(m_split, n_split, k_split)` factorization. Splits are captured via `parse_op_spec` hook. Cores used = product of factors; idle = 32 - cores_used. Wall time measured with per-iter sync.

**Stream-K hypothesis**: shapes with idle cores are leaving perf on the table. A planner that does 1D linearized work assignment (Stream-K-style) could activate idle cores at the cost of cross-core partial reductions or padding overhead.

| shape | use case | M, N, K | splits (size×cores) | cores/32 | idle | wall ms |
|---|---|---|---|---:|---:|---:|
| LoRA r=16 down decode | LoRA adapter | 1×16×4096 | `[16×1c, 4096×32c]` | 32/32 | 0 | 2.88 |
| LoRA r=16 down prefill | LoRA adapter | 128×16×4096 | `[128×32c, 16×1c, 4096×1c]` | 32/32 | 0 | 2.84 |
| LoRA r=64 down prefill | LoRA adapter | 128×64×4096 | `[128×32c, 64×1c, 4096×1c]` | 32/32 | 0 | 2.84 |
| L3-70B GQA TP=8 kv decode | Llama-70B GQA TP=8 | 1×128×8192 | `[128×2c, 8192×16c]` | 32/32 | 0 | 2.92 |
| L3-70B GQA TP=8 kv prefill | Llama-70B GQA TP=8 | 128×128×8192 | `[128×32c, 128×1c, 8192×1c]` | 32/32 | 0 | 2.90 |
| L3-8B GQA TP=4 kv prefill | Llama-8B GQA TP=4 | 128×256×4096 | `[128×32c, 256×1c, 4096×1c]` | 32/32 | 0 | 2.91 |
| DeepSeek-MoE inter=1408 prefill | DeepSeek-MoE per-expert | 192×1408×2048 | `[192×32c, 1408×1c, 2048×1c]` | 32/32 | 0 | 3.10 |
| Qwen3-MoE inter=1536 prefill | Qwen3-MoE per-expert | 128×1536×2048 | `[128×32c, 1536×1c, 2048×1c]` | 32/32 | 0 | 3.04 |
| Prime M=257 prefill | dynamic prefill | 257×4096×4096 | `[257×1c, 4096×32c, 4096×1c]` | 32/32 | 0 | 3.20 |
| Prime M=521 prefill | dynamic prefill | 521×4096×4096 | `[521×1c, 4096×32c, 4096×1c]` | 32/32 | 0 | 132.85 |
| L3-8B q_proj prefill (aligned) | reference / aligned | 128×4096×4096 | `[128×32c, 4096×1c, 4096×1c]` | 32/32 | 0 | 3.79 |
| L3-70B q_proj prefill (aligned) | reference / aligned | 128×8192×8192 | `[128×32c, 8192×1c, 8192×1c]` | 32/32 | 0 | 6.44 |

**Summary**: 0 of 12 measured shapes leave at least one core idle under the default planner.
