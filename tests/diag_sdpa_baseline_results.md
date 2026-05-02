# SDPA baseline diagnostic — flash-attention Phase 0a

PyTorch:        2.10.0+cpu
torch_spyre:    (editable)
Model shape:    B=1, H=32, H_kv=8, D=128 (Llama-3-8B GQA)
warmup iters:   3
measure iters:  20
is_causal:      True
per-iter sync:  torch_spyre.streams.synchronize() inside the timed loop

**Naive SDPA path** (decompositions.py:494): scale Q+K, QK matmul, softmax, AV matmul, output transpose. The (B,H,S,S) score intermediate flows through DDR three times (write, read for softmax, write softmax, read for AV).

**Flash-attention target**: tile Q-rows; per Q-tile, loop over KV-tiles with running max + running sum + running output pinned in scratchpad. Score tensor never materialized in DDR. Per-Q-tile latency dominated by Q+KV streaming, not S² intermediate.

| S | wall ms | kernels | score (S²) | total DDR | eff BW GB/s | TFLOPs/s | flops/byte AI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 46.53 | 14 | 17 MB | 61 MB | 1.3 | 0.09 | 70.6 |
| 1024 | 129.15 | 14 | 67 MB | 222 MB | 1.7 | 0.13 | 77.3 |
| 2048 | 445.71 | 14 | 268 MB | 847 MB | 1.9 | 0.15 | 81.1 |
| 4096 | 1554.82 | 14 | 1074 MB | 3305 MB | 2.1 | 0.18 | 83.2 |

**Arithmetic intensity** (flops/byte) is the key signal: low AI -> bandwidth-bound, flash attention helps a lot. AI converges to a constant for the naive path because total DDR scales like S² (matching compute scaling), so naive is bandwidth-bound at all S.
