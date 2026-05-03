# Broadcast-topology probe
# matmul (M=128, n*N_per=256*n, K=8192), forced (1, n, 1)
# per-core compute = 536,870,912 flops (constant across n)
# per-core unique B = 4096 KB; shared A = 2048 KB
# warmup=3 iters=20

# n=1 (SENCORES=1, N_total=256) …  3.095 ms
# n=2 (SENCORES=2, N_total=512) …  3.069 ms
# n=4 (SENCORES=4, N_total=1024) …  3.040 ms
# n=8 (SENCORES=8, N_total=2048) …  3.194 ms
# n=16 (SENCORES=16, N_total=4096) …  3.603 ms
# n=32 (SENCORES=32, N_total=8192) …  3.944 ms

## Results table

| n cores | wall ms | Δ vs n=1 |
|---:|---:|---:|
| 1 | 3.095 | +0.000 ms |
| 2 | 3.069 | -0.027 ms |
| 4 | 3.040 | -0.056 ms |
| 8 | 3.194 | +0.099 ms |
| 16 | 3.603 | +0.508 ms |
| 32 | 3.944 | +0.849 ms |

## Model fits to (Δ wall) vs n

  Ring model   (Δ ≈ -0.090 + +0.0304 * n) RMSE = 0.068 ms
  Tree model   (Δ ≈ -0.200 + +0.1715 * log2(n)) RMSE = 0.165 ms

## Verdict

  Linear fit is materially better (RMSE 0.068 vs 0.165). Consistent with **ring/chain** broadcast with t_hop ≈ 30.4 μs per A-broadcast.
