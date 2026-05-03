# Pure ring-share probe (LX-resident vs DRAM-bound)

## Phase: DRAM-bound (original sizing)
  M=128, K=8192, N_per=256
  shared A     = 2048 KB
  per-core B   = 4096 KB
  per-core C   = 64 KB
  per-core sum = 6208 KB (scratchpad limit: 2048 KB)
  per-core flops = 536,870,912

  n=1 (SENCORES=1, N_total=256) ...  3.189 ms
  n=2 (SENCORES=2, N_total=512) ...  3.089 ms
  n=4 (SENCORES=4, N_total=1024) ...  3.097 ms
  n=8 (SENCORES=8, N_total=2048) ...  3.258 ms
  n=16 (SENCORES=16, N_total=4096) ...  3.653 ms
  n=32 (SENCORES=32, N_total=8192) ...  3.997 ms

## Phase: LX-fit (small operands)
  M=128, K=2048, N_per=128
  shared A     = 512 KB
  per-core B   = 512 KB
  per-core C   = 32 KB
  per-core sum = 1056 KB (scratchpad limit: 2048 KB)
  per-core flops = 67,108,864

  n=1 (SENCORES=1, N_total=128) ...  2.850 ms
  n=2 (SENCORES=2, N_total=256) ...  2.847 ms
  n=4 (SENCORES=4, N_total=512) ...  2.898 ms
  n=8 (SENCORES=8, N_total=1024) ...  2.900 ms
  n=16 (SENCORES=16, N_total=2048) ...  2.970 ms
  n=32 (SENCORES=32, N_total=4096) ...  3.026 ms


## Side-by-side comparison

| n | DRAM-bound (original sizing) | LX-fit (small operands) |
|---:|---:|---:|
| 1 | 3.189 | 2.850 |
| 2 | 3.089 | 2.847 |
| 4 | 3.097 | 2.898 |
| 8 | 3.258 | 2.900 |
| 16 | 3.653 | 2.970 |
| 32 | 3.997 | 3.026 |

## Per-phase ring-fit (Δ wall vs n=1)

  DRAM-bound (original sizing):
    Δ ≈ -0.125 + +0.0301·n ms  (RMSE 0.073)
    Δ ≈ -0.229 + +0.1683·log2(n) ms  (RMSE 0.171)
    per-hop cost = 30.1 us  (operand size = 2.00 MB)
    per-hop per MB = 15.1 us/MB

  LX-fit (small operands):
    Δ ≈ +0.005 + +0.0057·n ms  (RMSE 0.016)
    Δ ≈ -0.024 + +0.0357·log2(n) ms  (RMSE 0.020)
    per-hop cost = 5.7 us  (operand size = 0.50 MB)
    per-hop per MB = 11.5 us/MB

## Verdict

  DRAM-bound max Δ at n=32:  0.808 ms
  LX-fit max Δ at n=32:      0.176 ms

  LX-fit broadcast cost is <50% of DRAM-bound. HMI contention contributed substantially to the original number — pure ring-share is faster than the 30 us/MB combined cost suggested.
