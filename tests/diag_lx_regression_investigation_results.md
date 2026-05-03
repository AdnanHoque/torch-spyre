# LX-budget regression investigation

# Forced split: (1, 32, 1)
# Fixed: M=128, K=4096; varied: N in [2048, 4096, 8192, 14336]
# warmup=3 iters=20

## Per-core operand total at (1, 32, 1) split, fp16

| N | per-core A | per-core B | per-core C | total | fits 2 MB? |
|---|---:|---:|---:|---:|---|
| 2048 | 1024 KB | 512 KB | 16 KB | 1552 KB | ✓ |
| 4096 | 1024 KB | 1024 KB | 32 KB | 2080 KB | ✗ |
| 8192 | 1024 KB | 2048 KB | 64 KB | 3136 KB | ✗ |
| 14336 | 1024 KB | 3584 KB | 112 KB | 4720 KB | ✗ |


## N=2048 (per-core total = 1552 KB)
  [1/20 t=0s] control ...  3.064 ms
  [2/20 t=10s] frac=0.2 ...  3.001 ms
  [3/20 t=20s] frac=0.4 ...  3.008 ms
  [4/20 t=31s] frac=0.8 ...  3.038 ms
  [5/20 t=41s] frac=0.95 ...  3.070 ms

## N=4096 (per-core total = 2080 KB)
  [6/20 t=51s] control ...  3.164 ms
  [7/20 t=61s] frac=0.2 ...  3.148 ms
  [8/20 t=71s] frac=0.4 ...  3.231 ms
  [9/20 t=82s] frac=0.8 ...  3.160 ms
  [10/20 t=92s] frac=0.95 ...  3.245 ms

## N=8192 (per-core total = 3136 KB)
  [11/20 t=102s] control ...  3.498 ms
  [12/20 t=113s] frac=0.2 ...  3.409 ms
  [13/20 t=123s] frac=0.4 ...  3.404 ms
  [14/20 t=134s] frac=0.8 ...  3.415 ms
  [15/20 t=145s] frac=0.95 ...  3.504 ms

## N=14336 (per-core total = 4720 KB)
  [16/20 t=155s] control ...  3.670 ms
  [17/20 t=166s] frac=0.2 ...  3.689 ms
  [18/20 t=176s] frac=0.4 ...  4.343 ms
  [19/20 t=187s] frac=0.8 ...  4.499 ms
  [20/20 t=197s] frac=0.95 ...  4.561 ms


## Median wall time per (N, config) — ms

| N | per-core total | fits | control | frac=0.2 | frac=0.4 | frac=0.8 | frac=0.95 |
|---|---:|---|---:|---:|---:|---:|---:|
| 2048 | 1552 KB | ✓ | 3.064 | 3.001 | 3.008 | 3.038 | 3.070 |
| 4096 | 2080 KB | ✗ | 3.164 | 3.148 | 3.231 | 3.160 | 3.245 |
| 8192 | 3136 KB | ✗ | 3.498 | 3.409 | 3.404 | 3.415 | 3.504 |
| 14336 | 4720 KB | ✗ | 3.670 | 3.689 | 4.343 | 4.499 | 4.561 |

## Speedup vs control (LX_PLANNING=0)

| N | per-core total | fits | frac=0.2 | frac=0.4 | frac=0.8 | frac=0.95 |
|---|---:|---|---:|---:|---:|---:|
| 2048 | 1552 KB | ✓ | 1.021x | 1.019x | 1.009x | 0.998x |
| 4096 | 2080 KB | ✗ | 1.005x | 0.979x | 1.001x | 0.975x |
| 8192 | 3136 KB | ✗ | 1.026x | 1.028x | 1.024x | 0.998x |
| 14336 | 4720 KB | ✗ | 0.995x | 0.845x ✗ | 0.816x ✗ | 0.805x ✗ |

## Verdict

  Shapes that FIT 2MB scratchpad (1): average frac=0.8 speedup = 1.009×
    N=2048: 1.009×
  Shapes that DON'T fit (3): average frac=0.8 speedup = 0.947×
    N=4096: 1.001×
    N=8192: 1.024×
    N=14336: 0.816×

  Hypothesis CONFIRMED: high frac helps shapes that fit LX, hurts shapes that don't. The regression correlates with per-core operand size exceeding 2 MB.

# Total wall time: 208s
