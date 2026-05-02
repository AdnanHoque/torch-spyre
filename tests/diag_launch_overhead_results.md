# Per-launch overhead diagnostic — flash-attention Phase 0b

PyTorch:        2.10.0+cpu
torch_spyre:    (editable)
warmup iters:   5
measure iters:  30
per-iter sync:  torch_spyre.streams.synchronize() inside the timed loop

**Method**: at a fixed mm shape, issue N back-to-back compiled mm calls. Per-call wall time = total / N. The asymptotic value as N grows large is the per-launch overhead floor for this shape. The ratio T(N)/T(1) tells us whether per-call overhead has plateaued.

## tiny work — `(M=1, N=512, K=128)` = 0.13M FLOPs/call

| N | total ms | per call ms | per call vs N=1 |
|---:|---:|---:|---:|
| 1 | 2.87 | 2.872 | 1.00× |
| 2 | 5.75 | 2.873 | 1.00× |
| 4 | 11.51 | 2.876 | 1.00× |
| 8 | 23.08 | 2.886 | 1.00× |
| 16 | 46.30 | 2.894 | 1.01× |
| 32 | 93.07 | 2.908 | 1.01× |
| 64 | 186.86 | 2.920 | 1.02× |

## flash-attention small tile — `(M=64, N=128, K=128)` = 2.10M FLOPs/call

| N | total ms | per call ms | per call vs N=1 |
|---:|---:|---:|---:|
| 1 | 2.96 | 2.960 | 1.00× |
| 2 | 5.88 | 2.942 | 0.99× |
| 4 | 11.76 | 2.939 | 0.99× |
| 8 | 23.51 | 2.939 | 0.99× |
| 16 | 47.06 | 2.941 | 0.99× |
| 32 | 93.71 | 2.928 | 0.99× |
| 64 | 187.63 | 2.932 | 0.99× |

## flash-attention larger tile — `(M=2048, N=2048, K=128)` = 1073.74M FLOPs/call

| N | total ms | per call ms | per call vs N=1 |
|---:|---:|---:|---:|
| 1 | 3.06 | 3.064 | 1.00× |
| 2 | 6.11 | 3.054 | 1.00× |
| 4 | 12.19 | 3.047 | 0.99× |
| 8 | 24.39 | 3.049 | 1.00× |
| 16 | 48.80 | 3.050 | 1.00× |
| 32 | 97.58 | 3.049 | 1.00× |
| 64 | 193.38 | 3.022 | 0.99× |

