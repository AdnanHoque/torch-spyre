# Reorder probe in LX-resident regime
# Shape: M=128, N=1024, K=2048, fp16
# warmup=5 iters=20

## Per-split LX-fit check + topology predictions

| split | per-core total | M-fast chain × shared | N-fast chain × shared |
|---|---:|---:|---:|
| (2, 16, 1) | 520 KB ✓ | 2 × 256 KB | 16 × 256 KB |
| (4, 8, 1) | 648 KB ✓ | 4 × 512 KB | 8 × 128 KB |
| (8, 4, 1) | 1096 KB ✓ | 8 × 1024 KB | 4 × 64 KB |
| (16, 2, 1) | 2088 KB ✗ | 16 × 2048 KB | 2 × 32 KB |

## Bench results

# (2, 16, 1)
  default:  2.998 ms
  reverse:  3.016 ms
  delta:    -0.017 ms  (speedup 0.994x)

# (4, 8, 1)
  default:  3.025 ms
  reverse:  3.029 ms
  delta:    -0.004 ms  (speedup 0.999x)

# (8, 4, 1)
  default:  3.057 ms
  reverse:  3.052 ms
  delta:    +0.005 ms  (speedup 1.002x)

# (16, 2, 1)
  default:  3.065 ms
  reverse:  3.069 ms
  delta:    -0.004 ms  (speedup 0.999x)


## Side-by-side

| split | per-core total | default ms | reverse ms | delta | speedup |
|---|---:|---:|---:|---:|---:|
| (2, 16, 1) | 520 KB | 2.998 | 3.016 | -0.017 ms | 0.994x |
| (4, 8, 1) | 648 KB | 3.025 | 3.029 | -0.004 ms | 0.999x |
| (8, 4, 1) | 1096 KB | 3.057 | 3.052 | +0.005 ms | 1.002x |
| (16, 2, 1) | 2088 KB | 3.065 | 3.069 | -0.004 ms | 0.999x |

## Verdict

  Max delta across splits: +0.005 ms
  Max speedup:             1.002x

  Reorder is flat (<2%) even in the LX-fit regime where ring-share is the dominant data-movement cost. The kernel templates' overlapped input fetch is hiding ring topology effects regardless of operand size. The lever is dead — close out the core-ordering project.
