# Per-axis sharing analysis

**Hypothesis**: with row-major core ordering (n-axis contiguous, m-axis spaced), pure n-split (1, n, 1) should be cheaper than pure m-split (m, 1, 1) on shapes that aren't launch-floor-bound.

**Test**: for each shape, find max-n pure n-split and max-m pure m-split (k=1 throughout). Compare wall times.

| shape | (M,N,K) | best (1,n,1) | best (m,1,1) | n-split / m-split |
|---|---|---|---|---:|
| L3-8B q_proj prefill | (128,4096,4096) | (1,32,1) @ 3.24ms | (32,1,1) @ 3.89ms | 0.833× |
| L3-8B MLP gate/up prefill | (128,14336,4096) | (1,32,1) @ 3.77ms | (32,1,1) @ 6.05ms | 0.623×  <- big |
| L3-8B MLP down prefill | (128,4096,14336) | (1,32,1) @ 4.64ms | (32,1,1) @ 6.04ms | 0.768×  <- big |
| L3-70B q_proj prefill | (128,8192,8192) | (1,32,1) @ 4.06ms | (32,1,1) @ 6.56ms | 0.619×  <- big |
| Mixtral down per-expert | (128,4096,14336) | (1,32,1) @ 4.72ms | (32,1,1) @ 6.07ms | 0.778×  <- big |

**All shapes**: mean ratio = 0.724× (n-split / m-split). Median = 0.768×.
**Big shapes only** (max wall > 4.5ms): mean ratio = 0.697×. Median = 0.696×.

Interpretation: ratio < 1 means n-split is faster than m-split → supports row-major sharing hypothesis. Ratio > 1 means m-split is faster (hypothesis violated).


## Detailed (m, n, 1) sweep per shape

### L3-8B MLP gate/up prefill (128,14336,4096)  best k=1 wall = 3.77ms

| (m, n, 1) | wall ms | vs best |
|---|---:|---:|
| ( 1,32,1) | 3.77 | 1.00× ←best |
| ( 2,16,1) | 4.78 | 1.27× |
| ( 4, 8,1) | 3.91 | 1.04× |
| ( 8, 4,1) | 3.95 | 1.05× |
| (16, 2,1) | 4.34 | 1.15× |
| (32, 1,1) | 6.05 | 1.60× |

### L3-8B MLP down prefill (128,4096,14336)  best k=1 wall = 4.24ms

| (m, n, 1) | wall ms | vs best |
|---|---:|---:|
| ( 1,32,1) | 4.64 | 1.09× |
| ( 2,16,1) | 4.82 | 1.14× |
| ( 4, 8,1) | 4.48 | 1.06× |
| ( 8, 4,1) | 4.24 | 1.00× ←best |
| (16, 2,1) | 4.31 | 1.02× |
| (32, 1,1) | 6.04 | 1.42× |

### L3-70B q_proj prefill (128,8192,8192)  best k=1 wall = 4.02ms

| (m, n, 1) | wall ms | vs best |
|---|---:|---:|
| ( 1,32,1) | 4.06 | 1.01× |
| ( 2,16,1) | 4.02 | 1.00× ←best |
| ( 4, 8,1) | 4.14 | 1.03× |
| ( 8, 4,1) | 4.07 | 1.01× |
| (16, 2,1) | 4.60 | 1.14× |
| (32, 1,1) | 6.56 | 1.63× |

### L3-70B MLP down prefill (128,8192,28672)  best k=1 wall = 8.03ms

| (m, n, 1) | wall ms | vs best |
|---|---:|---:|
| ( 1,32,1) | 10.89 | 1.36× |
| ( 2,16,1) | 9.23 | 1.15× |
| ( 4, 8,1) | 10.37 | 1.29× |
| ( 8, 4,1) | 8.23 | 1.02× |
| (16, 2,1) | 8.03 | 1.00× ←best |

### Mixtral down per-expert (128,4096,14336)  best k=1 wall = 4.27ms

| (m, n, 1) | wall ms | vs best |
|---|---:|---:|
| ( 1,32,1) | 4.72 | 1.11× |
| ( 2,16,1) | 4.87 | 1.14× |
| ( 4, 8,1) | 4.54 | 1.06× |
| ( 8, 4,1) | 4.27 | 1.00× ←best |
| (16, 2,1) | 4.35 | 1.02× |
| (32, 1,1) | 6.07 | 1.42× |

