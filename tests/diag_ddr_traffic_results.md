# Tile-ordering DDR-traffic diagnostic — Phase 0

PyTorch:        2.10.0+cpu
torch_spyre:    (editable)
SENCORES:       32 (default)
warmup iters:   5
measure iters:  20

**Theoretical DDR traffic** assumes each core independently reads its A-slice + B-slice + writes its (partial) C-slice, with no inter-core reuse. A_read = n·|A|, B_read = m·|B|, C_write = k·|C|.

**Effective BW** = traffic / kernel_time. Higher means we're closer to saturating LPDDR5 (~200 GB/s peak). Lower means kernel-launch / sync / compute overhead dominates and the matmul isn't actually moving bytes at peak rate.

## (M=2048, N=4096, K=8192)

| split (m,n,k) | A_read | B_read | C_write | total traf | median ms | TFLOPs/s | eff BW GB/s | vs default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (32, 1, 1) | 34MB | 2147MB | 17MB | **2198MB** | 5.97 | 23.00 | 367.8 | 1.00× |
| (16, 2, 1) | 67MB | 1074MB | 17MB | **1158MB** | 5.46 | 25.17 | 212.0 | 1.09× |
| (8, 4, 1) | 134MB | 537MB | 17MB | **688MB** | 5.05 | 27.19 | 136.1 | 1.18× |
| (4, 8, 1) | 268MB | 268MB | 17MB | **554MB** | 7.14 | 19.25 | 77.6 | 0.84× |
| (2, 16, 1) | 537MB | 134MB | 17MB | **688MB** | 5.29 | 25.97 | 130.0 | 1.13× |
| (1, 32, 1) | 1074MB | 67MB | 17MB | **1158MB** | 7.65 | 17.95 | 151.2 | 0.78× |
| (1, 1, 32) | 34MB | 67MB | 537MB | **638MB** | 10.66 | 12.89 | 59.8 | 0.56× |

## (M=128, N=8192, K=8192)

| split (m,n,k) | A_read | B_read | C_write | total traf | median ms | TFLOPs/s | eff BW GB/s | vs default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (32, 1, 1) | 2MB | 4295MB | 2MB | **4299MB** | 6.41 | 2.68 | 671.1 | 1.00× |
| (16, 2, 1) | 4MB | 2147MB | 2MB | **2154MB** | 4.50 | 3.82 | 478.3 | 1.42× |
| (8, 4, 1) | 8MB | 1074MB | 2MB | **1084MB** | 3.99 | 4.31 | 272.0 | 1.61× |
| (4, 8, 1) | 17MB | 537MB | 2MB | **556MB** | 4.00 | 4.29 | 138.8 | 1.60× |
| (2, 16, 1) | 34MB | 268MB | 2MB | **304MB** | 3.93 | 4.37 | 77.3 | 1.63× |
| (1, 32, 1) | 67MB | 134MB | 2MB | **203MB** | 3.98 | 4.31 | 51.1 | 1.61× |
| (1, 1, 32) | 2MB | 134MB | 67MB | **203MB** | 5.10 | 3.37 | 39.9 | 1.26× |

## (M=1024, N=1024, K=16384)

| split (m,n,k) | A_read | B_read | C_write | total traf | median ms | TFLOPs/s | eff BW GB/s | vs default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (32, 1, 1) | 34MB | 1074MB | 2MB | **1109MB** | 3.90 | 8.81 | 284.4 | 1.00× |
| (16, 2, 1) | 67MB | 537MB | 2MB | **606MB** | 3.67 | 9.37 | 165.2 | 1.06× |
| (8, 4, 1) | 134MB | 268MB | 2MB | **405MB** | 3.62 | 9.50 | 111.9 | 1.08× |
| (4, 8, 1) | 268MB | 134MB | 2MB | **405MB** | 3.61 | 9.51 | 112.0 | 1.08× |
| (2, 16, 1) | 537MB | 67MB | 2MB | **606MB** | 3.66 | 9.40 | 165.8 | 1.07× |
| (1, 16, 2) | 537MB | 34MB | 4MB | **575MB** | 10.90 | 3.15 | 52.7 | 0.36× |
| (1, 1, 32) | 34MB | 34MB | 67MB | **134MB** | 4.90 | 7.02 | 27.4 | 0.80× |

