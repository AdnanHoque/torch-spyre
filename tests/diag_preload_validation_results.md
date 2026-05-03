# LX scratchpad preload validation

# Shapes: [('L3-8B q_proj prefill', 128, 4096, 4096), ('L3-70B q_proj prefill', 128, 8192, 8192), ('L3-8B GQA kv_proj prefill', 128, 1024, 4096)]
# Configs:
#   LX_PLANNING=0 (control): {'LX_PLANNING': '0'}
#   LX_PLANNING=1, frac=0.2 (default): {'LX_PLANNING': '1', 'DXP_LX_FRAC_AVAIL': '0.2'}
#   LX_PLANNING=1, frac=0.5: {'LX_PLANNING': '1', 'DXP_LX_FRAC_AVAIL': '0.5'}
#   LX_PLANNING=1, frac=0.8: {'LX_PLANNING': '1', 'DXP_LX_FRAC_AVAIL': '0.8'}
# warmup=3 iters=25


## L3-8B q_proj prefill (128, 4096, 4096)
  LX_PLANNING=0 (control) ...  median=3.821ms first=3.834ms min=3.799ms
  LX_PLANNING=1, frac=0.2 (default) ...  median=3.783ms first=3.806ms min=3.772ms
  LX_PLANNING=1, frac=0.5 ...  median=3.608ms first=3.630ms min=3.588ms
  LX_PLANNING=1, frac=0.8 ...  median=3.498ms first=3.511ms min=3.483ms

## L3-70B q_proj prefill (128, 8192, 8192)
  LX_PLANNING=0 (control) ...  median=6.515ms first=6.517ms min=6.465ms
  LX_PLANNING=1, frac=0.2 (default) ...  median=6.419ms first=6.430ms min=6.396ms
  LX_PLANNING=1, frac=0.5 ...  median=5.719ms first=5.699ms min=5.683ms
  LX_PLANNING=1, frac=0.8 ...  median=5.452ms first=5.469ms min=5.425ms

## L3-8B GQA kv_proj prefill (128, 1024, 4096)
  LX_PLANNING=0 (control) ...  median=3.138ms first=3.138ms min=3.123ms
  LX_PLANNING=1, frac=0.2 (default) ...  median=3.122ms first=3.143ms min=3.102ms
  LX_PLANNING=1, frac=0.5 ...  median=3.065ms first=3.069ms min=3.051ms
  LX_PLANNING=1, frac=0.8 ...  median=3.113ms first=3.164ms min=3.093ms


## Summary: median wall time per (shape, config)

| shape | LX_PLANNING=0 (control) | LX_PLANNING=1, frac=0.2 (default) | LX_PLANNING=1, frac=0.5 | LX_PLANNING=1, frac=0.8 |
|---|---:|---:|---:|---:|
| L3-8B q_proj prefill (128, 4096, 4096) | 3.821 | 3.783 | 3.608 | 3.498 |
| L3-70B q_proj prefill (128, 8192, 8192) | 6.515 | 6.419 | 5.719 | 5.452 |
| L3-8B GQA kv_proj prefill (128, 1024, 4096) | 3.138 | 3.122 | 3.065 | 3.113 |

## Speedup vs control (LX_PLANNING=0)

| shape | LX_PLANNING=1, frac=0.2 (default) | LX_PLANNING=1, frac=0.5 | LX_PLANNING=1, frac=0.8 |
|---|---:|---:|---:|
| L3-8B q_proj prefill | 1.010x | 1.059x ✓ | 1.092x ✓ |
| L3-70B q_proj prefill | 1.015x | 1.139x ✓ | 1.195x ✓ |
| L3-8B GQA kv_proj prefill | 1.005x | 1.024x | 1.008x |

## First-iter vs median (within-process cache warmup)

If the first iteration is much slower than the median, weights are being fetched from DRAM on iter 0 and cached for subsequent iters. A small first/median gap means either DRAM fetch is fast or there's no caching.

| shape | config | first ms | median ms | first/median |
|---|---|---:|---:|---:|
| L3-8B q_proj prefill | LX_PLANNING=0 (control) | 3.834 | 3.821 | 1.00x |
| L3-8B q_proj prefill | LX_PLANNING=1, frac=0.2 (default) | 3.806 | 3.783 | 1.01x |
| L3-8B q_proj prefill | LX_PLANNING=1, frac=0.5 | 3.630 | 3.608 | 1.01x |
| L3-8B q_proj prefill | LX_PLANNING=1, frac=0.8 | 3.511 | 3.498 | 1.00x |
| L3-70B q_proj prefill | LX_PLANNING=0 (control) | 6.517 | 6.515 | 1.00x |
| L3-70B q_proj prefill | LX_PLANNING=1, frac=0.2 (default) | 6.430 | 6.419 | 1.00x |
| L3-70B q_proj prefill | LX_PLANNING=1, frac=0.5 | 5.699 | 5.719 | 1.00x |
| L3-70B q_proj prefill | LX_PLANNING=1, frac=0.8 | 5.469 | 5.452 | 1.00x |
| L3-8B GQA kv_proj prefill | LX_PLANNING=0 (control) | 3.138 | 3.138 | 1.00x |
| L3-8B GQA kv_proj prefill | LX_PLANNING=1, frac=0.2 (default) | 3.143 | 3.122 | 1.01x |
| L3-8B GQA kv_proj prefill | LX_PLANNING=1, frac=0.5 | 3.069 | 3.065 | 1.00x |
| L3-8B GQA kv_proj prefill | LX_PLANNING=1, frac=0.8 | 3.164 | 3.113 | 1.02x |

## Verdict

  Max speedup 1.19x — preload is firing and helping on at least one shape. Worth pursuing as a tuning project.
