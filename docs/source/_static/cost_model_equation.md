# Spyre matmul cost-model equation

Companion to [`docs/source/_static/cost_model_planner.html`](docs/source/_static/cost_model_planner.html). Source: [`torch_spyre/_inductor/work_division.py`](torch_spyre/_inductor/work_division.py), function `_matmul_split_cost`.

The planner enumerates every feasible `(b, m, n, k)` split (each factor divides its dimension; product ≤ 32 cores), scores each one with the formula below, and picks the lowest score.

## Equation

```
total_us = (compute_us + hbm_us + psum_us + target_m_us + lx_pressure_us) × batch_penalty
           + redistribution_us
```

## Per-term formulas

### `compute_us` — per-core MAC work, derated for PT-pipeline efficiency

```
per_core_MACs   = B × M × N × K / (b × m × n × k)
PT_passes       = max(1, (M / m) / 8)
pt_efficiency   = min(1.0, (PT_passes / 8) ** 0.5)            # sqrt below knee
effective_peak  = 1.536e6 × pt_efficiency                     # MACs/us/core
compute_us      = per_core_MACs / effective_peak
```

The PT (matrix) unit needs 8 rows per pass; pipeline fills over ~8 passes. Below 8 passes the efficiency derate is **sqrt-shaped, not linear** — calibration showed the linear `pt_passes/8` ramp was too pessimistic (predicted 50% derate at 4 passes; measured ~10–30%).

### `hbm_us` — input + output bytes over HBM bandwidth, with broadcast contention

```
bytes_total     = (B × M × K + B × K × N + B × M × N) × 2     # fp16
cohort          = max(m, n)
cohort_penalty  = max(1.0, cohort / 8)
hbm_us          = bytes_total × cohort_penalty / 204_800      # 204.8 GB/s
```

Linear penalty above a cohort knee of 8. **Known limit:** symmetric in `(m, n)` but real cost is asymmetric — m-heavy splits run ~2–3× faster than n-heavy at the same nominal cohort because B is LX-reusable while A is streamed. See "known limits" below.

### `psum_us` — K-split reduction across the ring

```
psum_us         = (k - 1) × (B × M × N) × 1.4e-4              # us/elem/hop
```

Each K-split adds one ring hop per output element. Coefficient fitted from a 7-shape sweep (Llama-7B QO/KV/Down, Granite MLP, Mistral MLP, Llama-70B QO, wideN) — implied coefficients cluster at 1.1–1.4e-4. For k=1 this term is 0.

### `target_m_us` — tie-breaker near PT-pipeline sweet spot

```
target_m        = clamp(4, max_cores/2, M / 64)
m_dist          = |log2(m / target_m)|
target_m_us     = m_dist × 50                                 # us/log2 step
```

Small penalty (50 µs/log2 step) that only matters when other terms tie. Magnitude fits big-M well (~48 µs/log2 measured), over-counts small-M by ~4× (real ~12 µs/log2); a scaled-by-`compute_us` variant was attempted but flipped QO so we deferred.

### `lx_pressure_us` — per-core RHS overflow penalty

```
per_core_RHS    = K × (N / n) × 2
lx_excess       = max(0, per_core_RHS − 2_097_152)            # 2 MB scratchpad
lx_pressure_us  = lx_excess × 5e-6                            # us/byte
```

Charges when per-core RHS slice exceeds the 2 MB on-core scratchpad. **The empirical mechanism is more complex than this formula** — a clean K-sweep up to 16 MB per-core RHS shows zero measurable penalty, yet MLP (4,8)-vs-(8,4) clearly has ~120 µs delta the term gets right. Useful kludge, not the underlying physics.

### `redistribution_us` — fusion-bundle penalty

```
redistribution_us = B × M × N × 2 × 1e-6                      # us/byte
                                                              # (only when split ≠ default
                                                              # AND matmul is in a fusion bundle)
```

When a matmul shares a fusion bundle with a non-matmul op (silu, add, etc.), a non-default split was previously assumed to incur an HBM round-trip to reshuffle output. Device measurement of fused `silu(linear)` bundles shows the actual cost is ≈ 0; the coefficient is kept at `1e-6` as a tie-breaker, not a hard gate. (Original `1e-4` over-penalized by ~100× and was blocking bundled matmul rewrites.)

### `batch_penalty` — multiplicative penalty for splitting batch across cores

```
batch_penalty   = b ** 1.4
```

For `b=1` (batch iterated) → 1.0. Power-law exponent fitted from a bmm[8,512,4096,512] sweep: measured `T(b)/T(1)` = 2.56× (b=2), 7.57× (b=4), 19.0× (b=8) → exponent fits at 1.36, 1.46, 1.42. The prior `1 + 0.6·(b-1)` linear form under-predicted by 3–4× at b=8.

## Hardware constants

| Constant | Value | Source |
|---|---|---|
| Peak compute | **98.304 TFLOPS** (DL16/fp16, MPE, dense) | `32 cores × 2 corelets × 8 rows × 8 cols × 8 SIMD × 1.5 GHz × 2 FLOPs/MAC`. **NOT** the public "300+ TOPS" figure — that's INT8 peak. |
| Per-core peak | 1.536e6 MACs/µs/core | derived: `98.304e12 / 2 / 32 / 1e6` |
| HBM bandwidth | 204.8 GB/s | LPDDR5 aggregate peak (Spyre paper, Chip Parameters table) |
| LX scratchpad | 2 MB per core | published in arch docs |
| PT array | 8 rows × 8 cols per corelet (2 corelets per core) | published |
| dtype | fp16 (2 bytes) | torch-spyre default |
| max cores | 32 | published |

## Known limits

After empirical validation, two structural limits remain:

1. **`hbm_us` is symmetric in (m, n)** — real cost is asymmetric. Physical mechanism: A (LHS) is streaming, B (RHS) is reused across the K reduction and stays in LX. Broadcasting B is cheap, broadcasting A is expensive. The model uses `cohort = max(m, n)` and doesn't distinguish.

2. **`lx_pressure_us` models only per-core RHS overflow** — per-core LHS overflow is unmodeled. For very wide N (e.g. N=20480), shrinking n grows per-core LHS, and at that scale the LHS overflow dominates. The planner picks `(m=4, n=8)` for N=20480 when `(m=8, n=4)` is actually ~40% faster.

A 7-parameter rework (per-tensor cohort + two-sided LX pressure) was attempted. The fit was degenerate against the available data, and a linear-in-excess RHS term cannot reproduce the empirically observed N=12800 → N=20480 winner-flip. Future revision will need a saturating RHS formulation + additional M-varying measurements.

## Validation status

The model picks the empirically-best split on every validated shape:

| Shape | Planner pick | Empirical winner |
|---|---|---|
| Llama-7B QO bs=1 (M=512, K=4096, N=4096) | `(mb=8, out=4, in=1)` | ✓ |
| Llama-7B KV bs=1 (M=512, K=4096, N=1024) | `(mb=8, out=4, in=1)` | ✓ |
| Granite MLP bs=1 (M=512, K=4096, N=12800) | `(mb=4, out=8, in=1)` | ✓ (15% faster than the prior heuristic) |
| MoE gate/up (B=8, M=128, K=2048, N=8192) | `(b=1, mb=4, out=8, in=1)` | ✓ |
| bmm large-K (B=8, M=512, K=4096, N=512) | `(b=1, mb=8, out=4, in=1)` | ✓ |
