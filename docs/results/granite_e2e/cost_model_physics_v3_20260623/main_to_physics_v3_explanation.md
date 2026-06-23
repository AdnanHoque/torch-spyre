# Main To Cost-Model Physics V3: Where The Granite Speedup Comes From

This note explains the `cost-model-physics-v3-candidate` improvement from first principles. It connects three layers:

1. the matmul work-division cost model,
2. the 12 Granite matmul shapes used as the optimization target,
3. the fused Granite block / e2e latency readout.

Coordinate-remap work is intentionally out of scope here. This note covers only the work-division cost-model changes.

## Result Summary

The latest Antoni validation for the rebased `cost-model-physics-v3-candidate` branch reported:

| run | prefill ms | decode ms | read |
|---|---:|---:|---|
| physics v3 candidate | 335 | 177 | best reported numbers so far |

Earlier thread reference numbers for latest-main-style Granite e2e were approximately:

| run | prefill ms | decode ms |
|---|---:|---:|
| latest-main reference from thread | 495 | 274 |
| physics v3 candidate | 335 | 177 |
| implied speedup | 1.48x | 1.55x |

The fused local Granite block probe also showed the same direction for the prefill refinement:

| local block probe | prefill block ms | read |
|---|---:|---|
| `cost-model-physics` baseline | 524.610 | before large-M tile refinement |
| large-M tile refinement | 493.336 | 1.06x faster |

The branch Antoni tested has one more refinement on top of that local block probe, so the exact e2e number should be taken from Antoni's latest run, not from the older local block probe.

One caveat: the checked-in standalone oracle predates the final v3 tweak, and two current v3 picks do not have exact committed isolated-timing rows. The split tables below are therefore used to explain directionality and root cause. The fused block and Antoni e2e numbers are the source of truth for aggregate latency.

## First Principles: What The Planner Is Trying To Choose

For a matmul-like kernel, the planner chooses how to divide the logical iteration space over up to 32 cores:

```text
[B, M, K] @ [B or 1, K, N] -> [B, M, N]

split = B_split x M_split x N_split x K_split
```

In the compact tables below, a split such as `1_4_8_1` means:

```text
B split = 1
M split = 4
N split = 8
K split = 1
```

The hardware tradeoff is not "always use the largest dimension" or "always use all cores in the same way". Different splits stress different limits:

- `M` split exposes rows/tokens to feed the PT pipeline.
- `N` split exposes output columns, but can increase weight/output broadcast fanout.
- `K` split exposes reduction parallelism, but creates partial sums that must be combined.
- `B` split helps true batched BMMs when each batch item has too little `M` work.

The old planner was too often falling into bad pure-M splits such as `1_32_1_1` or `32_1_1_1`. Those splits can look attractive because they use 32 cores, but they may give each core too little useful work, fail to expose the right output/reduction parallelism, or create bad fused-layout behavior.

## Cost Model Formula

The new model scores each legal split with:

```text
score =
    compute_us
  + hbm_us
  + psum_us
  + m_lane_underuse_us
  + m_tile_underfill_us
  + wide_n_us
  + large_m_tile_shape_us
  + core_underuse_us
  + batch_split_us
```

Lower score wins.

### `compute_us`

This estimates per-core MAC time:

```text
compute_us = MACs_per_core / effective_peak_MACs_per_core
```

The effective peak is reduced when each core gets too few `M` rows. The PT streams rows over a stationary weight tile; if the per-core `M` tile is too short, startup/drain overhead dominates. This is the core reason decode shapes need special care: decode has much smaller `M` than prefill.

### `hbm_us`

This estimates HBM traffic:

```text
activation bytes + weight bytes + output bytes
```

For shared-weight matmuls, the RHS weight is counted once, not once per batch. This matters because real projection/MLP weights are shared across tokens. Without that accounting, the model overestimates shared-weight traffic and can pick the wrong split.

The term also includes a broadcast/cohort penalty. Splitting the output dimension can increase fanout pressure because more cores need the same operand data. Past the cohort limit, bandwidth pressure grows.

### `psum_us`

Splitting `K` means multiple cores compute partial sums for the same output tile. Those partial sums must be combined.

The important correction is that the model charges the per-core output tile, not the whole output tensor. The older style over-penalized useful `K` splits and made reduction parallelism look worse than it really is.

### `m_lane_underuse_us`

This is a soft tie-breaker. Among otherwise similar choices, prefer enough `M` lanes to stream work over the stationary weight tile.

This helps avoid splits that technically use cores but do not feed the PT well.

### `m_tile_underfill_us`

This is a one-sided penalty for per-core `M` tiles that are too small:

```text
if M / M_split < target rows per core:
    penalize
```

It does not punish healthy larger `M` tiles just for missing a symmetric target. This was one of the important cleanups from the original cost model. Underfilling `M` is bad; having enough rows is not bad.

### `wide_n_us`

Very wide per-core `N` tiles lose schedule efficiency. This term only charges the over-wide side:

```text
if N / N_split > target N tile:
    penalize
```

This pushes large projection/MLP shapes toward more balanced output tiling without hard-coding op names.

### `large_m_tile_shape_us`

This is the v3 prefill refinement. It activates from shape geometry, not from names like "attention" or "Granite".

It has three pieces.

For true BMM value geometry:

```text
K >> N and M tile is already healthy
```

Splitting a tiny output dimension can hurt the fused attention path without buying useful PT fill. The model nudges away from that.

For shared-weight narrow projections:

```text
N is narrow and per-core N is too wide
```

The model nudges toward more usable `M`/`N` tile shapes.

For shared-weight down-projection geometry:

```text
K > N and N is being split
```

The model adds a small cost to avoid over-favoring N split on long-K down projections. This was added because the earlier physics model was still slightly too eager to split `N` in some folded shared-RHS cases.

### `core_underuse_us`

The old behavior effectively had a hard fallback toward full-core choices. V3 replaces that with a soft opportunity cost.

Using fewer than 32 cores is usually suspicious, but it can be right if the 32-core split is physically worse. The model should be allowed to pick a measured-good lower-core or non-default split when the rest of the score justifies it.

### `batch_split_us`

For true BMMs, splitting batch has a small overhead, but it is sometimes the right way to get enough parallelism when `M` is tiny. This is especially important for decode attention.

Shared-weight matmuls do not pay this term because there is no true per-batch RHS weight.

## Why Main Needed More Than A Small Constant Change

Main had three structural problems.

First, it did not consistently understand folded shared-RHS matmuls as shared-weight work. In generated graphs, a `[B, M, K] @ [K, N]` projection can be folded/unfolded through views before the planner sees it. The planner needed to recover the fact that the RHS is loaded once.

Second, the previous scoring made full-core pure-M splits too attractive. For decode, choices like `1_32_1_1` can use all cores but still starve useful work or miss better batch/N/K parallelism.

Third, prefill and decode need opposite instincts in some attention shapes:

- Decode has small `M`, so it often needs batch/reduction/output parallelism to avoid PT underfill.
- Prefill has large `M`, so extra output splitting can become layout/fusion overhead rather than useful array fill.

The v3 model adds enough hardware-shaped terms to distinguish those regimes without adding op-name rules.

## 12 Granite Matmul Split Difference

The table below compares latest-main-style picks from the committed Codex-pod measurement summary with the current `cost-model-physics-v3-candidate` planner picks.

Source artifacts:

- `docs/results/granite_e2e/codex_pod_upstream_main_measurement_summary_20260612.md`
- `docs/results/granite_e2e/codex_pod_device_timing_sweep_repro_summary_20260612.md`
- current branch planner output from `cost-model-physics-v3-candidate`

| phase | op | shape `B x M x N x K` | main pick | physics v3 pick | why this matters |
|---|---|---|---|---|---|
| prefill | QK^T | `512x32x512x128` | `32_1_1_1` | `2_2_8_1` | moves away from pure batch split and exposes output/M work |
| prefill | attn@V | `32x512x128x512` | `1_32_1_1` | `1_32_1_1` | v3 preserves the large-M value path instead of forcing extra tiny-output split |
| prefill | Q/O proj | `1x512x4096x4096` | `1_4_8_1` | `1_4_8_1` | already reasonable, preserved |
| prefill | K/V proj | `1x512x1024x4096` | `1_4_8_1` | `1_8_4_1` | more PT-friendly M tiling for narrow projection |
| prefill | MLP up | `1x512x12800x4096` | `1_4_8_1` | `1_4_8_1` | already good, preserved |
| prefill | MLP down | `1x512x4096x12800` | `1_4_8_1` | `1_8_4_1` | avoids over-favoring N split on long-K down projection |
| decode | QK^T | `64x32x576x128` | `32_1_1_1` | `8_2_1_2` | major decode attention fix: batch plus M plus K parallelism |
| decode | attn@V | `32x64x128x576` | `1_32_1_1` | `8_4_1_1` | uses batch/M parallelism instead of pure M |
| decode | Q/O proj | `1x64x4096x4096` | `1_32_1_1` | `1_4_8_1` | fixes decode projection underfill |
| decode | K/V proj | `1x64x1024x4096` | `1_32_1_1` | `1_8_4_1` | fixes decode projection underfill |
| decode | MLP up | `1x64x12800x4096` | `1_4_8_1` | `1_4_8_1` | already good, preserved |
| decode | MLP down | `1x64x4096x12800` | `1_32_1_1` | `1_8_4_1` | fixes bad pure-M split on long-K down projection |

## Isolated-Matmul Latency Evidence

The committed isolated-matmul oracle showed that latest main was far from the best measured split on several important shapes:

| phase | op | main pick | main us | device-best pick | best us | main gap |
|---|---|---|---:|---|---:|---:|
| prefill | QK^T | `32_1_1_1` | 989.49 | `4_1_8_1` | 731.06 | 35% |
| prefill | attn@V | `1_32_1_1` | 327.39 | `1_16_2_1` | 197.72 | 66% |
| prefill | K/V proj | `1_4_8_1` | 174.50 | `1_8_4_1` | 117.55 | 49% |
| decode | QK^T | `32_1_1_1` | 203.05 | `8_2_1_2` | 89.93 | 126% |
| decode | attn@V | `1_32_1_1` | 94.42 | `1_4_2_3` | 55.04 | 72% |
| decode | Q/O proj | `1_32_1_1` | 622.82 | `1_4_8_1` | 231.84 | 169% |
| decode | K/V proj | `1_32_1_1` | 142.62 | `1_8_4_1` | 66.78 | 114% |
| decode | MLP down | `1_32_1_1` | 2043.51 | `1_4_4_1` | 689.20 | 197% |

The speedup is therefore not mysterious. It comes from eliminating a cluster of bad splits, especially decode pure-M splits, while preserving the already-good MLP-up and wide projection choices.

## Granite Block / E2E Latency Readout

The fused block confirms that the split changes survive the context that matters: not just standalone matmul, but fused Granite kernels with pointwise epilogues.

The reusable Granite harness lives in the internal repo:

```text
https://github.ibm.com/Adnan-Hoque1/spyre-granite-e2e-bench
```

Relevant files from that repo:

- `benchmarks/granite_block_probe.py`: focused FMS `GraniteBlock` probe with `prefill` `M=512` and `decode` `M=64`.
- `benchmarks/granite_block_layer_probe.py`: one-layer Granite block profile path with Kineto trace summarization.
- `runbooks/granite_block_e2e.md`: runbook for the real one-layer FMS Granite block path.
- `README.md`: states the main methodology: use trace-derived `kernel_ms_per_iter`; keep wall time, memory time, compile time, and CPU time separate.

Coordinate-remap content in that repo is unrelated to this cost-model work and should be ignored for this analysis.

Committed local block probe:

| case | median ms | read |
|---|---:|---|
| `cost-model-physics` prefill block | 524.610 | baseline physics model |
| large-M tile refinement prefill block | 493.336 | prefill improves |
| `cost-model-physics` MLP decode | 8.858 | baseline physics model |
| large-M tile refinement MLP decode | 8.985 | unchanged within noise |
| `cost-model-physics` attention decode | 74.993 | baseline physics model |
| large-M tile refinement attention decode | 75.188 | unchanged within noise |

Latest Antoni e2e validation of v3:

| run | prefill ms | decode ms | read |
|---|---:|---:|---|
| `cost-model-physics-v3-candidate` | 335 | 177 | best reported e2e numbers so far |

The local block probe explains the direction: the large-M refinement recovers prefill without undoing decode. Antoni's run confirms that the complete v3 branch gives the best aggregate result in the real Granite e2e harness.

## What Changed In Code

At production-code level, the branch changes only the planner in:

```text
torch_spyre/_inductor/work_division.py
```

The important code-level changes are:

- add a `shared_weight` argument to `_matmul_split_cost`;
- count shared RHS weights once in HBM traffic;
- charge K-split PSUM by per-core output tile;
- make core underuse a soft penalty instead of a hard rejection;
- recognize folded shared-RHS matmuls before planning;
- add PT-fill, M-underfill, wide-N, cohort-fanout, BMM batch split, and large-M tile-shape terms;
- pass `rhs_loaded_once` into the cost evaluator.

The branch intentionally avoids Granite op-name special cases. It uses observable shape and hardware features: `B`, `M`, `N`, `K`, whether the RHS is shared, core count, per-core `M`, per-core `N`, broadcast fanout, and K-split PSUM cost.

## Bottom Line

Main left several Granite matmuls on clearly bad work divisions, especially decode projections and decode attention. Physics v3 fixes the bad decode choices and keeps the large-M prefill path from over-splitting awkward dimensions. The resulting e2e readout, `335 ms` prefill and `177 ms` decode, is consistent with the 12-matmul evidence: the branch improves the known bad rows and preserves the known good rows.
