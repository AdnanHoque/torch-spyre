# Large-M Tile-Shape Cost Model Refinement

This change builds on `cost-model-physics`. The goal is to keep the physics-based model simple while fixing the prefill behavior that the Granite block probe exposed: when `M` is large enough to feed the PT, some attention/projection splits create avoidable layout and fusion fallout without buying useful array fill.

## What Changed

The cost model now adds one large-M tile-shape term. It activates only when both conditions hold:

- logical `M` is large: `M >= _M_TILE_UNDERFILL_TARGET * _COHORT_LIMIT * 2`
- each M tile is already healthy: `m_t >= _M_TILE_UNDERFILL_TARGET`

With the current constants, that means `M >= 256` and at least `16` rows per M tile. Decode-style `M=64` shapes do not trigger this term.

The term has two pieces:

```python
large_m_tile_shape_us = true_bmm_value_split_us + shared_narrow_tile_us
```

### True BMM Value Geometry

For true BMMs, the term penalizes splitting a tiny output dimension when the reduction dimension is much larger:

```python
true_bmm_value_split_us =
    large_m_factor
    * max(0, log2(K / N))
    * log2(n)
    * _LARGE_M_TILE_SHAPE_PENALTY_US
```

This is meant for value-matmul geometry such as attention @ V. When `M` is already large, the array does not need extra `N` splitting to find parallelism. Splitting a very small output dimension can instead cause worse layout/restickify behavior in the fused attention path. The term is based on the observable shape ratio `K >> N`, not on an op name.

### Shared-Weight Narrow Projection Geometry

For shared-weight matmuls, the term penalizes overly wide per-core `N` tiles when the total output is narrow enough that more `N` lanes are available:

```python
shared_narrow_tile_us =
    large_m_factor
    * max(0, log2((_TARGET_N_TILE_ELEMS * _COHORT_LIMIT) / N))
    * max(0, log2(n_t / (_TARGET_N_TILE_ELEMS / 4)))
    * (_LARGE_M_TILE_SHAPE_PENALTY_US / 4)
```

This nudges large-M, narrow-output projections away from very wide per-core output tiles. For Granite prefill K/V-style projections, that moves the model toward the same tile family that works well for the block probe.

## Why This Is Still Physics-Based

The new term is not keyed on "Granite", "attention", "MLP", or any kernel name. It uses:

- `M` and per-core `m_t` to decide whether the PT already has enough rows
- `K / N` to identify value-matmul geometry where output is tiny relative to the reduction
- total `N` and per-core `n_t` to avoid overly wide output tiles for large-M narrow projections
- the existing hardware-shaped constants for PT fill, broadcast cohort size, and target N tile size

So the idea is structural: once M fills the array, stop chasing extra parallelism through awkward output tiling.

## Observed Effect

On the local Granite block probe:

| case | baseline `cost-model-physics` | with large-M tile term | read |
|---|---:|---:|---|
| full block prefill, `M=512` | `524.610 ms` | `493.336 ms` | about `1.06x` faster |
| MLP decode, `M=64` | `8.858 ms` | `8.985 ms` | no material change |
| attention decode, `M=64` | `74.993 ms` | `75.188 ms` | no material change |

The planner-level split changes are intentionally limited:

| shape family | physics pick | large-M tile pick | intent |
|---|---|---|---|
| QK prefill | `2_2_8_1` | `2_2_8_1` | unchanged |
| attention @ V prefill | `1_16_2_1` | `1_32_1_1` | avoid splitting tiny output when M already fills PT |
| K/V projection prefill | `1_8_4_1` | `1_4_8_1` | avoid overly wide per-core N tile |
| Q/O and MLP projections | unchanged | unchanged | preserve known good choices |
| decode families, `M=64` | unchanged | unchanged | preserve decode behavior |

## Caveats

The coefficient `_LARGE_M_TILE_SHAPE_PENALTY_US = 40.0` is still a calibrated cost-model constant. The structure is hardware/shape based, but the weight should continue to be validated against out-of-sample shapes before treating this as production-ready.

The block-level wall numbers are relative probe measurements, not a final e2e perf claim. They are useful because the same probe and runtime are used for both sides of the A/B.

