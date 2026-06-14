# Physics-Based Matmul Split Cost Model

Torch-Spyre must choose how to divide every matmul across the available Spyre cores. A split is described by four factors: batch, `M`, `N`, and `K`. The planner enumerates legal splits whose product fits within the core budget and scores each candidate. The goal of the physics-based cost model is to choose the split that is fastest on the hardware using structural properties of the matmul, not Granite-specific op names or shape special cases.

## What The Model Is Trying To Capture

A good matmul split balances four hardware effects:

- **Compute time:** how many MACs each active core must perform.
- **HBM time:** how many activation, weight, and output bytes must be streamed.
- **Partial-sum movement:** extra traffic introduced when the reduction dimension `K` is split across cores.
- **Array utilization:** whether each core gets enough `M` rows to keep the PT pipeline fed.

The high-level score is:

```text
score = compute_us + hbm_us + psum_us + soft_scheduling_terms
```

This is intentionally simple. It does not try to recognize QK, attention, MLP, or any other model-level role. It only looks at `B`, `M`, `N`, `K`, whether the RHS is shared weight or true batched weight, the candidate split, and hardware-derived quantities like bytes, cores, PT rows, and output tile width.

## Core Terms

### Compute Cost

For a candidate split, the model estimates per-core MACs:

```text
macs_per_core = B * M * N * K / active_cores
compute_us = macs_per_core / peak_macs_per_us_per_core
```

This favors using more cores, but only when the resulting tiles are still useful. More cores are not automatically better if the split creates tiny per-core tiles that underfill the PT pipeline.

### HBM Cost

The model estimates bytes read or written by the candidate split:

```text
activation bytes + weight bytes + output bytes
```

Shared-weight matmuls and true BMMs behave differently. For shared-weight projection-like matmuls, the RHS weight is reused across the batch. For true BMMs, each batch element has its own RHS slice, so batch splitting changes how weight traffic and parallelism behave.

The model also includes a fanout/cohort effect for HBM broadcasts. If too many cores contend for the same broadcast path, bandwidth efficiency falls. This is represented as a structural penalty based on the number of cores sharing a broadcast group, not as an op-specific rule.

For shared-weight matmuls the fanout term uses the larger of the `M` and `N`
splits, because a single RHS tile is reused across the streamed rows and output
columns. For true BMMs the fanout term uses the `N` split. Splitting `M` in a
true BMM mainly gives the array more independent rows to stream; it should not
be charged as if it were the same broadcast pressure as splitting output/weight
sticks.

### Partial-Sum Cost

Splitting `K` means multiple cores compute partial sums for the same output tile. Those partial sums must then be combined. The model charges an additional cost proportional to:

```text
(k_split - 1) * output_elements_per_core
```

This discourages unnecessary `K` splits, but does not ban them. For tiny-`M` attention-like BMMs, a `K` split can still be worthwhile if it helps expose enough parallel work.

### PT Underfill Cost

The PT array needs enough `M` rows per core to stay busy. If the `M` split is too aggressive, each core gets too few rows and the array underfills.

The earlier tuned model used a symmetric target-`M` penalty: it penalized being either below or above a target. That was harder to justify physically, because larger healthy `M` tiles should not be punished just for missing a target.

The physics-lite model makes this one-sided:

```text
penalty only when the candidate exposes too few M lanes
```

This captures the real issue: underfilling `M` is bad; having enough `M` is fine.

The current iteration also adds a small per-core `M` tile startup term:

```text
penalty when M / m_split < 16 rows per core
```

This catches tiny-decode cases where the compute estimate alone does not fully
account for PT startup/drain overhead. It is still structural: the term only
depends on the candidate's per-core `M` tile.

### Wide-N Tile Cost

Very wide per-core output tiles can be less efficient in the generated schedule. The model keeps a small penalty when each core receives an over-wide `N` tile. This acts as a tie-breaker between otherwise similar splits. It helps prefer more balanced `M`/`N` tiling for wide projections, without hard-coding projection shapes.

### Core Underuse Cost

The old planner often had hard fallback behavior around using the full core budget. The physics model treats unused cores as an opportunity cost instead of an absolute rule. If two splits have similar compute and memory costs, the one using more of the machine should usually win. But the model can still choose fewer cores if the extra cores would create poor tiles.

### True-BMM Batch Split Cost

True BMMs often have small `M`, especially in attention. In those cases, splitting only along `M` can starve the PT array. Batch and `K` parallelism can be the right way to expose enough work.

The earlier model used a large `K`-scaled batch-split penalty. That made batch splitting unattractive even when it was physically useful. The physics-lite model replaces it with a small additive overhead:

```text
batch_split_us = log2(batch_split) * small_overhead
```

That still acknowledges that splitting batch has some scheduling cost, but it no longer overwhelms the core compute and memory terms.

Small-output true BMMs get one additional structural adjustment. When `N` is
only a couple of sticks, `K` is long, and `M` is small, measured device timing
shows that `K` splitting can be a good way to expose useful parallelism even if
the split uses fewer than 32 cores. For that family the model reduces the
partial-sum coefficient and the soft core-underuse penalty. This is not keyed on
QK, attention, or Granite names; it follows from the shape: small output, long
reduction, true BMM.

## What Changed From The More Tuned Model

The simplified model keeps the useful structural pieces but removes or weakens the fit-heavy terms:

- Replaced symmetric target-`M` distance with one-sided `M`-lane underuse.
- Reduced partial-sum coefficients so `K` splitting is discouraged by real output movement, not a large magic penalty.
- Reduced the wide-`N` coefficient so it works as a tie-breaker.
- Replaced the large `K`-scaled true-BMM batch penalty with a small additive split cost.
- Charged true-BMM HBM fanout from `N` split instead of treating `M` split as
  equivalent broadcast pressure.
- Added a small one-sided startup penalty for very small per-core `M` tiles.
- Relaxed PSUM and core-underuse costs for small-output, long-`K` true BMMs
  where measured timing supports fewer-core `K`-split schedules.
- Kept all decisions based on observable matmul and hardware features, with no op-name conditionals.

This makes the model easier to explain: it is not trying to memorize twelve Granite shapes. It is trying to estimate the physical cost of each legal split.

## Validation On The +549 DeepTools Oracle

The model was validated against the measured +549 DeepTools oracle for the 12 Granite matmul anchor shapes.

| model | selected total | gap vs device-best |
|---|---:|---:|
| device-best oracle | 5148.19 us | 0.00% |
| generic tuned model | 5176.95 us | about 0.56% |
| first physics-lite model | 5189.43 us | about 0.80% |
| iterated physics model | 5166.56 us | about 0.36% |

The iterated physics model is now ahead of the more tuned model on the 12-shape
oracle while staying simpler and structural. It fixes the first model's two
decode-attention misses and restores the measured-best prefill `attn@V` pick.

The remaining known tradeoff is prefill QK. The iterated model picks
`2_2_8_1`, measured around 739 us, while the measured best is `4_1_8_1` at
about 731 us. That is roughly a 1% miss. Most projection and MLP shapes are at
or very close to the measured best split.

## Why This Is A Better Production Direction

For production, a cost model should generalize. A model that wins only because it has many calibrated constants for a small shape set is fragile. The physics-lite model deliberately gives up a tiny amount of anchor-set optimality to get a cleaner explanation:

- Use cores when they reduce real work per core.
- Keep enough `M` lanes to feed the PT array.
- Split `N` for wide outputs when it improves tiling.
- Split `K` only when the added partial-sum movement is worth it.
- Allow batch/K parallelism for true BMMs when tiny `M` would otherwise underfill the array.

That is the behavior we want from a hardware-backed planner.

## Current Caveats

This model should not be considered fully production-ready from the 12 Granite anchors alone. The next validation step is an out-of-sample oracle with shapes that vary `B`, `M`, `N`, and `K` across shared-weight matmuls and true BMMs. The acceptance bar should be that the model stays within a small margin of device-best on Granite while avoiding catastrophic misses outside the Granite anchor set.
