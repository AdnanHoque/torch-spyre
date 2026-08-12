# MoE Workstreams

## Purpose

This is a living assessment of where our compiler and core-to-core communication work can contribute to MoE. Additional recommendations from other investigations can be added and reconciled here.

Current context: issue 3565, RFC 34, the indirect-access work under issue 866, and the Gemma 4 effort under issue 3700.

## Recommended focus

Own the MoE expert-execution path, not the model glue:

```text
grouped token rows
    -> consumer-shaped expert-weight gather
    -> gate/up BMM
    -> activation
    -> down BMM
    -> inverse permutation
    -> fixed-K reduction
```

The strongest opportunity is to place data in the work division required by the next operation, then keep compatible intermediate activations in LX. This combines the indirect-access work needed by MoE with our ownership, relayout, and core-to-core communication experience.

The expert-weight transfer itself is primarily HBM-to-LX. Core-to-core communication becomes important when adjacent operations use different work divisions or when routed results are returned to token order.

## Why this angle matters

The current multicore gather direction in PR 2699 parallelizes over the index dimension. It explicitly leaves value-dimension parallelism for small-index workloads such as MoE as future work.

MoE frequently has one expert index selecting a large weight slab. A better mapping is:

```text
expert id -> selects the weight slab
core id   -> selects the shard needed by that core's BMM work division
```

Each core can pull its consumer-shaped shard directly instead of first materializing a large generic 3-D gather result. This may also avoid the large-offset failure represented by issue 3637.

This mapping does not automatically reduce HBM traffic. HBM-byte savings require grouping and reuse of an expert slab across multiple routed rows. Its immediate benefits are avoiding an unnecessary intermediate, distributing the large value dimensions, and matching the following BMM.

## Matmul assessment

MoE exposes two materially different matmul regimes. Treating both as one dense
BMM misses the main optimization opportunity.

### Sparse per-route BMM

The direct path repeats each token once per selected expert and gathers one
different expert weight for every routed row:

```text
activations [N, 1, H]
weights     [N, H, F]
             ----------
output      [N, 1, F]
```

This is a real unit-M batched-weight workload, not a benchmark artifact. It is
most relevant to decode, where only a few routes exist and different rows often
select different experts. In that regime, grouping may recover little or no
weight reuse. The direct BMM can be the right execution strategy, but it needs
a work-division and corelet policy designed for tiny M and distinct weights.

### Grouped expert GEMM

After sorting routed rows by expert, expert `e` owns `M_e` contiguous rows:

```text
activations [M_e, H]
weight      [H, F]
             --------
output      [M_e, F]
```

Within the segment, the expert weight is shared across all `M_e` rows. This
turns the expert computation back into the shared-weight matmul family, where
weight reuse and M/N tiling can be exploited. This is most promising in
prefill, larger batches, or skewed routing distributions that send many rows to
the same expert.

The central compiler primitive should therefore be a grouped expert GEMM, or an
equivalent tiled loop that selects one expert weight slab per tile. It should
consume grouped activations, segment boundaries or tile expert ids, and the
resident expert-weight stack without materializing a routed `[N, H, F]` weight
tensor.

### The missing planning problem

The dense matmul planner scores one rectangular `(B, M, N, K)` problem. MoE
instead presents a routing histogram:

```text
M_0, M_1, ..., M_(E-1)
```

That histogram determines the number of active experts, weight loads, useful
rows per tile, padding, available parallelism, and LX pressure. A MoE-aware
planner should compare the direct BMM and grouped-GEMM strategies, then choose
the expert-row tile and internal matmul split.

A first model should expose these terms directly:

```text
padded_rows(tile) = sum_e ceil(M_e / tile) * tile
weight_loads(tile) = sum_e ceil(M_e / tile)

cost(tile) =
    padded compute time
  + weight_loads * expert slab bytes / effective HBM bandwidth
  + PT underfill
  + core underuse
  + gather, scatter, and tile-loop overhead
```

The planner must also include the ungrouped per-route BMM as a candidate. This
is especially important for decode: with very few routes spread across many
experts, nearly every active segment may contain one row, so grouping adds
bookkeeping without reducing weight loads.

### Why tile size cannot be fixed

The following numbers are model estimates under uniform random routing, not
device measurements. They illustrate the padding risk of choosing a fixed
32-row expert tile:

| Workload | Useful routed rows | Expected rows computed with tile 32 | Padding ratio |
| --- | ---: | ---: | ---: |
| 128 experts, one token, top-4 | 4 | about 127 | about 31.6x |
| 128 experts, 64 tokens, top-4 | 256 | about 3,546 | about 13.9x |
| 128 experts, 64 tokens, top-8 | 512 | about 4,022 | about 7.9x |

Real routers may be skewed and produce more reuse than this model. The point is
not that tile 32 is always wrong; it is that tile selection must use the actual
or representative route distribution. Useful candidates are likely to include
small tiles for sparse decode and larger tiles when prefill provides enough
rows per active expert.

## Proposed workstreams

### 1. Partial-stick router reshape

Issue 3634 is the best contained first contribution. Router weights shaped like `[T, K]` must be flattened to routed rows, but the partial-stick dimension currently aborts layout handling.

Proposed outcome:

- Support the `[T, K]` to `[N, 1]` transition through a correct restickify path.
- Preserve exact row ordering for downstream grouping.
- Add focused correctness and emitted-transport evidence.

Why first: it is currently unowned, relatively self-contained, and blocks both MoE approaches described in RFC 34.

### 2. Consumer-shaped expert-weight gather

Extend the multicore indirect-access design so a small number of expert indices can address a large value tensor while the value dimensions are divided according to the consumer BMM.

Proposed outcome:

- One expert id selects a slab.
- Cores fetch disjoint shards of that slab.
- The destination ownership matches the gate/up or down BMM work division.
- No full generic 3-D gather result is required.
- Large address offsets are lowered safely.

This should be coordinated with PR 2699 rather than duplicating its index correctness and shared-base work.

### 3. Grouped expert GEMM and routing-aware planner

Define the expert matmul execution contract and select between direct and
grouped execution from the routing histogram.

Proposed outcome:

- A grouped expert GEMM or equivalent per-tile weight-selection primitive.
- No materialized `[N, H, F]` routed-weight tensor.
- A measured direct-BMM baseline for sparse decode.
- A planner that chooses direct versus grouped execution, expert-row tile, and
  internal matmul work division.
- Validation over balanced, skewed, and real routing histograms.

This is the strategic matmul workstream. It determines whether MoE merely runs
or turns expert-weight reuse into device-time savings.

### 4. LX-resident expert-tile pipeline

Keep the compatible region from the gate/up BMM through activation and down BMM in LX.

Proposed outcome:

- Align adjacent work divisions where possible.
- Use LX relayout only at real ownership boundaries.
- Allocate every source and destination with valid overlapping lifetimes.
- Prove the emitted LX payload and absence of unintended HBM or DMA transport.

This is the workstream most directly connected to our existing core-to-core relayout expertise.

### 5. Collision-free routed-result combine

Issue 3638 shows that repeated-index `index_add` or `scatter_add` is not yet a safe combine primitive.

A cleaner MoE-specific formulation is:

```text
expert-ordered results
    -> inverse-permute into unique [token, route] slots
    -> reshape to [T, K, H]
    -> reduce across the fixed, small K dimension
```

This does not eliminate the indirect transfer. It changes the transfer into a unique-destination permutation and leaves the reduction to an ordinary fixed-K operation, avoiding repeated-index collisions.

### 6. Measurement and attribution

Every optimization should report separate evidence for:

- semantic correctness;
- final work divisions and ownership;
- emitted HBM-to-LX and LX-to-LX payloads;
- allocation and spill behavior;
- HBM expert-weight bytes;
- padding ratio from expert grouping;
- kernel and end-to-end device time.

Planner telemetry or a correct output alone is not proof that the intended transport fired.

## Suggested execution order

1. Fix the partial-stick reshape in issue 3634.
2. Measure the direct per-route BMM for sparse decode shapes.
3. Use precomputed grouping tables to isolate one expert tile from router and grouping complexity.
4. Implement the consumer-shaped expert gather for that tile.
5. Establish the grouped expert GEMM contract and sweep row-tile candidates.
6. Add routing-aware selection between direct BMM and grouped GEMM.
7. Keep the grouped expert compute chain in LX and validate each ownership boundary.
8. Implement unique inverse permutation plus fixed-K reduction.
9. Add dynamic grouping only after the data-movement architecture is validated.
10. Measure the complete MoE layer and then the model.

## Suggested shipping shape

Organize this as one tactical contribution and one strategic track:

1. Router reshape unblocker: the partial-stick restickify required by both MoE approaches.
2. Expert matmul execution: direct-BMM characterization, grouped expert GEMM,
   routing-aware planning, consumer-shaped gather, LX-resident handoffs, and
   collision-free combine.

The strategic track should be shipped in reviewable pieces when the grouped
primitive, planner, indirect access, and LX ownership changes can be validated
independently.

## Areas already being handled elsewhere

Avoid duplicating active work unless coordination changes:

- TopK shape and K coverage: issue 3636 and PR 3070.
- TopK index dtype mismatch: issue 3635.
- Generic scatter layout enforcement: PR 3409.
- General indirect-access support: issue 866 and PR 2699.
- RISC-V grouping ABI: defer until the precomputed-grouping path proves the data-movement design.

## Open questions

- What exact gate/up and down BMM work divisions should define the destination shards?
- Can one gathered expert slab be reused across all rows in a tile without increasing LX pressure beyond the available budget?
- Where do work divisions change inside the expert compute chain, and which changes require a real relayout?
- Does unique inverse permutation have an existing multicore-safe primitive, or does it need a narrow addition?
- What tile size gives the best balance between expert reuse, padding, and LX capacity?
- At what route density or histogram shape does grouped execution overtake the direct BMM?
- Should runtime routing select among a small family of precompiled tile programs, or can one program express the required range efficiently?
- How should FP8 packing, scale traffic, and corelet legality change the grouped-matmul score?

## Additional recommendations

Add findings from other investigations here. For each recommendation, record:

- proposed workstream;
- expected impact;
- dependency or overlap with existing work;
- correctness and artifact proof required;
- measured result, model-only estimate, or untested hypothesis;
- decision: adopt, merge with an existing workstream, defer, or reject.
