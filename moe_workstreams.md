# MoE Workstreams

## Purpose

This is a living assessment of where our compiler and core-to-core communication work can contribute to MoE. Additional recommendations from other investigations can be added and reconciled here.

Current context: issue 3565, RFC 34, the indirect-access work under issue 866, and the Gemma 4 effort under issue 3700.

## Recommended focus

Own the MoE data-movement path, not the model glue:

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

### 3. LX-resident expert-tile pipeline

Keep the compatible region from the gate/up BMM through activation and down BMM in LX.

Proposed outcome:

- Align adjacent work divisions where possible.
- Use LX relayout only at real ownership boundaries.
- Allocate every source and destination with valid overlapping lifetimes.
- Prove the emitted LX payload and absence of unintended HBM or DMA transport.

This is the workstream most directly connected to our existing core-to-core relayout expertise.

### 4. Collision-free routed-result combine

Issue 3638 shows that repeated-index `index_add` or `scatter_add` is not yet a safe combine primitive.

A cleaner MoE-specific formulation is:

```text
expert-ordered results
    -> inverse-permute into unique [token, route] slots
    -> reshape to [T, K, H]
    -> reduce across the fixed, small K dimension
```

This does not eliminate the indirect transfer. It changes the transfer into a unique-destination permutation and leaves the reduction to an ordinary fixed-K operation, avoiding repeated-index collisions.

### 5. Measurement and attribution

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
2. Use precomputed grouping tables to isolate one expert tile from router and grouping complexity.
3. Implement the consumer-shaped expert gather for that tile.
4. Keep the expert compute chain in LX and validate each ownership boundary.
5. Implement unique inverse permutation plus fixed-K reduction.
6. Add dynamic grouping only after the data-movement architecture is validated.
7. Measure the complete MoE layer and then the model.

## Suggested shipping shape

Keep this to two contributions if possible:

1. Router reshape unblocker: the partial-stick restickify required by both MoE approaches.
2. Expert-tile data movement: consumer-shaped gather, LX-resident compute handoffs, and collision-free combine.

The second contribution can be split only if its indirect-access and LX ownership changes cannot be reviewed independently without obscuring correctness.

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

## Additional recommendations

Add findings from other investigations here. For each recommendation, record:

- proposed workstream;
- expected impact;
- dependency or overlap with existing work;
- correctness and artifact proof required;
- measured result, model-only estimate, or untested hypothesis;
- decision: adopt, merge with an existing workstream, defer, or reject.
