# Lineage, attribution, and scope

## Algorithmic premise

The Step 2 premise came from Mudhakar's MoE execution framing:

```text
evaluate one 512-token prefill chunk against all 128 experts;
reuse the 512-token activation;
apply runtime top-k router weights after expert evaluation;
reduce selected and zero-weight expert outputs into one token output.
```

This branch does not claim invention of that algorithmic idea.

## Team implementation context

Antoni and Swagath's dense all-expert work supplied the operational and semantic
reference for Step 2. The most concrete retained reference is hf-adapters
PR293 at head `672b2fc8b5f017a08c6b43b928deb3ccd0560761`, particularly
`hf_adapters/hf_gemma4_moe.py` and the functions `_moe_expert_chunk` and
`_moe_ffn_chunked`.

That path evaluates experts in four runtime chunks of 32 and correctly applies
post-down route weighting and accumulation. Its real-shape emitted placement
was used as a semantic and layout diagnostic. It was not copied into this
branch and is not claimed here as our implementation.

## Material we built on

Plain identifiers and paths used during development:

```text
Torch-Spyre base
  65508a025f557663c5694e3596c49b814d87517a
  Add LX relayout support, PR3439

Mudhakar grouped RFC context
  torch-spyre/RFCs PR34
  retained revision b2b7dbf

Antoni and Swagath Step 2 reference
  hf-adapters PR293
  head 672b2fc8b5f017a08c6b43b928deb3ccd0560761

Historical activation/LX transport experiments
  branch ah/communication-cost-model
  head 7dfeac33
  commits d98de4f8 and db4d4f73

DeepTools baseline used during the investigation
  82b0fa10bf8d9129b520f6d3baac462813f8c785

Prior shared-LHS DDL diagnostic
  local_runs/dense_shared_lhs_ddl_overlay_c1_20260816_01
```

The historical `d98de4f8` and `db4d4f73` work demonstrated useful LX-fed
projection mechanisms on an older compiler interface. It was studied but did
not apply cleanly to the pinned stack and did not handle the 64 shared X
consumers in the PR293 chunk.

## Our implementation contribution

This branch contributes the compiler realization and evidence needed to turn
the Step 2 premise into an executable activation-stationary program:

1. Shared-LHS expert projection contracts with no physical expert-batched X.
2. One flat static expert loop rather than four graph-level chunk calls.
3. A compact invariant X load hoisted into the same allocation plan as the
   loop.
4. Loop-carried X lifetime protection without unsafe LX aliasing.
5. Correct expert-bank pointer advances after unit-dimension tiling.
6. Direct streamed expert-weight operands instead of HBM-to-HBM staging.
7. Runtime `[E,T,1]` post-down weighting in LX.
8. A fixed LX accumulator with one final HBM drain.
9. Exact C32 ownership alignment and physical core-map verification.
10. Fail-closed source, SDSC, bundle, correctness, and timing validation.
11. Four-AIU full-shape measurement against the grouped candidate.

## Claim boundary

A defensible statement is:

```text
We implemented the Torch-Spyre compiler path that realizes Mudhakar's Step 2
activation-stationary dense MoE schedule as one static expert-loop bundle,
proved its full-shape storage and address behavior, validated runtime weighted
semantics, and measured it on four AIUs.
```

Do not claim sole ownership of the Step 2 algorithm or of Antoni and Swagath's
chunked dense implementation. The distinct contribution is the full
activation-stationary compiler realization, structural proof, and measurement.
