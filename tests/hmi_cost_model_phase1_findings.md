# HMI cost model — Phase 1 findings

## Status

Phase 1 deliverable from the project scope doc is **complete**:

> a script that takes (model, M) and prints per-op + end-to-end
> predicted wall time

The script `hmi_cost_model_phase1_block.py` composes the per-op cost
model into a serial sum over one decoder block. It covers attention
projections (q, kv, o), MLP projections (gate, up, down), norms,
residual adds, and a coarse attention compute term. Output: per-op
wall + class, block totals, top contributors, class breakdown, and
the Phase 1 qualitative readout ("how much of the block is
HMI-bound?").

## Headline numbers

Tested on Llama 3.1 70B and DeepSeek V3 across M ∈ {32, 128, 512, 2048}.
The two regimes the model identifies:

### Decode and decode-batching (M ≤ 512)

| model | M | block ms | HMI-bound % | top contributor |
|---|---:|---:|---:|---|
| Llama 3.1 70B | 32 | 79 | 72% | gate/up/down (15 ms ea) |
| Llama 3.1 70B | 128 | 81 | 72% | gate/up/down (15 ms ea) |
| Llama 3.1 70B | 512 | 84 | 71% | gate/up/down (15 ms ea) |
| DeepSeek V3 | 128 | 63 | 64% | gate/up/down (10 ms ea) |

Block wall is roughly constant from M=32 to M=512 — HMI for the MLP
projections is the floor, set by weight bytes (which don't scale with
M). The 72% HMI-bound fraction means **the block is HMI-bound in the
production decode regime that the planner targets**.

### Long prefill (M ≥ 1024)

| model | M | block ms | compute-bound % | top contributor |
|---|---:|---:|---:|---|
| Llama 3.1 70B | 2048 | 150 | 71% | gate/up/down (30 ms ea) |
| DeepSeek V3 | 2048 | 130 | 70% | gate/up/down (varies) |

At M=2048 the projections cross over to compute-bound — the per-core
M slice is large enough to fully utilize the PT array, and per-op
work exceeds HMI fetch time. **Scheduling around HMI is irrelevant
here; the block is compute-bound and the runtime is already saturating
arithmetic.**

## What this answers for Project B

The original Project B question:

> Would scheduling op orderings differently predict a lower
> wall time, and by how much?

Phase 1's answer is that **scheduling-driven HMI hiding is potentially
useful only in the decode regime (M ≤ 512)**. At M=128 Llama 70B:

- Block wall: 81 ms
- HMI-bound ops total: 58 ms
- Non-HMI ops total: 23 ms

If the runtime could perfectly overlap the 23 ms of non-HMI ops with
the 58 ms of HMI work, the block would shrink to 58 ms — **a 28%
saving**. That's the upper bound on Phase 2's "headroom" question.

In production, the runtime serializes per-bundle (Phase 3 preload
established this), so today we get the full 81 ms. The 28% gap is
exactly what Project B asked whether we could close.

## What Phase 1 doesn't say yet

- **Whether 28% is achievable** under the dependency graph. Some
  ops can't overlap because they're sequentially dependent
  (gate → silu → down). A concurrent simulator (Phase 2) needs to
  account for the dep graph to compute realistic headroom.
- **What runtime support is required**. Phase 1 just shows the
  prediction; Phase 2 + Phase 3 of the original scope cover the
  scheduler design and the runtime-side conversation.
- **How much of the prediction we trust**. The cost model is within
  ~13% on planner-natural rows (per Phase 0), good for
  qualitative classification but not tight enough for fine-grained
  ordering decisions.

## Files

- `hmi_cost_model_phase1_block.py`: the block-level predictor script
- `hmi_cost_model_phase1_block_results.txt`: example outputs for
  4 M values × 2 models = 8 configurations
- This doc: findings

## Next

Phase 2 (per the scope doc): build the concurrent simulator. Take
the per-op walls + dep graph and compute the gap between serial sum
(today's runtime) and concurrent ideal. If the gap is < 5%, project
closes. If 5-15%, marginal. If > 15%, worth pursuing the planner
heuristic + runtime conversation.

Phase 1 says the gap is ~28% at decode M, suggesting Phase 2 will
find substantial headroom — but only if the dependency graph allows
it. That's the question Phase 2 actually answers.
