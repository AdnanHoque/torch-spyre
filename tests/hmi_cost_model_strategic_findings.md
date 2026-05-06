# Strategic compare — k_fast PR vs. Phase 2 headroom

## TL;DR

The PR 1933 k_fast heuristic, as currently written, captures **near-zero
of the Phase 1 28% headroom** in production decoder blocks. Even with
the heuristic's n_sticks gate fully relaxed to fire on every matmul
in the block, empirical measurements show savings of at most **2–9%**
of block wall, depending on model. The remaining ~20–30% is the
target for Phase 2 (concurrent simulator + scheduling heuristic).

**Phase 2 is still well-motivated.** The k_fast PR and Project B
attack overlapping but largely distinct portions of the headroom
budget.

## Setup

Phase 1 said decode-regime decoder blocks are 64–72% HMI-bound, and
the perfect-overlap upper bound on scheduling savings is **28% (Llama
3.1 70B)** to **36% (DeepSeek V3)** of block wall at M=128. This
script asks: how much of that does the k_fast heuristic — the
companion PR currently in flight — already capture?

Three configurations compared per block:

- **(A) baseline**: planner-natural pure-M `(32, 1, 1)` on every matmul
- **(B) k_fast**: same, except matmuls matching PR 1933's heuristic
  use the `(1, n, k>1)` split it picks
- **(C) overlap bound**: Phase 1 perfect-overlap (block wall = sum of
  HMI-bound op walls; non-HMI ops fully hidden behind HMI ops)

## Result 1: PR-as-shipped captures ~0%

For both Llama 3.1 70B and DeepSeek V3 at M=128 (decode-batching
regime — the production target):

| model | M | (A) baseline | (B) k_fast | (C) overlap | k_fast savings | k_fast share of (A−C) headroom |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.1 70B | 128 | 80.62 ms | 80.64 ms | 57.86 ms | 0.0 ms | 0% |
| DeepSeek V3 | 128 | 63.47 ms | 63.54 ms | 40.63 ms | 0.0 ms | 0% |

The heuristic fires on **1 of 6 matmuls** (kv_proj only) because the
`n_sticks < 32` gate excludes everything else. kv_proj at M=128 is
already launch-floor-bound (~3.5 ms, dominated by the 3 ms LF), so
K-split has no headroom to save into.

The heuristic fires on:
- Llama 70B: kv_proj only (N=1024 → n_sticks=16)
- DSv3: kv_proj + q_a_proj (N=1536 → n_sticks=24)

It skips:
- All o_proj and q_proj (N=hidden_size, n_sticks ≥ 64)
- All MLP gate/up/down (N=intermediate_size, n_sticks ≥ 224)

## Result 2: Even relaxing the gate doesn't change much

What if the heuristic were less conservative — e.g., dropped the
n_sticks ≥ 32 gate so it fires on every matmul? The cost model
predicts the block becomes **slightly slower** (-1.6% on DSv3 M=128),
because under the current cost-model parameters the (1, n, k) splits
have nearly identical predicted wall to (32, 1, 1).

**This is a known cost-model bug, not a real result.** Phase 0's
residual pattern 1b documented that the model under-predicts k-split
benefit by ~95% on shapes like DSv3 o_proj M=128 — the per-cluster
bytes accounting and LX overflow factors aren't yet in the model.
The cost-model output here can't be trusted for this question.

## Result 3: Empirical sweep gives the trustworthy answer

The popular-models sweep (`diag_k_fast_popular_models_results.txt`)
measured forced (1, n, k>1) walls on real shapes. Using those
numbers directly, the maximum block-wall saving achievable if the
heuristic fired on every win-band shape:

### Llama 3.1 70B M=128 (block wall = 80.62 ms)

| op | shape | baseline ms | k_fast ms | savings | heuristic fires? |
|---|---|---:|---:|---:|:-:|
| kv_proj | (128, 1024, 8192) | 3.36 | 3.07 | 0.29 ms | ✓ |
| o_proj | (128, 8192, 8192) | 6.48 | 4.95 | 1.53 ms | ✗ |

**Savings: 0.29 ms shipped (0.4% block) → 1.82 ms with relaxed
heuristic (2.3% block).** Out of 22.76 ms (28%) headroom, k_fast
captures at most **8% of available headroom** even if relaxed.

### DeepSeek V3 M=128 (block wall = 63.47 ms)

| op | shape | baseline ms | k_fast ms | savings | heuristic fires? |
|---|---|---:|---:|---:|:-:|
| q_a_proj | (128, 1536, 7168) | 3.53 | 3.21 | 0.31 ms | ✓ |
| o_proj | (128, 7168, 16384) | 9.16 | 4.71 | 4.45 ms | ✗ |
| down_proj | (128, 7168, 2048) | 3.73 | 3.17 | 0.56 ms | ✗ |

**Savings: 0.31 ms shipped (0.5% block) → 5.32 ms with relaxed
heuristic (8.4% block).** Out of 22.84 ms (36%) headroom, k_fast
captures at most **23% of available headroom** even if relaxed.

DSv3 is the *best case* for k_fast because o_proj has unusually wide
B (235 MB) → bandwidth-saturated under pure-M, big benefit from
splitting. For Llama-family models with smaller B, the benefit is
much smaller.

## Decision

**Phase 2 (concurrent simulator) is well-motivated.** Two reasons:

1. **k_fast's reach is bounded by per-op K-split mechanics**, which
   only help shapes with the right (M, N, K) profile. The remaining
   headroom — overlapping HMI-bound ops with non-HMI ops in the
   block — is a different lever entirely.
2. **Even on the most favorable shape (DSv3 o_proj), k_fast captures
   only ~25% of the Phase 1 headroom** when the heuristic is
   maximally aggressive. The other ~75% of the headroom budget
   requires cross-op scheduling.

Concrete next-step recommendation: proceed to Phase 2.

## Side-finding: PR 1933's heuristic is too narrow vs. its motivation

The k_fast popular-models sweep showed the **biggest wins on shapes
the heuristic doesn't fire on** (o_proj, down_proj on DSv3 / Llama
70B). The heuristic's `n_sticks < 32` gate excludes them.

Reading the gate's comment:
```python
if n_sticks >= 32:  # pure-N (1, max_cores, 1) already valid; planner's choice
    return None
```
The gate's reasoning is that the planner could already pick pure-N
when N is wide enough. But the natural-pick verification in
`k_fast_planner_validation_findings.md` showed **the planner picks
pure-M (32, 1, 1) regardless of N width**. So the gate's premise is
wrong — the planner doesn't actually pick pure-N for these shapes.

Relaxing the gate to fire on the o_proj / down_proj shapes the sweep
measured wins on would yield 2–8% additional block savings (per the
empirical numbers above). That's a separate, smaller question worth
raising on the PR — but it doesn't change the Phase 2 conclusion.

## Files

- `hmi_cost_model_strategic_compare.py` — the comparison script
- This doc — findings
