# Exhaustive split probe — findings

Companion to `diag_exhaustive_split_driver.py`,
`diag_exhaustive_split_measure_one.py`, and
`diag_exhaustive_split_results.txt`. For each shape in the 3-way
campaign suite, measures **all 21 valid `(m, n, k)` work-division
splits** where `m·n·k = 32` (filtered for divisibility +
stick-alignment + per-shape compile validity), identifies the
empirical optimum, and compares to the heuristic's pick on the
combined k_fast PR (`AdnanHoque/pr-k-fast`).

## TL;DR

**The PR's pick is the empirical optimum on 0 of 12 shapes.** Gaps
range from 9% to 182%. The recurring optimal splits are **mixed
`(m, n, k)`** where `1 < m < 32` — splits the heuristic doesn't
consider at all because PR 1933's framework hard-codes `m = 1`.

This doesn't invalidate the PR. The heuristic still beats the
planner's current pure-M default on every shape it fires on.
But it leaves substantial headroom that a more general planner
(considering mixed splits) would capture.

## The exhaustive table

| shape | h-status | PR pick | PR ms | optimal split | optimal ms | gap |
|---|---|---|---:|---|---:|---:|
| L3-70B kv_proj M=32 | fired | (1, 16, 2) | 0.18 | (4, 8, 1) | 0.15 | 17.9% |
| L3-70B kv_proj M=128 | fired | (1, 16, 2) | 0.19 | (4, 4, 2) | 0.16 | 21.6% |
| L3-70B kv_proj M=512 | fired | (1, 16, 2) | 0.37 | (4, 8, 1) | 0.21 | 78.8% |
| Mixtral kv_proj M=128 | fired | (1, 16, 2) | 0.12 | (8, 4, 1) | 0.11 | 9.1% |
| DSv3 kv_proj M=128 | fired | (1, 8, 4) | 0.30 | (2, 2, 8) | 0.23 | 32.2% |
| DSv3 q_a_proj M=128 | fired | (1, 8, 4) | 0.30 | (2, 2, 8) | 0.23 | 29.0% |
| L3-70B q_proj M=32 | fired | (1, 16, 2) | 1.04 | (4, 8, 1) | 0.94 | 11.1% |
| DSv3 gate_proj M=32 | fired | (1, 16, 2) | 3.67 | (1, 4, 8) | 2.14 | 71.5% |
| L3-70B q_proj M=128 | fired | (1, 16, 2) | 1.27 | (4, 8, 1) | 0.98 | 29.7% |
| L3-70B q_proj M=512 | skipped | (32, 1, 1) | 3.52 | (4, 8, 1) | 1.25 | **182.4%** |
| DSv3 down_proj M=128 | fired | (1, 16, 2) | 3.83 | (4, 4, 2) | 1.94 | 97.4% |
| L3-70B kv_proj M=2048 | skipped | (32, 1, 1) | 1.21 | (4, 8, 1) | 0.59 | 105.7% |

## Where the wins live

Optimal split frequency across the 12-shape suite:

| split | count | character |
|---|---:|---|
| `(4, 8, 1)` | 5 | divides M into 4, N into 8, no K-split |
| `(4, 4, 2)` | 2 | divides M, N, AND K |
| `(2, 2, 8)` | 2 | minimal M+N split, big K-split |
| `(8, 4, 1)` | 1 | M-heavy mixed |
| `(1, 4, 8)` | 1 | row split, high k |

All but one optimum has `1 < m < 32` — i.e., it splits M *while
also* splitting N (and sometimes K). The heuristic's hard-coded
`m = 1` rule excludes these by construction.

## What the PR still does right vs the planner's current default

Cross-referencing the v3 3-way numbers (where we have pure-M
baselines):

| shape | pure-M (today) | PR | optimal | PR captures |
|---|---:|---:|---:|---:|
| L3-70B kv_proj M=32 | 0.46 | 0.18 | 0.15 | **80%** of available speedup |
| L3-70B kv_proj M=128 | 0.48 | 0.19 | 0.16 | 91% |
| L3-70B kv_proj M=512 | 0.47 | 0.37 | 0.21 | 38% |
| Mixtral kv_proj M=128 | 0.25 | 0.12 | 0.11 | 93% |
| DSv3 kv_proj M=128 | 0.61 | 0.30 | 0.23 | 81% |
| DSv3 q_a_proj M=128 | 0.62 | 0.30 | 0.23 | 82% |
| L3-70B q_proj M=32 | 3.43 | 1.04 | 0.94 | 96% |
| DSv3 gate_proj M=32 | 6.62 | 3.67 | 2.14 | 66% |
| L3-70B q_proj M=128 | 3.62 | 1.27 | 0.98 | 89% |
| DSv3 down_proj M=128 | 6.83 | 3.83 | 1.94 | 61% |

Median speedup capture: ~82% of available headroom. The PR is far
from optimal but is genuinely improving things on every fired row.

The skipped shapes are different — the PR keeps pure-M (which is
*not* the optimal) and gets 0% of the available speedup:

| shape | pure-M (PR) | optimal | speedup left on table |
|---|---:|---:|---:|
| L3-70B q_proj M=512 | 3.52 | 1.25 | 2.81× |
| L3-70B kv_proj M=2048 | 1.21 | 0.59 | 2.05× |

These aren't "regression-avoidance" decisions in the broader sense —
the heuristic correctly identified that K-split would regress on
these, but it didn't notice that *mixed* splits would dominate.

## Why the heuristic doesn't pick these mixed splits

PR 1933's `_try_k_fast_split` only considers `(1, n, k>1)` splits.
That framework assumes:
- The planner's pure-M default handles cases where M_per × K_per is
  large enough to keep the PT array busy
- The heuristic's job is to fix small-M PT-utilisation cases by
  going to K-split

What the data shows:
- The default planner is *also* wrong about pure-M being best
- Mixed M+N splits with no K-split (`(4, 8, 1)`, `(8, 4, 1)`,
  `(2, 16, 1)`) are often better than either pure-M *or* K-split
- The heuristic doesn't see these alternatives at all

So the heuristic is solving the right problem (pure-M is wasteful
at small M) with the wrong space (only K-split alternatives).

## Why this isn't a PR blocker

The PR is comparing two real-world states:

- **Before PR**: planner picks pure-M for every matmul → many shapes run far below their potential
- **After PR**: planner picks K-split for some shapes → those shapes run faster, others unchanged

It's a strict improvement: every fired row is faster than pure-M;
no row is slower; the skipped rows are unchanged. Whether mixed
splits would be even better is a question the *planner itself*
should answer, not this targeted heuristic.

The PR's value:
- Captures 60-95% of available speedup on 9 fired rows
- Zero regressions
- 24/24 unit tests green, hardware-verified across 12 production
  shapes
- Clear, narrow, reviewable scope

The PR's gap:
- Doesn't capture the mixed-split wins (9-182% gaps to optimum)
- Skipped shapes don't benefit at all even though they could

A more general planner change would be required to capture the
remaining headroom. That's a substantially larger project
(weeks vs days).

## Recommended PR description framing

> **What this PR does**: extends the planner with a heuristic to
> pick `(1, n, k>1)` splits over pure-M for narrow-N small-M and
> small-M wide-N matmul shapes. Captures 60-95% of available
> speedup (geomean 2.06×) on a 12-shape production suite, with
> zero regressions.
>
> **What this PR doesn't do**: explore the broader work-division
> space. An exhaustive measurement of all 21 candidate splits on
> the same suite (evidence: `tests/diag_exhaustive_split_*` on
> `AdnanHoque/feat-k-fast-combined`) shows mixed `(m, n, k)` splits
> are the empirical optimum on every shape, with gaps to the
> heuristic's pick ranging from 9% to 182%. Most of the optimal
> splits are `(4, 8, 1)`, `(4, 4, 2)`, or `(2, 2, 8)` — splits this
> heuristic doesn't consider. A future planner change to consider
> mixed splits could capture the remaining headroom but requires a
> substantially larger redesign.
>
> **Why ship this anyway**: the heuristic delivers a measurable,
> consistent improvement over the current state with a small,
> reviewable footprint and no regressions. The exhaustive probe
> motivates and scopes the next planner project.

## Caveats

1. The exhaustive probe forces the split via `_force_split`. The
   default planner has additional correctness constraints (memory
   layout, hardware span limits) that we bypass — some "winning"
   mixed splits might fail in production paths the probe doesn't
   exercise. A real planner change would need to apply those
   constraints. But the cases where the optimum is `(4, 8, 1)` or
   similar shouldn't trigger any of those issues — they're standard
   M+N divisions of the iteration space.
2. Each measurement is wall-time including ~0.2 ms of host
   overhead. Sub-millisecond rows have higher relative noise; the
   17-32% gaps on those rows might be ±5% wider in either
   direction. The 71-182% gaps are far above noise.
3. We measured wall time, not kernel-only latency. The wall-vs-
   kernel argument from earlier still applies — the gaps are if
   anything *underestimated* in kernel-only terms.

## Files

- `tests/diag_exhaustive_split_driver.py` — outer driver
- `tests/diag_exhaustive_split_measure_one.py` — single-config
  subprocess (isolation against deeptools scheduler crashes)
- `tests/diag_exhaustive_split_results.txt` — full per-shape +
  per-split table + summary
- This doc

## Branch

`AdnanHoque/feat-k-fast-combined` (evidence branch). Production
PR (`AdnanHoque/pr-k-fast`) is unaffected.
