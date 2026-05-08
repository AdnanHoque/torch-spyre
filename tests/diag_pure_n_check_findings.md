# Pure-N comparison probe — findings

Companion to `diag_pure_n_check.py` and
`diag_pure_n_check_results.txt`. Tests whether the heuristic's
chosen K-split is actually the fastest configuration on each shape,
or whether pure-N (1, 32, 1) — a configuration the heuristic doesn't
consider — would be even faster.

## TL;DR

**Pure-N beats the heuristic's K-split pick by 2-12% on every
small-M wide-N shape where pure-N is stick-aligned.** Our PR's wins
over pure-M are still substantial (1.79-3.28×), but they aren't
the empirical optimum; pure-N captures additional headroom on
three rows.

This is a **follow-up planner finding**, not a blocker for the
combined k_fast PR. The current planner picks pure-M for these
shapes (huge waste). Our PR overrides to K-split (huge improvement
over pure-M). A separate planner change to prefer pure-N for
small-M wide-N would be an additional ~7-12% on top.

## Data

| shape | (M, N, K) | pure-M | pure-N | K-split+kf (heuristic pick) | pure-N vs pure-M | pure-N vs K-split | best |
|---|---|---:|---:|---:|---:|---:|---|
| L3-70B q_proj M=32 | (32, 8192, 8192) | 3.43 | **0.97** | 1.04 | **3.54×** | 1.07× | pure-N |
| DSv3 gate_proj M=32 | (32, 18432, 7168) | 6.62 | **3.55** | 3.62 | 1.87× | 1.02× | pure-N |
| L3-70B q_proj M=128 | (128, 8192, 8192) | 3.62 | **1.13** | 1.27 | **3.20×** | 1.12× | pure-N |
| L3-70B q_proj M=512 | (512, 8192, 8192) | **3.51** | 4.39 | (heuristic skips) | 0.80× | — | pure-M ✓ |
| DSv3 down_proj M=128 | (128, 7168, 18432) | — | — | — | — | — | (N/32 not stick-aligned) |
| DSv3 down_proj M=512 | (512, 7168, 18432) | — | — | — | — | — | (N/32 not stick-aligned) |

Pure-N is invalid on the narrow-N kv_proj shapes (N=1024, 1536) and
on DSv3 down_proj shapes (N=7168 → N/32=224 → 224 % 64 ≠ 0). It's
valid on L3-70B q_proj and DSv3 gate_proj. Of the 4 valid cases,
pure-N is best on 3 (the small-M ones) and pure-M is best on 1
(the M=512 case the heuristic correctly skips).

## Why pure-N wins (cost-model rethink)

My earlier prediction was: K-split's per-cluster HMI bytes
((MK+KN)/k + MN) should beat pure-N's broadcast HMI bytes
(MK+KN+MN) by ~2× on the dominant traffic terms, so K-split should
win.

The data says otherwise. Looking at L3-70B q_proj M=32:
- pure-N HMI bytes = 134 MB, measured wall = 0.97 ms
- K-split HMI bytes = 67 MB, measured wall = 1.04 ms

The K-split has half the HMI bytes but a slightly *higher* wall.
Two effects I underweighted:

1. **B operand is multicast efficiently across all 32 cores under
   pure-N.** The HMI port reads B once and the data ring distributes
   it to all 32 consumers in parallel. The "broadcast" formula
   over-counts what HMI actually delivers — most of those bytes
   are multicast, not duplicated reads. K-split splits B across
   K-cohorts, which can't ride the same single-multicast pattern as
   efficiently.

2. **K-split adds PSUM chain coordination cost** that pure-N
   doesn't have. Even with k_fast emission (1-hop chain) and a
   small payload, the chain reduces parallelism in the output write
   path. Pure-N writes its output directly with no inter-core
   coordination.

The two effects together cost the K-split pick ~7-12% on these
shapes. Small relative to its win over pure-M, but still real.

## What this means for the combined k_fast PR

**The PR is still a clean win.** Today's planner picks pure-M for
all these shapes. Our PR overrides to K-split, which is 1.79-3.28×
faster than pure-M. That's the production lift we ship.

What we leave on the table: pure-N would be 1.02-1.12× faster than
K-split on three of those shapes. To capture that, a separate
planner change would need to:

1. Recognize that pure-N is stick-aligned for the shape
2. Prefer pure-N over pure-M when M is small (M_per < 8 under pure-M)
3. Resolve precedence with our K-split heuristic (which would also
   want to fire on these shapes)

**Recommended sequencing:** ship the combined k_fast PR as-is. The
extension's small-M wide-N firing is correct *vs the current
baseline*. Once it lands, a follow-up PR can introduce a pure-N
preference rule that takes precedence over K-split for the
small-M wide-N regime where pure-N is stick-aligned. The follow-up
would trim 7-12% off three production shapes.

The combined PR's evidence base remains intact — the speedups are
all over the planner's *current* output, not over a hypothetical
optimum. Reviewers asking "is this the fastest option?" should see
this finding documented but understand it's an orthogonal
optimization opportunity.

## What the heuristic's behaviour at L3-70B q_proj M=512 confirms

Pure-N at M=512 measures **0.80×** the speed of pure-M (4.39 vs
3.51 ms). The current heuristic correctly skips both K-split (which
also regresses) and pure-N (because the planner default is pure-M
and our heuristic only overrides to K-split, never to pure-N).

This is the right behaviour: at M=512, pure-M's PT util is full
(M_per=16 ≥ 8) and the dominant cost is HMI/compute, both of which
pure-M handles fine. Switching to pure-N would split N which gives
no PT-util benefit and loses HMI multicast efficiency.

So the M=128 → M=512 transition is real for pure-N too, not just
K-split: small-M wins from K-split *and* pure-N collapse around
M=256-512 where pure-M starts having decent PT util.

## Updated framing for the PR description

A defensible note to add to the PR description:

> **Note on optimal split selection:** for three of the four small-M
> wide-N shapes the extension targets, pure-N (1, 32, 1) is
> empirically 2-12% faster than the K-split pick this heuristic
> chooses (probe: tests/diag_pure_n_check.py on AdnanHoque/feat-k-
> fast-combined). The heuristic still delivers 1.79-3.28× over the
> planner's current pure-M default; pure-N preference would be an
> orthogonal follow-up planner change that takes precedence over
> this heuristic for stick-aligned wide-N shapes.

That's honest and doesn't undersell the PR's lift.

## Files

- `tests/diag_pure_n_check.py` — probe (tests pure-M, pure-N, K-split+kf)
- `tests/diag_pure_n_check_results.txt` — raw measurements
- This doc

## Branch

`AdnanHoque/feat-k-fast-combined` (evidence branch). Probe is NOT
on `AdnanHoque/pr-k-fast` — production PR stays clean.
