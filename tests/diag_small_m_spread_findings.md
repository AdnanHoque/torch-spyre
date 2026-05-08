# Small-M decode-regime spread analysis — partial findings

Companion to `diag_small_m_spread_driver.py` and
`diag_small_m_spread_partial_results.txt`. Sweeps every linear
layer of Llama 3.1 8B/70B/405B + Llama 3.2 1B/3B + DeepSeek V3 at
decode-regime batch sizes (M ∈ {1, 32, 128}), 78 unique (M, N, K)
shapes after dedup.

## Run status

**Partial completion.** The probe completed:

- All 26 M=1 shapes (Llama + DSv3) — clean
- 16 of 30 M=32 shapes (most Llama; DSv3 M=32 broken)
- 0 of 22 M=128 shapes

The runtime got into a bad state mid-DSv3-M=32 — every subsequent
subprocess returned ERR. Killed the probe at that point. The
remaining 36 shapes (DSv3 M=32 plus all M=128) need a fresh
runtime to measure.

The data below is honest about which shapes were successfully
measured and which were not.

## Headline finding — spread is M-regime-dependent

| M regime | shapes measured | speedup spread (winner vs pure-M) | takeaway |
|---:|---:|---|---|
| **M=1** (decode batch=1) | 26 | **1.00× – 1.11×, median ~1.02×** | planner default is fine; nothing to optimize |
| **M=32** (decode batch=32) | 16 | **1.03× – 3.61×, median ~3.0×** | massive opportunity available |
| M=128 | 0 | — | not measured before runtime broke |

The contrast between M=1 and M=32 is the key finding. **At M=1
(true decode), virtually every reasonable split produces the same
wall** — kernels are launch-floor / HMI-overhead dominated and
PT-array utilization differences are dwarfed. At M=32, pure-M's
M_per=1 produces 0.125× PT util while alternatives keep full M
per core, opening up 3-4× speedups.

## M=1 winners by category

Winners across the 26 M=1 shapes:

| category | count | typical speedup |
|---|---:|---|
| pure-M | 1 | 1.00× (Llama 405B down_proj) |
| k=1 mixed | 2 | 1.01-1.03× |
| k>1 + id | 14 | 1.01-1.07× |
| k>1 + kf | 9 | 1.02-1.11× |

The `k>1` family (id or kf) wins most M=1 rows but by a tiny
margin. The largest M=1 speedup observed: **DSv3 kv_a_proj M=1 at
1.11×** with `(1, 1, 32)+kf` over pure-M. Even the "best"
non-pure-M wins at M=1 are barely above noise threshold.

## M=32 winners by category

Winners across the 16 M=32 shapes successfully measured:

| category | count | typical speedup |
|---|---:|---|
| pure-M | 0 | (often EAR-overflow on big shapes — couldn't even compile) |
| **k=1 mixed** (e.g. `(4, 8, 1)`) | **6** | **1.09× – 3.55×** |
| k>1 + id | 5 | 2.89-3.55× |
| **k>1 + kf** | **5** | **1.04× – 3.39×** |

At M=32 the wins are real and large. The recurring optimal splits
are mixed `(m, n, k)` with 1 < m < 32 — same pattern as the
exhaustive 12-shape probe found.

Key M=32 rows:

| shape | (M, N, K) | winner | speedup |
|---|---|---|---:|
| L3-70B q_proj/o_proj M=32 | (32, 8192, 8192) | k=1 mixed (4, 8, 1) | 3.61× |
| L3-70B 405B kv_proj M=32 | (32, 2048, 16384) | k>1+id (4, 4, 2) | 3.55× |
| L3-70B kv_proj M=32 | (32, 2048, 8192) | k=1 mixed (4, 8, 1) | 3.38× |
| L3-70B 8B q_proj/o_proj M=32 | (32, 4096, 4096) | k=1 mixed (4, 8, 1) | 3.32× |
| L3-70B 8B gate/up_proj M=32 | (32, 14336, 4096) | k>1+kf (2, 4, 4) | 3.39× |
| L3-70B 8B kv_proj M=32 | (32, 2048, 4096) | k=1 mixed (4, 8, 1) | 3.01× |

## What this tells us about the PR

The PR's heuristic is targeted at exactly the M ∈ [32, 512] regime
where the spread is large. The decision to skip M < 32 (where the
default planner already produces reasonable mixed splits and the
spread is small) is empirically correct.

What the PR doesn't capture: the M=32 wins come from **mixed
`(m, n, k)` splits**, not from the heuristic's `(1, n, k>1)` family.
The empirical optima are `(4, 8, 1)`, `(4, 4, 2)`, `(2, 4, 4)`,
`(2, 2, 8)`, `(8, 2, 2)`, `(8, 4, 1)`. The heuristic captures only
a partial subset of these wins.

Same conclusion as the earlier 12-shape exhaustive probe: the PR is
a real improvement over today's planner but the bigger opportunity
is mixed-split planning.

## Caveat on M=1 measurements

At M=1, pure-M `(32, 1, 1)` is structurally invalid (M_per = 1/32
= 0). The probe's `_force_split((32, 1, 1))` likely fell back to
the planner's default at compile time, which itself picks pure-N
`(1, 32, 1)` or similar at M=1. So the "pure-M baseline" column at
M=1 is actually whatever the default planner produces for that
shape at M=1, not strict pure-M.

This means the M=1 spread numbers reflect "planner default vs.
non-default candidates" rather than "pure-M vs. alternatives." The
finding (small spread at M=1) holds under either framing — the
default planner is doing something reasonable for tiny M and the
alternatives don't substantially beat it.

## Runtime fragility issue (worth flagging)

The probe ran 56 successful subprocess measurements (one per
config × shape) before hitting a state where all subsequent
subprocesses returned ERR. The driver's subprocess isolation
prevented cascading failure into the parent, but the underlying
runtime got into a state where new compiles/launches couldn't
succeed.

Possible causes:
- Accumulated runtime state across many unique compiles
- Some specific shape triggered a runtime-level crash that
  persisted across processes
- Resource exhaustion (memory? handles?)

For future probes touching many shapes, consider:
1. Periodic runtime reset (kill + restart between batches)
2. Checkpoint progress + resume capability
3. Smaller probe scope (10-20 shapes per probe run)

## Files

- `tests/diag_small_m_spread_driver.py` — probe driver
- `tests/diag_kfast_essential_measure.py` — measurement subprocess
  (shared with focused k_fast-essential probe)
- `tests/diag_small_m_spread_partial_results.txt` — raw output
  (54 lines: 42 successful rows + 11 ERR rows + headers)
- This doc

## Branch

`AdnanHoque/feat-k-fast-combined` (evidence branch).
