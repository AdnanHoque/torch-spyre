# Core-emission reorder revisited — K-split PSUM + output reorder

## TL;DR

The earlier "lever is dead" call was overstated. After re-probing the
two regimes the original sweep didn't cover (K-split mixed splits,
plus a focused output-reorder retest on big shapes), we found two
configs where `core_emission_reverse=True` produces a real,
consistent speedup:

| shape | split | mean speedup (over 2 trials) | mechanism |
|---|---|---|---|
| L3-70B q_proj prefill `(128, 8192, 8192)` | `(4, 1, 8)` | **1.036×** | PSUM chain shortened 28→7 ring hops |
| L3-70B MLP down prefill `(128, 8192, 28672)` | `(16, 2, 1)` | **1.021×** | A becomes ring-shareable (458 KB fits LX); default mode neither operand shares |

Both replicate identically across default-first vs reverse-first
trial orders, so the signal is not in-process warmup.

The wins are real but **much smaller than the topology argument
predicts** (28→7 hops is 4×; we got 3.6%). Section "Why the win is so
small" below explains.

## What was actually new vs the original sweep

The original sweep (`diag_core_emission_sweep_results.md`) and the
LX-fit retest (`diag_core_emission_lx_fit_results.md`) both held
`k=1` everywhere. With `k=1` there is no PSUM chain and no SFP-ring
traversal — so the previous probes only ever tested input-fetch
sharing on the data ring, which is overlap-hidden by the kernel
template's chunk prefetch.

This run added two new dimensions:

- **Part A — K-split (`(m, 1, k)`, `k > 1`)**: PSUM reduction across
  K-collaborating cores travels the dedicated SFP ring (32 B/cycle),
  separate from the data ring. PSUM is on the critical path *after*
  compute — overlap can't hide it. Reverse emission packs K-chain
  cores into contiguous ring positions instead of striding by `m·n`.
- **Part B — Output reorder on big-K shapes**: a focused retest of
  `(m, n, 1)` swaps for shapes large enough that even the small
  per-core operand can matter (L3-70B MLP down, K=28672).

## Results

Initial single-trial probe (warmup=3, iters=15) found 4 candidate
signals at ≥1.013×. Replication probe (warmup=5, iters=30, two trial
orders per config):

| shape | split | trial1 sp | trial2 sp | mean | consistent? |
|---|---|---:|---:|---:|---|
| L3-70B q_proj prefill | `(4, 1, 8)` | 1.035× | 1.037× | **1.036×** | ✓ |
| L3-70B MLP down prefill | `(16, 2, 1)` | 1.021× | 1.021× | **1.021×** | ✓ |
| L3-70B MLP down prefill | `(2, 16, 1)` | 1.018× | 1.004× | 1.011× | weak |
| L3-8B  MLP down prefill | `(8, 1, 4)` | 0.993× | 1.004× | 0.998× | noise (flips sign) |

Two real consistent reverse-wins. The other two are weaker than the
trial-to-trial noise floor.

## Mechanism for the two real wins

### `(4, 1, 8)` K-split PSUM-chain shortening

Default M-fast emission for `(m, n, k)` assigns
`core_id = m_slice + m·(n_slice + n·k_slice)`. For `(4, 1, 8)`:

```text
core_id = m_slice + 4 · k_slice
K=0 chain (m varies, fix m=0): cores {0, 4, 8, 12, 16, 20, 24, 28}
spans ring positions 0..28 → 28 hops on the SFP ring
```

Reverse K-fast emission gives
`core_id = k_slice + k·(n_slice + n·m_slice)`. For `(4, 1, 8)`:

```text
core_id = k_slice + 8 · m_slice
K=0 chain: cores {0, 1, 2, 3, 4, 5, 6, 7}
spans ring positions 0..7 → 7 hops on the SFP ring
```

Predicted ratio: 28/7 = 4×. Measured wall-time gain: 1.036× = 3.5%.
Direction of effect matches the prediction.

### `(16, 2, 1)` output reorder — A becomes ring-shareable

Default M-fast for `(16, 2, 1)`: cores 0..15 share an N-band of B,
cores 16..31 share the other. With `K=28672, N=8192, n=2` the shared
B-band is `K · (N/n) · 2 B = 235 MB` — far larger than the 2 MB LX
scratchpad, so the runtime can't actually ring-share it. Each core
has to HMI-stream its own copy.

Reverse, which becomes N-fast for `(16, 2, 1)`: cores 0,1 share an
M-band of A, cores 2,3 share another, etc. The shared A-band is
`(M/m) · K · 2 B = 8 · 28672 · 2 = 458 KB`. Fits in LX → ring share
fires.

So default mode has *no* effective sharing on this shape; reverse
mode lets A flow through the ring. The 2.1% win is the ring-share of
A appearing as a non-zero data-movement saving.

## Why the wins are so much smaller than predicted

A 4× ring-hop reduction giving only 3.5% wall-time reduction is the
*right magnitude* once you decompose the wall time:

```
wall_time(4, 1, 8) ≈ launch_floor + max(compute, transit) + psum
                   ≈ 3 ms          + ~1 ms                + ~0.1-0.2 ms
                                                            ↑
                                                            this is what reorder
                                                            shrinks ~4×
```

PSUM at 28 hops × small per-hop cost on a dedicated 32 B/cycle ring
isn't a dominant share of the 4.3 ms wall time. Shrinking it to 7
hops removes a few hundred microseconds, not several milliseconds.
The lever exists but is bounded by *what fraction of wall time is
PSUM-or-ring-share*.

For the production shapes that the planner picks today (almost all
pure-N `(1, 32, 1)`), this fraction is essentially zero — there's no
PSUM chain and no inter-core sharing competition with the launch
floor + HMI streaming. That's why the original sweep saw only noise
on planner-picked splits.

## On novel ring-aware orderings (Morton, Hilbert, bit-reversal, etc.)

The follow-up question — is there a *better* ordering than just
flipping dim-iteration direction — is bounded by the same arithmetic.

For a ring topology, the asymptotically optimal broadcast/reduction
ordering is **already what the simple emitter produces**: contiguous
sequential cores along the ring direction. There is no reordering
that beats `0 → 1 → 2 → ... → N-1` on a sequential ring; bit-
reversal, Morton, and Hilbert orderings all deliberately scatter
neighbors and would *hurt* both ring share and PSUM chain.

The remaining theoretical gain comes from:

1. **Bidirectional ring exploitation** — splitting traffic across CW
   and CCW rings to halve worst-case latency. torch_spyre cannot
   express this today (see `bidirectional_ring_findings.md` — the
   knob lives below the SDSC layer).
2. **Tree-shaped reduction** — log-N hops instead of N-1 for PSUM.
   The SFP ring is structurally a ring, not a tree; this would
   require hardware-level changes.
3. **Rendezvous at HMI-adjacent core** — placing chain endpoints
   near the HMI port. We don't have evidence that HMI distance
   varies meaningfully across core_id, and we don't have a knob to
   force it if it did.

So the simple "swap dim order" reorder we already built captures
roughly all the practical ring-aware reordering the SDSC layer can
express. Other simple reorderings would be strictly worse.

## What this means for shipping

Two paths:

### Option A — Ship a planner heuristic that picks reverse selectively

Add a planner pass that picks `core_emission_reverse=True` when:
- Split is `(m, 1, k)` with `k ≥ 8` and `m ≥ 4` (PSUM-chain regime), OR
- Split is `(m, n, 1)` with `m·n = 32`, `m ≥ 8`, AND smaller operand
  fits in LX while larger doesn't (sharing-flip regime).

Expected win: 2-4% on a handful of (shape, split) combinations that
the existing planner doesn't usually pick anyway. Very narrow;
probably not worth the heuristic-maintenance cost.

### Option B — Close the project as bounded

Acknowledge that the simple emit-order lever produces wall-time
movement bounded by the PSUM-or-share fraction of total time, which
is ≤4% on the configurations where it fires at all and 0% on the
production-default pure-N splits. Document the mechanism in the
matmul-architecture reference so future investigations don't redo
this work, and move on.

### Recommendation

**Option B**, plus surface the `(4, 1, 8)` and `(16, 2, 1)` reverse-
wins as **manual tuning recommendations** for users who pin those
splits (not many, but there are some). The `core_emission_reverse`
config knob already exists and is opt-in.

If we ever revisit, the bigger lever is bidirectional ring traffic
or a different bundle structure (e.g., the Phase 3 preload-investigation
finding) — not core-ID emission order.

## Files

- [`tests/diag_core_emission_psum_chain.py`](diag_core_emission_psum_chain.py)
  — initial probe
- [`tests/diag_core_emission_psum_replicate.py`](diag_core_emission_psum_replicate.py)
  — replication with two trial orders
- existing: [`tests/diag_core_emission_lx_fit_results.md`](diag_core_emission_lx_fit_results.md),
  [`tests/diag_core_emission_sweep_results.md`](diag_core_emission_sweep_results.md)
- shipped knob: [`torch_spyre/_inductor/config.py:core_emission_reverse`](../torch_spyre/_inductor/config.py)
- emitter:
  [`torch_spyre/_inductor/codegen/superdsc.py:_get_core_to_slice_mapping`](../torch_spyre/_inductor/codegen/superdsc.py)
