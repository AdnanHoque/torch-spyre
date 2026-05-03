# Spyre interconnect topology — broadcast-cost characterization

A Phase 0 measurement of how cross-core operand sharing scales with the
number of cores receiving the same operand, using only one external
behaviour: per-call wall time.

## Question

When N cores all need the same operand, hardware can deliver it via
several broadcast patterns:

| pattern | broadcast cost | minimum wires per core |
|---|---|---|
| Ring / chain | linear in N (`(N-1)·t_hop`) | 2 (one to each neighbor) |
| Tree | `log₂(N)·t_hop` | up to log₂(N) |
| Bus / crossbar | constant up to saturation | shared / many |

We didn't know which Spyre uses. The `output_element_priority`
analysis used hand-wavy "neighbor sharing" arguments without evidence;
this probe was meant to ground them.

## Probe design

Hold per-core work constant while varying `n` (the number of cores
broadcasting the same A operand). Force a `(m=1, n, k=1)` matmul split
with `SENCORES = n` so exactly `n` cores are active.

For all `n ∈ {1, 2, 4, 8, 16, 32}`:

- per-core compute: `M · N_per · K = 128 · 256 · 8192 = 268 MFLOPs`
- per-core unique B: `K · N_per · 2 = 4 MB`
- per-core unique C: `M · N_per · 2 = 64 KB`
- **shared A (broadcast across n cores)**: `M · K · 2 = 2 MB`

Per-core compute and per-core unique data are constant. Anything that
changes between runs is *only* the broadcast cost of fanning A out to
more cores.

Linear least-squares fit of `Δ wall(n) = wall(n) − wall(1)` against
both `n` (ring) and `log₂(n)` (tree) discriminates the topology.

## Results

| n cores | wall ms | Δ vs n=1 |
|---:|---:|---:|
| 1 | 3.095 | +0.000 |
| 2 | 3.069 | −0.027 |
| 4 | 3.040 | −0.056 |
| 8 | 3.194 | +0.099 |
| 16 | 3.603 | +0.508 |
| 32 | 3.944 | +0.849 |

| model | fit | RMSE |
|---|---|---:|
| Ring | `Δ ≈ −0.090 + 0.030·n ms` | **0.068 ms** |
| Tree | `Δ ≈ −0.200 + 0.172·log₂(n) ms` | 0.165 ms |

Ring fit beats tree fit by 2.4× on residual error.

## Interpretation

**Spyre's cross-core operand sharing behaves as a ring (or linear
chain).** Each additional core receiving the broadcast adds ~30 μs of
wall time when fanning out a 2 MB operand. That implies a per-link
effective bandwidth of `2 MB / 30 μs ≈ 67 GB/s`.

For `n ≤ 4` the wall time is essentially flat (~3 ms) because:

- 3 ms is the per-launch floor measured in Phase 0b
- The broadcast cost for ≤ 4 cores is < 100 μs, which is fully hidden
  by the floor or by overlap with per-core compute (~2.7 ms)

The linear growth becomes visible from `n=8` onwards. By `n=32` the
broadcast cost is `31 · 30 μs ≈ 0.93 ms`, comparable to half of
per-core compute time.

## Implications for the cost model and the planner

1. **The `output_element_priority` heuristic still wins for the right
   reason.** Pure-N broadcasts a small A (cheap on a ring, ~1 ms
   per-call). Pure-M would broadcast a large B (which on a ring at
   67 GB/s would take ~64× longer for the same `n`, i.e. seconds).
   Operand-size asymmetry remains the dominant first-order story; ring
   topology amplifies it because broadcast cost scales with both `n`
   and operand size.
2. **The 32-core saturation regime now has a concrete cost.** For any
   matmul where all 32 cores share an A or B operand, expect ~30 μs
   per MB of broadcast operand at 67 GB/s. A future cost-model term
   for sharing should look like `t_share ≈ (n - 1) · |op| / 67 GB/s`,
   not a single global `α` factor.
3. **Mixed splits like `(2, 16, 1)` get a topology benefit** that
   `(1, 32, 1)` doesn't. With `m=2, n=16` only 16 cores share each
   A-row instead of 32, halving the chain length. Per-axis sharing is
   real and asymmetric in core-count, even if both axes use the same
   physical ring.

## What we still don't know

- **Where on the ring core 0 lives, and whether it's the sole DDR
  ingress point.** The `30 μs/MB` per-hop figure assumes the operand
  enters at one end and propagates. If multiple cores can pull from
  DDR concurrently, the effective topology is more complex.
- **Whether broadcasts overlap with compute as we'd hope.** The flat
  region at `n ≤ 4` suggests yes (broadcast hides under launch floor /
  compute), but past 8 cores the broadcast appears on the critical
  path. We didn't decompose how much overlap happens.
- **Whether N-axis and M-axis broadcasts use the same physical ring.**
  Per-axis analysis suggested asymmetry but this probe only measured
  N-axis (A-broadcast). A symmetric M-axis probe (varying `m` with
  `(m, 1, 1)` splits) would confirm.

These are good follow-ups but not blockers — the ring finding alone is
enough to inform v2 of the cost model and any future planner heuristic
that needs to predict broadcast cost.
