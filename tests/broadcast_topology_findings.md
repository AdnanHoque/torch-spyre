# Spyre interconnect topology — broadcast-cost characterization

A Phase 0 measurement of how cross-core operand sharing scales with the
number of cores receiving the same operand, using only one external
behaviour: per-call wall time.

## Architecture context

Spyre's on-chip interconnect:

- **Two counter-rotating data rings** (CW + CCW), each **128 B wide**.
  Total cross-core data-path width is `2 × 128 B = 256 B / cycle`.
- **A separate SFP ring at 32 B width** carries psum reduction, isolated
  from the data rings. Cross-core operand sharing uses the data rings;
  partial-sum reduction across psum-collaborating cores uses SFP.
- **HMI (DRAM interface) is a node on the data rings.** This is the
  single most important fact for interpreting these measurements:
  cores fetching operands from DRAM use the same ring infrastructure
  that carries cross-core sharing.
- **Cores are arranged 16×2** (two columns of 16, with the rings
  wrapping the perimeter through HMI/QGI at the top). Adjacent core IDs
  are physically adjacent on the ring, so our row-major
  `core_id → slice` mapping does walk the ring linearly.

Implication: **the 67 GB/s per-link figure measured below is a combined
ring-share + DRAM-streaming-on-the-same-ring cost.** Pure ring-share
(no DRAM contention) is likely faster — a follow-up probe with
LX-resident operands is needed to isolate it.

## Question

When N cores all need the same operand, hardware can deliver it via
several broadcast patterns:

| pattern | broadcast cost | minimum wires per core |
|---|---|---|
| Ring / chain | linear in N (`(N-1)·t_hop`) | 2 (one to each neighbor) |
| Tree | `log₂(N)·t_hop` | up to log₂(N) |
| Bus / crossbar | constant up to saturation | shared / many |

The probe was designed to discriminate ring vs tree vs bus from wall
time alone. The dual-ring topology described above means a ring fit
should map to the per-direction cost of one of the two rings — if both
directions are exploited concurrently, effective bandwidth could be 2×
what we measured.

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
chain), as expected from the dual-ring topology.** Each additional core
receiving the broadcast adds ~30 μs of wall time when fanning out a
2 MB operand. That implies a *combined* per-link effective bandwidth of
`2 MB / 30 μs ≈ 67 GB/s` — combined because the same ring is also
carrying per-core unique B from HMI during this probe (4 MB per core ×
32 cores = 128 MB through DRAM concurrent with the A broadcast).

For `n ≤ 4` the wall time is essentially flat (~3 ms) because:

- 3 ms is the per-launch floor measured in Phase 0b
- The broadcast cost for ≤ 4 cores is < 100 μs, which is fully hidden
  by the floor, by overlap with per-core compute (~2.7 ms), or by the
  built-in chunk-based overlapped input fetch in the WS dataflow

The linear growth becomes visible from `n=8` onwards. By `n=32` the
combined broadcast + DRAM-streaming cost is `31 · 30 μs ≈ 0.93 ms`,
comparable to half of per-core compute time.

## Implications for the cost model and the planner

1. **The `output_element_priority` heuristic wins by relieving HMI
   pressure.** Pure-N puts the 32× redundancy on the small input A
   (~MB scale), so total bytes through HMI on the ring stay small.
   Pure-M would put 32× on the large weight B (~hundreds of MB), which
   would saturate HMI for much longer. Since HMI sits on the same ring
   as cross-core sharing, both effects collapse into a single
   "bytes through the HMI-ring path" cost — and that's what the
   element-priority heuristic minimizes.
2. **A cost-model sharing term should distinguish ring-share from
   HMI-stream.** This probe couldn't separate them. Once we do
   (re-measurement #1: pure ring-share probe with LX-resident
   operands), the two costs should be modeled as competing for the
   same ring bandwidth budget.
3. **Mixed splits like `(2, 16, 1)` get a sharing benefit** that
   `(1, 32, 1)` doesn't only when the shared operand fits the
   ring-share envelope (small enough to actually broadcast across
   neighbors instead of streaming per-core from HMI). For large
   operands (B = hundreds of MB), the operand has to come from HMI
   per-core anyway, so reorder + ring-share doesn't change wall time —
   confirmed empirically by our flat reorder sweep results.

## Open follow-ups

- **Pure ring-share probe with LX-resident operands.** Pre-load A and
  B before benching. If the per-hop cost shrinks dramatically vs the
  67 GB/s combined number measured here, that confirms the
  HMI-on-ring contention model and gives us the *true* ring-only cost
  for the cost model.
- **Bidirectional ring exploitation.** The doc says CW + CCW are
  independent. Force two simultaneous broadcasts in opposite
  directions and compare to one at twice the volume. If we see ~2×
  bandwidth, dual-ring is exploitable in codegen.
- **SFP-ring-only psum probe.** Slide 30 of the doc says cross-core
  partial sum reduction uses the dedicated 32 B SFP ring, isolated
  from data ring traffic. That ring's per-hop cost has not been
  measured. Could matter for shapes where K-split is competitive with
  N-split.
- **N-axis vs M-axis symmetry.** This probe only measured N-axis
  broadcast (A across N-band cores). A symmetric M-axis probe would
  confirm whether the dual-ring topology treats both axes equally.
