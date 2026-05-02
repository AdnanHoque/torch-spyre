# DDR-traffic / tile-ordering — Phase 0 findings

This doc captures the result of the Phase 0 investigation into whether
matmul perf on Spyre is governed by per-core DDR-traffic (and therefore
whether the Inductor planner's choice of `(m_split, n_split, k_split)`
tuple is a perf lever via tile-ordering).

**TL;DR — no, not in the way the simple model predicts.** Spyre has
cross-core weight sharing that flattens the bandwidth curve. Tile-ordering
at the planner level is a 10-20% lever, not the 2-4× the naive bandwidth
model implied. Detailed below; the diagnostic at
`tests/diag_ddr_traffic.py` produced the data in
`tests/diag_ddr_traffic_results.md`.

## Hypothesis

For matmul `C[M,N] = A[M,K] @ B[K,N]` distributed across `num_cores` with
splits `(m, n, k)` such that `m·n·k = num_cores`, naive per-core DDR
traffic is:

- A read = `n × |A|` (each N-band reads its M-slice of A)
- B read = `m × |B|` (each M-band reads its N-slice of B)
- C write = `k × |C|` (k partial outputs per element)

Different `(m, n, k)` splits trade A-redundancy against B-redundancy. For
`(2048, 4096, 8192)` the traffic optimum is `(4, 8, 1)` at 554 MB; the
M-greedy default `(32, 1, 1)` carries 2198 MB — 4× more.

If matmul is bandwidth-bound on LPDDR5 (~200 GB/s peak), the perf gap
between traffic-optimal and traffic-worst splits should be ~4×.

## Method

`tests/diag_ddr_traffic.py` monkey-patches
`core_division.multi_dim_iteration_space_split` to force a specified
`(m, n, k)` tuple, then measures wall-time over 5 warmups + 20 timed
iterations with `torch_spyre.streams.synchronize()` per iter.

Three shapes, 7 splits each.

## Result 1: effective bandwidth far exceeds LPDDR5 peak

| Shape | Split | Theory traffic | Wall ms | Effective BW |
|---|---|---:|---:|---:|
| (2048, 4096, 8192) | (32, 1, 1) | 2198 MB | 5.97 | **367 GB/s** |
| (128, 8192, 8192) | (32, 1, 1) | 4299 MB | 6.41 | **671 GB/s** |
| (1024, 1024, 16384) | (32, 1, 1) | 1109 MB | 3.90 | **284 GB/s** |

LPDDR5 peak is ~200 GB/s. Observing 1.4-3.4× over peak proves Spyre is
doing **cross-core weight sharing** — when many cores want the same
DDR-resident weight bytes, they're not all going to DDR. The mechanism is
opaque from the Inductor side; plausible candidates are an on-chip ring
that forwards a single DDR fetch to multiple cores, a shared cache layer
between cores and DDR, or DDR-controller broadcast of repeated addresses.

## Result 2: traffic-optimal split is NOT wall-time-optimal

For `(2048, 4096, 8192)`:

| Split | Theory traffic | Wall ms | TFLOPs/s |
|---|---:|---:|---:|
| (4, 8, 1) — traffic optimum | 554 MB | **7.14** | 19.3 |
| (8, 4, 1) — wall-time optimum | 688 MB | **5.05** | 27.2 |
| (32, 1, 1) — current default | 2198 MB | 5.97 | 23.0 |

The wall-time-optimal `(8, 4, 1)` carries **24% more** theoretical
traffic than `(4, 8, 1)` but is **28% faster**. The gap between best and
worst `m·n=32` splits is ~50% on this shape, not the 4× the bandwidth
model predicted.

## Result 3: best m·n split varies by shape, but always within ~60%

For `(128, 8192, 8192)` — the M-skinny prefill case:

| Split | Wall ms | vs default |
|---|---:|---:|
| (32, 1, 1) default | 6.41 | 1.00× |
| (2, 16, 1) — best | **3.93** | **1.63×** |
| (1, 32, 1) | 3.98 | 1.61× |

For M=128, N-greedy splits are 60% faster than M-greedy because each core
gets only `M/32 = 4` rows under M-greedy — too few to amortize fixed
overhead. **But this regime is already addressed by the SplitK heuristic
we landed in `9978aa2`** (which routes (128, 8192, 8192) to a 32-way
K-split and gets +26% perf — though by a different mechanism, not by
optimizing m/n ratio).

For `(1024, 1024, 16384)`, all `m·n=32` splits cluster within ~10% of
each other. The planner's choice barely matters for balanced shapes.

## Implications for the project

The Phase 0 hypothesis was that bandwidth-aware tile-ordering at the
planner level is a 2-4× perf lever. **The data says it's a 10-60% lever
depending on shape, with the biggest gains in regimes already addressed
by other interventions** (SplitK heuristic for M-skinny prefill).

The infrastructure contribution — a forced-`(m, n, k)` planner harness +
a methodology for comparing theoretical traffic vs measured wall-time —
is durable independent of the project verdict.

## Why the bandwidth model was wrong

Two reasons:

1. **Cross-core weight sharing.** Naive model assumes each core
   independently reads its slice. Spyre has some sharing mechanism that
   absorbs redundancy when multiple cores want the same DDR addresses.
   Effective BW > 200 GB/s peak proves this empirically.
2. **Compute is not free.** Even if DDR-traffic differences were honored
   1:1, fixed compute-per-tile cost dominates for large matmul. We're at
   ~25 TFLOPs/s on `(2048, 4096, 8192)` — ~17% of fp16 peak — which
   suggests compute and bandwidth are roughly balanced, not bandwidth-
   dominated.

## What's still worth doing in the tile-ordering space

Smaller-scope follow-ups, in case anyone wants to pick this up later:

- **Document the cross-core sharing mechanism.** Coordinate with the
  Spyre HW/firmware team to characterize what's happening below the
  Inductor layer. Useful internal artifact for kernel optimization at any
  level.
- **Per-shape m/n split heuristic.** A planner pass that picks the best
  `(m, n, 1)` split (within `m·n=num_cores`) based on M, N, K. Lever is
  ~10-20% on bandwidth-leaning shapes. Smaller scope than originally
  pitched.
- **Cross-op tile-residency in scratchpad.** The existing
  `scratchpad_planning` pass (opt-in via `LX_PLANNING=1`) handles cross-op
  reuse for `max`, `sum`, `clone` outputs. Extending to matmul outputs
  (so a Linear's result can stay in scratchpad for a downstream activation)
  is a separate, cleanly-scoped project and probably has more lever than
  within-matmul tile ordering.

The decision to pivot away from tile-ordering as a flagship project is
based on this Phase 0. MoE grouped-GEMM moves up to the top of the
candidate list — it has bigger scope but addresses an actual missing-op
gap on Spyre rather than chasing a smaller-than-expected planner lever.

## Files

- `tests/diag_ddr_traffic.py` — Phase 0 diagnostic with forced-split
  planner monkey-patch.
- `tests/diag_ddr_traffic_results.md` — auto-regenerated bench output.
- `tests/ddr_traffic_phase0_findings.md` — this document.

## Reproducing

```sh
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
cd $DTI_PROJECT_ROOT/torch-spyre
python tests/diag_ddr_traffic.py
```
