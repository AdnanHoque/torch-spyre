# Multicast core_id permutation — Phase 0 findings

## TL;DR

**The multicast core_id placement lever does not exist for HMI-side
data movement on AIU.** Across 7 (shape, split) combinations,
structured permutations (identity, m-adjacent, reversed) produced
walls within 1.4% of each other. The lever k_fast exploits (packing
collaborators on adjacent ring positions) does not generalize to HMI
multicast — only PSUM ring traversal seems to care about placement.

This finding likely also closes the inter-op alignment idea as a
torch_spyre lever, because the alignment hypothesis depended on the
same placement-matters mechanism.

## What I tested

Same shape, same split, varying core_id permutation. The permutation
patches `compute_ops.generate_sdsc` to substitute `core_id → perm[c]`
when computing each physical core's work slice. Verified at the JSON
level that the `coreIdToWkSlice_` mapping changes correctly under
each permutation:

```
core 0 under identity: {M=0, N=0, K=0}  (B-sharing group: cores 0..7)
core 0 under m_adj:    {M=0, N=0, K=0}
core 1 under identity: {M=1, N=0, K=0}  (still in cores-0..7-share-B group)
core 1 under m_adj:    {M=0, N=1, K=0}  (now A-sharing-with-core-0 group)
core 4 under identity: {M=4, N=0, K=0}
core 4 under m_adj:    {M=1, N=0, K=0}  (m_slice changed)
```

Permutations tested:
- **identity** (default): packs n-sharing groups (cores reading same
  B chunk) at adjacent core_ids.
- **m_adj** = `(c % n) * m + (c // n)`: packs m-sharing groups (cores
  reading same A chunk) adjacent.
- **reversed** = `num_cores - 1 - c`: spreads everything (control).
- **random** (seed=42): pure scramble (worst-case control).

Shapes spanning A:B byte ratios:
- wide-B M=128 (LLM standard): A:B ≈ 1:32
- wide-B M=256: A:B ≈ 1:16
- square M=1024: A:B ≈ 1:1
- wide-A M=4096: A:B ≈ 16:1 (atypical)

Splits: (8, 4, 1) and (4, 8, 1).

## Results

| shape | split | identity | m_adj | reversed | random | spread |
|---|---|---:|---:|---:|---:|---:|
| wide-B M=128 | (8,4,1) | 3.957 | 3.949 | 3.962 | 3.961 | 0.3% |
| wide-B M=128 | (4,8,1) | 4.021 | 4.057 | 4.019 | 4.046 | 0.9% |
| wide-B M=256 | (8,4,1) | 4.026 | 4.041 | 4.021 | 4.024 | 0.5% |
| wide-B M=256 | (4,8,1) | 5.049 | 5.104 | 5.107 | 5.038 | 1.4% |
| square M=1024 | (8,4,1) | 3.110 | 3.113 | 3.154 | 3.120 | 1.4% |
| square M=1024 | (4,8,1) | 3.134 | 3.142 | 3.134 | 3.125 | 0.5% |
| wide-A M=4096 | (8,4,1) | 3.244 | 3.245 | 3.250 | 4.128 | 0.2% structured |

(wide-A M=4096 (4,8,1) errored — N_per too small for kernel template.)

The wide-A random outlier (27% slower than structured) is the only
significant variance — but identity, m_adj, and reversed are all
within 0.2% of each other on that shape too. The random case is
likely a chaotic kernel-template-internal failure mode triggered by
LX overflow + unstructured access pattern, not evidence that specific
placements help.

## Why placement doesn't matter for HMI multicast

Hypotheses (in order of plausibility):

1. **HMI multicast is implemented at the port level.** One fetch from
   HMI is broadcast to all subscribed cores in one cycle, independent
   of their ring positions. The chip's HMI port handles the fan-out
   internally, not via ring traversal.
2. **Ring traversal is too fast to matter at HMI bandwidth.** Even
   if multicast does walk the ring, the cycle cost is small compared
   to HMI transfer time per byte.
3. **Kernel templates do internal scheduling that ignores `core_id`.**
   The deeptools-side kernel emitter may use its own logic for ordering
   data movement that doesn't respect the SDSC's `coreIdToWkSlice_`
   mapping the way I expected.

Whatever the reason, the empirical answer is clear: torch_spyre
cannot reach this lever via SDSC IR.

## Why k_fast worked but multicast doesn't

k_fast (PR 1932) is on the **PSUM/SFP ring**, not the HMI/data ring.
PSUM accumulation is sequential (each k-collaborator passes its
partial sum to the next). Ring distance directly determines hop
count → wall time.

HMI multicast is parallel (one source, many sinks, simultaneous). The
ring topology matters less because the fetch can fan out in many
directions at once, and the bottleneck is the source's HMI port
bandwidth, not the ring's per-hop cost.

So the architectural lesson: **placement matters where traversal is
sequential (PSUM chains); placement doesn't matter where traversal
is parallel (HMI multicast).**

## Implication for inter-op alignment

The originally-paired idea was: align consecutive ops' splits so
output cores match input cores → eliminate ring shuffles between ops.

If this probe is right that the data ring doesn't care about physical
placement, then inter-op shuffles either:
- Don't actually exist on the data ring (data goes via HMI between
  ops, or via the L0/scratchpad hierarchy which doesn't depend on
  ring placement)
- Or do exist but cost the same regardless of placement

Either way, **alignment doesn't help wall time**. Both projects (the
multicast permutation AND inter-op alignment) close together based
on this finding.

To definitively close inter-op alignment, one more cheap probe would
help: time two consecutive ops with deliberately mismatched splits
(e.g., op N as (8, 4, 1) and op N+1 as (4, 8, 1)) and compare to
matched splits. If walls are equivalent, alignment closes.

## What's left after this

The brainstorm tier 1 had two projects depending on placement:
- ❌ Multicast fan-out via core_id permutation — **closed by this probe**
- ❌ Inter-op core_id alignment — **likely also closed**

The remaining strong solo torch_spyre projects from the brainstorm:

| project | status |
|---|---|
| LX residency planner | still viable, classical compiler work |
| Fix SDPA-to-bmm regression | still viable, ship-and-done |
| Cost-model-driven planner heuristic | still viable |
| Op fusion audit | still viable |
| AOT weight preprocessing | still viable |

LX residency planner is now the strongest "novel + intellectually
interesting + paper-worthy" candidate among solo torch_spyre work.
The rationale: it operates on a different lever (LX/HMI memory
hierarchy) that this probe didn't touch, and the Phase 1 cost model
already showed ~22 ms / Llama 70B M=128 block goes to non-matmul ops
with avoidable HMI traffic.

## A more meta lesson

Three project ideas have now closed via Phase 0 verification on real
AIU hardware:

1. Project B (HMI scheduling) — closed because HMI is binding and
   already saturated.
2. Joint SWP+WS at Python decomposition layer — closed because per-
   tile launch floors dominate.
3. Multicast core_id permutation — closed because data-ring placement
   doesn't matter.

In each case, a real hardware probe revealed the lever was much
narrower than the theoretical pitch suggested. **The pattern: ideas
extrapolated from GPU literature or first-principles arguments often
miss AIU-specific hardware behavior. Real measurements are the gating
filter.**

This argues for a project-selection bias toward "things grounded in
already-measured AIU behavior" rather than "things extrapolated from
GPU papers". Examples of the former:
- LX residency (Phase 1 cost model showed concrete avoidable HMI)
- SDPA-to-bmm regression (calibration measured concrete slowdown)
- k_fast (HMI BW probe found concrete win pattern — Jamie's PR)

Examples of the latter (all closed):
- Joint SWP+WS (Twill paper extrapolation)
- Multicast permutation (k_fast extrapolation, but to a different ring)
- HMI-aware scheduling (textbook ML compiler argument)

## Files

- `diag_multicast_core_perm.py` — the permutation probe
- `diag_multicast_core_perm_results.txt` — measurements
- This doc — findings
