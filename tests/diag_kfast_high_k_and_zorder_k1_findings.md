# k_fast wins at k>2 + Z-order at k=1 — findings

## Two questions, two answers

1. **Where does k_fast deliver wins beyond k=2?** → Mostly DSv3 q_a_proj
   shapes; less than the earlier "1.66-2.02×" measurements suggested
   for production-relevant cases; **regresses on small-N tiny-model
   kv_proj cases**.
2. **Does Z-order block-to-core help at k=1?** → No. ≤2% spread
   across 4 shapes confirms data ring is placement-invariant.

## Where the planner picks k > 2

Walking PR 1933's heuristic logic:
- `k = 2` when `n_sticks % 16 == 0` — most LLM kv_proj (N ∈ {1024, 2048, 4096})
- `k = 4` when `n_sticks ∈ {8, 24, ...}` — DSv3 q_a_proj (N=1536), L3.2 1B kv_proj (N=512)
- `k = 8` when `n_sticks ∈ {4, 12, 20, 28}` — TinyLlama-class kv_proj (N=256)
- `k = 16` when `n_sticks ∈ {2, 6, 10, ...}` — only at N=128 etc., very rare

**Attention compute matmuls** (QK^T, softmax·V) wouldn't trigger
because in production they're either fused or expressed as 4D bmm,
which the heuristic excludes via `len(it_space) != 3` gate.

## Benchmark — k>2 cases

5 shapes × 3 configs (pure-M, forced k-split without k_fast, with
k_fast). Cold cache + fresh process per measurement.

| label | shape | split | pure-M | kf-OFF | kf-ON | total win | k_fast contrib |
|---|---|---|---:|---:|---:|---:|---:|
| DSv3 q_a_proj M=128 | (128, 1536, 7168) | (1, 8, 4) | 3.53 | 3.23 | 3.16 | **1.12×** | 1.02× |
| L3.2 1B kv_proj M=128 | (128, 512, 2048) | (1, 8, 4) | 2.88 | 3.07 | 3.01 | **0.96×** ⚠️ | 1.02× |
| TinyLlama kv_proj M=128 | (128, 256, 2048) | (1, 4, 8) | 2.86 | 2.91 | 2.99 | **0.96×** ⚠️ | 0.98× |
| DSv3 q_a_proj M=256 | (256, 1536, 7168) | (1, 8, 4) | 3.48 | 3.61 | 3.17 | 1.10× | **1.14×** |
| TinyLlama kv_proj M=512 | (512, 256, 2048) | (1, 4, 8) | 3.00 | 3.01 | 2.93 | 1.02× | 1.03× |

### Three observations

**(a) Two production cases regress with the heuristic (4% slower).**
Llama 3.2 1B and TinyLlama at M=128 are LF-bound (~3 ms). K-split's
fixed overhead (PSUM ring traversal, even at minimum distance) eats
the K-split benefit.

**(b) k_fast contribution scales with PSUM payload, not k value alone.**
- Small-N small-M (LF-bound): 0.98-1.03× (noise)
- DSv3 q_a_proj M=256 (medium PSUM): **1.14×**
- Earlier (1, 8, 4) at M=2048, N=2048 from prior probe: 2.02×

The 1.66-2.02× wins from the prior Z-order probe were at *very*
large M and N. Production decode cases that pick k>2 have smaller
N (q_a_proj N=1536; tiny kv_proj N=256-512) and the PSUM payload
doesn't grow proportionally.

**(c) The heuristic is right when it picks (1, 8, 4) for q_a_proj
(real win) but wrong when it picks (1, 8, 4) for L3.2 1B kv_proj
(regression).** Both have n_sticks=8. The difference is K (q_a_proj
K=7168 vs L3.2 1B K=2048) and the resulting per-core PT util.

## Recommendation for PR 1933

Add a guard against the regression cases. Two options:

**Option A (simple)**: only fire `k>2` when `n_sticks ≥ 16`. This
restricts the heuristic to k=2 only. Loses the DSv3 q_a_proj win
(~10%) but eliminates the regression risk on small models.

**Option B (nuanced)**: only fire when `M_per_core × N_per_core ×
K_per_core` exceeds an LF-bound threshold (i.e., per-core compute
is large enough to amortize K-split overhead). Keeps q_a_proj win,
guards against tiny-model regression. More complex; needs cost-model
calibration.

Given the small magnitude of the regression (4%) and the win
(10-12%) in q_a_proj, the engineering cost-benefit favors Option A
unless DSv3 q_a_proj is a critical workload.

## Z-order block-to-core at k=1

4 shapes × 2 permutations (identity vs Z-order 2×2 block arrangement
of cells across cores). Cold cache per variant.

| shape (M, N, K) | identity | zorder | spread |
|---|---:|---:|---:|
| (128, 8192, 8192) | 3.957 | 3.986 | 0.7% |
| (128, 4096, 8192) | 3.450 | 3.518 | 2.0% |
| (256, 8192, 8192) | 3.976 | 3.998 | 0.6% |
| (512, 8192, 8192) | 4.324 | 4.361 | 0.9% |

All within measurement noise. **Z-order at k=1 delivers no benefit.**

This confirms (and refines) the earlier multicast permutation probe
(96 measurements, 0.6% median spread). The earlier probe tested
identity, m_adjacent, reversed, random — none of which is *strictly*
Z-order. Adding the Z-order arrangement specifically and re-running
cold-cache yields the same conclusion: **k=1 splits have no
SFP-ring traffic, and the data ring's parallel multicast doesn't
respond to block-to-core arrangement.**

Z-order is in fact slightly slower (1-2%) on every shape — it
disrupts whatever natural data-ring locality default emission has,
but the disruption is at noise level.

## Architectural restatement

The earlier paper draft's claim:
> "Placement matters when the ring is sequential AND has uniform
> per-hop cost. On rings that serve traffic in parallel, placement
> is invisible to wall-time."

is corroborated by this experiment for the k=1 case. For k>1 cases,
the prior Z-order block-to-core probe (at k=4) showed ring-distance
*does* matter, and we now know: **it matters more at high payload
(M=2048+) than at low payload (M=128 with small N)**.

The single-sentence rule:
- **Placement matters proportional to PSUM payload, scaled by
  ring traversal distance, modulo LF saturation at small walls.**

This is more nuanced than the paper draft's binary "matters / doesn't
matter" framing. Worth folding into the next revision.

## Files

- This doc — combined findings
- Prior data:
  - `diag_pr1932_top10_replication_results.txt` (k=2 cases)
  - `diag_zorder_block_to_core_findings.md` (k=4 large-payload cases)
  - `diag_multicast_core_perm_sweep_results.txt` (k=1 multicast cases)
