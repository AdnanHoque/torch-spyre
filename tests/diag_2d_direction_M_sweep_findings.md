# 2D direction + production-baseline M sweep — findings

## TL;DR (and a correction to my earlier alarm)

Re-ran the 2D direction probe across M ∈ {128, 256, 512, 1024, 2048}
with cold cache per variant, plus the production-relevant comparison
(pure-M vs forced (1,16,2)+k_fast). Three findings stack:

1. **Direction lever doesn't exist** at any M (≤3% across row/col packing).
2. **PR 1933's heuristic still delivers its claimed 1.09× at decode M**
   — reproduces original validation.
3. **My earlier alarm about a "shrinking 2.78×" was a misread** — the
   2.78× was vs a non-production baseline. The production win has
   been ~1.09× all along.

## Direction effect across M

Forced (1,16,2), comparing three permutations with cold cache per variant:

| M | default (2-hop col) | k_fast (1-hop row) | col_dir (1-hop col) | hop-count win | direction win |
|---:|---:|---:|---:|---:|---:|
| 128 | 3.078 | 3.073 | 3.087 | 1.00× | 1.00× |
| 256 | 3.262 | 3.080 | 3.094 | 1.06× | 1.00× |
| 512 | 3.508 | 3.296 | 3.283 | 1.06× | 1.00× |
| 1024 | 4.050 | 3.701 | 3.610 | 1.09× | **1.03×** |
| 2048 | 5.177 | 4.284 | 4.322 | 1.21× | 0.99× |

Direction win is consistently within ±3% across all M. The single
1.03× at M=1024 reverses to 0.99× at M=2048 — within measurement
noise. **No direction lever.**

Hop-count win (default → 1-hop) grows from 1.00× at M=128 to 1.21×
at M=2048 because PSUM cost scales with M while LF stays constant.

## Production comparison: pure-M vs (1,16,2)+k_fast

This is the comparison that matters for shipping PR 1932 + PR 1933:

| M | pure-M (planner's pick) | forced (1,16,2)+k_fast | win |
|---:|---:|---:|---:|
| 128 | 3.356 | 3.073 | **1.09×** ✓ |
| 256 | 3.331 | 3.073 | **1.08×** |
| 512 | 3.356 | 3.361 | 1.00× |

Compare to original validation set:
- M=128: 1.09× win (today: 1.09×) ✓ exact match
- M=512: 1.00× win (today: 1.00×) ✓ exact match

**PR 1933's heuristic delivers its claimed value today.** The combined
PR 1932 + PR 1933 path is ~9% faster than pure-M at M=128, dropping
to neutral by M=512. Matches validation.

## Correction to earlier alarm

In an earlier note I flagged that PR 1932's claimed 2.78× speedup had
shrunk to 1.20×. **That alarm was based on the wrong baseline.**

The original validation's 2.78× was:
- "+id" (forced (1,16,2), no k_fast permutation) at 10.93 ms
- "+kf" (forced (1,16,2), with k_fast permutation) at 3.94 ms

**No production system uses "+id"** (forced (1,16,2) without
permutation). It's a hypothetical baseline used to isolate the
permutation's effect, not a production comparison.

The production comparison is:
- pure-M (planner's natural pick)
- vs forced (1,16,2) + k_fast (heuristic override + permutation)

That comparison was ~1.09× in validation and is still ~1.09× today.
**The PR's production value is intact.**

## Refined understanding of where each PR contributes

- **PR 1933 (planner heuristic)** is the primary win-driver in production.
  At decode M, switching from pure-M to (1,16,2) puts more elements
  per core (M_per stays at M instead of M/32), giving better PT
  utilization. The 1.09× win at M=128 comes from this split choice.

- **PR 1932 (k_fast permutation)** has measurable effect only at
  larger M (M ≥ 1024) where PSUM cost dominates LF. In the heuristic's
  target regime (M ≤ 512), the permutation alone gives ≤6% benefit
  on top of the split choice.

In their target regime the two PRs are roughly co-equal in delivering
the headline 1.09×, but PR 1933 (the planner heuristic) is the bigger
contributor.

## Going forward

The "GPU-derived ideas don't translate" pattern still holds:
- ❌ Multicast permutation
- ❌ Inter-op alignment
- ❌ 2D direction
- ✓ k_fast hop-count (validated, ships)

But **PR 1933's heuristic itself is grounded in measured AIU behavior
and delivers production value**. It belongs in the "validated levers"
column, not the closed-as-extrapolation column.

## Files

- `diag_2d_direction_M_sweep_results.txt` — raw measurement output
- This doc — findings + corrected understanding
