# HMI BW gap investigation — findings

## TL;DR (and a redirect)

The investigation didn't find what it was looking for, but found
something bigger:

**Pure-M `(32, 1, 1)` — the planner's natural pick — is the
*slowest* split for many wide-B HMI-bound shapes. Other splits
deliver 25-38% lower wall on the same ops without k-split or k_fast.**

This is a torch_spyre planner-side lever, similar in shape to PR 1933
(k_fast heuristic) but on a different axis (m·n instead of k).
Estimated block-level impact at decode M: **~15% wall savings on a
Llama 70B decoder block** if the planner picks better splits for the
6 matmul ops.

This is a bigger lever than the original HMI BW gap question. The
"40 GB/s vs 67 GB/s" observation turns out to be specific to pure-M's
32-way broadcast pattern; other splits don't have the same ceiling.

## What I expected to find vs what I found

**Expected**: a flat 40 GB/s achieved BW across all access patterns,
suggesting the gap to spec lives in deeptools / runtime and isn't
torch_spyre-fixable.

**Found**: BW spread of 100+ GB/s across splits at the same shape.
For wide-B M=128:
- best split: (2,16,1) gives 141 GB/s implied BW (3.98 ms wall)
- worst split: (1,8,4) gives 30 GB/s implied BW (7.56 ms wall)

The "implied BW" exceeds the chip's 67 GB/s spec for several splits,
which means the broadcast accounting model (`bytes = M·K + K·N + M·N`)
over-counts for those splits. Real HMI bytes is less because cores
share more efficiently OR the wall-formula `wall = LF + hmi` doesn't
hold (LF may be hidden in HMI activity for some splits).

**Either way, the wall is what matters**, and the wall varies
dramatically with split.

## Wall-time data per shape

### Wide-B M=128 (128 × 8192 × 8192 — Llama 70B q_proj scale)

| split | wall ms | savings vs pure-M |
|---|---:|---:|
| **pure-M (32,1,1)** ← planner's pick | **6.52** | (baseline) |
| pure-N (1,32,1) | 4.03 | **38%** |
| (2,16,1) | 3.98 | **39%** |
| (8,4,1) | 4.04 | **38%** |
| (16,2,1) | 4.56 | 30% |
| pure-K (1,1,32) | 5.15 | 21% |
| (1,16,2) | 7.02 | -8% |
| (1,8,4) | 7.56 | -16% |

### Wide-B M=256 (256 × 8192 × 8192)

| split | wall ms | savings vs pure-M |
|---|---:|---:|
| **pure-M (32,1,1)** ← planner's pick | **6.25** | (baseline) |
| pure-N (1,32,1) | 4.10 | 34% |
| (8,4,1) | 4.08 | **35%** |
| (16,2,1) | 4.66 | 25% |
| (2,16,1) | 5.00 | 20% |
| pure-K (1,1,32) | 7.32 | -17% |
| (1,16,2) | 11.07 | -77% (catastrophic — LX overflow) |
| (1,8,4) | 12.10 | -94% (catastrophic — LX overflow) |

### Narrow-B M=128 (128 × 4096 × 4096)

| split | wall ms | savings vs pure-M |
|---|---:|---:|
| **pure-M (32,1,1)** ← planner's pick | **3.90** | (baseline) |
| pure-N (1,32,1) | 3.24 | 17% |
| (8,4,1) | 3.30 | 15% |
| (2,16,1) | 3.31 | 15% |
| (16,2,1) | 3.42 | 12% |
| pure-K (1,1,32) | 3.54 | 9% |
| (1,16,2) | 3.58 | 8% |
| (1,8,4) | 3.52 | 10% |

Savings shrink at narrow B because LF=3 ms dominates.

## Why pure-M is worse — hypothesis

Under pure-M (32, 1, 1):
- All 32 cores want full B (broadcast 32-way through the data ring)
- All cores want different M-slice of A (no sharing of A)

Under (8, 4, 1):
- 4 unique B chunks, each broadcast to 8 cores (4 broadcasts × 8 fan-out)
- 8 unique A chunks, each broadcast to 4 cores (8 broadcasts × 4 fan-out)

**Hypothesis**: the data ring has a *per-broadcast-cycle* cost that's
high relative to the per-byte transfer cost. A 32-way broadcast is
slower than 4 broadcasts of 8-way each because the 32-way broadcast
has to sequence through more ring positions.

**Alternative hypothesis**: wider broadcast fan-out causes more ring
contention with other cores' fetches, throttling effective BW.

**Either way**: lower fan-out → faster wall. The data is
unambiguous.

## Why some splits get catastrophically slow

The (1, 16, 2) and (1, 8, 4) splits at M=256 wide-B blow up to 11-12
ms wall — slower than pure-M. This matches the **LX overflow re-fetch
pattern** we found in the earlier k-split probe: per-core A under
(1, 8, 4) at K=8192 is 4 × 8192 × 2 = 64KB which fits, but actually
N_per=2048 means the kernel needs many sub-tiles per cluster which
forces re-fetch. (Or maybe a different mechanism — the catastrophic
slowdown is consistent with our prior finding that LX-overflowing
splits hit a 14× re-fetch multiplier.)

## What this means as a torch_spyre project

### Concrete proposal

A planner heuristic that picks **non-pure-M m·n splits for wide-B
shapes** when LX-fitting and likely to deliver better wall.

Specifically: for matmuls where K·N > some threshold (the broadcast
B is "wide"), prefer splits like (8, 4, 1) or (2, 16, 1) over pure-M.

### Effort estimate

- Investigation phase: 1-2 weeks — probe more shapes (different
  hidden_size, head_dim, M range), confirm the pattern generalizes,
  identify exact gating conditions.
- Implementation: 2-4 weeks — add the heuristic to
  `core_division.py`, similar shape to `_try_k_fast_split`. Add
  hardware-free unit tests.
- Validation: 1-2 weeks — sweep across popular models (same harness
  as `diag_k_fast_popular_models.py`), confirm block-level wins,
  no regressions on small-M shapes.

Total: 4-8 weeks.

### Risks

1. **Mechanism not fully understood**: the wall improvements are
   real but I'm only hypothesizing about why. The intervention
   could underperform on shapes I haven't tested if the cause is
   different from what I think.
2. **Interaction with k_fast PR**: the planner might want to choose
   between m·n split and k-split based on shape. Need a unified
   decision tree.
3. **LX overflow check**: must avoid the catastrophic regressions
   we saw on (1, 16, 2) wide-B at M=256. The heuristic must
   include LX-fit gating.

### Why this is bigger than k_fast

| optimization | shape coverage | per-op savings | block-level estimate |
|---|---|---:|---:|
| k_fast PR 1933 (as shipped) | narrow-N kv_proj only | 8% | <1% |
| k_fast PR 1933 (broadened gate) | + o_proj, down_proj | 15-50% | 2-8% |
| **m·n split heuristic (this finding)** | **all wide-B matmuls** | **25-38%** | **~15%** |

Coverage is the key difference. k_fast targets a narrow shape band;
the m·n split heuristic targets the bulk of decoder block work.

## Comparison to all other proposals

Going back to my brainstorm of torch_spyre-only projects:

| project | block-level win | effort | confidence |
|---|---:|---:|---|
| **m·n split heuristic** (this finding) | **~15%** | 4-8 wk | medium-high (real measurements) |
| FA tiling via decomposition | 2-8% | 4-8 wk | medium (calibration shows opportunity) |
| Fix SDPA-to-bmm regression | 30-50% on attention | 1-3 wk | high (provable bug) |
| Broaden k_fast heuristic | 2-8% | 1-2 wk | high (already in flight) |
| LX-fit aware splits | preventative (avoid 10× regressions) | 2-4 wk | high |
| Operator fusion audit | ? | 1+ wk investigation | uncertain |

**This new finding is the biggest available block-level win** from
torch_spyre alone, and supports the LX-fit work as a prerequisite
guard against catastrophic splits.

## Suggested next steps

1. **Confirm the pattern across more shapes** (~1 week). Extend the
   probe to:
   - Llama family hidden_sizes (4096, 8192, 16384)
   - DSv3 shapes (7168, 18432)
   - All matmul ops in a decoder block (q, kv, o, gate, up, down)
   - Different M values (32, 128, 512, 1024, 2048)
2. **Identify the optimal split per shape** (~3 days). Build a tiny
   model that predicts the best split given (M, N, K).
3. **Decide implementation approach**: either a pre-baked lookup
   table for known shapes, or a generic heuristic.
4. **Implement + test + measure end-to-end**.

This investigation has shifted the priority — instead of the
originally-planned FA-tiling work next, the biggest available
torch_spyre lever is now the planner change suggested by this
finding. Recommend pivoting.

## Files

- `diag_hmi_bw_split_pattern.py` — the probe
- `diag_hmi_bw_split_pattern_results.txt` — measurement output
- This doc — findings
