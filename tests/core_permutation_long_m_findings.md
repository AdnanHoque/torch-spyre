# Long-M generalization probe — M-trend narrow, kv_proj win is huge

## TL;DR — two corrections plus one major new finding

The broad-sweep "block_cyclic helps long-M prefill" hypothesis was
**wrong as stated**. After running 11 (shape, split) configs across
L3-70B, L3-8B, MLP types, and projection types at M ∈ {2048, 4096}:

1. **The M-trend is specific to L3-70B q_proj/o_proj geometry.** Doesn't
   reproduce on L3-70B MLP (different N or K), doesn't reproduce on
   L3-8B (smaller hidden dim). So it's a local pocket, not a general
   long-M lever.

2. **The "writeback DRAM banking" mechanism hypothesis from the broad
   sweep is unsupported.** The trend should reproduce wherever output
   bytes are large; it doesn't.

3. **MAJOR NEW FINDING: kv_proj `(1, 16, 2)` + block_cyclic = 2.76×.**
   This is the largest single permutation win across the entire core-
   ordering project. It's the same K-chain-shortening mechanism we
   identified earlier, just acting on a different split shape that
   the planner picks for narrow-N matmuls.

## Full results

| regime | shape | split | identity ms | block_cyclic ms | mean sp | consistent? |
|---|---|---|---:|---:|---:|---|
| trend | L3-70B q_proj M=4096 | (1, 32, 1) | 22.033 | 21.471 | **1.026×** | ✓ |
| l3-70B | L3-70B kv_proj M=2048 | **(1, 16, 2)** | 10.875 | 3.936 | **2.763×** | ✓ |
| l3-70B | L3-70B o_proj M=2048 | (1, 32, 1) | 12.257 | 12.141 | 1.010× | ✓ |
| l3-70B | L3-70B o_proj M=4096 | (1, 32, 1) | 21.898 | 21.415 | **1.023×** | ✓ |
| l3-70B | L3-70B mlp_gate M=2048 | (1, 32, 1) | 259.245 | 258.175 | 1.004× | ✓ |
| l3-70B | L3-70B mlp_down M=2048 | (1, 32, 1) | 70.243 | 70.211 | 1.000× | flat |
| l3-70B | L3-70B mlp_down M=4096 | (1, 32, 1) | 138.316 | 137.859 | 1.003× | ✓ |
| l3-8B | L3-8B q_proj M=2048 | (1, 32, 1) | 5.364 | 5.388 | 0.995× | ✓ |
| l3-8B | L3-8B q_proj M=4096 | (1, 32, 1) | 7.818 | 7.830 | 0.998× | ✓ |
| l3-8B | L3-8B mlp_down M=2048 | (1, 32, 1) | 18.115 | 18.167 | 0.997× | ✓ |
| l3-8B | L3-8B mlp_down M=4096 | (1, 32, 1) | 33.134 | 33.002 | 1.004× | flat |

Confirmed wins (≥2% mean, consistent direction):

- L3-70B q_proj M=4096 (continuation of the M=2048 finding) — 1.026×
- L3-70B o_proj M=4096 (sibling shape) — 1.023×
- **L3-70B kv_proj M=2048 with `(1, 16, 2)`** — **2.763×**

## The kv_proj 2.76× — what's going on

`(1, 16, 2)` is a K-split with `k=2`. The planner picks this shape for
narrow-N matmuls (kv_proj has N=1024, only 16 sticks; pure-N would give
0.5 sticks/core which is invalid; so it K-splits as a fallback).

PSUM-chain analysis:

| mode | K-pair physical positions | hops per chain | chains | total chain hops |
|---|---|---:|---:|---:|
| **identity** | logical 0,16 → physical (0, 16) | 16 | 16 | 256 |
| **block_cyclic** | logical 0,16 → physical (0, 1) | 1 | 16 | 16 |

**16× chain-hop reduction.** PSUM was a *huge* fraction of wall time
(7 ms of 11 ms = 64%) because the chain payload is large
(M·N/n·sizeof(fp32) ≈ 1 MB per chain) and each hop is sequential on
the SFP ring.

Block_cyclic compresses the chain to 1 hop; PSUM time becomes
negligible; wall time drops from 10.88 ms to 3.94 ms.

### Why we missed this in earlier probes

Every prior K-split test was on `(m, 1, k)` shapes (q_proj-style).
For those, the K-fast permutation is `stride2`. We never tested
`(1, n, k)` shapes — those aren't picked for q_proj. They're picked
for **kv_proj** (and any other narrow-N matmul where the planner
can't pure-N split).

So the K-chain shortening lever is **wider than we realized**: it
applies to *any* split with `k > 1`, with a different "K-fast"
permutation depending on the split shape:

| split shape | K-fast perm |
|---|---|
| `(m, 1, k)` (q_proj-style K-split) | `stride2` ≈ `core_emission_reverse` |
| `(1, n, k)` (kv_proj-style K-split) | `block_cyclic` |
| `(m, n, k)` mixed (both >1) | needs a generalized "k_fast" perm |

### Why the M-trend hypothesis was wrong

The broad sweep showed L3-70B q_proj `(1, 32, 1)` at M=2048 wins
1.024× with block_cyclic, scaling continuously up from M=128. We
hypothesized output-writeback DRAM banking. That hypothesis predicts
similar wins on any shape with comparable output bytes.

In this sweep, several shapes have similar output bytes but flat
results:
- L3-8B q_proj M=4096 (32 MB output): 0.998×
- L3-70B mlp_down M=4096 (64 MB output): 1.003×
- L3-70B o_proj M=2048 (32 MB output): 1.010× (slight)
- L3-70B q_proj M=4096 (64 MB output): **1.026×**

So the writeback hypothesis is unsupported — output size alone doesn't
predict the win. The L3-70B q_proj/o_proj wins likely come from some
*other* mechanism specific to that geometry (square N=K=8192 with
M ≥ 1024). Could be cache-line aliasing or some interaction with the
LX scratchpad layout. Without DRAM counters we can't isolate it, and
without a pattern that generalizes there's no shippable lever here
beyond manual tuning recommendation.

## The K-fast generalization is the real finding

Across all the work to date:

| finding | shape | win | mechanism |
|---|---|---:|---|
| `core_emission_reverse` (shipped knob) | various K-split `(m,1,k)` | ~3.6% | dim-iteration reverse acts as K-fast for `(m,1,k)` |
| `stride2` perm | K-split `(4,1,8)` | 3.9% | same mechanism via permutation |
| `block_cyclic` perm | K-split `(1,16,2)` (kv_proj) | **176%** | K-fast for `(1,n,k)` |
| L3-70B q_proj long-M | `(1,32,1)` | ~2.5% | unclear, L3-70B-specific |

The K-chain shortening lever is consistently real. The "K-fast"
permutation depends on split shape, but the underlying mechanism
(pack K-collaborators contiguously on the ring → minimize PSUM
chain-hop count) is the same.

A planner heuristic could pick the right K-fast permutation per
split. **For kv_proj specifically, this is a 2.76× win on a hot
matmul that's in every transformer.** That's the real takeaway.

## Implications

1. **Ship intent revised.** The single-percent wins on L3-70B q_proj
   long-M aren't worth a heuristic. The kv_proj win **is**.
2. **Mechanism well-understood.** PSUM chain-hop count is the lever.
   The right perm packs K-collaborators contiguously.
3. **Generalized "k_fast" perm should beat both stride2 and
   block_cyclic** on mixed splits where neither is the natural K-fast.
4. **Open question (still): does core_id distance actually match
   physical ring distance?** Today's kv_proj result is consistent
   with the linear ring-distance model at scale, but a direct probe
   would nail it.

## Files

- [`tests/diag_core_permutation_long_m.py`](diag_core_permutation_long_m.py)
- raw output: [`diag_core_permutation_long_m_results.txt`](diag_core_permutation_long_m_results.txt)
- prior context:
  [`core_permutation_findings.md`](core_permutation_findings.md),
  [`core_permutation_broad_findings.md`](core_permutation_broad_findings.md)

## Next

1. Pairwise-distance probe: vary K-pair distance d ∈ {1, 2, 4, 8, 16}
   on the kv_proj shape, measure wall time. If linear in d, direct
   verification of monotonic core_id ↔ ring position.
2. Generalized `k_fast` perm: a single permutation that adapts to
   any split `(m, n, k)` and packs K-collaborators contiguously.
3. Planner heuristic sketch: per-split-shape rule for picking the
   right K-fast perm.
