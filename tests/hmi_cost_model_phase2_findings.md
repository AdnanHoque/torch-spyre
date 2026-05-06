# HMI cost model — Phase 2 findings

## TL;DR

**Project B closes.** The concurrent decoder-block simulator shows
**0–1% wall-time savings** from cross-op HMI scheduling across every
(model, M) we measured. Per the scope doc rule (< 5% = close), there's
no headroom for an HMI-aware scheduler to attack.

| model | M | serial ms | concurrent ms | saved % |
|---|---:|---:|---:|---:|
| Llama 3.1 8B  | 128 | 47.83 | 47.83 | 0.0% |
| Llama 3.1 70B | 128 | 80.62 | 80.62 | 0.0% |
| Llama 3.1 405B | 128 | 198.90 | 198.90 | 0.0% |
| DeepSeek V3 | 128 | 63.47 | 63.47 | 0.0% |
| Llama 3.1 70B | 2048 | 150.29 | 149.73 | 0.4% |
| Llama 3.1 405B | 2048 | 467.19 | 463.72 | 0.7% |

The largest savings (0.7% on Llama 405B at M=2048) comes from the
compute-bound regime, where some op N+1 HMI can hide behind op N
compute. In every other regime, HMI is the binding constraint and
already runs at 100% utilization.

## Why Phase 1 over-stated the headroom

Phase 1 reported a 28% perfect-overlap upper bound for Llama 70B
M=128, calculated as `Σ wall(HMI-bound ops)`. That bound assumed
"non-HMI ops can fully hide behind HMI ops" — but **classifying an op
as 'non-HMI' just means HMI didn't dominate *that op's* wall. It
doesn't mean the op uses zero HMI.**

Concretely for Llama 70B M=128, the 7 non-HMI-bound ops still claim
on the HMI machine:

| op | t_hmi + LF | classification |
|---|---:|---|
| input_rmsnorm | 3.10 ms | HMI/LF |
| kv_proj | 3.48 ms | launch-floor-bound |
| attention | 3.21 ms | (compute-bound at high M) |
| post_attn_residual | 3.16 ms | HMI/LF |
| post_attn_rmsnorm | 3.10 ms | HMI/LF |
| silu_mul | 3.55 ms | HMI/LF |
| post_mlp_residual | 3.16 ms | HMI/LF |
| **Σ** | **22.76 ms** | |

Adding these to the HMI-bound ops' HMI claims:
`Σ HMI_total = 80.62 ms = block wall`. **HMI is at 100% utilization.**

The Phase 1 bound was apples-to-oranges: comparing a "wall sum"
(includes LF and small compute on every op) against a hypothetical
"HMI-bound only" total that ignored every other op's HMI demand.
Phase 2's per-resource simulator gets it right.

## What the simulator says about each regime

**Decode (M ≤ 128)**: HMI fully utilized. 0% scheduling headroom.
Block wall = `Σ (t_hmi + LF)` over all ops in the block.

**Decode-batching (M = 128–512)**: Same. HMI still fully utilized.

**Long prefill (M ≥ 1024)**: Some compute-bound matmuls (gate, up,
down at large M). Concurrent scheduling can hide their HMI behind
the next op's compute. **0.4–0.7% savings**, far below the 5% threshold.

## What this rules out

- An HMI-aware planner heuristic that *picks per-op splits* to balance
  HMI and compute across the block. The simulator says any time saved
  on one op is lost to another op's HMI claim — no zero-sum win.
- Cross-bundle execution overlap as a runtime feature targeting
  *operand HMI prefetch*. The bundles already run HMI back-to-back at
  full utilization.

## What this doesn't rule out (and is worth flagging)

The simulator assumes **launch floor (3 ms) sits on the HMI pipeline**
because Phase 0 found `wall ≈ LF + bytes/BW` empirically. For Llama
70B M=128, 12 ops × 3 ms LF = **36 ms — 45% of block wall**. If LF
could be hidden — e.g., by prefetching the kernel binary or
descriptor table for op N+1 during op N's actual data fetch — the
block could in principle drop to ~45 ms.

This is a **different problem** from operand HMI scheduling. It's
about overlapping *kernel-launch overhead* with *prior op's data
fetch*. Whether the runtime supports this prefetch path, or could,
is a question for the deeptools team that's separate from Project B's
original framing.

If LF prefetch is feasible, that's a **45% saving** — much larger than
anything cross-op operand scheduling could deliver. **Worth raising
as a follow-up question to deeptools** even though Project B itself
closes.

## What the simulator assumes (and where the assumptions could break)

- **HMI capacity 1**: only one op fetches at a time. Real hardware
  may allow multiple in-flight HMI requests with shared BW. If so,
  the relevant constraint is `Σ_weights / 40 GB/s` (irreducible),
  which for Llama 70B M=128 = 1.69 GB / 40 GB/s = **42 ms** — still
  most of the wall, but the per-op LF stack would not be there.
- **PT capacity 1**: only one op computes at a time. Reasonable —
  pure-M splits use all 32 cores.
- **LF on HMI pipeline**: empirically validated by Phase 0 probe.
- **Within-op compute/HMI overlap**: validated by the cost model fit.
- **No cross-op data dependency on HMI**: activations live in LX
  between ops. Reasonable for a single decoder block.

## Verdict

Per the scope doc's three-bucket rule:
- **< 5%**: HMI is essentially fully utilized. Project B closes.
- 5–15%: small headroom. Marginal.
- \> 15%: substantial headroom. Pursue.

We measure 0–0.7% across every (model, M). **Project B closes.**

The win-fraction rephrased: the simulator's perfect-overlap concurrent
schedule essentially matches the serial schedule that today's runtime
delivers. There's nothing to schedule around because the binding
constraint (HMI bandwidth) is already saturated by the planner's
sequential dispatch.

## Implications for the broader work

1. **k_fast PR (PRs 1932 + 1933) remains worth shipping**. Its win
   mechanism — picking K-split for narrow-N small-M shapes to reduce
   per-cluster HMI demand — directly attacks the HMI bottleneck this
   simulator says is binding. Different lever than scheduling.
2. **The 45% LF-prefetch follow-up is the bigger lever** if feasible.
   That's a deeptools/runtime question worth raising independently.
3. **The cost model itself remains useful** for predicting per-op walls
   — Phase 0 fit is good enough for production-regime predictions.
   Just don't use it to argue for cross-op scheduling.

## Files

- `hmi_cost_model_phase2_concurrent.py` — the simulator
- This doc — findings
