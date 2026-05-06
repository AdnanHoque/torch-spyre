# Joint SWP+WS — end-to-end block findings

## TL;DR

Per-op joint scheduling on attention is **a real per-op win (1.18-1.83×)
but flows through to only 0.5-4.5% block-level savings**, because
attention is a small fraction of decoder block wall at all M.

Severe Amdahl's law: attention compute is 4-10% of block wall in
prefill regimes and effectively zero at decode.

| regime | attention fraction of block | block savings (joint vs decoupled) |
|---|---:|---:|
| Decode (M=128, any model) | <0.1% | **0.0%** |
| Llama 70B prefill M=2048 | ~4% | **0.5–1.7%** |
| DSv3 prefill M=2048 | ~10% | **1.4–4.5%** |
| Llama 405B prefill M=2048 | ~3% | (extrapolated <1%) |

The DSv3 case is the upper bound — it has 128 attention heads vs
Llama 70B's 64, making attention a larger fraction of the block.

## Per-regime breakdown

### Decode (M ≤ 128) — attention is zero

At M=128, attention compute per core is ~0.03 ms (essentially the
launch floor + tiny compute). The block wall is **80 ms = sum of
HMI claim times**, dominated by MLP weight fetches. **No amount of
joint scheduling on attention changes the block wall.**

This is consistent with Project B's verdict that decode-regime blocks
are HMI-bound. Joint SWP+WS doesn't escape that constraint.

### Prefill (M = 2048) — attention is small

For Llama 70B M=2048:

| attention mode | attention ms | block ms (serial runtime) |
|---|---:|---:|
| serial | 6.75 | 150.69 |
| decoupled | 5.79 | 149.73 |
| joint | 4.27 | 148.20 |

Joint saves **1.5 ms per block per core, ~1% of block wall**. The
per-op number (5.79 → 4.27 = 26% saving on attention) is real but
attention is just 4% of block wall. Saving 26% of 4% = ~1%.

For DSv3 (more attention heads), the same dynamic gives ~3% block
savings — better but still modest.

### Where the block wall actually goes

For Llama 70B M=2048, per Phase 1 cost model:

| op | wall ms | % of block |
|---|---:|---:|
| gate_proj | 30.07 | 20.0% |
| up_proj | 30.07 | 20.0% |
| down_proj | 30.07 | 20.0% |
| q_proj | 8.59 | 5.7% |
| o_proj | 8.59 | 5.7% |
| attn_qkt_softmax_v | 6.36 | 4.2% |
| silu_mul, residuals, norms | rest | ~24% |

**The MLP projections (gate, up, down) are 60% of the block.** They
are matmuls without FA-style PT/SFP overlap structure — joint SWP+WS
doesn't apply to them. Attention is one of several mid-sized ops.

## What this changes about the project's value

The earlier framing — "joint SWP+WS on FA: 1.18-1.83× per-op
speedup" — is mathematically correct but misleading at the block
level. Honest summary:

- **As a generic compiler optimization shipped across all ops**: only
  attention has the right structure (PT/SFP overlap across iterations).
  Block-level wins are 0.5-4.5% at prefill and 0% at decode.
- **As a paper / patent contribution**: the per-op FA result is the
  strong claim. Generalizing Twill from 4 GPU warpgroups to 9 AIU DAE
  units is a meaningful contribution even if real-world block-level
  impact is modest, because:
  - Other workloads with different op mixes (e.g., heavy attention
    decoder-only, retrieval-heavy, reasoning models with multi-turn
    attention) would see proportionally bigger gains
  - The infrastructure enables future ops with PT/SFP overlap structure

## Honest project verdict

This shifts the original Phase 0 assessment:

- **From the generic prototype**: 7-9% per-op gain on compute-balanced
  workloads. Marginal.
- **From the FA prototype**: 18-58% per-op gain on attention. Strong.
- **From the end-to-end analysis**: 0.5-4.5% block gain on prefill,
  0% on decode. Marginal-to-small.

For a 12-16 week project, 0.5-4.5% block savings on prefill workloads
puts it on the edge of the project's bar. The patent/paper case is
stronger than the perf case at the block level.

A honest pitch for the project would now read:
1. **Generalizes Twill to 9 AIU units** (academic contribution).
2. **Delivers 18-58% per-op speedup on attention compute** (FA-3
   ping-pong recovered automatically).
3. **Block-level wins are 1-5% on prefill, ~0% on decode** unless
   workload mix shifts toward attention-dominant patterns.
4. **Patent-grade, MLSys/CGO-tier paper potential.**

## Suggested deeptools-owner question — refined

Now that we know FA is the only op that meaningfully benefits, the
deeptools owner question narrows:

**"For Llama 70B M=2048 attention compute on AIU today: what fraction
of attention wall is PT busy vs SFP busy vs neither?"**

If owner says "PT runs ~70% of attention wall, SFP runs 30%, with
significant idle gaps" — joint scheduling could close those gaps,
matching our prototype's predictions. If owner says "PT and SFP
already overlap to ~90% of attention wall" — even the per-op win
shrinks and the project closes.

## Files

- `joint_swp_ws_block_e2e.py` — end-to-end script
- `joint_swp_ws_block_e2e_results.txt` — sweep output
- `joint_swp_ws_fa_prototype.py` — per-op FA prototype (called by e2e)
- This doc — end-to-end findings

The project status is now:
1. **Phase 0.A** (codebase analysis): premise verified — scheduler IS
   decoupled.
2. **Phase 0.B** (ILP prototype): tractable with horizon decomposition;
   per-op gains are workload-dependent.
3. **Phase 0.C** (FA prototype): per-op attention savings of 18-58%.
4. **Phase 0.D** (end-to-end, this doc): block-level savings of 0-5%
   under realistic op mixes.

**Phase 0 verdict**: technically interesting, marginally valuable
at end-to-end. Pursue if the academic/patent case justifies 12-16
weeks; close if the bar is "5%+ block-level wall reduction in
production workloads."
