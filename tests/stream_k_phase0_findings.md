# Stream-K-on-Spyre — Phase 0 findings

The Phase 0 probe (`tests/diag_wave_quant.py`) characterizes wave-
quantization losses on Spyre across 12 production matmul shapes (LoRA,
GQA-TP=8 kv_proj, MoE per-expert, prime-M dynamic prefill, aligned
references). Headline: **wave-quantization is essentially a non-issue
on Spyre. The default planner already saturates 32 cores for every
measured shape via mixed M/N/K splits.**

This pivots the Stream-K project framing significantly. The wave-
quantization motivation that drove my original scope doesn't apply.
Detailed below.

## Method

For each shape, run a single matmul through `torch.compile` and capture
the planner's chosen `(m_split, n_split, k_split)` factors via the
`parse_op_spec` SDSC hook (same one we use for SplitK + DDR-traffic +
flash-attention diagnostics). Compute cores-used = product, idle =
32 - cores-used. Measure wall time per call.

## Results

| Shape | Use case | M, N, K | Splits (size×cores) | Cores | Wall ms |
|---|---|---|---|---:|---:|
| LoRA r=16 down decode | LoRA adapter | 1×16×4096 | `[16×1c, 4096×32c]` | 32/32 | 2.88 |
| LoRA r=16 down prefill | LoRA adapter | 128×16×4096 | `[128×32c, 16×1c, 4096×1c]` | 32/32 | 2.84 |
| LoRA r=64 down prefill | LoRA adapter | 128×64×4096 | `[128×32c, 64×1c, 4096×1c]` | 32/32 | 2.84 |
| L3-70B GQA TP=8 kv decode | Llama-70B | 1×128×8192 | `[128×2c, 8192×16c]` | 32/32 | 2.92 |
| L3-70B GQA TP=8 kv prefill | Llama-70B | 128×128×8192 | `[128×32c, 128×1c, 8192×1c]` | 32/32 | 2.90 |
| L3-8B GQA TP=4 kv prefill | Llama-8B | 128×256×4096 | `[128×32c, 256×1c, 4096×1c]` | 32/32 | 2.91 |
| DeepSeek-MoE inter=1408 prefill | per-expert | 192×1408×2048 | `[192×32c, 1408×1c, 2048×1c]` | 32/32 | 3.10 |
| Qwen3-MoE inter=1536 prefill | per-expert | 128×1536×2048 | `[128×32c, 1536×1c, 2048×1c]` | 32/32 | 3.04 |
| Prime M=257 prefill | dynamic | 257×4096×4096 | `[257×1c, 4096×32c, 4096×1c]` | 32/32 | 3.20 |
| **Prime M=521 prefill** | dynamic | 521×4096×4096 | `[521×1c, 4096×32c, 4096×1c]` | 32/32 | **132.85** |
| L3-8B q_proj (aligned ref) | reference | 128×4096×4096 | `[128×32c, 4096×1c, 4096×1c]` | 32/32 | 3.79 |
| L3-70B q_proj (aligned ref) | reference | 128×8192×8192 | `[128×32c, 8192×1c, 8192×1c]` | 32/32 | 6.44 |

**Summary**: 0 of 12 measured shapes leave any core idle.

## What this finding actually means

The default planner does **mixed-dim spillover** — when one output dim
can't absorb 32 cores within stick alignment, leftover cores get
allocated to other dims (including K). Examples:

- LoRA r=16 down decode (`1×16×4096`): N=16 fits in 1 core (N is sub-stick
  size); K=4096 absorbs the remaining 32 cores → `[1, 1, 32]`-style split.
  *Looks like our SplitK heuristic but already happening at the default
  planner level.*
- L3-70B GQA TP=8 kv decode (`1×128×8192`): N=128 = 2 sticks → 2 cores;
  K=8192 = 128 sticks → 16 cores → `(1, 2, 16)` mixed split = 32 total.

This is exactly the kind of "Stream-K-style spillover" the original
project would have built — except it already exists. Spyre's compile-
time planner does what GPU runtime work-queues do for Stream-K. **The
GPU "wave quantization" problem doesn't have an analog here because
work assignment is decided once at compile time with full visibility.**

## The unexpected outlier — prime M=521 is 42× slower than prime M=257

| M | Wall ms | M_iter splits |
|---:|---:|---|
| 257 (prime) | 3.20 | `[257×1c, 4096×32c, 4096×1c]` |
| **521 (prime)** | **132.85** | `[521×1c, 4096×32c, 4096×1c]` |

Same N, K. Same prime-ness. Same split factor pattern (`m_split=1`,
`n_split=32`). But M=521 takes **42× longer** than M=257. Both should be
in the same regime — neither divides 32, both spill to N-split. Yet a
~2× difference in M produces a ~42× difference in wall time.

This is **not** the wave-quantization problem the Stream-K project
would have addressed. It's a different class of issue — likely a
pathological codegen path in the dxp_standalone backend for specific M
values, or a stick-alignment-related compiler issue. Worth filing as a
separate investigation but it's not a Stream-K project per se.

## What this means for Stream-K-on-Spyre

The project as originally framed (3 components: padding, smart 2D
wave-quant-aware split, 1D linearized partial reductions) **doesn't
have the perf headroom I projected**, because the planner already does
the spillover work. Specifically:

| Component | Stream-K rationale | Spyre reality |
|---|---|---|
| Wave-quant-aware split | GPU SMs idle in last wave | Spyre planner already saturates 32 cores |
| 2D linearized work assignment | Use any (m, n) ratio | Schema already supports this; planner already searches this space |
| Padding for divisibility | Activate idle cores | No idle cores to activate |
| Cross-core partial reductions | Boundary tiles | Already used for SplitK; not a Stream-K-unique need |

**There's no Stream-K-shaped perf gap on Spyre.** The architectural
fit between dataflow + compile-time planning is genuinely different
from GPU's runtime-queue model in a way that makes Stream-K's main
value prop already-implemented.

## Pivots / what's still worth doing

Two adjacent investigations remain interesting based on this data:

1. **The M=521 anomaly.** A 42× slowdown for a specific prime M is
   suspicious. Worth a focused probe sweeping M over primes and
   not-primes to map the pathology, then file with the dxp/dsm team.
   This is a bug-investigation project, not a Stream-K project. ~1 week.

2. **Awkward-N shapes specifically.** I sampled N values that all hit
   nice stick fractions. What about N values that don't divide 64 (one
   stick)? Could be a small-but-real perf gap there. ~3 days probe.

Neither matches the original "Stream-K" framing or the perf headroom
I'd hoped for. The honest project verdict: **fold this finding back
into the broader Spyre-perf knowledge base. Stream-K as a flagship
project doesn't pan out on this architecture.**

## Decision

**Stream-K project paused.** The empirical wave-quantization gap
doesn't exist. The M=521 anomaly is a worthwhile follow-up but not
the same project. The findings + diagnostic harness remain useful
artifacts.

Flash-attention (the parallel project on `AdnanHoque/diag-flash-
attention`) has a 30-50× perf gap that IS real and IS frontend-
addressable. That's the bigger lever; this branch can stay paused
unless the M=521 anomaly turns into something deeper.

## Files

- `tests/diag_wave_quant.py` — Phase 0 wave-quantization sweep
- `tests/diag_wave_quant_results.md` — auto-regenerated bench output
- `tests/stream_k_phase0_findings.md` — this document
