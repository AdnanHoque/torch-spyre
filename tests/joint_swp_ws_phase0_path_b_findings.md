# Joint SWP+WS scheduling — Phase 0 Path B findings (ILP prototype)

## TL;DR

The OR-tools CP-SAT prototype establishes three things:

1. **ILP scaling**: fine for small problems (<0.5s @ 128 iters with
   pinned stages), but **hits the 30-second timeout at iters=32 once
   WS choice is enabled**. Horizon decomposition is mandatory for any
   real-world block.
2. **Joint vs decoupled gap**: 0% on pinned-stage workloads;
   **7-9% on compute-balanced workloads with one WS choice point**.
   Falls in the "marginal" bucket per the original scope criteria.
3. **Joint vs serial gap**: 1.8-2.6× consistently. **The proposal's
   value depends critically on what today's compiler actually does** —
   if closer to "serial" (no per-unit pipelining), this is a huge
   win; if closer to "decoupled per-unit greedy", it's marginal.

The single biggest open question coming out of Phase 0 is therefore
**not** "does joint scheduling help?" — it's "what's the baseline?"
That requires a deeptools-side measurement we cannot do alone.

## Prototype design

A K-tiled matmul as a 4-stage pipeline:

| stage | unit | duration (cycles) — HMI-dom | duration — compute-balanced |
|---|---|---:|---:|
| HMI fetch B | HMI | 10 | 4 |
| LX stage | LX | 2 | 1 |
| PT compute | PT | 8 | 10 |
| SFP post | SFP | 3 | 8 |

Constraints:
- intra-iteration deps (stage s waits for s-1)
- per-unit no-overlap (each unit runs one task at a time)
- PT serialization across iterations (PSUM accumulation)

Three modes:
- **serial**: each iteration fully completes before next starts.
  Worst-case "today's compiler with no pipelining" approximation.
- **decoupled**: per-unit greedy in iter order, but cross-iteration
  overlap on different units allowed. Approximates "today's
  compiler doing within-unit SWP but no cross-unit WS choice".
- **joint**: full SWP+WS — units pipeline across iterations subject
  only to deps and PT serialization. The Twill formulation.

WS-choice variant: an optional 5th "post" stage (4 cycles) that can
run on either PT or SFP per iteration. Decoupled forces a global
assignment; joint can alternate per-iteration.

## Headline results

### HMI-dominant profile (decode regime — Llama 70B q_proj M=128 ratio)

Without WS choice:

| iters | serial | decoupled | joint | joint solve s | joint vs decoupled |
|---:|---:|---:|---:|---:|---:|
| 16 | 368 | 173 | 173 | 0.007 | **1.00×** |
| 128 | 2944 | 1293 | 1293 | 0.301 | **1.00×** |

With WS choice:

| iters | serial | decoupled | joint | joint solve s | joint vs decoupled |
|---:|---:|---:|---:|---:|---:|
| 16 | 432 | 177 | 177 | 0.013 | **1.00×** |

When HMI is the binding bottleneck, joint scheduling cannot improve
on decoupled — same lesson as Project B Phase 2: you can't schedule
around the binding constraint. The 4-cycle post stage fits in
available slack on either unit, so the WS choice doesn't matter.

### Compute-balanced profile (prefill regime — attention QK·V)

Without WS choice:

| iters | serial | decoupled | joint | joint vs decoupled |
|---:|---:|---:|---:|---:|
| 16 | 368 | 173 | 173 | **1.00×** |

With WS choice — **the joint advantage emerges**:

| iters | serial | decoupled | joint | joint solve s | joint vs decoupled |
|---:|---:|---:|---:|---:|---:|
| 4 | 108 | 63 | 59 | 0.006 | **1.07×** |
| 8 | 216 | 111 | 103 | 0.012 | **1.08×** |
| 16 | 432 | 207 | 191 | 0.115 | **1.08×** |
| 32 | 864 | 399 | 367 | **30.22 (TIMEOUT)** | 1.09× |
| 64 | 1728 | 783 | 719 | **30.26 (TIMEOUT)** | 1.09× |
| 128 | 3456 | 1551 | 1435 | **30.44 (TIMEOUT)** | 1.08× |

The joint schedule alternates which unit (PT vs SFP) runs the
optional post stage per iteration, balancing load between two
near-saturated units. Per the Gantt traces, both PT and SFP stay
busy nearly all wall, while decoupled has visible gaps on SFP.

**Wall-time win: 7-9%** in this regime. **Falls in the "marginal"
bucket per scope (5-15%).**

### ILP scaling

| profile | ws choice | iters | solve time |
|---|:-:|---:|---:|
| HMI-dominant | — | 128 | 0.30 s |
| HMI-dominant | ✓ | 128 | 0.50 s |
| compute-balanced | — | 128 | 0.30 s |
| compute-balanced | ✓ | 16 | 0.12 s |
| compute-balanced | ✓ | 32 | **30 s timeout** |
| compute-balanced | ✓ | 128 | **30 s timeout** |

CP-SAT scales fine without WS choice. With WS choice + balanced units
(the regime where joint actually wins), scaling cliff at ~32 iters.

The pitch flagged this risk and proposed "horizon decomposition" as
mitigation. The data here confirms it's not a defensive concern but a
real engineering requirement — full-block joint scheduling at
realistic iter counts is not solvable by direct ILP.

## What this means for the proposal

### Verified
- The scheduler IS structurally decoupled (Phase 0.A finding).
- Joint scheduling does find strictly better schedules in regimes
  with multiple near-bottleneck units + WS choice points.
- ILP scaling is real but tractable with horizon decomposition.

### Refined
- The 7-9% gap between joint and decoupled is **smaller than the
  pitch implies**. The pitch's headline benefit ("recover patterns
  no current AIU compiler emits") doesn't directly translate to
  large wall-time wins — it depends on (a) how decoupled today's
  scheduler really is, and (b) whether the AIU has meaningful WS
  choice points (i.e., stages that can actually go to multiple units).
- The pitch references "PT runs current GEMM tile while SFP
  normalizes the previous tile while LX prefetches the next-next
  tile" — this pattern emerges naturally from per-unit greedy
  scheduling under decoupled mode (because each stage is pinned to
  one unit). It does **not** require joint optimization.

### Still unknown (gating the project)
- **What does today's deeptools scheduler actually do?** Specifically:
  is its per-unit pipeline depth 1 (close to my "serial" mode) or
  meaningful (close to my "decoupled" mode)? If the former, joint
  beats today by 2-2.6× — a clear pursue. If the latter, joint
  beats today by 7-9% — a marginal pursue.
- **Where does AIU have WS choice points?** GPU FA-3 alternates
  warpgroups between math and softmax — fungible warp roles. AIU has
  fixed-function units (PT for matmul, SFP for special functions).
  WS choice on AIU is more constrained: maybe LX vs L0 routing, or
  PT-row-reduction asymmetry, or auxiliary ops with multiple paths.
  The proposal needs to enumerate these explicitly.

## Recommended next step

Path A (deeptools owner conversation) is now the gating question.
Specific asks:

1. **What's the per-unit utilization on a Llama 70B q_proj M=128 compile
   today?** If PT is at 70%+ utilization with adjacent units idle
   most of wall, today's scheduler is closer to "serial" → big
   joint win likely. If 4-5 units already overlap at 60%+, today's
   scheduler is closer to "decoupled" → marginal gain.

2. **What WS choice points exist for the AIU?** Concrete question:
   "Are there ops in your op library where the placement (PT vs SFP
   vs RIU vs Mni) is genuinely a planner choice, or is each op
   pinned to a fixed unit at registration?"

3. **Has anyone tried joint scheduling, even informally?** What
   blocked it — solver scaling, lack of demonstrated win, or
   higher-priority work?

If the deeptools owner says "today's scheduler runs each iter to
completion before starting the next" (i.e., no SWP at all), the
project's expected win jumps from 8% to 2.4× and Phase 0 closes with
a clear pursue verdict. If they say "we already pipeline SWP within
each unit and have looked at WS but didn't find wins", the project
likely closes near the 8% margin.

## Files

- `joint_swp_ws_phase0_findings.md` — Phase 0.A (codebase analysis)
- `joint_swp_ws_ilp_prototype.py` — Phase 0.B prototype
- `joint_swp_ws_ilp_prototype_results.txt` — full sweep output
- This doc — Phase 0.B findings
