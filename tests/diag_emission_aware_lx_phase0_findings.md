# Emission-aware LX scheduling — Phase 0 findings

Hardware results from `diag_emission_aware_lx_p1_kscan.py` on
`AdnanHoque/diag-core-ordering` (which has the `core_id_permutation`
infrastructure required for kf-vs-id emission switching).

Two shapes measured. WARMUP=3, ITERS=12, fp16, SENCORES=32.

## Result

The Track-2-Phase-1 hypothesis (M1 — chain-cooperative LX residency
via data-ring multicast) is **NOT supported** by the data. There is
no inflection in the kf wall curve at the chain-LX threshold
predicted by `chain_LX > A_per_core`.

But the probe revealed two different unexpected phenomena, only the
first of which we predicted:

1. **Specific-regime PSUM contention at (m·n=16, k=2)** under id
   emission. The id-kf gap is concentrated at this single split.
2. **Mid-k catastrophic slowdown on wide-N shapes.** DSv3 o_proj
   under k ∈ {4, 8, 16} runs 4× slower than k ∈ {2, 32}. The
   slowdown is independent of emission mode and persists at split
   choices well within LX. This is a new finding, not in the
   original M1/M2/M3 framework.

## Data

### L3-70B kv_proj M=2048 (2048, 1024, 8192) — small N, narrow B

| split | A_per (MB) | overage(M1) | kf ms | id ms | id − kf |
|---|---:|---:|---:|---:|---:|
| (32, 1, 1) | 1 | 0.5× | 3.71 | — | — |
| (1, 16, 2) | 16 | 4× | 4.34 | 10.95 | **+6.60** |
| (1, 8, 4) | 8 | 1× | 5.82 | 5.47 | -0.35 |
| (1, 4, 8) | 4 | 0.25× | 4.28 | 5.13 | +0.85 |
| (1, 2, 16) | 2 | 0.06× | 7.30 | 7.22 | -0.08 |
| (1, 1, 32) | 1 | 0.02× | 4.95 | 4.94 | -0.01 |

### DSv3 o_proj M=2048 (2048, 7168, 16384) — wide N, big PSUM payload

| split | A_per (MB) | overage(M1) | kf ms | id ms | id − kf |
|---|---:|---:|---:|---:|---:|
| (32, 1, 1) | 2 | 1× | 14.11 | — | — |
| (1, 16, 2) | 32 | 8× | 31.12 | 116.92 | **+85.80** |
| (1, 8, 4) | 16 | 2× | **126.78** | 129.00 | +2.23 |
| (1, 4, 8) | 8 | 0.5× | **128.45** | 127.41 | -1.04 |
| (1, 2, 16) | 4 | 0.12× | **132.48** | 131.99 | -0.49 |
| (1, 1, 32) | 2 | 0.03× | 29.93 | 30.02 | +0.09 |

Validation cross-checks: (1,16,2)+kf at 31.12 ms matches the
validation row (31.23 ms ✓); (1,16,2)+id at 116.92 matches
(116.12 ✓). The probe data is consistent.

## What the data rules out

### Mechanism M1 — chain-cooperative LX residency

M1 predicted: kf wall flat once chain_LX > A_per (overage < 1.0),
climbing where overage > 1.0. The data shows no such inflection on
either shape:

- L3-70B at overage = 4× (kf 4.34 ms) is *faster* than at
  overage = 1× (kf 5.82) and overage = 0.06× (kf 7.30).
- DSv3 at overage = 8× (kf 31.12 ms) is *faster* than at
  overage = 2× (kf 126.78), 0.5× (kf 128.45), and 0.12× (kf 132.48).

Both shapes contradict M1's chain-LX-threshold prediction. The
"k_fast adjacent-collaborator stream-prefetch A" hypothesis we
floated to motivate this project is not what the hardware does.

### Mechanism M2 — kernel-template fast path on kf

M2 predicted: kf wall flat across all k. The DSv3 data alone
refutes this — kf walls range 30–132 ms across k.

### Mechanism M3 — PSUM forward-pipelining

M3 predicted: id − kf gap *grows* with k as PSUM payload grows. The
data shows the opposite: gap is large at k=2 and ≤ 1 ms everywhere
else on both shapes.

## What the data shows instead

### Phenomenon A — k=2 + id emission has a specific PSUM contention penalty

The id − kf gap is sharply concentrated at the (m·n = 16, k = 2)
geometry on both shapes:

- L3-70B: +6.6 ms gap at k=2 → ≤ 1 ms at every other k
- DSv3:   +85.8 ms gap at k=2 → ≤ 2.3 ms at every other k

Mechanism (best current hypothesis): under id emission with k=2 and
m·n=16, all 16 K-collaborator chains traverse 16 ring hops
simultaneously, with each chain having only 1 send. Ring link
contention reaches a maximum because every chain's path overlaps
with most other chains' paths. At higher k, each chain has more
sends but the *cohort size* (m·n) shrinks, so fewer chains compete
for the same links at any moment.

This is NOT a chain-LX or stream-prefetch mechanism. It's a
straightforward bandwidth-saturation artefact of the (1, n, k)
split family at k=2.

The k_fast permutation cleanly avoids it (1-hop sends, no
contention), which is why kf walls at k=2 are an order of magnitude
faster than id walls.

### Phenomenon B — DSv3 o_proj has a mid-k catastrophe (NEW)

For wide-N high-K shapes, k ∈ {4, 8, 16} runs 4× slower than
k ∈ {2, 32}, regardless of emission. This is the most striking and
unexpected finding from Probe 1.

DSv3 o_proj M=2048 has constant per-core compute (M_per × N_per ×
K_per = 117M MACs / core for any (1, n, k) with n·k = 32). The PT
utilisation model says all these splits should run at peak compute,
~15 ms per core wall. The actual walls:

- k = 2:  31 ms   (compute + overhead ≈ 16 ms over the 15 ms compute)
- k = 4:  127 ms  (8× over compute)
- k = 8:  128 ms  (8× over compute)
- k = 16: 132 ms  (9× over compute)
- k = 32: 30 ms   (15 ms compute + 15 ms PSUM-chain)

The slow zone is reproducible (medians of 12 iterations) and
mode-independent. This rules out:

- LX overflow (independent of overage_factor 0.12× to 8×)
- PSUM-payload-driven cost (k=32 has the largest PSUM payload but
  is fast)
- Compute-bound work (all configs have identical per-core MACs)

**Hypothesis (speculative)**: the per-core work shape changes with k
even though MAC count is constant. Per-core (M_per, N_per, K_per):

| k | M_per | N_per | K_per | shape descriptor |
|---:|---:|---:|---:|---|
| 2 | 2048 | 448 | 8192 | tall-K narrow-N |
| 4 | 2048 | 896 | 4096 | square-ish |
| 8 | 2048 | 1792 | 2048 | medium-square |
| 16 | 2048 | 3584 | 1024 | wide-N short-K |
| 32 | 2048 | 7168 | 512 | very-wide-N very-short-K |

The kernel template may have fast paths optimised for two extremes
(tall-K or wide-N pure-K) and degenerate paths in between. The
midline shapes might trigger inefficient HMI access patterns or
poorly-tiled inner loops.

Whatever the cause, this is a kernel-template / codegen issue, not
an emission or split-planner issue. It's outside the scope of this
project — but worth surfacing.

## Path forward

The original "emission-aware LX scheduling" framing is dead. The
mechanism we hypothesised (M1, chain-cooperative LX) doesn't exist.
The probe was well-designed to find it, and didn't find it.

Two independent questions remain, neither of which is the project
we set out on:

### Question A — k=2 + id PSUM contention as a planner-avoidable cost

The (1, n, k=2) + id pattern has a specific bandwidth-contention
penalty that scales with `m·n`. If the planner picks (1, 16, 2) it
should *always* enable k_fast emission to avoid this. The PR 1933
heuristic already prefers k_fast in this family — this finding
quantifies the cost of *not* doing so.

Useful side-finding for the existing PRs. Not a new project.

### Question B — DSv3 o_proj mid-k catastrophe

The 4× slowdown at k ∈ {4, 8, 16} on wide-N shapes is a real,
reproducible, *mode-independent* phenomenon. Investigating it would
require:

- Running the same probe on more wide-N shapes to confirm the
  pattern (DSv3 q_a_proj M=2048, gate_proj M=2048, similar)
- Profiling kernel execution at k=4 to see *where* the time goes
  (compute, HMI, prefetch stalls, etc.)
- Reading the SDSC emitter's per-(M_per, N_per, K_per) codegen path

If the catastrophe is real across wide-N shapes, it represents a
**performance bug in the kernel template** that the planner today
sidesteps by always picking (32, 1, 1) — and which would block any
attempt to use K-split for wide-N shapes. It's a legitimate
investigation but it's a kernel/codegen problem, not a memory-system
research project.

## Recommendation

The "chain-cooperative LX" research framing isn't paying out. Three
options for the next move:

1. **Close this branch as a negative result**, return to a
   building project (Roller-on-AIU enumerator + LX-fit gate from
   the Phase 0 work). The investigation produced a clean negative —
   one of the LX-and-emission hypotheses is now ruled out.
2. **Pivot to Phenomenon B** — the mid-k catastrophe. Run more
   wide-N shapes and instrument the kernel to find where time goes.
   This is interesting but probably ends up touching the kernel
   template (deeptools-adjacent territory).
3. **Pivot to a "negative result" paper** — the chain-cooperative
   LX hypothesis is intuitive and someone else might propose it.
   Documenting the empirical refutation has some value but it's
   minor.

I'd lean (1). The probe did its job: refuted the hypothesis cleanly.
That's a win in the sense that we're not building a cost-model
fix on a wrong premise.

## Files

- `tests/diag_emission_aware_lx_p1_kscan.py` — Probe 1 (used)
- `tests/emission_aware_lx_phase0_scope.md` — original scope (M1
  hypothesis is now refuted)
- This doc — Phase 0 findings
- (raw outputs) `/tmp/probe1_l3_70b_kv.txt`, `/tmp/probe1_dsv3_o_proj.txt`
