# Project B — HMI-bandwidth-aware op scheduling — kickoff plan

## Hypothesis

HMI is the chip-level bottleneck for prefill matmul. Today the planner
picks per-op splits to *minimize per-op HMI bytes* (via
`output_element_priority` and now `k_fast`). It does **not** consider
HMI utilization across ops in a graph. Two distinct opportunities live
in this gap:

1. **Reorder independent ops** so an HMI-heavy op runs in parallel
   with a compute-bound op, hiding HMI behind compute (or vice versa).
2. **Prefetch weights for op N+1 during op N's execution**, even when
   N+1 depends on N's output — N+1's weights don't.

Either gives wall-time savings without changing per-op work; both
require that the runtime expose the necessary concurrency.

## Why this might be small or impossible — read first

Two reality checks before designing experiments:

### Reality check 1 — runtime concurrency model

torch_spyre compiles **one matmul per dxp bundle** and dispatches each
bundle via `subprocess.run(["dxp_standalone", "--bundle", ...])`. The
resulting `SpyreSDSCKernelRunner` invocations look serialized at the
Python level. We need to verify whether:

- Bundles can be in-flight concurrently (overlap across bundles)
- HMI streaming for bundle N+1 can begin while bundle N's compute
  is still running
- The dxp runtime exposes any "asynchronous launch" or "stream"
  semantics

If the answer is "no overlap at all across bundle boundaries," this
project hits the same wall as Phase 3 preload — the lever exists at
deeptools/dxp level but isn't reachable from torch_spyre.
**Phase 0 must answer this question before we build anything.**

### Reality check 2 — independent-op scarcity in LLM forward

A decoder block looks like:

```
norm → q_proj ─┐
norm → k_proj ─├→ attn → out_proj → norm → gate ─┐
norm → v_proj ─┘                          → up   ─┴→ silu·mul → down
```

Mutually-independent pairs: (q_proj, k_proj, v_proj), (gate, up).
**But each pair is the same shape with the same HMI/compute profile.**
Reordering them doesn't change which is HMI-heavy vs which is
compute-bound.

Compute-bound ops (norm, softmax, silu, residual) are *between*
matmuls because they consume matmul outputs. They're not independent
of their surrounding matmuls; you can't move them.

So **opportunity (1) — reorder independent ops to interleave** is
limited to the (qkv, gate-up) micro-batches, where the ops are too
similar to benefit from reordering.

That leaves opportunity (2) — **cross-op weight prefetch** — as the
real lever. Which is functionally equivalent to cross-call weight
preload (Phase 3). So Project B may need to be reframed as "Phase 3,
take 2: a fresh look at runtime support for op-N+1 weight prefetch
during op-N execution."

## Investigation phases

### Phase 0 — characterize HMI utilization

Build a probe that measures, during a real LLM forward pass:

- HMI active vs idle time (if exposable via dxp telemetry; otherwise
  inferred from per-op timings + theoretical HMI demand)
- Compute active vs idle time per core
- Per-op breakdown: HMI cost, compute cost, "slack" (compute > transit)
- Identify any inter-op gaps where HMI sits idle while a downstream
  op's weights could have been prefetched

**Predicted outcome**: HMI is busy ~70-90% of the time on prefill
matmul (the dominant cost). The 10-30% idle periods correspond to
either (a) launch-floor overhead between bundles, or (b) compute-bound
non-matmul ops between matmuls.

If (a) dominates → opportunity is "tighten the launch-floor pipeline"
(maybe op fusion, maybe eager-launch).
If (b) dominates → opportunity is "prefetch the next matmul's weights
during norm/softmax" — which is opportunity 2.

**~1 day to write the probe, ~1 day to run + interpret.**

### Phase 1 — runtime concurrency feasibility

Empirically determine whether the runtime supports any form of
overlap across bundle boundaries. Two probes:

1. **Two-bundle launch race**: launch two independent matmuls (e.g.,
   two different shapes on different inputs) back-to-back. Measure
   whether their wall times sum sequentially or partially overlap.
   If `wall(A+B) < wall(A) + wall(B)`, there's some concurrency.
2. **Async API hunt**: read the `SpyreSDSCKernelRunner` source and
   any dxp Python bindings to see if there's an async-launch path.
   Phase 3 preload investigation got close to this; revisit with
   fresh eyes.

**Predicted outcome (pessimistic)**: bundles are strictly serialized.
**Predicted outcome (optimistic)**: there's a queue depth ≥ 2 in dxp
that lets one bundle's HMI begin while the previous bundle's compute
finishes.

**Off-ramp**: if pessimistic, write up findings (similar to Phase 3
preload), engage deeptools team. Project enters "blocked on runtime
support" state.

**~2 days probe + analysis.**

### Phase 2 — proof of concept (only if Phase 1 says go)

Pick a synthetic graph with two ops:
- Op A: HMI-heavy matmul (e.g., 1 MB/sec compute, 50 MB/sec HMI demand)
- Op B: compute-bound elementwise (e.g., 10 MB/sec compute, 1 MB/sec HMI)

A and B independent (separate inputs). Measure:
- Sequential wall time (A then B)
- "Overlap-attempted" wall time (whatever the runtime exposes)

If overlap fires, design a scheduler heuristic that detects (HMI-bound
A, compute-bound B) pairs in real graphs and emits them in the
overlap-friendly order.

### Phase 3 — production sweep

If POC works, apply the heuristic to real LLM forward graphs:
- L3-70B prefill at M ∈ {2048, 4096}
- DSv3 prefill at M=2048 (lots of HMI-bound matmuls)
- Mixtral 8x7B (per-expert + attention)

Measure end-to-end speedup. Target: 5-10% on prefill where HMI is the
bottleneck.

## What "success" looks like

| outcome | impact |
|---|---|
| Phase 0 shows HMI is fully utilized | Project closes; no opportunity to address. |
| Phase 0 shows large HMI-idle gaps but Phase 1 confirms no runtime concurrency | Project closes from torch_spyre; bumps to deeptools team with measurements. Still useful as a diagnostic. |
| Runtime supports overlap, scheduling helps a few % on real graphs | Ship the heuristic. ~5% prefill win across all GQA models. |
| Runtime supports overlap, scheduling helps double digits | Big win. Probably the most consequential perf change post-k_fast. |

## Risks / pre-mortem — three failure modes ranked by likelihood

1. **Most likely: bundle serialization is enforced at dxp level**.
   Same wall as Phase 3 preload. Rationale: dxp_standalone is
   designed for AOT model compilation; it likely assumes whole-model
   bundles where overlap is internal to the bundle, not across
   `dxp_standalone` invocations.

2. **Likely: HMI is already near-saturated by intra-bundle prefetch**.
   The kernel template already overlaps HMI fetch with compute within
   a single matmul. If that's hitting ~90% HMI utilization, there's
   only a few % left for cross-op scheduling to grab.

3. **Possible: independent ops too rare to matter**. Even if the
   runtime supports overlap, we don't have many (HMI-heavy,
   compute-light) independent pairs to exploit. Prefetch (which
   doesn't require independence) might be the only viable lever, and
   that's just Phase 3 preload by another name.

## Sequencing recommendation

Run **Phase 0 first**, as a small standalone probe (~2 days). The
output is either:

- A clean "HMI is mostly idle X% of the time, here are the gaps"
  characterization → motivates Phase 1
- A "HMI is fully saturated; no gaps to exploit" result → close the
  project with one writeup

Either way, Phase 0 produces a useful document about HMI utilization
patterns even if the project doesn't continue.

**Don't commit to Phase 1+ until Phase 0 says there's something
worth chasing.**

## Connection to existing work

- [`preload_phase3_results.md`](preload_phase3_results.md): same
  underlying constraint — runtime support for cross-bundle overlap.
  If Phase 1 here confirms what Phase 3 preload found, the projects
  collapse.
- [`session_summary.md`](session_summary.md): meta-pattern that big
  remaining levers cross repo boundaries. Project B fits the pattern
  unless Phase 1 finds a torch_spyre-side lever we missed.
- The matmul reference doc
  ([`matmul_first_principles_v3.md`](../docs/source/architecture/matmul_first_principles_v3.md))
  Part 6 / 11 already discusses HMI as the dominant bottleneck for
  weight-streaming matmul; this project is the operational follow-up.

## Open question to resolve at kickoff

Before Phase 0, decide: is this project framed as *"HMI-aware
scheduling"* or as *"cross-op weight prefetch"* (Phase 3 preload, take
2)? They're related but the empirical work is different:

- HMI-aware scheduling: needs HMI utilization probe, needs concurrency
  model probe, scheduler heuristic at end.
- Cross-op weight prefetch: needs runtime async-launch probe, needs
  per-op weight-residency tracking, codegen change at end.

My recommendation: **start with cross-op weight prefetch framing**
because (a) the lever is closer to the existing planner architecture,
(b) Phase 3 work is already partially done so we have some runway,
(c) the "reorder independent ops" framing has fewer real targets in
LLM forward graphs.

But run Phase 0 either way — the HMI utilization characterization is
a prerequisite for both framings.
