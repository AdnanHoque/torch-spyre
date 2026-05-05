# Project B — scoping via per-op HMI cost-model simulation

## Why a simulator first

The original Project B kickoff
(`project_b_hmi_aware_scheduling_plan.md`) framed HMI-aware scheduling
as needing hardware-runtime cooperation: cross-bundle execution
overlap to interleave HMI-heavy and HMI-light ops. Phase 3 preload
established that the runtime does not expose that concurrency at the
torch_spyre layer.

A static **HMI cost-model simulator** sidesteps the runtime question.
Instead of asking "can we schedule ops to overlap HMI?", we ask
"would scheduling them differently *predict* a lower wall time, and
by how much?" If the simulator says no, the project closes cleanly
without needing runtime support. If yes, we have a quantified target
to take to the deeptools team and a basis for designing planner
heuristics.

The simulator is also useful regardless of scheduling:

- Score the current planner's split choices per op
- Predict end-to-end wall time for new shapes before running on hardware
- Identify which ops are HMI-bound vs compute-bound vs launch-floor-bound
- Empirical calibration target: a "AIU performance reasoning" reference
  with first-principles cost model, validated against hardware

## What the simulator computes

### Inputs

- **Op graph**: list of ops with type, shapes, dtypes, dependency edges.
  Initial scope: 1 transformer decoder block, ~10-15 ops.
- **Hardware params**: SENCORES=32, HMI bandwidth (67 GB/s combined,
  88 GB/s pure-ring), launch floor (~3 ms), per-core fp16 throughput
  (~0.1 TFLOPS achieved, ~1 TFLOPS peak), LX size (2 MB), SFP ring
  bandwidth (32 GB/s), PT SIMD width (64).

### Per-op model — matmul

For matmul `C[M, N] = A[M, K] @ B[K, N]` with planner split `(m, n, k)`:

```
per_core_C_bytes  = (M/m) * (N/n) * dtype_bytes
per_core_B_bytes  = (K/k) * (N/n) * dtype_bytes
per_core_A_bytes  = (M/m) * (K/k) * dtype_bytes
per_core_macs     = (M/m) * (N/n) * (K/k)

# HMI demand: weights (B) stream from DRAM unless they fit in LX;
# activations (A) ditto. Outputs (C) write back to DRAM.
hmi_bytes = total_unique_B_bytes(split) + ...   # depends on sharing pattern

# Compute time, gated by PT array utilization
simd_util         = min(1.0, (M/m) / 64)        # SIMD width fudge
t_compute         = per_core_macs / (PT_PEAK * simd_util)

# HMI time
t_hmi             = hmi_bytes / HMI_BW

# PSUM time on SFP ring
psum_chain_hops   = chain_hops_for_split(split, emission_mode)
t_psum            = psum_chain_hops * (per_chain_payload / SFP_BW)

# Wall: launch floor, compute and transit overlap, PSUM serial
wall_time         = max(LAUNCH_FLOOR, max(t_compute, t_hmi) + t_psum)
```

The constants and fudge factors (`simd_util`, `PT_PEAK`, etc.) are
calibrated against the measurements we already have:

- Pairwise-distance probe: directly characterized PSUM chain cost as
  `~0.48 ms / hop` for `(1, 16, 2)` kv_proj
- M-sweep on real workloads: characterized pure-M wall times across
  M ∈ {32, 128, 512, 1024, 2048} for kv_proj-like shapes
- DSv3 cross-model sweep: characterized o_proj, down_proj, q_a_proj
  wall times at multiple M values

If the simulator predicts the measured wall times within ~10%, the
cost model is good enough to drive scheduling decisions.

### Per-op model — non-matmul (norm, softmax, activation, residual)

Simpler. Mostly compute-bound, small HMI footprint:

```
t_compute = elements / (per_core_throughput * num_cores)
t_hmi     = io_bytes / HMI_BW
wall_time = max(LAUNCH_FLOOR, max(t_compute, t_hmi))
```

### Graph model

Two simulators worth building:

1. **Serial model**: `wall_total = sum(wall_op[i] for i in ops)`. This
   is what today's runtime does (per-bundle serialization). Useful as
   a baseline.
2. **Concurrent model**: model HMI as a shared resource with a budget
   per timestep; multiple ops can be in flight if they don't exceed
   the HMI budget. Lets us simulate hypothetical scheduling.

Difference between the two = upper-bound on what cross-op scheduling
could buy us. If the gap is small, the project closes.

## Phases

### Phase 0 — calibrate the per-op cost model

Build the matmul cost model. Fit/tune the constants
(`simd_util`, achieved-throughput-fraction, HMI BW under different
sharing regimes) against the data we already have:

- Pure-M wall times from `diag_k_fast_real_workloads_msweep_results.txt`
- K-split wall times from same file
- Pairwise PSUM cost from `diag_core_pairwise_distance_results.txt`

**Output**: a Python module that takes `(M, N, K, split, dtype)` and
returns predicted wall time, with predictions within X% of measured.

**Effort**: ~1-2 days. No hardware needed for development; uses
saved measurements for validation.

**Risk**: low. The hardest part is fitting the SIMD-utilization fudge
factor for small M. We have the M=32, 128, 512, 1024, 2048 data
points to anchor it.

### Phase 1 — extend to a transformer block

Take the matmul cost model and extend to non-matmul ops. Walk through
a Llama 3.1 70B decoder block (kv_proj, q_proj, o_proj, gate, up,
down, plus norms / softmax / activation / residual). For a chosen M,
compute per-op wall time and total serial wall time.

**Output**: a script that takes (model, M) and prints per-op +
end-to-end predicted wall time.

**Effort**: ~1-2 days.

**Risk**: low. Non-matmul ops are simpler models.

### Phase 2 — concurrent simulation + identify scheduling headroom

Build the concurrent simulator. Compute the gap between serial and
concurrent end-to-end wall time:

- If gap is < 5%: HMI is essentially fully utilized already. Project
  B closes — there's nothing to schedule around.
- If gap is 5-15%: small headroom. Possibly worth a heuristic; weigh
  against complexity.
- If gap is > 15%: substantial headroom. Worth pursuing the
  scheduling heuristic and the runtime-side conversation.

**Output**: a "scheduling headroom across model families" report.
Same shape as the popular-models speedup table — for each (model, M),
how much wall time could perfect HMI scheduling save.

**Effort**: ~3-5 days. Concurrent simulator is the most non-trivial
piece; the analysis report is templated.

**Risk**: low. The simulator is software-only, validated against
known measurements.

### Phase 3 — design and ship a planner heuristic (only if Phase 2 says go)

If Phase 2 says there's headroom, design a heuristic that picks
op orderings (within the constraints of dependency edges) to minimize
predicted wall time. Same shape as the k_fast planner heuristic —
empirical thresholds, hardware-free unit tests, gated by config flag.

**Output**: a planner-side change + tests, paired with a follow-on
HMI-aware split-choice cost model.

**Effort**: ~1-2 weeks if Phase 2 finds a clear win.

**Risk**: medium. Real-world schedulings will need to land alongside
runtime support for cross-op overlap, which is the same wall as
Phase 3 preload. We'd be shipping infrastructure ahead of runtime
cooperation.

## What this avoids

Things the simulator-first approach lets us defer or skip:

- **Real-hardware HMI utilization probe** (Phase 0 of the original
  kickoff). The simulator's predictions answer the same question
  ("is HMI saturated?") with less effort and more nuance.
- **Runtime concurrency feasibility** (Phase 1 of the original
  kickoff). We don't need it for the simulator to produce useful
  predictions — only for shipping the scheduling heuristic itself.
- **Speculative cross-op weight prefetch implementation**. The
  simulator can predict whether prefetch would help before we
  invest in the codegen.

## What it doesn't avoid

- The simulator is only as good as its calibration. Phase 0 has to
  produce predictions accurate enough to trust. A 50% prediction
  error makes Phase 2's "is there headroom?" question unanswerable.
- The simulator can't predict effects we don't model (e.g., cache
  warmth, ring contention beyond first-order, runtime queue depth
  effects). Those become unmodelled uncertainty in the final
  answer.
- A simulator-validated win still needs runtime support to ship.
  Phase 3 is gated on deeptools cooperation.

## Suggested next move

Start with Phase 0 (calibrate cost model) since it's the smallest,
lowest-risk, and produces a useful artifact (validated cost model)
regardless of whether the rest of the project proceeds.

The Phase 0 deliverable is concrete: a Python module
`hmi_cost_model.py` that takes `(M, N, K, split, dtype)` and returns
predicted wall time, validated to within X% of the
`diag_k_fast_real_workloads_msweep_results.txt` and
`diag_k_fast_popular_models_results.txt` measurements we already have
on disk.

If Phase 0 validates within 10%, proceed to Phase 1. If it doesn't,
the validation gap itself tells us something about the AIU's actual
runtime behaviour vs. our first-order model — which is also useful.

## Open questions to resolve before Phase 0 starts

1. **What's "HMI bandwidth" exactly?** We measured 67 GB/s (combined
   with cross-core sharing) and 88 GB/s (pure-ring). The cost model
   needs to pick one or model the regime transition.
2. **What's the SIMD utilization curve?** At M=4 (M/m at 32 cores),
   we measured pure-M wall times far above the HMI floor — implying
   compute under-utilization. We need a parametric form, e.g.:
   `simd_util(M) = min(1, M / 64)` or a tabulated curve.
3. **Do we model launch floor as additive or as a `max`?** Empirically
   wall_time at very small ops bottoms out at ~3 ms regardless of
   compute/HMI — suggests `max(launch_floor, ...)` not additive.
4. **fp32 PSUM accumulation: is the SFP ring payload fp32 or fp16?**
   Affects per-chain payload size by 2x.

These are answerable from the existing measurement data; resolving
them is part of Phase 0's calibration.
