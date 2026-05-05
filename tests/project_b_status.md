# Project B — status after Phase 0

Companion to `project_b_hmi_simulator_scope.md` (original plan) and
`hmi_cost_model_phase0_findings.md` (Phase 0 detail). This doc states
where the project stands, what we now know empirically, and the path
forward.

## Where we are

Phase 0 (per-op cost model + calibration) is **partially complete**.

- `hmi_cost_model.py` predicts wall time for a single matmul given
  shape, split, dtype, and emission mode.
- `diag_hmi_cost_model_calibrate.py` validates against 30 measured
  rows from prior diag-branch probes.
- Current fit: 12/30 rows within 10%, median 13.3%, max 99%.

The original Phase 0 exit gate was "predictions within 10%". We are
not there. But Phase 0 has produced something useful regardless: an
empirical map of *which* model assumptions are correct and which are
wrong, anchored against measured wall times.

## What we know empirically now

These are the load-bearing facts to carry into Phase 1+:

1. **B is broadcast under pure-M, not 32× replicated.**
   Probe `diag_hmi_bw_pure_m.py` ruled out the replicated model
   (would give 1300–3400 GB/s implied BW, absurd). Total HMI bytes =
   M·K + K·N + M·N for `(32, 1, 1)`.

2. **Achieved HMI BW under matmul-template ring broadcast is ~40 GB/s,
   not the 67 GB/s spec headline.**
   Probe consistently measures 40 GB/s for B ≥ 128 MB at M=64 (where
   compute is small enough not to interfere). The 67 GB/s number is
   the chip's HMI port nameplate; the kernel template under-utilizes
   it by ~40%.

3. **Wall formula: `max(compute, hmi + LF) + psum`.**
   LF (3 ms) is serial with HMI but overlaps compute. Mechanism is
   probably that LF *is* HMI activity (kernel binary + descriptor
   table fetch) so it can't run in parallel with the operand HMI
   transfer, but compute runs on PT independently.

4. **Cost model classifies correctly on ~80% of rows.**
   The label (launch-floor / HMI / compute / PSUM-bound) is right
   even when the absolute prediction is off by 20–50%. Useful for
   Phase 1 qualitative questions ("which ops are HMI-bound"), not for
   Phase 2 quantitative scheduling decisions.

## What is still wrong with the model

After the k-split probe (`diag_hmi_bw_k_split.py`), three structural
gaps remain:

- **LX overflow re-fetch under k-split.** When per-core A or B
  exceeds the 2 MB LX scratchpad, the operand is re-fetched from HMI
  per N-chunk. Pure-M (32, 1, 1) keeps A_per_core = M_per·K bytes,
  which fits LX for typical decode shapes. K-split scales A_per_core
  by k, blowing past LX for high-K shapes (e.g. DSv3 o_proj at K=16384).
  Measured wall on DSv3 o_proj M=2048 (1,8,4)+kf is 124 ms vs cost
  model's 21 ms — the 6× factor matches an estimated 14× A re-fetch
  multiplier. **This is the dominant missing factor for wide-shape
  k-split predictions.**

- **Per-cluster HMI bytes for shapes that fit LX.** For shapes where
  A_per_core stays within LX (e.g. L3-70B kv_proj M=2048), the
  per-cluster bytes model `(M·K + K·N)/k + M·N` fits to 2% — clean
  win over full-broadcast. For LX-overflowing shapes, both models
  fail because LX overflow dominates.

- **k_fast adjacency benefit collapses at k=4.** Empirical id−kf
  delta drops from 5–85 ms at (1,16,2) to ~0 ms at (1,8,4). At k=4
  the SFP ring is throughput-saturated, not latency-bound, and the
  per-hop savings of kf vanish. The model needs a regime switch
  for high-k.

These are tractable but require structural additions (LX-fit gate,
split-aware bytes, regime-switched PSUM), not parameter tweaks.

## Project B viability

Original Project B framing: "is there scheduling headroom from
HMI-aware op ordering?" The answer requires the simulator to predict
end-to-end wall time accurately enough that comparing two orderings
gives a trustworthy delta.

After Phase 0:

- For **HMI-bound pure-M shapes**, predictions are within 13% — good
  enough for Phase 1's qualitative answer ("here are the HMI-bound
  ops in a transformer block").
- For **K-split shapes**, predictions are ±50% — *not* good enough.
  Comparing two orderings where some ops use k-split would give noise
  rather than signal.

In short: Phase 0 closed the simpler half of the cost model. The
remaining half (k-split, PSUM identity) is what stands between us and
a Phase 2 scheduling question that can be answered.

## Path forward (sequenced)

1. **(DONE) K-split HMI BW probe.** Identified LX overflow as the
   dominant missing factor; per-cluster bytes works for LX-fitting
   shapes; k_fast adjacency collapses at k=4.
2. **Phase 1 with caveats** (NEXT). Extend the model to a transformer
   block (kv_proj, q_proj, o_proj, gate, up, down, plus norms).
   Predict per-op + serial wall time for one decoder block. Use Phase 1
   to answer: "where in the block does HMI dominate?" — qualitative
   answer, doesn't need split-mode accuracy yet. Gate Phase 1 outputs
   on whether each op fits LX under its planner-chosen split, since
   that's the dominant factor we now know.
3. **(Conditional) Add LX-fit gate to the cost model.** Only if
   Phase 1 needs split-mode predictions; for the planner-pure-M
   regime that's the production default, LX always fits and the
   current model suffices.

The original Project B question — "is there scheduling headroom?" —
is still answerable from a model that's accurate in the pure-M regime
even if k-split predictions are noisy, because the planner picks pure-M
in production. K-split residuals matter for "should the planner choose
differently?" which is a separate question (handled by the k_fast
heuristic PR).

## Files

- `project_b_hmi_simulator_scope.md` — original plan
- `hmi_cost_model.py` — per-op cost model
- `diag_hmi_cost_model_calibrate.py` — calibration harness
- `diag_hmi_bw_pure_m.py` + `*_results.txt` — Phase 0 BW probe
- `hmi_cost_model_phase0_findings.md` — Phase 0 detail
- This doc — Phase 0 status + path forward
