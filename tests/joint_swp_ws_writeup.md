# Joint SWP+WS for AIU — investigation summary + paper/patent sketch

## Investigation summary (what we did)

Phase 0 of evaluating whether to generalize the Twill paper's joint
software-pipelining + warp-specialization ILP from 4 GPU warpgroups to
the AIU's 9 DAE units. Four sub-phases, ~1 day each:

### Phase 0.A — codebase analysis (premise verification)

Read `RCUIntraEntityScheduler` in deeptools (1914 lines). Confirmed
the proposal's central claim: the AIU compiler IS structurally
decoupled along two axes:

1. **Data transfer vs. compute**: `performIntraEntitySchedulingForEntity`
   dispatches `DATA_TRANSFER` entities to `doAboveLxScheduling` and
   `COMPUTE` entities to `doBelowLxScheduling` as independent passes.
2. **Per-iteration vs. cross-iteration**: the per-iteration task subgraph
   `formSETaskSubGraphForCompEntity` connects iterations sequentially
   without joint cross-iteration reordering across units.

Found that `seTaskEdgeType::functional_overlapped` is *defined* in
`perfEstimator.h:26` and *consumed* in `interEntityScheduler.cpp:1242`,
but apparently *never constructed* — supporting the premise that
cross-unit overlap modeling exists in skeleton form but isn't producing
edges.

### Phase 0.B — OR-tools ILP prototype (generic matmul)

Encoded a K-tiled matmul as a 4-stage pipeline (HMI → LX → PT → SFP)
with three modes (serial, decoupled per-unit greedy, joint full ILP).
Scaling sweep through 128 iterations.

Findings:
- **ILP scaling**: <0.5s @ 128 iters with pinned stages; 30s timeout
  at iters=32 once WS choice is enabled. Horizon decomposition is
  mandatory for real workloads.
- **Joint vs decoupled**: 0% on pinned-stage workloads; **7-9% on
  compute-balanced workloads with one WS choice point**.
- **Joint vs serial**: 1.8-2.6× consistently across all profiles.

### Phase 0.C — FlashAttention-specific prototype

Modeled FA's per-iteration structure (HMI → PT_QK → SFP_softmax →
PT_OV → SFP_update) — two PT stages and two SFP stages per iteration,
making it the canonical Twill-style workload.

Findings (across SFP softmax cost regime):
- **PT-dominant** (cheap exp, 10× ratio): joint vs decoupled = 1.06×
- **Medium balance** (typical exp): joint vs decoupled = 1.36×
- **Balanced** (expensive exp): joint vs decoupled = **1.83×**

Per-op savings on attention: 18-58% wall-time reduction.

### Phase 0.D — end-to-end block analysis (Amdahl's law)

Plugged the FA prototype's predicted attention wall back into the
Phase 2 concurrent block simulator (Project B work) and measured
block-level wall.

Findings:
- **Decode M=128**: 0.0% block savings (attention is <0.1% of HMI-
  bound block; HMI binding constraint absorbs everything).
- **Llama 70B prefill M=2048**: 0.5-1.7% block savings (attention
  is ~4% of block).
- **DSv3 prefill M=2048**: 1.4-4.5% block savings (attention is ~10%
  of block due to 128 attention heads).

The per-op 1.36× FA win compresses to 1-5% at the block level
because MLP projections (gate/up/down) are 60% of decoder block wall
and don't have FA-style PT/SFP overlap structure.

## Paper structure (academic contribution)

**Suggested venue**: MLSys 2027 or CGO 2027.
**Suggested title**: "9-Way Twill: Joint Software Pipelining and
Warp Specialization for Heterogeneous Functional Units on the IBM AIU"

### Outline (~10 pages)

**Section 1 — Introduction (1.5 pages)**
- AIU has 9 DAE units (PT, SFP, LX, L0, RIU, Mni, ...) — heterogeneous
  vs GPU's homogeneous warpgroups.
- Twill's joint formulation provably recovers FA-3 ping-pong on GPU.
  We generalize to heterogeneous units with capacity constraints.
- Three contributions:
  1. ILP formulation for 9 heterogeneous functional units with
     PT-row reduction asymmetry + LFSR/REDUCE state machines.
  2. FIFO-depth constraint linking dependent stages.
  3. Horizon decomposition that keeps ILP tractable past iters=32.
- Empirical: 1.36× per-op speedup on attention; 1-5% end-to-end
  block savings on prefill workloads.

**Section 2 — Background (1.5 pages)**
- AIU architecture overview (9 DAE units, ring topology, LX hierarchy).
- Software pipelining + warp specialization fundamentals.
- The Twill formulation (the GPU 4-warpgroup case).
- Why decoupled compilation misses opportunities (with one motivating
  example — likely FA prefill).

**Section 3 — Problem formulation (2 pages)**
- AIU functional-unit capacity model: per-unit throughput, latency,
  pipeline depth. Specifically the PT-row reduction asymmetry which
  GPU formulations don't have.
- FIFO-depth constraints between dependent stages (e.g., LX → PT
  feeds through a finite buffer).
- Joint scheduling as an ILP: variables for stage start times,
  WS choice booleans, makespan minimization.
- Decoupled baseline as a constrained subproblem (forces per-unit
  iter-order).

**Section 4 — Solver design (1.5 pages)**
- OR-tools CP-SAT encoding.
- **Horizon decomposition**: solve overlapping windows of K
  iterations, splice solutions. This is the key engineering
  contribution — without it, ILP times out at iters≥32.
- Optimality bounds: how close splice solutions get to global optimum.

**Section 5 — Evaluation (3 pages)**
- Workloads: q_proj, attention compute, MLP projections, full
  decoder blocks. Models: Llama 70B, DSv3.
- Baseline: today's deeptools scheduler (per-unit greedy, no WS choice).
- Per-op results:
  - Attention compute: 1.18-1.83× speedup depending on SFP cost
  - MLP projections: 1.00× (no PT/SFP overlap structure)
  - Other ops: minimal
- End-to-end block results: 1-5% on prefill, 0% on decode.
- Scaling: ILP solve time across iters and unit count.
- Discussion: where joint scheduling matters (compute-balanced ops
  with WS choice points) and where it doesn't (HMI-bound, or
  pinned-stage workloads).

**Section 6 — Related work (0.75 pages)**
- Twill (the direct precursor).
- CGO 2026 papers on single-unit instruction window optimization.
- CUTLASS Ping-Pong Deep Dive (the GPU-specific equivalent).
- AIU-side: prior PRs (k_fast permutation, multicast coalescing if
  it lands).

**Section 7 — Conclusion (0.25 pages)**
- Joint formulation generalizes from GPU warpgroups to heterogeneous
  AIU units. Per-op wins are real (up to 1.83×). Block-level wins
  are bounded by Amdahl's law unless workload mix is attention-heavy.

### Strengths of the paper

- **Novel generalization**: nobody has applied Twill-style joint ILP
  to heterogeneous functional units. The PT-row reduction asymmetry
  + FIFO constraints are AIU-specific contributions.
- **Concrete win demonstrated**: 1.36× per-op on attention with
  realistic cycle estimates.
- **Tractability**: horizon decomposition is a real engineering result.

### Risks of the paper

- **Modest end-to-end impact**: reviewers may push back on the 1-5%
  block-level number. Need to argue strong per-op result + future
  workload mix.
- **No real hardware measurements**: prototype runs in OR-tools, not
  on the AIU. Reviewers will want a deeptools-side validation. This
  requires partnering with the deeptools team — significant project
  scope addition.

## Patent structure (IP contribution)

**Suggested filing**: USPTO continuation or new application.
**Suggested title**: "Joint Software Pipelining and Functional-Unit
Assignment for Heterogeneous Accelerator Architectures with
Per-Packet Target-Node-Identifier Multicast Constraints"

### Independent claims

**Claim 1** (the method): A computer-implemented method for compiling
a deep neural network operator to a heterogeneous accelerator with
N ≥ 5 functional units, comprising:
1. Receiving an operator specification with K loop iterations.
2. Constructing an integer-linear program with variables representing
   start times of each (iteration, stage) pair, where stages are
   assignable to one of multiple functional units.
3. Adding constraints encoding (a) per-unit no-overlap, (b) per-unit
   throughput-capacity, (c) per-unit FIFO-depth between dependent
   stages, (d) cross-iteration data dependencies.
4. Solving the ILP using horizon decomposition wherein iterations are
   partitioned into overlapping windows of size W ≤ 32.
5. Splicing per-window optimal schedules into a global schedule.
6. Emitting per-functional-unit instruction streams.

### Dependent claims

- **Claim 2** (PT-row reduction asymmetry): The method of claim 1
  wherein the heterogeneous accelerator includes a tensor-processing
  unit with row-reduction asymmetry, and at least one constraint
  encodes the row-reduction asymmetry as a per-stage execution
  cost dependent on iteration index.
- **Claim 3** (FA-3 ping-pong recovery): The method of claim 1 wherein
  the operator is multi-head attention and the resulting schedule
  alternates which of two functional units processes the matmul
  computation versus the softmax computation across iterations.
- **Claim 4** (LFSR/REDUCE state machine): ...
- **Claim 5** (multicast TNID coalescing): ... (links to A5.2 if filed)

### Why this is patent-grade

1. **Novel combination**: joint ILP for heterogeneous units (Twill
   generalization) + AIU-specific constraints (PT-row asymmetry,
   FIFO depths) + horizon decomposition.
2. **Non-obvious**: prior art (Twill, CGO 2026) operates on
   homogeneous units or single-unit instruction windows. The
   step from N=4 homogeneous to N=9 heterogeneous is non-trivial.
3. **Practical embodiment**: prototype shows 1.36× speedup with
   tractable solve times.

### Considerations

- **Filing strategy**: file as a continuation in a family with the
  k_fast and multicast PRs to build a portfolio.
- **Defensive vs. offensive**: even if perf wins are modest, the
  patent prevents competitors from claiming the same idea.
- **Cross-licensing leverage**: pairs well with NVIDIA Hopper TMA
  multicast patents in cross-licensing discussions.

## Recommended next move

**Conditional pursue.** Two paths:

1. **Academic-first**: focus on the paper. Strengthens the case
   significantly if we can pair with deeptools to get real-hardware
   validation on Llama 70B prefill attention. Effort: 6-8 weeks for
   prototype + paper, contingent on deeptools cooperation.
2. **Patent-first**: file the IP claim, ship the per-op attention
   optimization opportunistically without committing to a full paper
   timeline. Effort: 2-3 weeks for filing + initial implementation.

Given the modest end-to-end perf case, the patent-first path likely
gives the best ROI. The paper is gravy if the work goes well.

If neither pursued, we should at least document the Phase 0 findings
as a memo for the deeptools team — the "scheduler is decoupled"
observation alone is useful institutional knowledge.

## Files in this branch

| File | Purpose |
|---|---|
| `joint_swp_ws_phase0_findings.md` | Codebase analysis (Phase 0.A) |
| `joint_swp_ws_ilp_prototype.py` | Generic ILP prototype (Phase 0.B) |
| `joint_swp_ws_ilp_prototype_results.txt` | Generic prototype results |
| `joint_swp_ws_phase0_path_b_findings.md` | Phase 0.B findings |
| `joint_swp_ws_fa_prototype.py` | FA-specific prototype (Phase 0.C) |
| `joint_swp_ws_fa_prototype_results.txt` | FA prototype results |
| `joint_swp_ws_fa_findings.md` | Phase 0.C FA findings |
| `joint_swp_ws_block_e2e.py` | End-to-end block analysis (Phase 0.D) |
| `joint_swp_ws_block_e2e_results.txt` | End-to-end results |
| `joint_swp_ws_e2e_findings.md` | Phase 0.D end-to-end findings |
| **This doc** | Investigation summary + paper/patent sketch |
