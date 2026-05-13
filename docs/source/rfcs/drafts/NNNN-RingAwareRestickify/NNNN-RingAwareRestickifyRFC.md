# Ring-Aware Restickify

**Authors:**

* @AdnanHoque

**Status:** Draft

## Summary

Make `restickify` placement and core-id mapping aware of the cost of moving
data on Spyre's **RIU data ring**. Today, the inductor backend chooses each
op's `core_id_to_work_slice` mapping independently, so producer/consumer pairs
can end up on opposite sides of the 32-core ring, paying up to half-ring of
hops per stick of data restickified between them. This RFC proposes a phased
plan to add ring-locality awareness to the work-division and restickify
passes, starting with telemetry and an MVP that aligns consumer mappings to
producer mappings when possible.

This is a follow-up to the k_fast PR ([#1986](https://github.com/torch-spyre/torch-spyre/pull/1986)),
which added ring-locality awareness for the **SFP ring** (PSUM reduction).
The same general technique — choosing physical core IDs to minimize ring
traversal — is applied here to the RIU ring (data movement).

## Motivation

The Spyre AIU 1.0 has 32 cores connected by three logical ring fabrics:

| Ring | Purpose | Approx. BW |
|---|---|---:|
| SFP | PSUM reduction (bichain) | 35.2 GB/s/dir |
| RIU | Data movement (A/B reads, restickify, broadcast) | 166 GB/s |
| HBM | Off-chip reads/writes; bank-parallel | shape-dependent |

The k_fast work demonstrated that picking a core-id-to-slice mapping that
places **K-cohort cores adjacent on the SFP ring** delivers a measured
geomean speedup of 1.73× across 20 Granite / L3-70B / Mixtral / DSv3
shapes ([diag_pr_vs_main_findings.md](../../../../tests/diag_pr_vs_main_findings.md)
on branch `AdnanHoque/feat-k-fast-combined`).

The same locality argument applies to the **RIU ring** for restickify:

* `restickify` is the data-movement kernel that converts one stick layout
  into another. Mechanically, each consumer core loads its required slice
  from arbitrary source cores via `lower_restickify`
  ([lowering.py:485-508](../../../../torch_spyre/_inductor/lowering.py)).
* The bytes physically traverse the RIU ring. Cost ≈
  `bytes_per_stick × ring_hops(producer_core, consumer_core)`.
* Today the cost model in `optimize_restickify_locations`
  ([optimize_restickify.py:495](../../../../torch_spyre/_inductor/optimize_restickify.py))
  is the **element count moved**. It is ring-blind: a restickify between
  core 0 and core 31 (~16 hops) costs the same as one between core 0 and
  core 1 (1 hop).

Why this matters: restickify is one of the dominant data-movement
primitives in the backend. Most non-trivial models incur multiple
restickifies per layer (post-matmul transpose, attention reshape,
elementwise fusions across mixed layouts, etc.). Even modest improvements
in per-restickify ring traffic should compound across a model.

The architectural gap is that the existing cost model **cannot** include
ring hops, because:

* Restickify costs are computed at `optimize_restickify_locations`, which
  runs **before** `work_distribution`.
* Core-id-to-slice mappings (the inputs needed to compute ring hops) are
  finalized at `work_distribution`, **after** restickify placement is fixed.

```
propagate_layouts          attach candidate STLs + restick cost fn
        ↓
optimize_restickify_loc    pick committed_stl per op  (cost = elements)
        ↓
finalize_layouts           record restickify_plan
        ↓
insert_restickify          splice restickify ops into FX
        ↓
span_reduction
        ↓
work_distribution          ← core mappings finalized here
        ↓
k_fast_override
        ↓
codegen reads core_id_to_work_slice
```

This RFC proposes closing that gap.

## Proposed Implementation

The proposal is structured in **four phases**, each gated on evidence from
the previous one. Phase 0 and Phase 1 are firmly in scope. Phases 2 and 3
are presented for design discussion but should not be committed to until
Phase 1 data is in hand.

### Phase 0 — Telemetry (~2 days)

Goal: confirm there's signal worth optimizing, and produce a top-K list of
expensive restickifies on real models.

Add a **read-only diagnostic pass** that runs after `work_distribution` and
before codegen. For each restickify op, the pass:

1. Resolves the producer's `core_id_to_work_slice` and the consumer's
   `core_id_to_work_slice`.
2. Computes an estimated `ring_hop_cost` = `Σ_c bytes_consumer_c ×
   hops(producer_of_slice_c, c)` where `hops(a, b) = min(|a - b|, 32 - |a - b|)`
   (signed ring distance).
3. Logs `restickify_ring_cost(op_name) = total_bytes, total_hops,
   estimated_ring_us`.

A second logging point (in `optimize_restickify_locations`) records the
**element-count cost** used today, so we can compare the two cost models on
the same data.

**Deliverables**:

* New file: `torch_spyre/_inductor/restickify_telemetry.py` (~150 lines)
* Hook: `passes.py` schedules the pass after `k_fast_override`.
* Output: structured log + optional JSONL via `SPYRE_RESTICKIFY_TELEMETRY=path`.
* Profile run: Granite 3.3 8B forward pass; table of top-10 restickifies by
  estimated ring cost; estimate `total_restickify_ring_time / total_kernel_time`.

**Kill criterion**: if total estimated restickify ring time is under 2% of
total kernel time on production models, descope to telemetry-only and close
the project.

### Phase 1 — Producer-aligned consumer mappings (~1 week, MVP)

Goal: deliver the smallest behavior change that reduces ring traffic.

Modify `superdsc.py:_get_core_to_slice_mapping` to accept an optional
`producer_mapping` hint:

```python
def _get_core_to_slice_mapping(
    iteration_space: dict[Symbol, Expr],
    dim_splits: dict[Symbol, int],
    num_cores: int,
    producer_mapping: dict[Symbol, Expr] | None = None,  # NEW
) -> dict[Symbol, Expr]:
    """If a compatible producer_mapping is provided, align consumer cores
    to it so restickify between them is on-core (or short-hop). Falls back
    to the default row-major mapping when alignment is infeasible.
    """
```

A new pre-codegen pass (`align_consumer_mappings`) walks ops in topological
order. For each op that is a direct consumer of a producer with a known
mapping, it checks whether the producer's iteration-space dims are a subset
of the consumer's. If yes, it passes the producer's mapping as a hint.

**Alignment rule (initial, conservative)**:

* Both ops have the same `iteration_space` keys (allowing extras on the
  consumer side).
* The consumer's split factors for shared dims equal the producer's split
  factors.
* Producer's `core_id_to_work_slice` is reusable as-is (the consumer's
  extra dims, if any, become innermost).

When the rule doesn't match, fall back to the default. **No behavior change
on non-matching ops.**

Gate behind a new config flag: `align_restickify_consumer_mappings`,
defaulting to `False` for initial rollout. Flip to `True` after Phase 1
benchmarks pass.

**Deliverables**:

* Modified `superdsc.py` (~30 lines).
* New pass `align_consumer_mappings` (~80 lines) in `work_division.py` or
  new file `torch_spyre/_inductor/mapping_alignment.py`.
* Config flag in `config.py`.
* Unit tests in `tests/inductor/test_inductor_ops.py` covering at least:
  * matmul → pointwise → restickify chain (alignment fires)
  * matmul → reduction (different cohort, alignment skipped)
  * bmm → matmul (3-vs-4 dim, alignment skipped)
* Re-run Phase 0 telemetry to confirm reduction in `total_ring_us`.

**Success criterion**: measurable reduction in `total_ring_us` on the top
restickifies identified in Phase 0, with **zero regressions** on the 20-shape
end-to-end benchmark from k_fast PR.

### Phase 2 — Post-pass restickify re-optimization (~2–3 weeks, conditional)

Goal: re-evaluate **placement** of restickifies given the actual core mappings.

This phase is conditional on Phase 1 evidence. If Phase 1's producer
alignment captures most of the available win, Phase 2 may not be worth the
complexity.

Approach: after `work_distribution`, run a second optimization pass that:

1. Re-costs each restickify with ring hops as the dominant term.
2. Considers moving restickifies earlier (closer to producer) or later
   (closer to consumer) if doing so reduces total ring cost.
3. Considers cancelling redundant restickifies that were inserted under the
   element-count cost model but turn out to be no-ops under the ring-cost
   model.

The risk here is interaction with the existing greedy / beam-search
optimizer ([optimize_restickify.py:281-492](../../../../torch_spyre/_inductor/optimize_restickify.py)).
Adding ring cost as a fourth search dimension could blow up the K=64 beam.

**Defer until Phase 0 + 1 data confirms it's needed.**

### Phase 3 — Cross-op work-division coordination (out of scope)

Goal: the full joint optimization: choose `core_id_to_work_slice` for each
op such that total ring traffic across the whole graph is minimized.

This is the architecturally pure answer but a multi-week refactor of the
work-division pass. It is **explicitly out of scope** for this RFC. If
Phases 0–2 leave large wins on the table, this is the natural next step.

## Metrics

Primary metrics:

* **Total restickify ring time / total kernel time** on Granite 3.3 8B M ∈
  {32, 128, 512} forward pass. Baseline measured in Phase 0; target 50%+
  reduction post-Phase 1.
* **End-to-end matmul + restickify kernel latency** on the 20-shape suite
  from k_fast PR. Target: zero regressions, at least 5 shapes show ≥ 5%
  improvement.

Secondary metrics:

* Number of restickifies inserted, before and after.
* Average ring hops per restickified stick.

## Drawbacks

* **Adds complexity to the pass pipeline.** A new alignment pass and a new
  telemetry pass mean two more places that downstream changes have to
  preserve.
* **Increases coupling between work_division and restickify.** Today these
  are separable; Phase 1 introduces a directed dependency from
  `_get_core_to_slice_mapping` to upstream `producer_mapping`.
* **Risk of interaction with concurrent layout work.** The scratchpad
  refactor [#1941](https://github.com/torch-spyre/torch-spyre/pull/1941)
  just landed; further restickify-area work is in flight. This RFC needs to
  align with that direction.
* **Cost model fidelity.** Hops × bytes is a first-order approximation. It
  ignores ring contention (multiple concurrent restickifies sharing the same
  ring segment) and the actual ring topology details. May need a calibration
  probe similar to PSUM ring cost.

## Alternatives

1. **Pipeline reorder.** Pull `work_distribution` before
   `optimize_restickify_locations` so the existing optimizer can include
   ring cost. Rejected as Phase 3 / out-of-scope — too disruptive given the
   active layout-pass refactoring.

2. **Pure observability** (Phase 0 only). Ship telemetry, then iteratively
   patch hot spots by hand. Considered as a fallback if Phase 0 evidence is
   weak.

3. **Static mapping templates.** Define a small set of canonical mappings
   (row-major, K-cohort-adjacent, N-cohort-adjacent, ...) and have the
   planner pick the best per op. Less flexible than producer-aligned but
   simpler to reason about. May converge to roughly the same outcomes for
   matmul-heavy graphs.

4. **Doing nothing.** The cost is real but not catastrophic — k_fast PR's
   end-to-end measurements include the unoptimized restickify cost and
   still show 1.73× geomean. The opportunity cost of inaction is bounded
   by the Phase 0 telemetry.

## Prior Art

* **k_fast PR ([#1986](https://github.com/torch-spyre/torch-spyre/pull/1986))**:
  the direct analogue on the SFP ring. Same general technique (choose
  physical core IDs to minimize ring traversal) applied to PSUM reduction
  instead of restickify data movement. Provides the methodology for
  telemetry, A/B/C decomposition, and benchmarking that this RFC reuses.

* **GPU literature on ring-allreduce** (e.g. NCCL): well-established that
  ring topology matters for collective ops; choosing ranks adjacent on the
  ring saves bandwidth. Spyre's restickify is structurally similar to a
  reshape-and-allgather.

* **TPU XLA's collective-permute optimization**: when XLA emits a
  `collective-permute` for transpose-on-mesh, it chooses source/dest pairs
  to minimize on-mesh distance. Spyre's restickify is the analogous op
  on the AIU's ring.

## How we teach this

The user-facing concept is `core_id_to_work_slice` and the principle that
**adjacent cores on the ring share data more cheaply than distant cores**.
This will be added to:

* `docs/source/compiler/work_division_planning.md` — explain the
  producer-alignment rule and when it fires.
* `docs/source/getting_started/how_torch_spyre_works.md` — mention ring
  locality as a backend optimization concern.

A new diagram in the existing `fig4b-sdsc-example.svg` style, showing the
producer/consumer alignment, will be added.

## Unresolved questions

* **What's the right alignment policy for ops with mismatched iteration
  spaces?** Phase 1 picks the conservative "exact match" rule, but the
  more interesting cases (matmul → transpose → matmul, attention
  reshape) involve dim permutations. The right rule probably involves
  computing a per-stick "source core" function and matching that
  function across producer/consumer.
* **How does this interact with bmm support?** k_fast currently excludes
  bmm. Restickify is heavily used in bmm-driven attention; the
  alignment rule has to define a sensible bmm policy.
* **Should the alignment pass run before or after `k_fast_override`?**
  `k_fast_override` changes matmul mappings, so the alignment pass needs
  to see post-override mappings to align correctly. Order: span_reduction
  → work_distribution → k_fast_override → align_consumer_mappings.
* **What's the cost model fidelity ceiling?** First-order hops × bytes
  may underestimate cost in high-contention scenarios. A calibration
  probe (one shape, one restickify, varying producer/consumer distance)
  would establish whether the linear model holds.

## Resolution

To be filled in after review.

### Level of Support

To be filled in after review.

#### Tracking issue

To be filed.

#### Next Steps

1. Circulate this draft to the layout-team owners (Olivier and the team that
   landed PRs #1941, #1989) for early feedback on whether Phase 1
   conflicts with their direction.
2. If green-lit, implement Phase 0 telemetry and produce the top-10
   table on Granite 3.3 8B.
3. Convert this draft into a formal RFC in the
   [torch-spyre/rfcs](https://github.com/torch-spyre/rfcs) repo with a
   formal number.
