# Joint producer-consumer placement for ring communication

Joint placement is the compiler decision that maps a connected set of logical
workers to physical ring cores together, instead of choosing each operation's
physical embedding independently. It changes where work executes while keeping
the logical tensor slices and numerical program unchanged.

The measured motivation and experimental protocol are in
[LX ring measurement methodology](lx_ring_measurement_methodology.md). Candidate
placements should be ranked with the
[communication cost model](communication_cost_model.md).

## Why placement matters

Work division answers:

```text
Which logical tensor slice does worker c own?
```

Physical placement answers:

```text
At which physical ring position does worker c execute?
```

Represent the latter as a permutation `P(c)`. Changing `P` does not change the
mathematics, but it changes every source-destination distance, directed-link
hotspot, and opportunity for local delivery.

In the measured Flash case, the default embedding interleaves four attention
heads around the 32-core ring:

```text
head = physical_core % 4
```

The experimental inner-first embedding makes each eight-core head cohort
contiguous:

```text
head = physical_core // 8
```

Contiguity is valuable only when the producer, SHUFFLE destination, and
downstream consumers agree on the same embedding. Moving just one endpoint does
not optimize a relational communication problem.

## What the experiment established

The source-only placement experiment showed no detected timing win, while total
modeled hop units remained 2,048. It therefore did not support source-only
placement as the next optimization; this is not an equivalence claim.

The minimal passing `joint_all` candidate applied one common physical embedding
to a closed score path:

- Q-side producer work involved in the first matrix multiplication;
- the synthetic K restickify source;
- the LX SHUFFLE destination and first matrix multiplication;
- score and stable-softmax operations; and
- the second matrix multiplication consuming the probabilities.

Unrelated upstream scaled-K work remained at its default embedding. Structural
gates proved that the targeted operations changed only their physical embedding,
the non-targeted operations stayed semantically unchanged, the expected LX
allocation components remained present, the emitted SHUFFLE maps matched the
adjacent roots, and a deliberately wrong route failed correctness.

| Metric | Default | `joint_all` | Change |
|---|---:|---:|---:|
| Total hop units | 2,048 | 672 | -67.2% |
| Mean remote distance | 9.143 | 3.000 | -67.2% |
| Maximum directed-link units | 40 | 16 | -60.0% |
| Maximum combined-segment units | 64 | 32 | -50.0% |
| Whole fused LX time | 213.527 us | 203.140 us | -10.387 us |

The paired five-block effect was `10.3867 us`, with a descriptive t95 interval
of `[10.0617, 10.7117]`. That is **4.864%**, or **1.051x**, and every block
favored joint placement.

This measurement is deliberately described as a whole-region placement effect.
It is not a causal measurement of SHUFFLE time alone because placement can also
change compute scheduling, locality, and synchronization.

## What exists in the current substrate

The current feature branch already supplies most of the transport and allocation
mechanics needed by a production planner:

- `torch_spyre/_inductor/lx_relayout.py` derives exact producer and consumer
  per-core views and creates an `LXRelayoutPlan`.
- `torch_spyre/_inductor/scratchpad/allocator.py` models S1 and S2 lifetimes,
  allocates them atomically, and falls back without leaving partial LX state.
- `torch_spyre/_inductor/op_spec.py` carries explicit allocation ownership maps
  and fold geometry.
- `torch_spyre/_inductor/spyre_kernel.py` materializes the explicit SHUFFLE
  immediately before the consumer.
- `torch_spyre/_inductor/codegen/superdsc.py` translates allocation maps into
  the backend representation and rejects explicit distributions outside LX.

These pieces answer “how to materialize a legal LX relayout.” They do not yet
answer “which physical embedding minimizes the cost of the connected region.”
That is the production feature described below.

Several seams need hardening before automatic policy can rely on them:

- Physical core mapping is currently limited to today's default/K-fast choices;
  emitted active IDs are contiguous `range(num_cores)`, and OpSpec has no
  general placement field. Generalize the existing shared K-fast helpers into
  one compiler-neutral selector used by per-core-view analysis, the planner,
  and emission. “Default” must mean the mapping actually realized today,
  including K-fast behavior and K-split PSUM traffic—not raw identity.
- Restickify optimization does not retain sufficiently strong semantic operand
  identity for a production operand-scoped placement decision. Carry an exact
  consumer input identity through propagation, optimization, insertion, and
  synthetic restickify construction so the placement contract cannot migrate
  to a merely layout-compatible operand.
- Per-core-view caches currently key work splits, not placement. Add a placement
  fingerprint to each relevant cache key, or make placement an explicit pure
  function input, before evaluating same-split/different-placement candidates.
- `LXRelayoutPlan` and the synthetic S2 name are source-oriented. Add a stable
  edge identity containing source, consumer, operand ordinal/access signature,
  and target view. Otherwise the same buffer used by two operands or readers can
  collide. V1 should retain the existing one-writer/one-reader/one-view boundary;
  generic multi-consumer collectives require edge-scoped S2 names and widened
  lifetime semantics.

The experimental work already demonstrated three prerequisite ideas—typed
root placement, operand-scoped relayout-source placement, and semantic operand
hardening. They should be ported and reviewed as substrate, not enabled as a
manual Flash-specific policy.

## Production architecture

### 1. Add a typed physical-placement contract

Introduce a first-class immutable type, for example:

```python
@dataclass(frozen=True)
class PhysicalCorePlacement:
    active_core_ids: tuple[int, ...]
    logical_to_physical: tuple[int, ...]
    topology_version: str
    reason: str
```

Attach it to operations or to a named placement region. Do not pass it through
environment variables, operation names, or test-only string matching. Validate
that it is a bijection over the active cores, define the inverse mapping used by
emission, and serialize both the active IDs and placement explicitly through the
OpSpec/backend boundary. If that transport is not in the first patch, restrict
V1 to all 32 contiguous active cores and known permutations; smaller or
non-contiguous active sets fail closed.

The backend must return or emit the realized placement. Requested placement is
not enough for either correctness or cost accounting.

### 2. Form closed placement regions

After work division is known, construct dependence regions around typed
communication edges. A region must include every operation whose physical
embedding affects:

- the producer allocation map;
- the SHUFFLE source and destination maps;
- consumers that reuse the destination allocation;
- in-place or shared-allocation chains; and
- boundary traffic whose cost can regress when the region moves.

Use explicit graph dependence and allocation relationships, not Flash-specific
operation names. Split a candidate region when an operation has an incompatible
active-core set, an externally fixed placement, an unsupported backend contract,
or a boundary whose semantics cannot be preserved.

### 3. Generate a bounded candidate set

Start with candidates that have predictable compile cost:

- the currently realized default, including K-fast where applicable;
- dimension-major permutations derived from each operation's work division;
- ring rotations and reflection of those permutations;
- producer-major and consumer-major cohort-contiguous embeddings; and
- previously chosen placements on adjacent closed regions when compatible.

Deduplicate candidates by their exact logical-to-physical tuple. Avoid an
unbounded `32!` search. If needed later, use beam search or CP-SAT over structured
permutations, not arbitrary placement variables in the first release.

### 4. Route exact emitted demands on the global ring

For each candidate:

1. apply the candidate permutation to exact producer and consumer per-core
   ownership maps;
2. build every source-destination relation and its physical byte count;
3. use the realized routing and multicast rules;
4. count clockwise and counterclockwise load on every directed link;
5. count injection and drain bytes independently; and
6. include all region-boundary demands.

Use a single global 32-core ring. A logical eight-core cohort is not an isolated
ring: treating its endpoints as adjacent across a subgroup boundary invents a
physical link that may not exist.

### 5. Rank whole-region cost

Call the pure communication model with exact route loads, then add:

- predicted compute or locality changes;
- synchronization and layout costs;
- spill/memory penalties;
- legal overlap on the dependency critical path; and
- an uncertainty guardband for uncalibrated transport kinds.

Select a non-default candidate only if it is legal and beats default by a
configurable margin. A tie selects default, which prevents churn and keeps the
feature fail-safe while coefficients mature.

### 6. Commit transactionally before LX allocation

The natural pipeline point is after `_distribute_work`, when per-core views are
known, and before `_maybe_scratchpad_planning`, where
`collect_lx_relayout_plans` and the allocator consume those views.

Plan the entire closed region without mutating the graph. Use a non-mutating
allocator feasibility check when capacity or banks affect candidate legality.
Then validate and commit all placement metadata in one transaction. If any
operation, allocation, or backend capability fails, discard the candidate and
preserve the existing default path. Never leave a source changed while its
consumer or S2 allocation falls back.

The current allocator's atomic S1/S2 fallback is the right model for this
transactional behavior.

Late backend failure must also fail closed. Today, late SHUFFLE materialization
raises `Unsupported`; transactional recompile-to-default is not implemented.
Add a capability version and realized-path result before committing the graph,
or implement and test an explicit default retry. Do not surface an experimental
placement failure as a user compilation error, and do not silently time a
fallback as LX.

## Legality contract

A production candidate is legal only when all of these hold:

- the mapping is a bijection over the exact active physical cores;
- logical per-core slices and tensor values are unchanged;
- every targeted operation supports the mapping;
- producer, SHUFFLE, destination, and shared-allocation users agree;
- in-place and lifetime constraints remain valid;
- S1 and S2 remain disjoint and within the usable-LX contract;
- graph inputs/outputs and fixed external interfaces keep valid ownership;
- backend code generation realizes the requested placement and route;
- multicast and direction choices are supported rather than inferred; and
- every boundary edge is accounted for in correctness and cost.

The first release must also preserve the existing relayout gates: representable
views, no partial reductions, supported uniform geometry and matmul operand,
and a single writer/reader/view. Scratchpad graph-boundary clones are inserted
after placement and currently propagate only allowlisted metadata, so placement
must be explicitly propagated to a clone and checked against its first consumer.

Unknown capability is a rejection, not permission to guess.

## Diagnostics required for every compile

Emit a structured record containing:

```text
region members and boundary edges
candidate placements considered
chosen logical-to-physical mapping
requested and realized transport
source and destination allocation maps
CW and CCW per-link byte loads
source injection and destination drain loads
fixed, service, sync, layout, spill, and overlap cost terms
uncalibrated assumptions and guardbands
rejected candidates and reasons
fallback reason, if any
cost-model version and coefficient provenance
```

This record makes a performance decision explainable from first principles and
lets hardware measurements recalibrate the model without reverse-engineering
the compiler's choice.

## Test strategy

### Pure topology and cost tests

- Exhaust small rings and compare route loads with a brute-force reference.
- Prove rotation/reflection invariants.
- Check conservation of injected, drained, delivered, and link-carried bytes.
- Test 16-hop tie behavior explicitly.
- Distinguish expanded unicast from true multicast path sharing.
- Reject group-local wraparound on the global ring.

### Compiler structural tests

- Identity placement emits byte-identical default maps.
- A head-contiguous candidate emits the expected 32-core permutation.
- Closed-region membership includes allocation and in-place dependents.
- Partial support causes an all-or-nothing fallback.
- Requested and realized placement round-trip through code generation.
- Same splits with different placements cannot hit a stale per-core-view cache.
- The realized K-fast/default baseline is byte-identical before optimization.
- Duplicate-buffer two-operand and multi-reader cases reject or receive unique
  edge/view identities.
- Boundary clones preserve placement or force a priced boundary relayout.
- Capacity failure removes the complete relayout rather than one endpoint.

### Hardware correctness tests

- Compare full outputs against the default path.
- Check the complete destination window and source immutability.
- Require exact SHUFFLE count and allocation maps.
- Preserve the wrong-route negative so a no-op or stale-route implementation
  cannot pass by coincidence.

### Performance tests

- Run counterbalanced fresh-process default/joint pairs.
- Require a positive paired effect on the allowlisted Flash case.
- Track compile time, memory fallback frequency, and unrelated-kernel
  regressions.
- Use a placement-by-handoff factorial against both default and joint
  no-SHUFFLE oracles before claiming how much oracle residual placement closes.

## Rollout plan

1. **Library and shadow mode.** Land pure route/cost types and compute candidate
   decisions without changing code generation.
2. **Typed backend round-trip.** Add active-core IDs, requested/realized
   placement and route serialization, and fail-closed capability checks. Torch
   currently emits ownership maps but cannot observe the realized STCDP
   direction, path, or multicast sharing; route-based selection remains shadow
   only until this round-trip exists.
3. **Allowlisted Flash A/B.** Enable only the structurally proven closed region
   under an opt-in flag.
4. **Paired oracle decomposition.** Run the placement-by-handoff factorial and
   add root-scoped transfer/sync/consumer markers.
5. **Automatic selection.** Enable the candidate search for supported regions
   only when shadow prediction matches measured ordering and the expected win
   exceeds its guardband.
6. **Broaden collectives.** Add native multicast, dual injection, and other
   shuffle shapes one calibrated transport kind at a time.

## Immediate next implementation slice

The smallest production-quality slice is:

1. add pure `RingTopology`, `PhysicalCorePlacement`, route-load, and cost types;
2. adapt `LXRelayoutPlan` source/destination maps into exact transfer demands;
3. implement identity plus dimension-major/rotation/reflection candidates;
4. run the planner between work distribution and scratchpad planning in shadow
   mode;
5. emit a complete decision record; and
6. validate its predicted default-versus-`joint_all` ordering against the
   packaged structural and timing evidence.

Only after that shadow result is stable should the pass be allowed to mutate
placement metadata. This keeps the current LX-relayout substrate as the common
foundation for Flash and later collectives instead of embedding another
Flash-only scheduling exception.
