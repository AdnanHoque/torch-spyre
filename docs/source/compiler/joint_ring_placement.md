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

Represent the latter as a permutation `P(c)`. Changing `P` preserves the
mathematics only when the change covers a closed region: every unshuffled local
producer/alias/consumer edge must retain one realized mapping, and every mapping
change must cross an explicit materialized bridge. A legal change alters
source-destination distance, directed-link hotspots, and opportunities for local
delivery without changing which logical bytes an operation sees.

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

Contiguity is valuable only when every actual producer of a locally reused LX
view, the SHUFFLE endpoints, aliases, and downstream consumers agree on the same
embedding. Moving or relabeling one endpoint is not merely a weak optimization;
without a materialized bridge, it can reinterpret bytes and change values.

## What the experiment established

The original `joint_all` prototype was not a closed region. It changed the
synthetic K relayout-source view but left the actual scaled-K producer at the
default mapping, even though both accessed the same local LX allocation without
a transfer between them. A strengthened high-contrast device test found
241,384/262,144 mismatches at `atol=rtol=1e-2` and maximum absolute error
`0.943985`. Its historical timing is therefore invalid for any performance
claim.

This negative is itself useful: a source-view annotation cannot define physical
truth independently of the operation that produced the bytes. The earlier
low-amplitude allclose input and CPU-only wrong-route negative did not expose
that boundary error.

The corrected experiment established closure incrementally:

- relayout source alone failed with the same high-contrast corruption;
- actual scaled-K producer plus source view passed;
- query producer, first BMM, score path, and second BMM passed as an independent
  destination-side closed region; and
- one coherent mapping across all six points passed full-device correctness in
  both compilation orders and an adversarial V-coded test sensitive to head,
  token, channel, second-BMM ingress, and output ordering.

The corrected coherent candidate retained the modeled route improvement:

| Metric | Default | Coherent | Change |
|---|---:|---:|---:|
| Remote relations | 224 | 224 | 0 |
| Total hop units | 2,048 | 672 | -67.2% |
| Maximum directed-link units | 40 | 16 | -60.0% |

Those values are equal-payload shortest-path proxies derived from emitted maps,
not measured link traffic. The corrected five-block placement-by-handoff
factorial supplies the elapsed-time evidence:

| Condition | Mean process median |
|---|---:|
| LX default | 210.7515 us |
| Oracle default | 180.8223 us |
| LX coherent | 199.0549 us |
| Oracle coherent | 181.8865 us |

Coherent placement reduced observed LX whole-kernel time by `11.6966 us`, t95
`[10.7843, 12.6089]`, or **5.550% / 1.05877x**. The preregistered
difference-in-differences was `12.7608 us`, t95 `[11.9086, 13.6130]`, closing a
mean **42.6177%** of the default graph-level handoff residual. The coherent
residual is `17.1684 us`; its graph is at **91.3749%** of paired-oracle
inverse-time performance and has at most **1.09440x** further graph-level
speedup. None of these numbers is a SHUFFLE-root timer or device-peak ring
utilization. The t95 intervals are descriptive process-block intervals from one
device, not inference across independent devices.

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
- Placement closure does not yet follow the complete producer/alias/allocation
  chain. The corrected experiment showed that the real scaled-K producer can be
  outside a superficially plausible score region while still defining the
  bytes read by its synthetic view. Region construction must prove mapping
  compatibility on every local no-transfer edge before cost ranking.

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
- every actual producer, alias, transpose/view, and allocation user on a local
  no-transfer path;
- the SHUFFLE source and destination maps;
- consumers that reuse the destination allocation;
- in-place or shared-allocation chains; and
- boundary traffic whose cost can regress when the region moves.

Use explicit graph dependence and allocation relationships, not Flash-specific
operation names. A region may split only at an explicit materialized bridge, a
fixed external boundary, or an edge whose exact per-buffer physical ownership
and byte maps are already equivalent. An incompatible active-core set,
externally fixed placement, unsupported backend contract, or unprovable edge
therefore rejects the candidate; it must never create an incoherent local-LX
split.

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
- every unshuffled local-LX producer-to-consumer edge has the same realized
  per-buffer physical ownership and byte map at both ends; equality of only the
  operation-level placement tuple is insufficient;
- every mapping change crosses an explicit materialized bridge with compatible
  source and destination maps;
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

- The currently realized default, including K-fast behavior, emits byte-identical
  baseline maps when the optimizer declines a candidate.
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

- Compare full device outputs against CPU and default using high-contrast data,
  not only near-uniform softmax inputs.
- Check the complete destination window and source immutability.
- Require exact SHUFFLE count and allocation maps.
- Compile ordinary and oracle-prefixed graphs in both orders and require stable
  identities, so cached placement state cannot create a false pass.
- Encode head, token, and channel in V to expose second-BMM ingress or output
  permutations.
- Preserve a source-only negative and require actual-producer-plus-source
  closure to restore correctness.
- Preserve the wrong-route negative so a no-op or stale-route implementation
  cannot pass by coincidence.

### Performance tests

- Run counterbalanced fresh-process default/coherent pairs.
- Treat correctness, structure, trace quality, and provenance as execution
  gates; report the preregistered performance effect regardless of sign.
- Track compile time, memory fallback frequency, and unrelated-kernel
  regressions.
- Use a placement-by-handoff factorial against both default and coherent
  no-SHUFFLE oracles before claiming how much oracle residual placement closes.

## Rollout plan

1. **Library and shadow mode.** Land pure route/cost types and compute candidate
   decisions without changing code generation.
2. **Typed backend round-trip.** Add active-core IDs, requested/realized
   placement and route serialization, and fail-closed capability checks. Torch
   currently emits ownership maps but cannot observe the realized STCDP
   direction, path, or multicast sharing; route-based selection remains shadow
   only until this round-trip exists.
3. **Allowlisted Flash A/B.** Reproduce the corrected coherent closure as a typed
   compiler region—not environment-driven test hints—under an opt-in flag.
4. **Root-scoped decomposition.** The placement-by-handoff factorial is now
   complete; add transfer/sync/consumer markers to localize the remaining
   `17.1684 us` graph residual.
5. **Automatic selection.** Enable the candidate search for supported regions
   only when shadow prediction matches measured ordering and the expected win
   exceeds its guardband.
6. **Broaden collectives.** Add native multicast, dual injection, and other
   shuffle shapes one calibrated transport kind at a time.

## Immediate next implementation slice

The smallest production-quality slice is:

1. add pure `RingTopology`, `PhysicalCorePlacement`, route-load, and cost types;
2. adapt `LXRelayoutPlan` source/destination maps into exact transfer demands;
3. implement the currently realized default plus
   dimension-major/rotation/reflection candidates;
4. run the planner between work distribution and scratchpad planning in shadow
   mode;
5. emit a complete decision record; and
6. validate its predicted default-versus-coherent ordering against the
   packaged structural and timing evidence.

Only after that shadow result is stable should the pass be allowed to mutate
placement metadata. This keeps the current LX-relayout substrate as the common
foundation for Flash and later collectives instead of embedding another
Flash-only scheduling exception.
