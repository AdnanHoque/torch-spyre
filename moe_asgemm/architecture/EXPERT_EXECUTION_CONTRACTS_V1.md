# Expert execution contracts v1

## Document status

```text
status                 proposed
contract version       1 draft 0
implementation status  none
prototype authority    none
sign-off status        pending
```

This document defines the proposed inter-phase contracts for MoE expert
execution.  It does not authorize a production branch.  The contract becomes
an implementation authority only after the sign-off gate in
`moe_asgemm/architecture/PHASE1_REVIEW_AND_SIGNOFF.md` is complete.

## Design rule

Execution strategy is explicit data chosen once.  Every compiler phase has one
decision class, consumes an immutable input contract, and produces a new
immutable output contract.  A downstream phase may accept or reject its input;
it may not silently patch an upstream decision.

The staged documents are:

```text
MoEFFNSemanticsV1
  -> ExpertExecutionIntent
  -> TemporalLoopPlan
  -> DivisionPlan
  -> PlacementPlan
  -> LoweredExecutionPlan
  -> BackendExpertExecutionContractV1
```

Every document carries:

```text
schema_version
plan_attempt_id
parent_digest
semantic_fingerprint
producer_phase
```

The parent digest makes a phase transition independently testable and prevents
in-place refinement from becoming a new hidden mutation channel.

## 1. MoEFFNSemanticsV1

### Owner

Model/compiler semantic boundary.

### Purpose

Represent a bounded family of MoE FFN equations without selecting an execution
strategy.

### Required fields

```text
schema_version = 1

topology.kind
  gated
  plain

topology.activation
  gelu_tanh
  silu
  explicitly_registered_v1_activation

dimensions
  experts E
  tokens T
  hidden H
  intermediate F when topology is gated
  top_k K

weights
  gated: gate, up, down
  plain: projection schema declared by topology version

routing
  weights tensor
  logical shape
  weighting position
  selected-expert semantics

numeric
  input dtype
  weight dtype
  accumulator/reference policy
```

### Semantic rule for gated v1

```text
Y[t,h] = sum over e of alpha[e,t,0] * (
    activation(X[t,:] @ Wg[e]) * (X[t,:] @ Wu[e])) @ Wd[e]
```

The explicit singleton in `alpha[E,T,1]` is a physical ABI choice only after a
strategy is selected.  The semantic routing representation may begin as
`[T,E]`; the reference decomposition defines the exact conversion.

### Reference requirement

Every accepted semantic descriptor must have an eager decomposition that is:

- the CPU fallback;
- the silicon-independent correctness oracle;
- independent of execution strategy; and
- versioned with the descriptor.

### Rejection rule

Unknown topology, activation, weighting position, or routing semantics must
fail at the semantic boundary.  V1 does not carry arbitrary expert subgraphs.

## 2. ExpertExecutionIntent

### Owner

Expert execution strategy selector and cost model.

### Purpose

Choose a strategy and state requirements without choosing a concrete work
division, allocation, address, or backend schedule.

### Required fields

```text
strategy
  per_token
  dense_activation_stationary
  active_dense
  grouped
  ordinary_fallback

selection_reason
machine_constant_set_id
measurement_provenance_id
debug_override if present

loop_requirement
residency_requirements
ownership_constraints
binding_requirements
combine_requirement
```

For dense activation-stationary v1, the requirements are:

```text
static temporal expert loop
X loop-invariant in LX
output accumulator loop-carried in LX
Wg, Wu, Wd streamed from HBM
runtime alpha streamed from HBM and applied after down
equal token-row ownership across the activation path
one final output drain
```

The selector records requirements, not the physical answer.

## 3. TemporalLoopPlan

### Owner

Coarse tiling and temporal loop formation.

### Purpose

Turn the selected strategy into explicit temporal execution state.

### Required fields

```text
loop_id
loop_domain
trip_count_source
trip_count
induction_variable

invariant_operands
streamed_operands
loop_local_values
carried_reductions
preheader_operations
postloop_operations
```

### InvariantOperandPlan

```text
operand_id
logical_view
copy_scope
owning_loop_id
required_residency
lifetime = preheader_start through loop_end
```

### StreamedOperandPlan

```text
operand_id
binding_requirement_id
consumer_group
logical_view_per_iteration
required_residency = HBM_streamed for v1 weights and alpha
```

### CarriedReductionPlan

```text
reduction_id
identity
contribution
state_shape
state_dtype
fill_scope
combine_scope
drain_scope
required_residency
```

The fixed accumulator transformation is additive and plan-gated.  Flat
reductions with no `CarriedReductionPlan` retain legacy behavior.  Same-loop
consumers are not globally prohibited.

## 4. LoopOperandBinding

### Owner

Temporal planning defines the requirement.  Lowering resolves it.  Codegen
consumes it.  The backend contract receives it.

### Required fields

```text
binding_id
operand_id
loop_id
kind
address_unit
base_source
trip_count_source
bounds
```

### SequentialAffineBindingV1

```text
kind = sequential_affine
base_source
induction_variable
step
address_unit = device_elements or bytes, never implicit
minimum_offset
maximum_offset
```

### IndexedBindingV1 reservation

The schema reserves, but v1 does not implement:

```text
kind = indexed
index_operand
index_dtype
index_bounds
table_base
table_stride
```

Unsupported binding kinds must fail.  They must never become a fixed-base
operand.  Codegen may derive an address expression only from the authoritative
binding.  TensorArg analysis verifies the emitted expression; it does not
independently define the binding.

## 5. DivisionConstraint and DivisionPlan

### Owner

Work divider and cost model.

### Constraint input

```text
constraint_id
kind = equal_logical_axis_ownership
logical_axis = token_rows
operation_group
required_core_count if fixed by strategy
legal_mapping_properties
```

The constraint states that an operation group must have equal row ownership.
It does not state `M32`, a physical core order, or a split answer.

### DivisionPlan output

```text
operation_work_divisions
physical_core_maps
constraint_satisfaction_proof
cost_model_terms
rejected_candidates and reasons
```

Work division is the sole author of these choices.  Relayout and scratchpad
phases may verify them or reject the plan.  They may not replace them.

## 6. PlacementPlan

### Owner

Scratchpad planner and allocator.

### Required fields

```text
allocations
  buffer
  memory kind
  address
  size
  alignment
  owner core map
  lifetime interval

relayouts
capacity proof
alias proof
residency requirement satisfaction
```

Loop-carried lifetime facts come from `TemporalLoopPlan`.  The allocator may
realize them through a general lifetime-end mechanism.  It must not infer them
from workload-specific markers.

Placement failure returns `PlanFailure`; it never rewrites the division,
strategy, or loop plan.

## 7. LoweredExecutionPlan

### Owner

Torch-Spyre lowering and code generation preparation.

### Required fields

```text
resolved_loop_specs
resolved_operand_bindings
resolved_core_maps
resolved_allocations
operation_sequence
backend_capability_requirements
all parent plan digests
```

This is a compiler-internal artifact.  It is not the backend ABI.

## 8. BackendExpertExecutionContractV1

### Owner

Joint frontend/backend contract.  The backend owns execution after acceptance.

### Required fields

```text
contract_version
loop bounds and trip-count source
ordered operation invocations
operand bindings with explicit address units
physical core maps
memory allocation kinds and addresses
required backend capabilities
semantic fingerprint
lowered-plan digest
```

The backend accepts or rejects the contract as a unit.  It must not silently
drop a loop binding, change a core map, or substitute a fixed operand base.

## 9. PlanFailure and selector re-entry

### Failure document

```text
plan_attempt_id
strategy
failed_phase
reason_code
human_message
recoverable
failed_predicate
relevant capacity or ownership values
parent_plan_digest
```

Initial reason codes include:

```text
unsupported_semantics
unsupported_binding_kind
unsupported_backend_capability
unrepresentable_work_division
unrepresentable_ownership
lx_capacity
alias_conflict
legacy_behavior_conflict
```

### Re-entry rule

On recoverable failure:

1. preserve the failed immutable plan attempt;
2. emit a degradation event;
3. return to the selector with the failure document;
4. choose a different strategy or fail compilation; and
5. create a new `plan_attempt_id`.

No downstream phase may patch an upstream plan.  Retries are bounded and a
strategy may not be attempted twice for the same semantic fingerprint and
machine-constant set.

## 10. Provenance contract

Every compilation emits:

```text
semantic_descriptor_created
strategy_selected
plan_stage_accepted or plan_stage_rejected
strategy_degraded when applicable
backend_contract_accepted or backend_contract_rejected
```

Each event contains:

```text
timestamp
semantic_fingerprint
plan_attempt_id
strategy
phase
reason code
input and output plan digests
compiler source identity
machine constant set identity
```

The final result exposes the selected strategy, every rejected strategy, and
the exact reason for fallback.  Logging verbosity may vary; the structured
provenance artifact may not disappear.

## 11. Versioning

1. Semantic and backend contract versions are explicit integers.
2. Unknown major versions fail closed.
3. Additive optional fields require documented defaults that do not change
   semantics.
4. New topology or binding kinds require a minor schema revision and named
   capability negotiation.
5. A field may be removed only in a new major version.
6. Serialized plans include all version numbers and source identities.

## 12. V1 acceptance invariants

For the dense activation-stationary strategy, acceptance requires:

- one semantic descriptor and one selected strategy;
- one temporal expert loop and no model-authored strategy operation;
- one invariant activation preheader;
- authoritative sequential affine bindings for Wg, Wu, Wd, and alpha;
- one divider-produced row-ownership plan satisfying the group constraint;
- invariant activation and carried accumulator in LX;
- zero internal HBM-pool intermediates;
- zero HBM restickify operations;
- runtime routing weights applied after down;
- one final output drain;
- complete plan provenance; and
- correctness against the eager semantic decomposition.

Performance acceptance is a matched remeasurement against the frozen baseline
form on one exact software and hardware stack.  No literal historical latency
is a permanent gate.

## 13. Architecture litmus test

Adding active-dense must require:

- a selector strategy;
- an authoritative binding kind or trip-count capability; and
- no model-specific source changes or duplicate loop, placement, or binding
  infrastructure.

Adding grouped execution may require routing preparation, combine semantics,
and a backend primitive.  It must still reuse the semantic, selector, staged
planning, failure, placement, binding, and provenance infrastructure.
