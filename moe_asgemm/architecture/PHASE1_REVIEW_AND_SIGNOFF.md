# Phase 1 review and sign-off

## Gate status

```text
design drafted             complete
prototype frozen           complete
frontend sign-off          pending
work-division sign-off     pending
allocator sign-off         pending
runtime/codegen sign-off   pending
backend sign-off           pending
model-path sign-off        pending
implementation authorized  no
```

Review target:

```text
contract path    moe_asgemm/architecture/EXPERT_EXECUTION_CONTRACTS_V1.md
revision         v1 draft 0
SHA-256          8cb856a482bd7b928de7a6a199d981c2251aa1a73be9ac90db2a94d0d5b882cf
```

No sign-off is inferred from prior conversations, repository ownership, code
review, or prototype participation.  A reviewer must record an explicit
decision against the contract revision and digest.

## Proposed review roles

### Semantic and strategy framing

Proposed reviewers:

```text
Mudhakar
Antoni
Swagath
```

Questions:

- Does `MoEFFNSemanticsV1` describe the intended Step 1, Step 2, and future
  Step 3 decision space without selecting a strategy?
- Is the bounded topology descriptor sufficient for the known model family?
- Is the eager decomposition the authoritative semantic definition?

### Torch-Spyre frontend and lowering

Proposed reviewer pool from current repository ownership:

```text
dgrove-oss
avery-blanchard
cyang49
marnold-ibm
tardieu
```

Questions:

- Are the staged immutable boundaries consistent with the current compiler?
- Is the semantic operation appropriately bounded?
- Can shared-LHS contracts remain internal lowering targets?
- Does plan-gated temporal reduction preserve legacy behavior?

### Work division and cost model

Designated owner: pending confirmation.

Questions:

- Is `DivisionConstraint` an acceptable new input to the divider?
- Does equal logical row ownership express a requirement without embedding the
  prototype's M32 answer?
- Which cost-model registry owns measured machine constants and provenance?
- How are unsatisfiable constraints reported without a post-hoc scheduler?

### Scratchpad planning and allocation

Proposed reviewer pool:

```text
avery-blanchard
dgrove-oss
tardieu
```

Questions:

- Are explicit lifetime intervals and residency requirements sufficient?
- Is the general lifetime-end override acceptable?
- Can placement return structured failure without mutating upstream plans?
- Are capacity and alias proofs serializable enough for offline tests?

### Code generation and runtime

Proposed reviewer pool from current runtime ownership:

```text
JRosenkranz
thoangtrvn
ani300
```

Questions:

- Is `LoopOperandBinding` authoritative at the correct stage?
- Are address units, bounds, and trip-count sources explicit enough?
- Is selected physical core-map propagation represented safely?
- Where should structured plan provenance be emitted and retained?

### DeepTools/backend contract

Proposed reviewers:

```text
Swagath
designated DXP/DDC owner pending confirmation
designated DDL owner pending confirmation
```

Questions:

- Is `BackendExpertExecutionContractV1` the smallest stable ABI surface?
- Can sequential affine bindings be accepted and verified as specified?
- What capability negotiation is needed for indexed binding and dynamic trip
  count later?
- Which failures are frontend-recoverable and which are fatal backend errors?

### Model-path integration

Proposed reviewers:

```text
Antoni
Swagath
```

Questions:

- Can the adapter invoke only the semantic operation?
- Are routing weights, topology, K, and weight packing represented without
  selecting dense, active-dense, or grouped execution?
- What model behavior must remain outside compilation?

## Sign-off ledger

Record one row per role.  Names below are proposals, not approvals.

| Role | Required reviewer | Revision | Decision | Date | Conditions |
|---|---|---|---|---|---|
| Semantics and strategy | Mudhakar or delegate | v1 draft 0 | pending | — | — |
| Model path | Antoni and Swagath or delegates | v1 draft 0 | pending | — | — |
| Frontend and lowering | named compiler owner | v1 draft 0 | pending | — | — |
| Work division and cost model | designated owner | v1 draft 0 | pending | — | — |
| Scratchpad allocation | designated owner | v1 draft 0 | pending | — | — |
| Codegen and runtime | designated owner | v1 draft 0 | pending | — | — |
| DeepTools/backend ABI | designated backend owner | v1 draft 0 | pending | — | — |

Accepted decisions are:

```text
approve
approve with enumerated conditions
request revision
reject
```

Silence is not approval.  Approval against one revision does not transfer to a
different contract digest.

## Required decisions before implementation

1. Approve the semantic descriptor boundary and versioning policy.
2. Approve staged immutable plan documents instead of one mutable plan.
3. Approve selector re-entry as the only recovery path.
4. Approve `DivisionConstraint` as input to the existing divider.
5. Approve `LoopOperandBinding` as authoritative and non-optional.
6. Approve the narrow backend ABI projection.
7. Approve structured selected/declined/degraded provenance.
8. Choose the implementation base using
   `moe_asgemm/architecture/IMPLEMENTATION_BASE_DECISION.md`.

## Completion rule

Phase 1 completes only when all required roles have approved the same contract
revision and the implementation-base decision is recorded.  Until then:

- no production implementation branch is created;
- no prototype policy is promoted as architecture;
- no model adapter is changed to a new strategy boundary; and
- contract experiments remain docs or isolated throwaway tests.
