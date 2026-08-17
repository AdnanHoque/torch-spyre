# Implementation base decision

## Status

```text
decision  pending cross-owner review
default   do not create the production implementation branch
```

This decision is intentionally deferred until the contracts are reviewed.  A
branch created earlier would make the existing code, rather than the signed
contract, the de facto architecture.

## Option A: then-current upstream main

### Benefits

- Starts from the code owners' current abstractions.
- Avoids carrying prototype-specific compatibility debt.
- Makes differential legacy tests meaningful on the target codebase.
- Produces reviewable general mechanisms rather than a long-lived fork.

### Costs

- Requires re-measuring the frozen baseline form on the new SHA.
- May require adapting sound prototype mechanisms to changed compiler APIs.
- Contract work must complete before implementation starts.

### Assessment

Recommended default after sign-off.

## Option B: docs-approved private beta fork from then-current main

### Benefits

- Allows staged integration before requesting upstream review.
- Gives frontend and backend owners a shared branch for schema experiments.
- Can keep incomplete contract consumers away from ordinary main behavior.

### Costs

- Accumulates rebase cost for every upstream change.
- Risks becoming a second product branch.
- Can hide compatibility failures until late.
- Requires an explicit lifetime and merge/abandon criterion.

### Required price if selected

```text
named branch owner
weekly rebase or merge policy
maximum fork lifetime
exit criteria
prohibited production dependencies
matched-main validation cadence
```

### Assessment

Acceptable only if several owners need a shared integration surface before
upstreaming.  It must be time-bounded.

## Option C: refactor prototype-v0 in place

### Benefits

- Lowest immediate coding effort.
- Preserves the already measured program with fewer intermediate regressions.

### Costs

- Preserves the post-hoc scheduler, reduction hazard, attribute transport, and
  dual read-copy dialect as compatibility constraints.
- Makes it difficult to prove that legacy behavior is unchanged.
- Encourages architecture to follow existing code rather than signed contracts.
- Weakens prototype-v0 as an immutable oracle.

### Assessment

Rejected.

## Decision record template

```text
selected option
decision date
contract revision and digest
approvers
base commit
expected fork lifetime if applicable
matched oracle protocol
conditions
```

## Measurement rule

Whichever base is selected, acceptance compares on one exact stack:

```text
frozen baseline graph form rebuilt on selected SHA
new contract-driven implementation on the same SHA
same native extension
same DeepTools stack
same AIU
same deterministic tensors
same timing protocol
```

The historical 42.444 ms value is a reference, not the pass/fail threshold.
