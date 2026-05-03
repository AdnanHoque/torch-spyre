# Project D — Per-corelet work assignment, validated by code reading

## TL;DR

**The lever is real and the impact would be substantial — potentially
2× parallelism on every matmul.** But exploiting it requires
coordinated changes across torch_spyre + the SDSC schema + the
deeptools backend compiler, all of which we'd need to either own or
get cooperation from.

This investigation closes Project D as **infeasible from torch_spyre
alone**. Documenting the finding here so the work can be picked up if
the right cross-team coordination becomes available.

## What we set out to test

The IBM AIU has 32 RaPiD cores, each with 2 corelets (CL0 and CL1)
that share a 2 MB LX scratchpad but have independent PT compute
arrays. Hypothesis: today's planner might engage only one corelet per
core, leaving 32 corelets idle out of 64 physically available.

## What the code says

Three pieces of evidence converge:

### 1. `SENCORES` is hardcapped at 32 cores

[`core_division.py:663-665`](../torch_spyre/_inductor/core_division.py#L663-L665):

```python
max_cores = config.sencores
if max_cores > 32 or max_cores < 1:
    raise Unsupported(f"invalid SENCORES value {max_cores}")
```

The hardware has 64 corelets (32 cores × 2). Hardcap of 32 means we
can't tell the planner "use 64 work slices."

### 2. `numCoreletsUsed_=1` is hardcoded in the SDSC bundle

In [`compute_ops.py`](../torch_spyre/_inductor/codegen/compute_ops.py)
at lines 52, 127, 186, 226, 241, and 315 — **six occurrences**:

```python
{"factor_": 1, "label_": "corelet"},
"numCoreletsUsed_": 1,
```

The schema explicitly tells the backend: use 1 corelet per core. The
other corelet sits idle every kernel call. This is across all six
op-type templates — every primary op torch_spyre emits.

### 3. The planner has no concept of corelets

[`superdsc.py:131-149`](../torch_spyre/_inductor/codegen/superdsc.py#L131-L149)
maps `core_id ∈ [0, num_cores-1]` to work slices. The mapping
function takes a single linear `core_id` and produces dimension slice
indices. There's no place to attach "and which corelet of that core."

The work_division_planning.md docs reference "cores" as the unit of
parallelization throughout — "corelet" is not mentioned at all.

## What it would take to exploit this

To use both corelets per core, three changes would be needed:

| layer | change |
|---|---|
| torch_spyre planner | extend `core_id` to `(core_id, corelet_id)` pairs; index 0..63; track per-corelet splits |
| SDSC schema | extend `coreIdToWkSlice_` to specify per-corelet assignment; remove the `numCoreletsUsed_=1` hardcode in compute_ops.py |
| deeptools backend | schedule different work slices onto CL0 vs CL1 of the same core (sharing the LX); plausibly mostly the same kernel code but with different slice indices |

The backend constraint is the hardest to evaluate from outside. The
two corelets share LX, which is good (operands fit together) but
their PT arrays are independent. The dataflow patterns described in
the architecture doc (slides 7-32) are written assuming a single PT
array per corelet — so doubling work onto two corelets per core means
two parallel kernel instances running on shared scratchpad, which has
its own scheduling complexity.

## Compare with Project E (bidirectional ring)

Both projects close as "blocked by abstraction layer," but they're
qualitatively different:

| | Project E (dual ring) | Project D (per-corelet) |
|---|---|---|
| Lever exists? | yes | yes |
| Lever visible from torch_spyre? | no — hidden in deeptools / RIU | yes — we can point at compute_ops.py |
| Maximum potential gain | ~2× ring bandwidth (small impact since ring is 24% of cost) | ~2× compute parallelism (large impact) |
| Shippable from torch_spyre alone? | no | no |
| Cross-team coordination needed? | yes — deeptools team | yes — deeptools team + SDSC schema owner |

**Project D is the bigger and more concrete opportunity.** But same
multi-repo blocker.

## Implication for the project pattern

Adding to the running pattern from the session summary:

| project | layer | outcome |
|---|---|---|
| `output_element_priority` | planner | shipped |
| Cost model | planner | not enough headroom |
| Core-ordering | planner | dead lever |
| LX budget | runtime config | mixed |
| K-split / PSUM | planner | narrow |
| Bidirectional ring | runtime/HW (hidden) | infeasible from our layer |
| **Per-corelet** | **codegen + backend** | **infeasible from our layer alone** |

Three of the six post-shipping projects close because the lever lives
outside the torch_spyre repo. **The next phase of work probably
needs to be either (a) cross-repo coordination project to claim these
big multi-layer levers, or (b) a layer-shift to where torch_spyre has
self-contained authority.**

The cross-call preload investigation flagged in the session summary
falls into category (a) — it requires understanding the runtime path
that lives outside torch_spyre. Cross-op fusion / scheduler work
falls into category (b).

## Recommendation

**Don't pursue Project D as a torch_spyre-internal effort.** If
cross-team coordination with the deeptools / SDSC owners becomes
possible, this is the highest-impact item on the list — a 2× compute
parallelism win on every matmul would dwarf everything we've shipped
so far. But it's a different kind of project (multi-repo, multi-team)
and shouldn't be attempted unilaterally.
