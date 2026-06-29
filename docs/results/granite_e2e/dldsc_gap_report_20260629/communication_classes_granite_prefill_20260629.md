# Granite Prefill LX Communication Classes - 2026-06-29

## Context

Goal: reproduce the prior ~1.2x Granite block prefill speedup with the
dldsc-based LX relayout implementation.

Current best measured result:

| Variant | Kernel ms/iter | Median wall ms | Speedup vs baseline |
|---|---:|---:|---:|
| Baseline, relayout off | 12.4741 | 19.1460 | 1.000x |
| dldsc relayout, boundary clones, full Torch LX | 10.9780 | 17.7715 | 1.136x |

A 1.2x result against this baseline would need about 10.395 ms/iter, so the
remaining gap is roughly 0.58 ms/iter.

## Implemented Class

### Scatter Resident Remap

Producer owns final tensor slices in LX, consumer wants the same tensor in a
different per-core resident ownership. Torch records the producer residency in
dl-dsc allocation coordinates; Deeptools inserts an LX relayout and materializes
the post-relayout resident view before the consumer.

This is the PR1 dldsc LX relayout class.

## Bounded Variant

### Resident Fanout

Producer ownership can feed multiple resident consumer slices when the
post-relayout pieces fit in LX. This is still a resident materialization class:
Torch must reserve space for the post-relayout LX view, and Deeptools must be
able to allocate the materialized pieces.

## Missing Class

### Non-Resident Matmul Operand Collective

Signature:

- producer operand is sharded across cores;
- consumer matmul compute is parallel, but this operand's consumer view is not
  partitioned by that compute split;
- if lowered as resident scatter remap, each consumer core would need a full operand
  piece.

Observed Granite prefill case:

| Edge | Planner kind | Communication pattern | Evidence |
|---|---|---|---|
| `buf21 -> buf22` | `matmul_operand_broadcast` | `all_gather_replicate` | producer dims are sharded; consumer operand dims are unsliced |

DXP-only repro:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_gap_report_20260629/buf21_dxp_repro`

Important diagnostic:

```text
out_form_size=4.1943e+06 out_piece_size=4.1943e+06
no_lx_space sdsc=16_batchmatmul lds=Tensor1 core=0
```

Interpretation: current Deeptools resident relayout would materialize roughly
4 MiB per consumer core for this operand. That is not the intended on-chip
communication class and cannot reproduce the missing speedup.

## Ruled-Out Cheap Alternative

Forcing the attention value matmul to split across the batch/head axis changed
the AV matmul from `mb=32` to `x=32`, but regressed performance:

| Variant | Kernel ms/iter | Median wall ms |
|---|---:|---:|
| Default best dldsc relayout | 10.9780 | 17.7715 |
| Forced value-BMM batch split | 12.1928 | 19.1263 |

This rules out a simple work-division override as the route to 1.2x.

## Contract Boundary

The relayout planner should classify communication classes:

- `scatter`: implemented by current dldsc relayout path.
- `resident_fanout`: implemented only when materialized pieces fit.
- `matmul_operand_broadcast` / `all_gather_replicate`: classified, not
  realized by PR1.
- reduction-aware movement: future class for partial/K-split producers.
- layout-changing movement: future class for restickify/reformat movement.

Working Set Reduction can own how much data is resident at a time and how the
movement is staged. The relayout planner should still expose the communication
class and cost signal so work division and backend lowering can make coherent
decisions.
