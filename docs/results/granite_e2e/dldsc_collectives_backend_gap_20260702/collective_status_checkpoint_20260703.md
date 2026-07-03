# DLDSC Collective Status Checkpoint - 2026-07-03

This checkpoint records the current Granite/flash communication-class findings from the three pod lanes. It is intentionally artifact-level: what we tested, what passed, what failed, and what that implies for the next implementation step.

## Granite Edge Inventory

Current Granite prefill artifacts show no remaining in-scope non-weight `ReStickifyOpHBM` rows. The obvious non-weight HBM scatter round trips are covered by PR1-style DLDSC scatter metadata. Remaining work is now in larger LX collective/form-changing classes, not the original one-to-one scatter class.

See `granite_edge_inventory_clc_20260703.tsv`.

| Edge | Class | Status |
|---|---|---|
| `buf13 -> buf46` inside old `9_ReStickifyOpHBM` | scatter | Covered by PR1 scatter; not remaining. |
| `buf9 -> buf14` into old `10_batchmatmul` | scatter | Covered by PR1 scatter; not remaining. |
| `mul/buf46 -> batchmatmul/buf14` | all-gather + layout-restickify/form-changing | Needs staged `ReStickifyOpLx`/form-changing path. |
| `clone/buf21/Tensor1 -> batchmatmul/buf22` | all-gather/broadcast | Needs grouped same-layout LX collective in matmul operand path. |
| `buf45`, `buf47`, `buf48`, `buf49` restickifies | weight/prelayout | Out of scope; should be handled by offline weight preload/prelayout. |

## Existing Primitive Coverage

The dev-pf standalone sweep shows Deeptools already has working small examples for multicast, gather, and LX restickify. It does not have an all-gather standalone sample, and no reduce/all-reduce sample was found.

See `devpf_primitive_coverage_20260703.tsv`.

Key result: broadcast/multicast and gather are expressible in existing standalone samples, but flash/Granite need larger grouped all-gather-like patterns than those basic samples cover.

## Flash Grouped All-Gather Result

The flash score-prefix edge is not a pure same-layout copy. It is a grouped all-gather plus layout/restickify-shaped handoff. The exact standalone descriptor matches the full flash debug descriptor:

- producer pieces: 256
- consumer pieces: 2048
- fanout: 8
- source LX base: `131072`
- destination LX base: `0`
- representative full shape: `out=4096`, `in=128`, `x=4`

### What Passed

Individual `in=16` slices pass end to end for every offset:

- starts tested: `0,16,32,48,64,80,96,112`
- `DataOpStandalone`: pass
- `senpcfg`: pass
- `dcc-opt` senprog: pass
- `dcc-opt` smc: pass
- `senulator -v store`: pass

See `flash_layout_allgather_20260703/grouped_allgather_sweep/slice_offsets_store_summary.tsv`.

### What Fails

Putting all eight `in=16` slices into one scheduled data-op program fails:

- descriptor generation: pass
- `senpcfg`: pass
- `dcc-opt` senprog: aborts after program verification failure
- SMC lowering reports: `Require larger IBUFF`, `Max IBUFF(256) Current IBUFF(328)`
- `senulator -v store`: misses tail writes, notably on LX31

See `flash_layout_allgather_20260703/grouped_allgather_sweep/split_dataops_summary.tsv`.

### Fission Granularity

A follow-up variable-count sweep shows the safe chunk size for this flash-sized grouped all-gather:

- `count=1`: pass
- `count=2`: pass
- `count=4`: pass, including nonzero start `64`
- `count=8`: fail with the same DCC/program capacity and store verification failure

See `flash_layout_allgather_20260703/grouped_allgather_sweep/split_count_summary.tsv`.

This means the current full edge can likely be represented as two standalone relayout SDSCs, each carrying four `in=16` slice rows.

## Interpretation

The slice-by-slice result is the critical positive result: the ring movement is correct for every destination slice and every nonzero `in` offset. The current failure is not that the transfer is impossible. It is that fusing the whole flash-sized grouped all-gather into one generated L3LU program exceeds backend program capacity and then fails store verification.

So the next backend design should not be “one huge grouped all-gather row.” Viable directions are:

1. Emit several smaller standalone relayout SDSCs/kernels, each under the L3LU instruction-buffer limit.
2. Add backend fission for large grouped collectives so the frontend can express the logical handoff while Deeptools splits the physical movement safely.
3. Later, add scheduling/overlap so those chunks can pipeline with compute rather than becoming a serialized tax.

## Current Scope Statement

PR1 scatter remains useful and correct for the simple one-to-one shard mismatch class. It does not cover the remaining flash/Granite classes by itself. The next production-worthy extension should target grouped all-gather/broadcast and form-changing relayout, with explicit backend chunking/fission constraints.

