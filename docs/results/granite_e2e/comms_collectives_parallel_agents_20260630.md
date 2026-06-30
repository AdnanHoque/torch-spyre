# Granite LX Relayout Parallel Agent Run - 2026-06-30

This note records the two-lane implementation split for removing the remaining
non-weight HBM round trips in the Granite prefill block.  Weight restickifies
are intentionally out of scope because offline weight prelayout/preload owns
those rows.

## Agent Lanes

| lane | pod | implementation direction | branch target |
|---|---|---|---|
| dl-dsc backend-derived relayout | `adnan-spyre-dev-pf` | Torch emits tensor residency / compute coordinate contract; Deeptools synthesizes movement | `ah/comms-collectives-dldsc-agent` |
| explicit data-op / STCDPOpLx relayout | `adnan-cdx-spyre-dev-pf` | Torch enumerates source/destination movement ranges and schedules explicit transfer rows | `ah/comms-collectives-stcdp-agent` |

## Current Non-Weight Targets

The guarded spill inventory identifies two remaining in-scope gaps after PR1
scatter relayout:

| target | SDSC evidence | communication class | current interpretation |
|---|---|---|---|
| computed attention activation layout restickify | attention `sdsc_9 ReStickifyOpHBM`, downstream `sdsc_10 batchmatmul` | `layout_restickify_activation` | true layout/stick transform plus downstream operand movement, not pure owner-core scatter |
| attention matmul operand movement | attention `sdsc_18 batchmatmul` | `matmul_operand_broadcast` / `all_gather_replicate` | producer has 32 shards while the consumer wants the full operand; full resident replication is too large, so it needs staged or loop-scoped movement |

## Known Baseline From Artifact Branch

- `scatter`: 14 classifications already realized through dl-dsc relayout.
- `layout_restickify_weight`: 4 rows, out of scope.
- `layout_restickify_activation`: 1 row, in scope.
- `matmul_operand_broadcast`: 1 row, in scope.

Relevant source artifacts:

- `docs/results/granite_e2e/comms_collectives_guarded_spill_inventory_20260630.md`
- `docs/results/granite_e2e/comms_collectives_guarded_spill_inventory_20260630.csv`
- `docs/results/granite_e2e/comms_collectives_reproduce_runbook_20260630.md`

## Questions Each Lane Must Answer

1. Can this communication class be expressed without reintroducing HBM?
2. Does the representation scale, or does it produce oversized transfer tables?
3. Is the gap frontend classification/metadata, backend lowering, DCC legality,
   runtime safety, or LX capacity?
4. Does the focused SDSC replay pass?
5. Does full Granite prefill run without a runtime fence?
6. What kernel and wall-time delta does the change produce?

## Environment Reminder

For full-LX Torch planning with backend relayout insertion, use the split-env
wrapper pattern:

```bash
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=1
```

Torch sees full frontend LX planning, while the DXP subprocess gets backend LX
space for inserted relayout pieces.
