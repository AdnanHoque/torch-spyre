# Overnight Comms Collectives Progress - 2026-07-07

## Goal

Build the DLDSC LX communication substrate for Granite without duplicating WSR. The current focus is bounded on-chip communication classes: scatter/permutation, broadcast/multicast, gather/all-gather. Full-tensor streaming/tile scoping remains WSR-owned.

## What Changed In This Session

The clean Deeptools `ah/comms-collectives` branch was failing the bounded M16 `matmul_operand_broadcast` replay with:

```text
DtException: wrong locale for dst operand
```

The same Torch-generated SDSC bundle passed with the older green backend. Comparing plan artifacts showed the current backend selected:

```text
realization_strategy = loop_scoped_input_fetch
```

while the old green backend selected:

```text
realization_strategy = gather_then_restickify
physical_lowering_status = lowered_gather_then_restickify
```

Root cause: under `SPYRE_LX_PLANNER_RELAYOUT=1`, the DXP decision order routed `matmul_operand_broadcast` into the loop-scoped/kernel-neighbor path before considering the safe gather/restickify materialization. That path is not currently safe for this KERNEL operand case and trips DCC/DXP lowering.

## Patch

Archived patch:

```text
docs/results/granite_e2e/comms_collectives_20260707/patches/deeptools_gather_restickify_priority_20260707.diff
```

Patch effect:

- Prefer `attachMatmulOperandBroadcastGatherThenRestickify(...)` for `communication_pattern == all_gather_replicate` when layout conversion is required.
- Only fall through to loop-scoped/kernel-neighbor after that path is unavailable.
- This keeps the bounded all-gather/re-stickify path on the safe two-stage lowering.

## Validation

### Bounded all-gather/re-stickify replay

Run:

```text
logs/dxp_replay_priority_narrow_20260707_134642
```

Result:

```text
rc = 0
physical_lowering_status = lowered_gather_then_restickify
realization_strategy = gather_then_restickify
logical_transfer_count = 16
```

This is the same semantic path as the older green backend, now reproduced on the clean/current backend branch.

### LayoutAllgatherRestickify unit tests

Run:

```text
logs/unit_layout_allgather_20260707_134457
```

Result:

```text
32/32 passed
```

### DXP focused filter

Run:

```text
logs/unit_dxp_core_work_div_narrow_20260707_134642
```

Result:

```text
3/5 passed
```

Passing:

- `MatmulOperandBroadcastChunkCapFailsClosed`
- `CoreWorkDivIncomptLxRelayout`
- `CoreWorkDivIncomptLxRelayoutCardinality`

Still failing:

- `MatmulOperandBroadcastPatternBroadcastCompiles`
- `MatmulOperandBroadcastPatternMulticastCompiles`

Failure mode after narrowing:

```text
DtException: wrong locale for dst operand
```

This means broadcast/multicast still fall into a backend path that DCC cannot lower safely.

## Broadcast/Multicast Probe

Run:

```text
logs/pattern_high_cap_probe_20260707_135029
```

I forced broadcast/multicast through gather/restickify with a high diagnostic chunk cap.

Result:

```text
broadcast:  physical_lowering_status=lowered_gather_then_restickify, rc=134
multicast:  physical_lowering_status=lowered_gather_then_restickify, rc=134
```

DCC failure was no longer wrong-locale. It became program-size pressure:

```text
broadcast:  Require larger IBUFF, Max IBUFF(256) Current IBUFF(6446)
multicast:  Require larger IBUFF, Max IBUFF(256) Current IBUFF(412)
```

Interpretation:

- Broadcast/multicast are representable in the DLDSC metadata and can reach gather/restickify artifact emission.
- The naive resident materialization explodes backend program size for this fixture.
- This is exactly the boundary with WSR/tile-scoping or a more compact backend loop form.

## Status Against Goal

Completed / solid:

- Scatter/permutation PR1 path remains the production-shaped baseline.
- Bounded all-gather/re-stickify now has a clean current-backend replay passing DXP.
- The old/current backend mismatch is explained and minimized to a DXP path-selection bug.
- Artifacts prove the all-gather case is using on-chip `STCDPOpLx` gather plus `ReStickifyOpLx`, not HBM fallback.

Open:

- Broadcast/multicast need backend work beyond metadata parsing.
- Current direct kernel-neighbor path still produces wrong-locale for these cases.
- Gather/restickify materialization for broadcast/multicast can be emitted, but is too large without WSR/tile-scoping or compact loop lowering.
- Full Granite spill removal remains blocked on broadening these communication classes and on WSR for oversized activations.

## Next Best Step

Keep the narrow all-gather priority patch. Then decide the broadcast/multicast backend strategy:

1. Add a compact bounded lowering for broadcast/multicast so the unit compile tests do not exceed IBUFF, or
2. Mark broadcast/multicast full-resident materialization as unsupported without WSR and add fail-closed diagnostics, while leaving smaller/tiled cases for WSR integration.

Given the project scope, option 2 is cleaner unless a compact loop form is already available in Deeptools.
