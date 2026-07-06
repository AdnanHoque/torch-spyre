# DLDSC LX Communication Class Status

Date: 2026-07-06

This is a status map for the `ah/comms-collectives` exploration. It separates three tiers of evidence:

1. Lower-level Deeptools carrier/unit support.
2. DLDSC relayout contract/planner support.
3. Workload proof on Granite or flash.

## Current Matrix

| Communication class | Lower-level evidence | DLDSC/planner evidence | Workload evidence | Status |
|---|---|---|---|---|
| Scatter / permutation | DXP `CoreWorkDivIncomptLxRelayout*` | PR1-style DLDSC tensor-vs-compute mismatch relayout | Granite PR1-style path, not the newly removed Granite spill | Supported for 1:1 ownership reassignment. |
| Broadcast / multicast | DCG `stcdpLibtest.multicastSimple*` passes | Not yet exposed as a general DLDSC planner class | Not yet proven as a standalone Granite spill removal class | Carrier exists underneath; planner plumbing remains. |
| Gather | Utility gather-index tests pass | Covered only as part of matmul operand gather/restickify paths | Flash compile probe and Granite attention path use gather-like movement | Partial; not a generic many-to-one planner primitive yet. |
| All-gather / replicate | `LayoutAllgatherRestickify.*` passes | `layout_allgather_restickify` and `matmul_operand_broadcast` contracts exist | Granite S512 attention spill removed; flash compile probe passes with gather carrier | Best-supported non-scatter class today. |
| Form-changing restickify | DCG STCDP relayout tests pass | `ReStickifyOpLx` used as local layout conversion stage | Granite attention handoff becomes `ReStickifyOpLx`; flash compile probe emits LX restickifies | Supported for current all-gather/restickify path. |
| Reduce | No arithmetic reduction relayout evidence in this path | Not represented by copy-only DLDSC relayout | Not proven | Requires a reduction primitive, not just movement. |
| All-reduce | Same as reduce | Not represented by copy-only DLDSC relayout | Not proven | Requires reduction plus distribution/broadcast. |

## Tests Run On CDX

```text
Deeptools branch: Adnan-Hoque1/deeptools:ah/comms-collectives
Deeptools SHA: 3095cdb33
Pod: adnan-cdx-spyre-dev-pf
```

Focused gates:

```text
dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
2/2 passed

util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
27/27 passed

dcg_unit_test --gtest_filter="stcdpLibtest.multicast*"
2/2 passed

dcg_unit_test --gtest_filter="stcdpLibtest.relayout*"
2/2 passed

util_unit_test --gtest_filter="*gatherIdx*"
3/3 passed
```

Logs are archived next to this file.

## Workload Evidence

Granite S512 causal prefill:

```text
baseline attention handoff: 7_ReStickifyOpHBM
enabled attention handoff:  7_ReStickifyOpLx
remaining HBM restickifies: weights only
```

Flash compile/lowering probe:

```text
carrier: gather_then_restickify
returncode: 0
ReStickifyOpHBM string hits: 0
backend plans: 32
communication_pattern: all_gather_replicate
logical_transfer_count per plan: 256
```

Kernel-neighbor/input-fetch flash carrier:

```text
default: fails on double-buffering + input-neighbor-fetch coexistence guard
diagnostic: gets past the guard, then fails fold solving for lxlu -> ptrow0
```

## Engineering Read

The copy-only communication substrate is moving in the right direction for scatter and all-gather/restickify. The strongest current workload proof is removing the Granite attention activation/layout HBM spill and replacing it with `ReStickifyOpLx`.

The current flash result says carrier selection matters. The `gather_then_restickify` carrier works structurally. The newer kernel-neighbor/input-fetch carrier is promising but not production-safe yet because it collides with double buffering and has a fold-solving failure after the diagnostic guard is bypassed.

Reduce/all-reduce are separate work. Coordinates can describe where partials live and where results should go, but the backend still needs an arithmetic reduction realization; copy-only STCDP/LX relayout is not enough.

