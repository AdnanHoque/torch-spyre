# DLDSC Broadcast / Multicast Contract Checkpoint

Date: 2026-07-06

This checkpoint records the first explicit DLDSC-side support for naming
`broadcast` and `multicast` communication patterns in the existing matmul
operand relayout contract.

## Code Under Test

- Deeptools fork branch: `Adnan-Hoque1/deeptools:ah/comms-collectives`
- Deeptools SHA: `b1c93212d`
- Commit: `[DXP] classify matmul operand broadcast and multicast`

The change is intentionally narrow:

- `communication_pattern` now accepts:
  - `all_gather_replicate`
  - `broadcast`
  - `multicast`
- Pattern-only metadata is normalized to the existing
  `matmul_operand_broadcast` movement plan.
- `broadcast` is fail-closed to one source chunk and one destination group.
- `multicast` is fail-closed to one source chunk per destination group.
- The existing all-gather/restickify behavior is unchanged.

## Evidence

The saved logs in this directory came from CDX:

`/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/broadcast_multicast_contract_20260706_172235`

Passing tests:

- `LayoutAllgatherRestickify.*`: 31/31 passed
- `DxpTestFixture.CoreWorkDivIncomptLxRelayout*`: 2/2 passed
- `stcdpLibtest.multicast*`: 2/2 passed

The new utility tests prove exact logical transfer expansion for:

- global broadcast: one source core to all destination cores;
- grouped multicast: one source core per group to that group's destination
  subset;
- pattern-only multicast metadata, without an explicit `kind` field.

## Current Meaning

This does not yet mean every Granite broadcast/multicast spill is removed.
It means the Deeptools-side DLDSC movement contract can now name these
communication classes directly and reject invalid cardinalities before physical
lowering.

The next step is Torch-side classification: emit `broadcast` or `multicast`
from the LX planner when producer/consumer coordinate maps show one-to-many
ownership instead of overloading everything as `all_gather_replicate`.

## Files

- `deeptools_b1c93212d_broadcast_multicast.patch`: exact Deeptools patch.
- `deeptools_b1c93212d_stat.txt`: patch stat and commit subject.
- `layout_allgather_restickify_tests.log`: utility contract tests.
- `dxp_core_work_div_tests.log`: DXP relayout regression tests.
- `dcg_multicast_tests.log`: lower-level STCDP multicast smoke.
- `env.txt`: run metadata from the CDX pod.
- `final_metadata.txt`: pushed Deeptools branch/SHA metadata after commit.
