# Broadcast/Multicast High-Cap Probe Excerpts

Run source on CDX pod:
`/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/pattern_high_cap_probe_20260707_135029`

Purpose: test whether the `matmul_operand_broadcast` DLDSC hook can be forced through the safer gather-then-restickify materialization path when the direct loop-scoped/kernel-neighbor path hits `wrong locale for dst operand`.

Environment used for the probe:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1
DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_MAX_CHUNKS=10000
DXP_LX_FRAC_AVAIL=1
```

## Broadcast

Result: `rc=134`.

Backend plan was emitted:

```text
communication_pattern: broadcast
physical_lowering_status: lowered_gather_then_restickify
realization_strategy: gather_then_restickify
logical_transfer_count: 32
source_core_count: 1
```

DCC failure excerpt:

```text
Require larger IBUFF
Max IBUFF(256) Current IBUFF(6446) for unit:
%1111 = dataflow.get_unit {core = 0 : i32, name = "l3su", num_folds = 1 : i32, type = "l3su"} : index
error: Unable to lower successfully the module for sdsc: 20_batchmatmul
terminate called after throwing an instance of DtException
  what():  DtException: DCC causes the compilation failure, file .../dcc/src/Driver/dcc.cpp line 563
```

## Multicast

Result: `rc=134`.

Backend plan was emitted:

```text
communication_pattern: multicast
physical_lowering_status: lowered_gather_then_restickify
realization_strategy: gather_then_restickify
logical_transfer_count: 32
source_core_count: 4
```

DCC failure excerpt:

```text
Require larger IBUFF
Max IBUFF(256) Current IBUFF(412) for unit:
%1105 = dataflow.get_unit {core = 0 : i32, name = "l3su", num_folds = 1 : i32, type = "l3su"} : index
error: Unable to lower successfully the module for sdsc: 20_batchmatmul
terminate called after throwing an instance of DtException
  what():  DtException: DCC causes the compilation failure, file .../dcc/src/Driver/dcc.cpp line 563
```

## Interpretation

The DLDSC metadata is sufficient to classify and plan these edges, and the gather-then-restickify carrier can emit a physical plan for broadcast/multicast. The current lowered program is too large, so these cases are not production-ready through naive full-resident materialization.

This is different from the bounded all-gather flash case, which succeeds with the narrow priority fix and emits `lowered_gather_then_restickify` without the `wrong locale` failure.

Next backend question: add a compact/bounded lowering for broadcast/multicast or rely on WSR to tile the region before materializing the communication. This branch should not grow a custom full-tensor streaming system.
