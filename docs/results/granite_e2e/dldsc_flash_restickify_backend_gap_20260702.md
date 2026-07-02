# Flash DLDSC Restickify Backend Gap - 2026-07-02

## Reproducer

The failing flash SuperDSC bundle was copied from CLC to CDX for backend
inspection:

```text
/home/adnan-cdx/codex-isolated/flash_dldsc_replay_20260702/
  sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_ihw4zzb8/
```

Source run on CLC:

```text
/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/
  latest_after_zero_stick_split_lx_layout_allgather_20260702_085442
```

The run used the split-LX environment that keeps Torch planning at full LX while
giving DXP backend workspace:

```bash
DXP_LX_FRAC_AVAIL=0 \
DXP_BACKEND_LX_FRAC_AVAIL=1 \
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
```

## Observed Failure

DXP fails on the standalone LX restickify row:

```text
DtException: Scheduler failed to find a suitable op mapping for sdsc: 2_ReStickifyOpLx
```

The representative copied artifact is `sdsc_104.json`:

```text
top-level SDSC: 104_ReStickifyOpLx
compute opFuncName: ReStickifyOpLx
lxRelayoutClassifications_: []
```

The downstream consumer is `sdsc_105.json`:

```text
top-level SDSC: 105_batchmatmul
lxRelayoutClassifications_: one layout_allgather_restickify entry
communication_class: all_gather
communication_pattern: layout_allgather_restickify
producer_work_slice_dims: {0: 4, 2: 8}
consumer_work_slice_dims: {0: 4}
max_fanout: 8
max_fanin: 8
transfer_count: 256
requires_staged_realization: true
restickify_op: ReStickifyOpLx
```

## Interpretation

The frontend is now expressing the hard flash activation handoff correctly as a
DLDSC contract on the downstream batchmatmul. It identifies the edge as a
layout-aware grouped all-gather: the producer has `{mb:4,out:8}` ownership, the
consumer collapses the `out` grouping, and each produced slice must feed eight
consumer placements.

The backend branch also has a generic coordinate-incompatibility relayout path:

- `dxp/SdscRelayoutInsertion.cpp` scans LX-pinned inputs whose allocation
  coordinates differ from consumer compute coordinates.
- It inserts an internal `STCDPOpLx` data-op SuperDSC before the consumer.
- `DxpTestFixture.CoreWorkDivIncomptLxRelayout` passes on CDX.

The flash failure is therefore more specific. Before the consumer batchmatmul
can benefit from relayout insertion, DXP still has to compile the standalone
`ReStickifyOpLx` SDSC. That op is present as a DL compute row, but existing
backend support for LX restickify appears to live in data-op / relayout paths,
not as a schedulable standalone DL op in DDC.

## CDX Replay Update

After building `dxp_standalone` from the current Deeptools branch on CDX, the
copied bundle reproduced the failure directly:

```bash
DXP_LX_FRAC_AVAIL=1 \
DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR=/tmp/flash_dldsc_dxp_debug_20260702 \
./build-dxp-focused/dxp/dxp_standalone \
  -d /home/adnan-cdx/codex-isolated/flash_dldsc_replay_20260702/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_ihw4zzb8 \
  -b sentient
```

The replay generated 32 `layout_allgather_restickify_plan.json` files before
aborting on `2_ReStickifyOpLx`. That proves the downstream consumer hook is
active; the remaining failure is the earlier standalone LX restickify row.

One additional Deeptools bug was found and fixed on the fork branch
`Adnan-Hoque1/deeptools:ah/comms-collectives`:

```text
commit e2ee21b1f2203a08f43e788a28fad0305582767c
[DXP] Preserve string core counts in relayout metadata
```

Before that fix, SuperDSC metadata count fields arrived as strings, so the
backend ignored `consumer_core_count=32` and fell back to the sparse
`consumer_work_slice_dims` product of 4. The generated plan therefore had only
32 logical transfers. After the fix, the same real flash metadata produces:

```text
consumer_core_count: 32
consumer_cores_per_group: 8
replication_factor: 8
logical_transfer_count: 256
```

## Immediate Backend Gap

One of these contracts has to become true:

1. `ReStickifyOpLx` as a standalone SuperDSC compute op is schedulable by DDC;
2. DXP recognizes standalone `ReStickifyOpLx` SDSCs and internally routes them
   through the same data-op / relayout realization path used for inserted LX
   relayouts;
