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


## CDX Backend Unblock Update

The standalone `ReStickifyOpLx` backend gap was narrowed and unblocked on the
Deeptools fork branch:

```text
repo: git@github.ibm.com:Adnan-Hoque1/deeptools.git
branch: ah/comms-collectives
sha: 00a37826a8c8e1b32f97c7d6edbc2527f1359076
commit: [DDC] Support LX restickify DDL mapping
```

Two issues were found:

1. The CDX replay binary was using rebuilt Deeptools libraries, but
   `DEEPTOOLS_PATH` still pointed at `/opt/ibm/spyre/deeptools/share`, so DDC
   loaded the installed DDL templates instead of the patched checkout. Replays
   must pin `DEEPTOOLS_PATH` to the matching Deeptools checkout or install tree.
2. The DD1/DD2 `restickify.ddl` template had a `ReStickifyOpHBM` bind but not a
   `ReStickifyOpLx` bind. Once the LX bind was added, DDC matched the op but
   exposed a schedule finalization issue: DDL-only staging dimensions can appear
   as `PrimaryDimTypesCount`, which should not be used as tensor-storage offset
   dimensions.

The accepted backend delta is intentionally small:

- add `ReStickifyOpLx` to `ddc/ddl_templates/restickify.ddl` beside the existing
  `ReStickifyOpHBM` bind;
- skip DDL-only staging dimensions when materializing loop/constant offsets in
  `dsc/dsc2.cpp`;
- keep the existing generic `STCDPOpLx` relayout data-op path for the downstream
  batchmatmul relayout.

Focused validation on CDX passed:

```bash
# Build
cmake --build build-dxp-focused \
  --target dxp_standalone util_unit_test dxp_unit_test -j 8

# Unit tests
build-dxp-focused/util/util_unit_test \
  --gtest_filter="LayoutAllgatherRestickify.*"
# 14/14 passing

build-dxp-focused/dxp/dxp_unit_test \
  --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout"
# 1/1 passing

# DDL sanity
DEEPTOOLS_PATH=/home/adnan-cdx/codex-isolated/dldsc_backend_path_20260702_074814/deeptools \
  build-dxp-focused/ddc/ddl/ddl_standalone \
  -d ddc/ddl_templates/restickify.ddl
# exits 0 and shows both ReStickifyOpHBM and ReStickifyOpLx binds

# Full copied flash SuperDSC replay
DEEPTOOLS_PATH=/home/adnan-cdx/codex-isolated/dldsc_backend_path_20260702_074814/deeptools \
DXP_LX_FRAC_AVAIL=1 \
DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR=/tmp/flash_dldsc_dxp_debug_final_20260702 \
  ./build-dxp-focused/dxp/dxp_standalone \
  -d /home/adnan-cdx/codex-isolated/flash_dldsc_replay_20260702/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_ihw4zzb8 \
  -b sentient
# exits 0 and writes 32 layout_allgather_restickify plan artifacts
```

A representative generated plan now contains the expected flash cardinality:

```text
consumer_core_count: 32
consumer_cores_per_group: 8
replication_factor: 8
logical_transfer_count: 256
```

This means the copied flash DLDSC/SuperDSC bundle now compiles through DXP. The
next required gate is full AIU execution/profiling with the same patched
Deeptools checkout or install path pinned into Torch runtime.
