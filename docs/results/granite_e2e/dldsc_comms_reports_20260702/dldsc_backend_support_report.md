# DLDSC communication-class backend support evidence - 2026-07-02

Checkout inspected on pod `adnan-cdx-spyre-dev-pf`:

- Repo: `/home/adnan-cdx/codex-isolated/deeptools-comms-collectives-clean`
- Branch: `ah/comms-collectives`
- Commit: `966f1149e9e6cb02f8c5a2a102a9e6cc01083fc3`
- Commit subject: `[DXP] Sync LX relayout LBR addresses`
- Git status before this report: clean tracked tree, with existing untracked `layout_allgather_restickify_backend_report.md`

## Support matrix

| Class/path | Backend support observed | Evidence |
|---|---|---|
| Broadcast/multicast | Yes, literal `STCDPOpLx` coordinate-overlap realization. | `stcdpLibtest.multicastSimple` and `stcdpLibtest.multicastSimpleZP` pass and assert `myDataOpDsc.op->name == OpFuncs::STCDPOpLx`; generated SDSC for `multicastSimple*` has op name `STCDPOpLx`. |
| Scatter | Backend generation and `senpcfg` pass for LX-overlap scatter cases, but the realized op is `ScatterOpHBM`, not literal `STCDPOpLx`. | `ScatterHBMSimpleABLXOpted` and `ScatterHBMSimpleALXOpted` pass `dcg_standalone` internal program verification plus `senpcfg`; generated SDSC op name is `ScatterOpHBM` with `l3su/l3lu` LX/HBM ring-DT PCFG loops. |
| Gather | Backend generation and `senpcfg` pass for LX-overlap gather cases, but the realized op is `GatherOpHBM`, not literal `STCDPOpLx`. | `GatherHBMLX` and `GatherHBMLX128` pass `dcg_standalone` internal program verification plus `senpcfg`; generated SDSC op name is `GatherOpHBM` with L3LU ring-DT PCFG loops. |
| Narrow `layout_allgather_restickify` | Supported as a fail-closed contract checker and backend-facing logical movement-plan artifact only; not wired into full DXP/DSM physical lowering. | `LayoutAllgatherRestickify.*` passes 13/13. The plan emits `ReStickifyOpLx`, `consumerOperandDsType=KERNEL`, stages `restickify_layout_on_chip`, `grouped_all_gather`, `bind_bmm_kernel_operand`, and 256 logical transfers for the tested 4x8x8 case. Header comments state byte ranges/LX addresses are still filled by the DXP mutation point using imported SDSC allocation data. |

## Passing commands

From `/home/adnan-cdx/codex-isolated/deeptools-comms-collectives-clean`:

```sh
cmake --build build-codex-util --target dcg_unit_test -j 8
cmake --build build-codex-util --target dcg_standalone -j 8
cmake --build build-codex-util --target senpcfg_standalone -j 8

build-codex-util/util/util_unit_test --gtest_filter=LayoutAllgatherRestickify.*
# 13 tests passed

build-codex-util/dcg/dcg_unit_test --gtest_filter=stcdpLibtest.relayoutComplex:stcdpLibtest.relayoutDynMVLoop:stcdpLibtest.multicastSimple:stcdpLibtest.multicastSimpleZP
# 4 tests passed
```

From `/home/adnan-cdx/codex-isolated/deeptools-comms-collectives-clean/build-codex-util`:

```sh
for op in GatherHBMLX GatherHBMLX128 ScatterHBMSimpleABLXOpted ScatterHBMSimpleALXOpted multicastSimple multicastSimpleWithZP; do
  rm -rf dataDSC
  SENARCH=mpw4 DT_OPT="dtversion=1" ./dcg/dcg_standalone -o "$op" -s
  SENARCH=mpw4 ./senulator/senpcfg -p dataDSC -g -dformat hex
done
# Each case passed dcg_standalone program verification and senpcfg reported success.
```

## Reduce/all-reduce gap

Reduce/all-reduce is not supported by the DLDSC communication-class coordinate-overlap path as a transfer-only `STCDPOpLx` realization. Current DSM collective code supports explicit `AllReduce` fission algorithms (`LinearBcast`, `BiTreeBcast`, `ReduceScatterAllGather`, `GatherSumBcast`, `PairwisePow2`, etc.) and `ReduceScatter` algorithms, but those require compute/reduction envelopes (`COMPUTE`, `COMPUTE_TREE`) in addition to data movement. The `layout_allgather_restickify` checker intentionally accepts only `communication_class=all_gather`; reduce/all-reduce metadata is rejected or routed through the existing collective planner, not this coordinate-overlap movement helper.

## Torch metadata

No Torch metadata appears required beyond DLDSC coordinate/layout fields and the narrow `layout_allgather_restickify` contract metadata. The helper accepts either named DLDSC dimensions (`mb`, `x`, `out`, `in`) or compact staged dimension indices (`0`, `2`) for the flash case. Required fields are the communication class/pattern, producer/restickify/consumer op names, staged-realization flag, producer/restickify/consumer work-slice dims, three layout/stick-dim-order contracts, and `dimension_rename`. Explicit core counts are optional when they can be derived from split products.

## Files changed

No backend source files were changed. This report file was added.
