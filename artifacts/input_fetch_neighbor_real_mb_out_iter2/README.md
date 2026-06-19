# InputFetchNeighbor real mb/out probe

Second-iteration backend probe for the `input_fetch_neighbor` carrier.

This directory contains a single-edge `mb/out` IFN artifact built from full
torch-spyre-generated SDSCs, not the small unit-test fixture:

- `sdsc_0.json`: producer `batchmatmul`, patched so output `Tensor2` is LX.
- `sdsc_1.json`: consumer IFN SDSC with one `0_OnChipMoveIFNDataOpLx` and
  trigger row `[[0, 0, 0, 0]]`.
- `bundle.mlir`: minimal two-SDSC bundle used for the `dxp_standalone --bundle`
  probe.
- `summary.json`: exact pod, commands, and blocker sequence.

The dedicated IFN standalone reaches the Deeptools IFN path, then fails before
value execution:

```text
DtException: mySDscMain.dscs_.at(0).primaryDsInfo_.count(DsTypes::INPUT)
file /project_src/deeptools/dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp line 16
```

Compatibility probes show additional stock-helper assumptions after that:
all tensors must appear LX/ring-pinned, legacy `coreStateInit_` must be present,
and synthetic `coreStateInit_` then hits an empty loop-order assertion. The
normal DXP bundle path rejects the `datadscs_` entry with:

```text
DtException: Datadsc not allowed, use dldsc
file /project_src/deeptools/dxp/SdscTree.cpp line 152
```

Status: blocked. No value-correct runtime smoke was reached.
