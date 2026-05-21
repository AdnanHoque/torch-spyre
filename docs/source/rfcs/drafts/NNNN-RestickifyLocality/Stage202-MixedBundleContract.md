# Stage 202: Mixed Bundle Contract Path

## Goal

Proceed with Path 1:

```text
keep the LX restickify bridge as Deeptools data ops,
and make the normal bundle path accept a mixed DL + data-op SuperDsc.
```

This is the most promising path because the real movement already exists in
the data-op/PCFG lowering.  The remaining issue is packaging.

## Current Torch-Spyre State

Stage199 already added the default-off Torch-Spyre lowering hook:

```text
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
```

For an eligible adjacent:

```text
producer -> ReStickifyOpHBM -> consumer
```

Torch-Spyre emits:

```text
producer SDSC
mixed bridge+consumer SDSC
```

The mixed SDSC contains:

- one consumer DL DSC in `dscs_`;
- `ReStickifyOpWithPTLx` data op;
- `STCDPOpLx` data op;
- `coreIdToDscSchedule` entries that run both data ops before the consumer DL
  op.

For the high-signal 2048 case, the schedule is:

```json
[[0, -1, 0, 1], [1, -1, 1, 1], [-1, 0, 1, 0]]
```

Meaning:

- step 0: data-op 0, before sync;
- step 1: data-op 1, before sync;
- step 2: DL op 0.

## Stage202 Refresh

I reran the mixed path against the current installed pod image.

The installed DXP still rejects mixed bundle SDSCs at import:

```text
DtException: Datadsc not allowed, use dldsc
file /project_src/deeptools/dxp/SdscTree.cpp line 152
```

I updated the offline probe to classify this case explicitly as:

```text
installed-dxp-missing-mixed-import-support
```

This makes future runs easier to interpret: a failure at this point is a DXP
bundle import limitation, not evidence that the mixed contract is invalid.

## DCC Evidence

The 2048 mixed SDSC still lowers with standalone DCC:

```text
dcc_rc = 0
has_hbm = false
work_op_count = 822
```

Lowered unit summary:

```text
l0lu0:32
l0su0:32
l3lu:993
l3su:932
lxlu0:32
lxsu0:32
pe0:32
ptrow0_0..ptrow7_0:32 each
sfp0:32
pt_slice_mask_arf_write:8
```

This is the important signal: the mixed SDSC is not an empty shell.  It lowers
to real LX/PT/SFP work and does not contain an HBM marker in the emitted DCC
ProgIR.

Artifacts:

```text
artifacts/stage202_mixed_bundle_contract/summary_2048_dcc_mixed.json
artifacts/stage202_mixed_bundle_contract/sdsc_1_MixedReStickifyOpWithPTLxConsumer_2048.json
```

## Deeptools Contract Delta

The local Deeptools source tree already contains the minimal shape of the
needed DXP-side change as an uncommitted patch:

```text
artifacts/stage202_mixed_bundle_contract/deeptools_mixed_schedule_support.patch
```

It does two things:

1. allow an imported SDSC to contain `dataOpdscs_` when it also has `dscs_` and
   a populated `coreIdToDscSchedule`;
2. route scheduled SDSCs through `DcgManager::runDcgForDataOpsDlOps`.

Patch shape:

```text
dxp/SdscTree.cpp:
  allow dataOpdscs_ iff dscs_ and coreIdToDscSchedule are present

dxp/dxp.cpp:
  if every core has a non-empty coreIdToDscSchedule:
      runDcgForDataOpsDlOps
  else if data-op only:
      runDcg
  else:
      runDcgForDlOpsStandalone
```

I did not push Deeptools changes.

## Small-Shape Note

The 512 offline mixed-schedule matrix currently fails in DCC before the DXP
import question is interesting:

```text
myOutPiece.second.dimToSize_.at(dimName) >= op->outLds->dimToStickSize_.at(dimName)
```

This is consistent with the PT-aware bridge having a minimum per-core piece
shape requirement.  The Path 1 contract should first target the known 2048
case where DCC already emits the intended bridge program.

## Conclusion

Path 1 is still the right path.

We are no longer asking whether LX restickify movement exists, or whether
Torch-Spyre can represent it.  The current blocker is very specific:

```text
build or install a Deeptools/DXP that accepts scheduled mixed SuperDscs in
normal bundle import.
```

Once that DXP-side support is available, the next validation is:

1. rerun the 2048 `SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1` compile;
2. confirm DXP bundle generation succeeds;
3. launch the value-correct tuple fixture;
4. compare against the stock `ReStickifyOpHBM` path for kernel time and HBM
   traffic.

## Validation

Local:

```text
python3 -m py_compile tools/restickify_mixed_schedule_probe.py
```

Pod:

```text
python3 tools/restickify_mixed_schedule_probe.py \
  --bundle-dir /tmp/stage201-seed512/kernel_code/computed_transpose_adds_then_matmul_tuple_512/0001_sdsc_fused_add_t_0 \
  --output-dir /tmp/stage202-mixed-contract-512-v2 \
  --run-deeptools
```

Known-good 2048 mixed DCC check:

```text
dcc_standalone --input-mode=sdsc --kEmitProgIR \
  /tmp/torchinductor_1000800000/tmpz7xxnbj0/inductor-spyre/sdsc_fused_add_t_0_np3vpchr/sdsc_1_MixedReStickifyOpWithPTLxConsumer.json
```
