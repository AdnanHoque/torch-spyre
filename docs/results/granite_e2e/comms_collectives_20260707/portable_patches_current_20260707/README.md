# Portable gather/restickify patch bundle - 2026-07-07

This directory contains portable patches for the current gather/restickify
prototype.  These are intended for another developer to apply onto their own
Torch/Deeptools checkouts for experimentation.  They are not PR-sized patches.

## Bases And Heads

- Torch base: `665d0a6ca2cfd30d5cfb90dc98c9508273251483`
- Torch head: `0c8ead7e12695e972c8f83a995c2ecd672dc2e4c`
- Deeptools base: `ff1c7c676cdc8f319f90fe7baa666db2a1103327`
- Deeptools head: `a5ff55eee627c5c2bd4b7b0518bb0cbaad385952`

## Apply

From a Torch checkout based on current upstream main:

```bash
git apply /path/to/torch_spyre_gather_restickify_vs_upstream_main_20260707.patch
```

From a Deeptools checkout based on current upstream master:

```bash
git apply /path/to/deeptools_comms_collectives_vs_upstream_master_20260707.patch
```

## Runtime Flag

The public feature flag is:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

For full Granite/full-LX local reproduction, keep using the split LX wrapper
setup:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

`DXP_BACKEND_LX_FRAC_AVAIL` must be translated to `DXP_LX_FRAC_AVAIL` inside the
DXP subprocess wrapper.  This is a capacity/workaround setting, not a separate
feature flag.

## Included Late Updates

- Torch defaults full-tensor matmul operand collectives to a 1 MiB bounded cap
  under `SPYRE_LX_PLANNER_RELAYOUT=1`; larger full activations preserve the HBM
  fallback until WSR/tile-scoping can make the movement bounded.
- The attempted source-core-aware Deeptools chunking patch is intentionally not
  in this portable patch bundle.  It helped diagnose Granite full-activation
  pressure, but it regressed the saved flash replay into an IBuff failure.

## Validation Snapshot

- Torch focused tests:
  `tests/inductor/test_lx_relayout_dldsc.py -k "matmul_operand_contract_budget or coordinate_topology"`
  passed: 7/7.
- Deeptools focused tests:
  `DxpTestFixture.MatmulOperandBroadcast*:DxpTestFixture.CoreWorkDivIncomptLxRelayout*`
  passed: 7/7.

See `../granite_failclosed_checkpoint_20260707/` for the Granite fail-closed
checkpoint that motivated the bounded default.
