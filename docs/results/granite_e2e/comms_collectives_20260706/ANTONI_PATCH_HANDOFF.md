# Antoni patch handoff

Use the one-gate patch set in:

- `docs/results/granite_e2e/comms_collectives_20260706/portable_onegate_patches_20260707/torch_spyre_gather_restickify_onegate.patch`
- `docs/results/granite_e2e/comms_collectives_20260706/portable_onegate_patches_20260707/deeptools_comms_collectives_onegate.patch`

The older branch-relative patch paths have also been refreshed to the same one-gate contents:

- `docs/results/granite_e2e/comms_collectives_20260706/torch_patch_rebased_main_20260706/torch_spyre_gather_restickify_rebased_on_main.patch`
- `docs/results/granite_e2e/comms_collectives_20260706/deeptools_patch_rebased_master_20260706/deeptools_ah_comms_collectives_rebased_on_master.patch`

Runtime gate:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

For Granite/full-LX reproduction, keep the same public gate and let the Torch patch drive the backend LX setting. If the local environment overrides LX manually, use:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```


The refreshed Deeptools patch includes the staged matmul operand coordinate fix verified against flash and Granite DXP replay.
