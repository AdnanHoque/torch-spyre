# Portable one-gate patches

These are the portable patches for the gather/restickify and loop-scoped LX collective prototype.

Apply both patches, then enable the feature with one public flag:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

For full Granite/full-LX reproduction, the Torch side patch also defaults the frontend/backend LX split when the feature flag is enabled. If an environment overrides LX capacity manually, use the split-env form:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

The Deeptools patch also honors legacy prototype flags for compatibility, but Antoni-facing runs should use `SPYRE_LX_PLANNER_RELAYOUT=1`.

Generated from:

- Torch clean branch: `gather-restickify` at `cc29ba25`
- Deeptools branch: `ah/comms-collectives` at `911b3877d`

## Verification after one-gate refresh

Verified on `adnan-cdx-spyre-dev-pf` with `SPYRE_LX_PLANNER_RELAYOUT=1`, frontend `DXP_LX_FRAC_AVAIL=0`, and backend `DXP_BACKEND_LX_FRAC_AVAIL=1`:

- DXP replay passed for the formerly failing flash bundle and all Granite S=512 bundles.
- Flash compile probe passed with `SUCCESS`.
- Flash compile-probe SDSCs contained no `ReStickifyOpHBM`; movement rows were `ReStickifyOpLx` plus `matmul_operand_broadcast`.

Latest replay root:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_onegate_chunk_alloc_20260707_022823/dxp_replay_status_20260707_031920
```
