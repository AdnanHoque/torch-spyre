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
- Deeptools branch: `ah/comms-collectives` at `4ec7b9ae9`
