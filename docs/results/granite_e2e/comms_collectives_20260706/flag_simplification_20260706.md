# LX Relayout Flag Simplification

Generated: 2026-07-06

Torch branch `AdnanHoque/torch-spyre:gather-restickify` now treats `SPYRE_LX_PLANNER_RELAYOUT=1` as the umbrella switch for the Torch-side DLDSC relayout research lane.

## Current Torch-side invocation

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

With that flag set, the following subfeatures default on:

- computed-source ReStickifyOp output LX eligibility
- collective/matmul-operand relayout planning
- matmul operand DLDSC metadata contract
- layout all-gather plus ReStickifyOpLx metadata lane

The old subflags are still accepted as diagnostic overrides. Set a subflag to `0` to disable that slice while keeping the umbrella enabled.

## Backend/runtime flags

Deeptools now honors the same umbrella flag, so no separate Deeptools feature flag is required:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

The old `DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1` flag remains as a compatibility alias only. Do not ask new users to set it.

For Granite-style full-LX experiments, keep the LX capacity split as a separate runtime setup detail, not as a feature flag:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

The split LX setting remains important: Torch sees `DXP_LX_FRAC_AVAIL=0` for full frontend LX planning, while the DXP wrapper rewrites `DXP_BACKEND_LX_FRAC_AVAIL=1` for the backend subprocess.

## Historical artifacts

Older run directories in this folder still show the five explicit Torch-side subflags. Those are accurate for the recorded runs and were not rewritten. New runs should prefer the umbrella form above.
