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

For patched Torch-launched Granite/flash experiments, the old split-LX wrapper is not required. With `SPYRE_LX_PLANNER_RELAYOUT=1`, Torch defaults frontend planning to full LX and passes `DXP_LX_FRAC_AVAIL=1` only to the `dxp_standalone` subprocess.

Only direct manual SDSC replay that bypasses Torch needs explicit backend workspace:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1 DXP_LX_FRAC_AVAIL=1 dxp_standalone -d path/to/sdsc_bundle
```

## Historical artifacts

Older run directories in this folder still show the five explicit Torch-side subflags. Those are accurate for the recorded runs and were not rewritten. New runs should prefer the umbrella form above.
