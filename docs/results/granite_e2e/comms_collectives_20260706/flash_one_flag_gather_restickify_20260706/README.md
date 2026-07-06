# Flash one-flag gather/restickify structural replay

Generated: 2026-07-06

This run verifies that Torch branch `AdnanHoque/torch-spyre:gather-restickify` only needs the umbrella Torch-side flag for the flash gather/restickify structural transformation.

## Branches

- Torch branch head: `c80af0d688a477c4d67b62d3090c2258cfd9adbe`
- Deeptools branch head used by run: see `roots.txt`

## Torch-side flag

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

The old Torch subflags were unset in this run.

## Backend/runtime setup

No separate Deeptools feature flag is required with the rebased Deeptools patch; it also keys off `SPYRE_LX_PLANNER_RELAYOUT=1`. The old `DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1` flag is only a compatibility alias.

For Granite-style full-LX runs that use the split DXP wrapper, keep the capacity split:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

## Result

See `one_flag_structural_summary.json`.

Key result:

- return code: 0
- root `ReStickifyOpHBM`: 0
- root `ReStickifyOpLx`: 32
- matmul operand plans: 32
- realized strategy: `gather_then_restickify`
- communication pattern: `all_gather_replicate`

This is structural compile/runtime evidence. Flash value correctness remains blocked by the independent baseline zero-stride/broadcast issue in the current test, so this artifact should not be used as value-correctness proof.
