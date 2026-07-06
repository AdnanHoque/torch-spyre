# Flash one-gate gather/restickify structural proof

Generated: 2026-07-06

Run directory on CDX:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_onegate_gate_path_fix_20260706_233821
```

This run used the current Torch gather-restickify branch and Deeptools ah/comms-collectives at 5faca11015753999da1d26b87ecccdb08e4975ce. The only feature gate was SPYRE_LX_PLANNER_RELAYOUT=1; legacy Deeptools flags were explicitly unset.

## Result

- return code: 0
- SDSC JSON files: 550
- ReStickifyOpHBM files: 0
- ReStickifyOpLx files: 64
- backend plans: 64
- communication pattern: {'all_gather_replicate': 64}
- realization strategy: {'gather_then_restickify': 64}
- physical lowering: {'lowered_gather_then_restickify': 64}
- logical transfers: 40960

This proves the current one-gate DLDSC path removes the explicit flash HBM restickify rows structurally and lowers the remaining matmul operand movement as all_gather_replicate plus local ReStickifyOpLx.

## Caveat

The run is structural: PATCH_MODE=no_h2d,skip_cpu_ref. Flash value correctness remains gated by the separate baseline zero-stride/broadcast-view issue, so this artifact is evidence of compile/lowering behavior and HBM-spill removal, not a standalone numerical correctness proof.
