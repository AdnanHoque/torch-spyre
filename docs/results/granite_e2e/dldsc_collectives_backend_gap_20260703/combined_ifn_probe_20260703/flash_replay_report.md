# Flash DLDSC layout classification replay

Pod: adnan-spyre-dev-pf
Run root: /home/adnan/codex-isolated/flash_attention_dldsc_20260703_170000/runs/replay_layout_classification_20260703_174650
Source script: /home/adnan/codex-isolated/flash_attention_dldsc_20260703_170000/test-spyre-scripts/test_flash.py
Wrapper: /home/adnan/codex-isolated/flash_attention_dldsc_20260703_170000/runs/replay_layout_classification_20260703_174650/run_flash_replay.sh

## Commands

Top-level:

```bash
kubectl exec adnan-spyre-dev-pf -- bash -lc 'RUNROOT=$(cat /home/adnan/codex-isolated/flash_attention_dldsc_20260703_170000/latest_replay_layout_classification.txt); "$RUNROOT/run_flash_replay.sh"'
```

Per mode command inside the wrapper:

```bash
timeout 900 /home/adnan/dt-inductor/.venv/bin/python3 <mode>/bootstrap.py
```

Both modes used `PATCH_MODE=no_h2d,skip_cpu_ref`, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, `TORCHINDUCTOR_FX_GRAPH_CACHE=0`, and `DXP_BACKEND_LX_FRAC_AVAIL=0.2`. `DEEPTOOLS_ENABLE_UNSAFE_MATMUL_OPERAND_BROADCAST` was left unset.

## SHAs

- torch-spyre: 79e9958e2f7cf3e8a35c639c10f80f582edc00ed (ah/comms-collectives)
- deeptools: 9397aa9c439060155082f2fb6c132a70b7b32e0e (ah/comms-collectives)
- test-spyre-scripts: afda166e58b23519d0b4ca871350b011b56d91a3 (main)

Note: the Deeptools source tree reports `M dxp/SdscRelayoutInsertion.cpp` after the run; I did not edit source. The wrapper used `/home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108/build/deeptools/dxp/dxp_standalone` via `/home/adnan/codex-isolated/flash_attention_dldsc_20260703_170000/runs/replay_layout_classification_20260703_174650/../../tools/dxp_standalone`.

## Modes

| mode | rc | key relayout env |
| --- | ---: | --- |
| baseline | 0 | `SPYRE_LX_PLANNER_RELAYOUT=0`, collectives/layout-allgather/matmul-contract/restickify-outputs all `0`, `LX_BOUNDARY_CLONES=0` |
| metadata | 0 | `SPYRE_LX_PLANNER_RELAYOUT=1`, `SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1`, `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1`, `SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1`, `SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=0`, `LX_BOUNDARY_CLONES=1` |

## SDSC counts

Counts are parsed from top-level op names and nonempty `lxRelayoutClassifications_` arrays in `cache/inductor-spyre/**/sdsc_*.json`.

| mode | sdsc json files | ReStickifyOpHBM | ReStickifyOpLx | files with nonempty lxRelayoutClassifications_ | classification entries | kinds |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 550 | 32 | 0 | 0 | 0 | none |
| metadata | 550 | 0 | 32 | 32 | 32 | `layout_allgather_restickify:32` |

Metadata backend plan artifacts: 32 `*_layout_allgather_restickify_plan.json` files in `metadata/backend_plans`.
No `matmul_operand_broadcast` strings or `DEEPTOOLS_ENABLE_UNSAFE_MATMUL_OPERAND_BROADCAST` strings were found under either run directory.

## Result

Both baseline and metadata/classification modes completed with `SUCCESS`. This was a compile-probe replay: host-to-device copies were patched out and CPU reference/assert-close were skipped, matching the existing flash replay setup.
