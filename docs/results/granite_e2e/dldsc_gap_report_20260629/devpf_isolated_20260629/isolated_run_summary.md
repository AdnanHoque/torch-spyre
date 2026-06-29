# Isolated Granite Prefill Run - adnan-spyre-dev-pf - reservation fix

Run root: `/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114/runs/granite_prefill_artifact_merged_20260629_182634`

| Variant | Status | kernel ms/iter | median wall ms | kernel speedup vs baseline | wall speedup vs baseline | Notes |
|---|---:|---:|---:|---:|---:|---|
| `baseline_off` | pass | 14.697693 | 34.857512 | 1.000000 | 1.000000 | saved baseline from first isolated run |
| `boundary_relayout_lxfrac_0p2_graphfix` | pass | 14.581861 | 34.120083 | 1.007944 | 1.021613 | saved 0.2 endpoint from first isolated run |
| `boundary_full_torch_lx_backend1_graphfix` | pass | 12.014579 | 31.895638 | 1.223074 | 1.092861 | replan + corrected relayout reservation, DXP backend split wrapper, local GraphEditor fix |

## SDSC proof

The corrected optimized run emits 3 non-empty allocation `coreIdToWkSlice_` rows, proving dl-dsc coordinate relayout metadata is present in the before-DXP SDSCs.

## Key local hacks

- DXP split wrapper: Torch sees `DXP_LX_FRAC_AVAIL=0`; DXP sees `DXP_BACKEND_LX_FRAC_AVAIL=1` remapped to `DXP_LX_FRAC_AVAIL=1`.
- Local GraphEditor ReinterpretView wrapper preservation patch.
- Profiler-enabled `torch_spyre._C.so` overlay from `/home/adnan/dt-inductor/torch-spyre/torch_spyre/_C.so`.
