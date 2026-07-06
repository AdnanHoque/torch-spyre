# Reduce / All-Reduce Boundary For DLDSC LX Relayout - 2026-07-06

## Branches

- Torch branch: `gather-restickify`
- Torch SHA: `360d3abeb3c257bf48334ac45164ed7d4474800b`
- Deeptools branch: `gather-restickify`
- Deeptools SHA: `e3e265d22c7283054dd36e147a7e7ec919606441`

## Summary

The DLDSC LX relayout path is a copy-movement substrate. It can represent scatter/permutation, broadcast, multicast, gather, and all-gather shaped movement when the producer values are already final tensor values.

Reduce and all-reduce are different: they require arithmetic combining. They must not be implemented by blindly gathering partial sums as if they were final values.

## Current Contract

Torch intentionally skips relayout planning when the producer has partial reduction output:

- `torch_spyre/_inductor/lx_relayout.py:442-451`
- guard test: `tests/inductor/test_lx_relayout_dldsc.py:test_partial_reduction_outputs_are_not_copy_relayout_candidates`

Deeptools has separate DL scheduler machinery for cross-core reductions:

- `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2989-3025` detects cross-core reductions and builds reduction core groups.
- `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3028-3045` selects output transfer cores for cross-core reduction outputs.
- `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3446-3515` conditionally scopes transfer nodes to reduction output cores.
- `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7475-7503` customizes LX output allocation coordinates for cross-core reduction output tensors.

That means reduce/all-reduce should be treated as a producer-op arithmetic collective/lowering problem, not as another `STCDPOpLx` copy relayout case.

## Verification

Torch focused test:

```bash
cd /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/torch-spyre
source /home/adnan-cdx/dt-inductor/.venv/bin/activate
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=$PWD:${PYTHONPATH:-}
python3 -m pytest tests/inductor/test_lx_relayout_dldsc.py -q
```

Result: `20 passed`.

Deeptools generic relayout test:

```bash
cd /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools/build-deeptools
export DEEPTOOLS_PATH=/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools
./dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
```

Result: `2 passed`.

## Consequence For The Granite Spill Goal

For the current Granite S512 profiled run, remaining visible `ReStickifyOpHBM` rows classify as weight/prelayout rows, not missing copy collectives. The currently proven copy-collective substrate is enough for the in-scope attention activation handoff we observed.

Future reduce/all-reduce work should start from workload evidence that a non-weight HBM spill is caused by arithmetic partials escaping a producer op. The implementation shape should then extend the producer reduction lowering/scheduler contract, not the copy-only LX relayout path.
