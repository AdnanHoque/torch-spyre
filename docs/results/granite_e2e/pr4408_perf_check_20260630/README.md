# PR 4408 Granite Prefill Perf Check

Workspace: `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905`
Run/artifacts root: `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/runs/granite_prefill_pr4408_20260630_232800`

## SHAs
- Torch: `df26b2ec7c14159e835a288d3369e7971661c43b`
- spyre-granite-e2e-bench: `76cd51426ba1de6e99dd8fbf613cb0f32b71e87f`
- Deeptools PR 4408 head: `4502294b344bed3c1955e3cd276a7219151505c3` ([DXP] relayout insertion -- code cleanup and minor misc fixes)
- PR base from `gh pr view`: `0a9da5eb19d08712383312bb7dec18fbd7caf711`
- Upstream `master` from `git ls-remote` at fetch time: `9b22ca42792d8d2a9f72e3f3869d4132e6710069`
- PR merge ref fetched: `323dea254349e5285c0b088dab3b532b644c5773`

## Build
```bash
cmake -S $ROOT/deeptools -B $ROOT/deeptools/build-dxp-pr4408 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMANAGE_LLVM=false \
  -DLLVM_PROJ_SRC=/home/adnan/dt-inductor/llvm-project \
  -DLLVM_PROJ_BUILD=/home/adnan/dt-inductor/build/llvm \
  -DCMAKE_INSTALL_PREFIX=$ROOT/deeptools/install-pr4408
cmake --build $ROOT/deeptools/build-dxp-pr4408 --target dxp_standalone -j 16
```
DXP binary: `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/deeptools/build-dxp-pr4408/dxp/dxp_standalone`
Split wrapper: `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/tools/dxp-split-wrapper-pr4408/dxp_standalone`

## Runtime Env
Common: `PYTHONPATH=$TORCH:$TORCH/tests/inductor:$FMS`, `PATH=$ROOT/tools/dxp-split-wrapper-pr4408:$PATH`, `DEEPTOOLS_PATH=$ROOT/deeptools`, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, and LD pin starting with `/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib`.
- Baseline: `SPYRE_LX_PLANNER_RELAYOUT=0`, `LX_BOUNDARY_CLONES=0`, `DXP_LX_FRAC_AVAIL=0.2`, `DXP_BACKEND_LX_FRAC_AVAIL` unset.
- Optimized: `SPYRE_LX_PLANNER_RELAYOUT=1`, `LX_BOUNDARY_CLONES=1`, `DXP_LX_FRAC_AVAIL=0`, `DXP_BACKEND_LX_FRAC_AVAIL=1`, `SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES` unset.

## Command
```bash
python benchmarks/granite_block_layer_probe.py \
  --fms-root /home/adnan/dt-inductor/foundation-model-stack \
  --run-root <variant-run-root> \
  --case prefill \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 5 \
  --warmups 1 \
  --profile \
  --no-profile-memory
```

## Results
| variant | status | kernel ms/iter | wall median ms | kernel speedup | wall speedup |
|---|---:|---:|---:|---:|---:|
| baseline relayout off | pass | 14.685101 | 27.928829 | 1.000000x | 1.000000x |
| full Torch LX + backend LX=1 | pass | 12.023446 | 24.392128 | 1.221372x | 1.144994x |

Conclusion: 1.2x kernel speedup is preserved (`1.221x` observed). No compile/import/runtime failure occurred; DXP replay was not needed.

## Artifacts
- baseline relayout off result: `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/runs/granite_prefill_pr4408_20260630_232800/baseline_off/block_prefill/result.json`
- baseline relayout off trace summary: `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/runs/granite_prefill_pr4408_20260630_232800/baseline_off/block_prefill/trace_summary.json`
- full Torch LX + backend LX=1 result: `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/runs/granite_prefill_pr4408_20260630_232800/full_lx_backend1/block_prefill/result.json`
- full Torch LX + backend LX=1 trace summary: `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/runs/granite_prefill_pr4408_20260630_232800/full_lx_backend1/block_prefill/trace_summary.json`
- Build log: `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/logs/build_dxp_pr4408.log`
- Fetch/metadata logs: `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/logs/pr4408_fetch.log`, `/home/adnan/codex-isolated/pr4408_perf_check_20260630_231905/logs/pr4408_metadata.log`
