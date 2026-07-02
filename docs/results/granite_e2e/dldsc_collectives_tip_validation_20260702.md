# DLDSC Collectives Tip Validation - 2026-07-02

## Scope

Pod: `adnan-spyre-dev-pf`.

Pod-local workspace: `/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302`.

Branches pulled and validated:

- Torch: `AdnanHoque/torch-spyre ah/comms-collectives` at `687340f742d32e3589991ffda1b94110be94f6a5` (`docs: record dldsc mixed ifn route checkpoint`).
- Deeptools: `Adnan-Hoque1/deeptools ah/comms-collectives` at `3c7b754f04a032bd181d4cb43fa35aaf74a4686f` (`[DXP] Route scheduled mixed input-fetch SDSCs`).
- Granite harness: `/home/adnan/codex-isolated/comms_collectives_20260629/spyre-granite-e2e-bench` at `76cd51426ba1de6e99dd8fbf613cb0f32b71e87f`.
- FMS root: `/home/adnan/dt-inductor/foundation-model-stack`.
- Python: `/home/adnan/dt-inductor/.venv/bin/python3`.
- Deeptools DXP build: `/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/build-deeptools/dxp/dxp_standalone`.

A local runtime `_C.so` build artifact was copied into the fresh Torch checkout from `/home/adnan/codex-isolated/comms_collectives_20260629/torch-spyre/torch_spyre/_C.so`; it is ignored and was not committed.

## Build

Commands recorded in `/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/commands_build.txt`.

```bash
git clone --single-branch --branch ah/comms-collectives git@github.com:AdnanHoque/torch-spyre.git torch-spyre
git clone --single-branch --branch ah/comms-collectives git@github.ibm.com:Adnan-Hoque1/deeptools.git deeptools
/home/adnan/dt-inductor/.venv/bin/cmake -S /home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/deeptools -B /home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/build-deeptools -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/install-deeptools -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DMANAGE_LLVM=0 -DLLVM_PROJ_SRC=/home/adnan/dt-inductor/llvm-project -DLLVM_PROJ_BUILD=/home/adnan/dt-inductor/build/llvm -DDT_BUILD_DOCUMENTATION=OFF -DDT_DR5_USE_IREE=OFF -DDT_DR5_USE_TORCH_SCRIPT=OFF -DDT_USE_DCC_DDC=ON
/home/adnan/dt-inductor/.venv/bin/cmake --build /home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/build-deeptools --target dxp_standalone -j 32
```

Build result: pass. Logs are in `/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/logs/`.

## Runs

| Run | Result | Key env | Evidence |
|---|---|---|---|
| Baseline Granite prefill, no profile | Pass, `rc=0`, wall `102s`, 44 SDSC JSONs, 4 bundles | `DXP_LX_FRAC_AVAIL=0.2`, relayout flags unset | `/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/runs/granite_prefill_baseline_20260702_113342` |
| Optimized DLDSC split Granite prefill, no profile | Fail, `rc=1`, wall `95s`, 0 SDSCs, 0 bundles, 0 backend plan files | Torch `DXP_LX_FRAC_AVAIL=0`; wrapper maps backend `DXP_BACKEND_LX_FRAC_AVAIL=1` to DXP `DXP_LX_FRAC_AVAIL=1`; `SPYRE_LX_PLANNER_RELAYOUT=1`; `LX_BOUNDARY_CLONES=1`; `SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1`; `SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1`; `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1`; `SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1` | `/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/runs/granite_prefill_optimized_split_20260702_113116` |

The full expanded commands are in each run directory as `command.txt`. The logs are `stdout.log`, `stderr.log`, and `block_prefill/result.json`.

Baseline output matched the probe success shape:

```text
output_shape: [1, 512, 4096]
cache_shape: [[1, 8, 512, 128], [1, 8, 512, 128]]
median wall ms, one measured iteration: 28.7017822265625
```

This is not a speedup number; it is a non-profile wall-sync smoke run.

## First Blocker

The optimized split-DXP run fails before SDSC emission and before DXP/backend plan generation. The first blocker is a Torch frontend scratchpad graph-edit assertion while replacing `buf11`, which is a `ReinterpretView` graph output from the RoPE path.

Exact log path:

```text
/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/runs/granite_prefill_optimized_split_20260702_113116/block_prefill/result.json
/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/runs/granite_prefill_optimized_split_20260702_113116/stdout.log
/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/runs/granite_prefill_optimized_split_20260702_113116/stderr.log
```

Key traceback excerpt:

```text
torch_spyre/_inductor/scratchpad/graph_editor.py", line 71, in _replace_matching_buffer
  assert isinstance(buffer, StorageBox), (
torch._inductor.exc.InductorError: AssertionError: unexpected buffer type <class torch._inductor.ir.ReinterpretView> while replacing buf11
...
File "/home/adnan/dt-inductor/foundation-model-stack/fms/modules/positions.py", line 359, in adjusted_qk,
  .sum(4, keepdim=True),
```

The optimized run had `0` `sdsc_*.json`, `0` `bundle.mlir`, and `0` files under `backend_plans`, so this is not yet the Deeptools mixed-IFN or operand-aware IFN backend blocker described in the branch notes. The current branch tip is blocked earlier in Torch scratchpad graph output replacement under the split DLDSC relayout setup.

## Speedup Status

The existing `1.19-1.2x` speedup does not reproduce on the latest tips in this validation. The optimized split-DXP path fails before profiler timing can start, so there is no optimized `kernel_ms_per_iter` to compare against baseline.
