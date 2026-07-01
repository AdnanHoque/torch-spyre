# Explicit grouped-remap checkpoint 20260701

## Summary

The grouped explicit-remap path advanced past the DCC stitcher failure. The clean replay of the saved grouped `sdsc_10` bundle completed with `rc=0` and empty stderr after a one-line logic fix plus local scheduling guard in `dcc/src/Stitcher/ModuleStitcher.cpp`.

## Root cause

The failure is schedule-step/unit reuse in DCC stitching for mixed DL/data SDSC schedules, not data-op naming and not grouped row expansion shape by itself.

The grouped explicit STCDP-LX data op produced two DCC dataflow modules for `10_batchmatmul`:

- `stitcher_dataflow_ir_datadsc_0.mlir`: DL-side L3 PCFG module from `sdsc_.pcfgPool_`.
- `stitcher_dataflow_ir_datadsc_1.mlir`: explicit STCDP-LX data-op module from the grouped remap.

For several destination-only cores, the core schedule only contains the explicit data module. Before the fix, `ModuleStitcher::fillStitchMapWithFunction` initialized `idx = module_idx`; if the module was not present in `coreid_to_module_idx[coreId]`, it left `idx` unchanged and inserted units from an unscheduled data module into that core schedule slot. The explicit data module then inserted into the same slot and hit `unit already set for associated schedule step`.

## Patch

Files patched on pod:

- `/home/adnan/codex-isolated/explicit_range_agent_20260630/deeptools/dcc/src/Stitcher/ModuleStitcher.cpp`
- `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools/dcc/src/Stitcher/ModuleStitcher.cpp`

Patch location: `fillStitchMapWithFunction`, around lines 259-272. For non-DLDSC modules, the stitcher now searches the per-core module order and skips units when that data module is absent from the core schedule.

Diff artifacts:

- `diff_runtime_source.patch`
- `diff_real_attention_source.patch`

## Rebuild

Rebuilt the runtime tree that the workspace binary links against:

`cd /home/adnan/codex-isolated/explicit_range_agent_20260630/deeptools; /usr/bin/ninja -C build-explicit-range-agent-nollvm dxp/dxp_standalone -j16`

The final clean rebuild after removing diagnostics relinked in 5 steps in both the runtime source tree and the known workspace source tree: `ModuleStitcher.cpp.o`, `libStitcher.a`, `libdcc.so`, `libdxp.so`, and `dxp_standalone`.

## Replay

Final replay directories:

- Runtime binary: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/explicit_grouped_checkpoint_20260701_051726/replay_clean_patch`
- Workspace binary: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/explicit_grouped_checkpoint_20260701_051726/replay_workspace_clean_patch`

Command shape:

`DEEPTOOLS_PATH=/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/deeptools DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1 /home/adnan/codex-isolated/explicit_range_agent_20260630/deeptools/build-explicit-range-agent-nollvm/dxp/dxp_standalone --bundle -d /home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/explicit_range_grouped_sdsc10_20260701_024349/bundle_input`

Results: runtime binary `rc=0`, workspace binary `rc=0`, stderr lines: 0 for both.

## Artifacts

- Original grouped run: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/explicit_range_grouped_sdsc10_20260701_024349`
- DCC IR dumps used for diagnosis: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212/runs/explicit_grouped_checkpoint_20260701_051726/replay_explicit_dump/codegen_dumps/10_batchmatmul`
- Diagnostic replay after patched runtime: `replay_old_runtime_diagnostic`, result `rc=0`
- Final clean replay after removing diagnostics with runtime binary: `replay_clean_patch`, result `rc=0`
- Final clean replay after workspace relink: `replay_workspace_clean_patch`, result `rc=0`

## Status

Explicit grouped-remap advanced past the stitcher error. No push was performed.
