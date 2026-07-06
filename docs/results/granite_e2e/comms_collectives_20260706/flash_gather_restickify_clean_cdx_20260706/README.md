# Flash Attention Gather-Restickify Clean Branch Probe, 2026-07-06

This records the clean CDX validation of the `gather-restickify` split branches.

## Branches

- Torch: `AdnanHoque/torch-spyre:gather-restickify`
  - SHA: `b84528d7e32ad0aea5f31d7de107344b35617695`
- Deeptools: `Adnan-Hoque1/deeptools:gather-restickify`
  - SHA: `393403f8205a089045e364a4e98ab7291e584618`
- Pod: `adnan-cdx-spyre-dev-pf`
- Clean root: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236`

## Backend Unit Gates

From the clean CDX Deeptools build:

- `./util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"`: 27/27 passed
- `./dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"`: 2/2 passed

## Flash Compile Probe

Script:

- Source: `github.ibm.com/aviros/test-spyre-scripts/test_flash.py`
- Shape: `B=1, H=32, D=128, Lq=4096, Lk=4096`
- Work division in script: `{H:4, Lq:8, Lk:8}`
- Mode: compile/runtime probe with host-to-device copies replaced by `empty_strided` and CPU reference/assert skipped.

This is not a value-correctness claim for the flash kernel. The known baseline zero-stride/broadcast issue is being investigated separately. This run only validates that the on-chip communication path compiles and lowers without introducing an additional DXP failure.

### Required Environment

The successful run needed all of these categories pinned:

- Clean Torch source on `PYTHONPATH`
- Clean Torch `_C.so` rebuilt in the clean worktree
- Clean Deeptools `dxp_standalone` first on `PATH`
- Clean Deeptools libraries first on `LD_LIBRARY_PATH`
- `DEEPTOOLS_PATH` set to the clean Deeptools source root
- `DEEPTOOLS_INSTALL_DIR` set to the clean Deeptools install root
- Split LX fraction handling:
  - `DXP_LX_FRAC_AVAIL=0`
  - `DXP_BACKEND_LX_FRAC_AVAIL=1`

Important gotcha: the pod default was `DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share`. With that default, the clean DXP binary looked for `restickify_lx.ddl` in the system install and failed. Pinning `DEEPTOOLS_PATH=$ROOT/deeptools` fixed the DDL lookup.

### Feature Flags

```bash
SPYRE_LX_PLANNER_RELAYOUT=1
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
```

## Results

### Metadata-Only Control

Run directory:

`/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_clean_gather_restickify_20260706_115524`

This run had `SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS` omitted.

- Return code: 0
- `ReStickifyOpHBM`: 32
- `ReStickifyOpLx`: 0
- SDSCs with matmul operand metadata: 32
- Backend plan files: 0

Interpretation: metadata was present, but the original HBM restickify remained, so the backend did not need to materialize the on-chip all-gather/restickify path.

### LX-Restickify Enabled Run

Run directory:

`/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_clean_gather_restickify_20260706_120319_restick_lx_pinned_dt`

- Return code: 0
- Printed `SUCCESS` in compile-probe mode
- SDSC files: 550
- `ReStickifyOpHBM`: 0
- `ReStickifyOpLx`: 64 string hits, 32 top-level op rows
- SDSCs with communication metadata: 32
- Backend plan files: 32
- Backend plan kind: `matmul_operand_broadcast`
- Communication pattern: `all_gather_replicate`
- Realization strategy: `gather_then_restickify`
- Logical transfers per plan: 256

This is the expected structural result: the activation-side HBM restickify handoff is removed, the producer side remains LX-resident, and Deeptools emits the staged gather/restickify plan for the matmul operand.

## Archived Artifacts

This directory includes:

- `metadata_only_structural_summary.json`
- `lx_restickify_structural_summary.json`
- `lx_restickify_roots.txt`
- `lx_restickify_stdout.log`
- `lx_restickify_stderr.log`
- `sample_105_batchmatmul_sdsc.json`
- `sample_105_matmul_operand_broadcast_plan.json`

The representative SDSC is `sample_105_batchmatmul_sdsc.json`. It shows:

- root op `105_batchmatmul`
- `kind = matmul_operand_broadcast`
- `communication_pattern = all_gather_replicate`
- producer context `104_ReStickifyOpLx`

The representative backend plan is `sample_105_matmul_operand_broadcast_plan.json`. It shows:

- `carrier_hint = lx_all_gather_then_local_restickify`
- movement stage `all_gather_replicate`
- conversion stage `local_restickify_to_kernel`
- 256 logical transfers

## Next Checks

- Run a smaller value-correct flash case only after the known zero-stride/broadcast TensorArg issue is fixed or bypassed.
- Convert the compile-probe into a profiled run once correctness is not masked by baseline flash issues.
- Keep CLC untouched while Claude owns it; continue using DEV/CDX for this lane.
