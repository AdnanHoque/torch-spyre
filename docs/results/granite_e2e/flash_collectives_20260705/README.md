# Flash Attention Relayout Smoke - 2026-07-05

Pod: `adnan-spyre-dev-pf`
Namespace: `a6-quantization`
Scope: pod-local experiment/artifact repos only. No PR branches were checked out or modified.

## Provenance

- Experiment root: `/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129`
- Torch runtime checkout: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/torch-spyre`, branch `ah/comms-collectives`, sha `8960d88af18e31033a75e36450d8b6efcf9cf301`, status clean.
- Deeptools runtime checkout: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/deeptools`, branch `ah/comms-collectives`, sha `352919bf3f9c0efb2430568c667111aeb0a99e95`, dirty file `util/LayoutAllgatherRestickify.cpp`.
- Flash script checkout: `/tmp/test-spyre-scripts`, remote `git@github.ibm.com:aviros/test-spyre-scripts.git`, branch `main`, sha `afda166e58b23519d0b4ca871350b011b56d91a3`, file sha256 `622622e262e9829868d82cbb5632522625073bde9d49945d7a1385370e0a7818`.
- Artifact repo: `/home/adnan/codex-isolated/artifact_push/torch-spyre-ah-comms-collectives`, branch `ah/comms-collectives`.

## Result Matrix

| Variant | Run dir | RC | Correctness | HBM restickify | LX restickify | Backend plans | Notes |
|---|---|---:|---|---:|---:|---:|---|
| Relayout off | `/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/baseline_relayout_off_backend1_20260705_082034` | 0 | skipped by `PATCH_MODE=no_h2d,skip_cpu_ref`; stdout has skip line and `SUCCESS` | 33 files / 97 occurrences | 0 files / 0 occurrences | 0 | Existing 2026-07-05 comparison run. |
| Relayout on | `/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_backend1_20260705_082639` | 0 | skipped by `PATCH_MODE=no_h2d,skip_cpu_ref`; stdout has skip line and `SUCCESS` | 0 files / 0 occurrences | 33 files / 97 occurrences | 32 | Existing 2026-07-05 comparison run. |
| Relayout on, current main smoke | `/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_current_main_backend1_20260705_191932` | 0 | skipped by `PATCH_MODE=no_h2d,skip_cpu_ref`; stdout has skip line and `SUCCESS` | 0 files / 0 occurrences | 33 files / 97 occurrences | 32 | Fresh run against current `/tmp/test-spyre-scripts` main at `afda166e58b23519d0b4ca871350b011b56d91a3`; `RuntimeStream::synchronize` warning seen, no SIGSEGV/SIGABRT. |

All three runs generated 550 SDSC JSON files. The fresh relayout-on run has 32 files named `*_matmul_operand_broadcast_plan.json`; the backend-plan count above comes from `backend_plan_files.txt` / filename count.

## Env Flags

Relayout-off comparison:

```text
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=1
SPYRE_LX_PLANNER_RELAYOUT=0
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=0
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=0
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=0
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=0
LX_BOUNDARY_CLONES=0
PATCH_MODE=no_h2d,skip_cpu_ref
```

Relayout-on comparison and fresh smoke:

```text
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=1
SPYRE_LX_PLANNER_RELAYOUT=1
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
LX_BOUNDARY_CLONES=1
PATCH_MODE=no_h2d,skip_cpu_ref
```

## Commands And Artifacts

- `baseline_relayout_off_backend1_command.txt` and `relayout_on_backend1_command.txt` archive the existing off/on comparison commands and branch metadata.
- `relayout_on_current_main_backend1_exact_command.txt` contains the expanded fresh relayout-on smoke command with captured `PATH`, `LD_LIBRARY_PATH`, and `PYTHONPATH`.
- `relayout_on_current_main_backend1_run_command.sh` is the fresh run command wrapper copied from the run directory.
- Per-run summaries are copied into this directory as `*_run_summary.txt`.
- Fresh stdout/stderr tails are copied as `relayout_on_current_main_backend1_stdout_tail.txt` and `relayout_on_current_main_backend1_stderr_tail.txt`.

## Interpretation

Relayout-off keeps flash attention restickify ops on HBM and emits no backend plans. Relayout-on converts the same restickify evidence to LX and emits 32 backend plan files. The fresh current-main smoke confirms the relayout-on path still passes the compile/runtime probe with correctness skipped. This is not numerical correctness evidence because the bootstrap replaces H2D movement and skips `torch.testing.assert_close`.
