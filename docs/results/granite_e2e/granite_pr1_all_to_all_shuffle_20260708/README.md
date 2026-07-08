# Granite PR1 `all_to_all_shuffle` Artifacts

This directory archives the fresh whole-Granite-block prefill run used to explain what PR1 changes structurally.

## Open First

- `granite_pr1_all_to_all_shuffle_before_after.md`: concise before/after explanation using simple formulas such as `S = Q @ K.T` and tensor residency tables.
- `granite_pr1_rescue_sdsc_edge_report.md`: lower-level SDSC edge classification from the same run.
- `sdsc_reports/`: Jamie-style `summarize-sdsc` reports for each generated SDSC directory.

## Timing

| Variant | Kernel ms/iter | Wall median ms | Speedup |
|---|---:|---:|---:|
| Baseline, relayout off | 14.7002 | 28.0849 | 1.000x |
| PR1 `all_to_all_shuffle` on | 12.0353 | 24.9376 | 1.221x kernel / 1.126x wall |

Primary metric is Kineto trace-derived `kernel_ms_per_iter`.

## What PR1 Removes

PR1 removes two non-weight HBM activation handoffs in the Granite attention path:

1. `Q` input to the score matmul, `S = Q @ K.T`: `0_hbm -> 0_lx`.
2. `C` input to the output projection, `O = C @ W_o`: `0_hbm -> 0_lx`.

It does not remove MLP/SwiGLU activation spills in this run. Those are PR2/all-gather-restickify or later WSR-shaped work.

## Source Run

The source run was produced on the pod at:

`/home/adnan/codex-isolated/pr1_rescue_compare_20260708/runs/granite_rescue_device_20260708_200414`

The copied run metadata includes:

- `rescue_perf_summary.json`
- `rescue_run_summary.json`
- `baseline_off/block_prefill/trace_summary.json`
- `rescue_full_torch_lx_backend1/block_prefill/trace_summary.json`
- `run_env_common.sh`
