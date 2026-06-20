# LX Coordinate Remap Benchmark Artifacts - 2026-06-20

These artifacts capture the profiler rerun after range-encoded coordinate-remap lowering.

Actual SHAs used:

- Upstream Torch main: `3db9efbae0182cc916ab7f5f36f38ffbbb05cc25`
- Torch branch `swiglu-ws-co-remap`: `763c2314f106d526be508aea3d801894cded3b83` for earlier branch runs; relay-fix FMS runs use `3ac4c1ed1d3564969fcfd15f07a0c7a5b9645d0b`.
- Deeptools coordinate-remap checkout: `83f9320cd6924833950c1aa362dfdb9abe0c29d7`
- `spyre-perf-suite` `jamie/dev`: `d73ea9b9d653f28c4391184eaf84e45e3b6fdfb5`

Upstream-main measurement note:

- A pristine current-main checkout did not run in this pod's PyTorch 2.12
  profiler stack: current main required newer `_C.so` bindings than the stale
  main binary, and after using a compatible profiler overlay it hit PyTorch's
  joint-graph lazy attention-pattern `no_dispatch()` failure.
- The `upstream-main` variant therefore uses current main Python at the SHA
  above, the profiler-enabled `_C.so` overlay from the known-good profiler
  checkout, and a measurement-only `torch_spyre/_inductor/patches.py` shim that
  disables `joint_graph.lazy_init` for Spyre.  The shim is archived as
  `upstream_main_measurement_patch.diff`.

Primary metric is trace/perf `kernel_ms_per_iter`; wall time is not used for speedup claims.

See `results_table.md` and `results.csv` for the summary.  See
`../../lx_coordinate_remap_swiglu_snapshot_2026_06_20.md` for the current
first-principles snapshot of how the FMS fused SwiGLU speedup was produced.

Each case/variant directory includes:

- `perf.txt`, `benchmark.log`, `env.json`, `commands.json`, `run_status.json`
- `artifacts/trace_summary.json`, `artifacts/sdsc_summary.json`, `artifacts/sdsc_table.md`, `artifacts/sdsc_table.csv`
- raw `sdsc_*.json` files under `sdsc_json/`
- raw Kineto trace JSON under `trace/` when available
- `sdsc_breakdown_jamie_style.md` and `.csv`, the original table-format dump
- `sdsc_jamie_summary.md`, `sdsc_jamie_table.md`, and `sdsc_jamie_table.csv`
  for newly generated Jamie-style operation summaries and screenshot-shaped
  tables
- `sdsc_hbm_roundtrip_comparison.md` for coordinate-remap variants that have a
  branch baseline
- `baseline_diff/sdsc_diff.md` for non-baseline variants where a branch baseline exists
- `upstream_diff/sdsc_diff.md` for non-upstream variants where an upstream-main baseline exists

Notes:

- `branch-baseline` is now explicitly separated from `upstream-main`.
- `prefill_bmm` and `jamie_mlp` show a roughly 4-5% trace kernel-time improvement with coordinate remap versus both upstream main and branch baseline, but memory-transfer time increases.
- `decode_bmm` emits no coordinate-remap rows and does not improve kernel time.
- The FMS namespaced `fms_granite_micro.swiglu` attempt did not produce artifacts: it stayed in baseline compile/tracing for roughly nine minutes, so it is not included as a timing row.
- `fms_swiglu_prefill_relayfix` is the FMS empty-weight fused SwiGLU prefill rerun after local relay chunking. Coordinate remap now realizes two mixed SDSCs and improves trace kernel time by 19.53% versus branch baseline.
- `fms_swiglu_decode_relayfix` is the B=1 S=1 control. It emits no remap rows and shows no meaningful speedup.
