# LX Coordinate Remap Benchmark Artifacts - 2026-06-20

These artifacts capture the profiler rerun after range-encoded coordinate-remap lowering.

Actual SHAs used:

- Torch branch `swiglu-ws-co-remap`: `763c2314f106d526be508aea3d801894cded3b83`
- Deeptools coordinate-remap checkout: `83f9320cd6924833950c1aa362dfdb9abe0c29d7`
- `spyre-perf-suite` `jamie/dev`: `d73ea9b9d653f28c4391184eaf84e45e3b6fdfb5`

Primary metric is trace/perf `kernel_ms_per_iter`; wall time is not used for speedup claims.

See `results_table.md` and `results.csv` for the summary.

Each case/variant directory includes:

- `perf.txt`, `benchmark.log`, `env.json`, `commands.json`, `run_status.json`
- `artifacts/trace_summary.json`, `artifacts/sdsc_summary.json`, `artifacts/sdsc_table.md`, `artifacts/sdsc_table.csv`
- raw `sdsc_*.json` files under `sdsc_json/`
- raw Kineto trace JSON under `trace/` when available
- `sdsc_breakdown_jamie_style.md` and `.csv`, matching the comparison columns from the shared screenshot
- `baseline_diff/sdsc_diff.md` for non-baseline variants where a branch baseline exists

Notes:

- `prefill_bmm` and `jamie_mlp` show a roughly 4-5% trace kernel-time improvement with coordinate remap, but memory-transfer time increases.
- `decode_bmm` emits no coordinate-remap rows and does not improve kernel time.
- The FMS namespaced `fms_granite_micro.swiglu` attempt did not produce artifacts: it stayed in baseline compile/tracing for roughly nine minutes, so it is not included as a timing row.
- Upstream-main profiling was attempted, but current main was not ABI-compatible with the profiler `_C.so` overlay and failed before trace generation.
