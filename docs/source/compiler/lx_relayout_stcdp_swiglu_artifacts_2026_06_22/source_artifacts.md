# Source Artifacts: STCDPOpLx LX Relayout Fused SwiGLU Prefill

The key archived run is:

## FMS Fused SwiGLU Prefill LX-Relayout Run

- Shape: `B=1 S=512 E=4096`
- Torch production branch: `pr-lx-planner-relayout-extension`
- Torch current branch SHA: `952d4baf5957ed44ca089902b8158e2caf53487d`
- Artifact-generation checkout SHA: `0ce25ae5baa9fb21545eb9f2b08325889f64c458`
- Tree hash shared by both Torch SHAs: `4a9b8c8dbad13bd838e65b84e3ec638fcc7c39ca`
- Upstream base SHA for the squashed PR branch: `c6b357a`
- Deeptools STCDPOpLx prototype SHA: `29254c37d3f2ee5c96a7323fdfd701026b63546c`
- spyre-perf-suite branch: `jamie/dev`
- spyre-perf-suite SHA: `d73ea9b9d653f28c4391184eaf84e45e3b6fdfb5`
- Primary structural metric: Jamie-style SDSC tables plus explicit ranged `STCDPOpLx` movement counters
- Timing note: this fresh artifact run used the direct FMS SwiGLU empty-weight path and did not capture nonzero Kineto kernel events in the short profiler run. Use the prior archived Kineto rerun for timing claims; use this bundle for current-branch SDSC evidence.

## Jamie-Style SDSC Artifacts

- [Prefill baseline summary](prefill_baseline_summary.md)
- [Prefill baseline table](prefill_baseline_table.md)
- [Prefill baseline table CSV](prefill_baseline_table.csv)
- [Prefill LX relayout summary](prefill_lx_relayout_summary.md)
- [Prefill LX relayout table](prefill_lx_relayout_table.md)
- [Prefill LX relayout table CSV](prefill_lx_relayout_table.csv)
- [Prefill HBM round-trip comparison](prefill_hbm_roundtrip_comparison.md)
- [STCDPOpLx movement counters](prefill_stcdp_lx_relayout_counters.json)
- [Prefill baseline run output](prefill_baseline_perf.txt)
- [Prefill LX relayout run output](prefill_lx_relayout_perf.txt)

## Current Structural Result

- Baseline: `9` SDSCs, `23` Jamie-table rows, `0` STCDPOpLx chunks.
- LX relayout: `9` SDSCs, `28` Jamie-table rows, `5` STCDPOpLx chunks in `1` mixed SDSC.
- On-chip movement: `524` movement ranges, `6600` expanded movements, `13516800` bytes moved through ranged `STCDPOpLx`.
- HBM round trip eliminated for the first projection half feeding `neg` and `realdiv`.
- Second-half `mul` input and final pointwise product output remain HBM-backed in this PR1 production branch.

## Prior Coordinate-Remap Reference

For comparison with the older relay-fix coordinate-remap artifact, see [the June 20 snapshot](https://github.com/AdnanHoque/torch-spyre/blob/b9b8f30ffa987e22babd8f253fd0d468ffa61f79/docs/source/compiler/lx_coordinate_remap_swiglu_snapshot_2026_06_20.md).
