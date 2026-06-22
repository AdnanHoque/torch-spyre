# Source Artifacts: STCDPOpLx LX Relayout Fused SwiGLU Prefill

The key archived run is:

## FMS Fused SwiGLU Prefill LX-Relayout Run

- Shape: `B=1 S=512 E=4096`
- Torch production branch: `pr-lx-planner-relayout-extension`
- Torch fixed branch SHA: `0f9bbcb682c80d9de9a1c2708a3b336b50f6868c`
- Upstream base SHA for the squashed PR branch: `c6b357a`
- Deeptools STCDPOpLx prototype SHA: `29254c37d3f2ee5c96a7323fdfd701026b63546c`
- spyre-perf-suite branch: `jamie/dev`
- spyre-perf-suite SHA: `d73ea9b9d653f28c4391184eaf84e45e3b6fdfb5`
- Profiler harness SHA: `76cd51426ba1de6e99dd8fbf613cb0f32b71e87f`
- Primary timing metric: archived Kineto trace-derived `kernel_ms_per_iter`
- Primary structural metric: Jamie-style SDSC tables plus explicit ranged `STCDPOpLx` movement counters

## Jamie-Style SDSC Artifacts

- [Prefill three-way kernel comparison](prefill_three_way_kernel_comparison.md)
- [Prefill three-way kernel comparison CSV](prefill_three_way_kernel_comparison.csv)
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
- [Prefill coordinate-remap reference run output](prefill_coordinate_remap_perf.txt)
- [Raw fixed-branch profiler artifacts](latest_profile_fixed_branch/)

## Current Structural Result

- Baseline: `9` SDSCs, `23` Jamie-table rows, `0` STCDPOpLx chunks.
- LX relayout: `9` SDSCs, `33` Jamie-table rows, `10` STCDPOpLx chunks in `2` mixed SDSCs.
- On-chip movement: `1048` movement ranges, `13200` expanded movements, `27033600` bytes moved through ranged `STCDPOpLx`.
- HBM round trips eliminated for both fused-projection halves feeding `neg`/`realdiv` and `mul`.
- Final pointwise product output remains HBM-backed for the downstream matmul; down-projection fan-out/streaming remains follow-up work.

## Current Timing Result

| Variant | Kernel ms/iter | Kernel speedup vs upstream main | Wall mean ms | Wall median ms |
| --- | ---: | ---: | ---: | ---: |
| Upstream main `c6b357a` | `16.153035` | `0.00%` | `19.600` | `17.544` |
| STCDPOpLx relayout disabled control `0f9bbcb` | `16.306749` | `-0.95%` | `20.055` | `17.841` |
| STCDPOpLx fixed relayout `0f9bbcb` | `13.174297` | `18.44%` | `15.717` | `14.844` |
| Coordinate-remap reference `b9b8f30f` | `13.145061` | `18.62%` | `22.994` | `22.274` |

## Prior Coordinate-Remap Reference

For comparison with the older relay-fix coordinate-remap artifact, see [the June 20 snapshot](https://github.com/AdnanHoque/torch-spyre/blob/b9b8f30ffa987e22babd8f253fd0d468ffa61f79/docs/source/compiler/lx_coordinate_remap_swiglu_snapshot_2026_06_20.md).
