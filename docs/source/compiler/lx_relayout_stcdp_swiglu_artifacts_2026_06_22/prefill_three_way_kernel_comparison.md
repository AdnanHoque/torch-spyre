# Fused SwiGLU Prefill Three-Way Kernel Comparison

Shape: `B=1 S=512 E=4096`.

Primary metric is archived Kineto trace-derived `kernel_ms_per_iter` from the profiler-enabled run at `/tmp/lx_relayout_three_way_profile_20260622_081800`. Wall time is shown separately and should not be used as the primary speedup claim.

| Variant | Torch SHA | Deeptools SHA | Kernel ms/iter | Kernel speedup vs upstream main | Wall mean ms | Wall median ms | Memory ms/iter | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Upstream main baseline | `c6b357a` | `29254c37d3` | `16.153035` | `0.00%` | `19.600` | `17.544` | `0.246932` | pass |
| STCDPOpLx branch, relayout disabled control | `0f9bbcb` | `29254c37d3` | `16.306749` | `-0.95%` | `20.055` | `17.841` | `0.269957` | pass |
| STCDPOpLx LX relayout fixed branch | `0f9bbcb` | `29254c37d3` | `13.174297` | `18.44%` | `15.717` | `14.844` | `0.255713` | pass |
| Coordinate-remap reference branch | `b9b8f30f` | `83f9320cd6` | `13.145061` | `18.62%` | `22.994` | `22.274` | `0.287262` | pass |

## Interpretation

- The fixed STCDPOpLx/LX-planner-relayout branch recovers the same kernel-time win as the older coordinate-remap branch.
- The STCDPOpLx fixed branch is only `0.22%` slower than coordinate-remap on trace-derived kernel time in this run, which is within noise for this benchmark.
- The STCDPOpLx fixed branch has better wall time than the coordinate-remap reference in this sample. That supports using the STCDPOpLx carrier for the production branch, but the headline performance claim should remain trace-derived kernel time.
- The disabled-control run is within `0.95%` of upstream main, so the branch itself is not introducing material overhead when relayout is disabled.

## Artifact Pointers

- Upstream main: `latest_profile_fixed_branch/upstream_main_current/branch-baseline`
- STCDPOpLx disabled control: `latest_profile_fixed_branch/stcdp_disabled/branch-baseline`
- STCDPOpLx fixed relayout: `latest_profile_fixed_branch/stcdp_enabled/branch-baseline`
- Coordinate-remap reference: `latest_profile_fixed_branch/coordinate_remap_enabled/branch-baseline`
