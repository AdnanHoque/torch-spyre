# fms_swiglu_decode_relayfix

spyre-perf-suite `jamie/dev` FMS empty-weight fused SwiGLU, relay-fix build, B=1 S=1 E=4096.

This decode-shaped run is included as a control. The coordinate-remap planner does not emit remap rows for this shape, and kernel time is effectively unchanged.

Jamie-style comparison artifacts:

- `branch-baseline/sdsc_jamie_summary.md`
- `branch-baseline/sdsc_jamie_table.md`
- `branch-baseline/sdsc_jamie_table.csv`
- `coordinate-remap/sdsc_jamie_summary.md`
- `coordinate-remap/sdsc_jamie_table.md`
- `coordinate-remap/sdsc_jamie_table.csv`
- `coordinate-remap/sdsc_hbm_roundtrip_comparison.md`

The comparison file is expected to show no HBM-trip removal: both variants have zero remap chunks, and the pointwise consumers remain HBM-backed.
