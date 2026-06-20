# fms_swiglu_prefill_relayfix

spyre-perf-suite `jamie/dev` FMS empty-weight fused SwiGLU, relay-fix build, B=1 S=512 E=4096.

This is the first FMS fused prefill run where coordinate-remap realizes both useful subviews of the fused projection:

- first-half projection remap before `neg`;
- first-half LX reuse for `realdiv`;
- second-half projection remap before `mul`.

The upstream-main run is not included for this case because current `origin/main` at `eb86364` trips the empty-weight wrapper in Dynamo with a `no_dispatch()` fake-tensor failure. A real-weight upstream retry was stopped after several minutes of silent setup; use branch-baseline versus coordinate-remap for the pass delta here.

Jamie-style comparison artifacts:

- `branch-baseline/sdsc_jamie_summary.md`
- `branch-baseline/sdsc_jamie_table.md`
- `branch-baseline/sdsc_jamie_table.csv`
- `coordinate-remap/sdsc_jamie_summary.md`
- `coordinate-remap/sdsc_jamie_table.md`
- `coordinate-remap/sdsc_jamie_table.csv`
- `coordinate-remap/sdsc_hbm_roundtrip_comparison.md`

The key proof is that baseline `neg`, `realdiv`, and the second `mul` input read HBM-backed projection data, while coordinate remap inserts `LXCoordinateRemapOp` rows and patches those consumer inputs to LX.
