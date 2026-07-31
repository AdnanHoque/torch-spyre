# Torch-Spyre DD2 FP8 Q/O matmul benchmark

This benchmark exercises the Granite Q/O projection:

```text
[M, 4096] @ [4096, 4096]
M = 1, 2, 4, ..., 2048
```

The default serialized sweep runs:

- FP16 matmul;
- automatic FP8 with static weight prepacking and activation conversion in the
  timed graph; and
- raw FP8 with both activation and weight prepacked outside the timed graph.

The raw control can be disabled with `FP8_INCLUDE_RAW_CONTROL=0`. Optional
diagnostics are `FP8_INCLUDE_SCALED=1` for the explicit scale-application path
and `FP8_FORCE_ORACLE_AB=1` for the old forced work-division A/B.

Effective throughput is `2*M*K*N` divided by total Kineto
`cat == "kernel"` time. Compilation, transfers, and prepacking graphs are not
included. Scale derivation is excluded and unit FP16 quantization scales are
supplied, so this is not yet the production dynamic-scaled contract.

## Run

Activate the pinned DD2 environment, then run:

```bash
STUDY_ROOT=/path/to/results \
bash benchmarks/torch_spyre_fp8_matmul/run_qo_fp8_poc_sweep.sh
```

Useful controls:

```text
FP8_M_VALUES="512"
FP8_WARMUPS=5
FP8_REPS=20
FP8_CORELET_PRELOAD=/path/to/libdeeptools_fp8mb_corelet_poc.so
FP8_DXP_PRELOAD=/opt/ibm/spyre/deeptools/lib/libdxp.so
```

The runner refuses an environment containing `1p5`, executes cases serially,
and never launches two benchmarks on one device.

Accepted results and limitations are documented in
[`../../docs/results/granite_e2e/torch_spyre_fp8_planner_20260731/`](../../docs/results/granite_e2e/torch_spyre_fp8_planner_20260731/).
