# Torch-Spyre DD2 FP8 Q/O matmul benchmarks

These benchmarks exercise the Granite Q/O projection:

```text
[M, 4096] @ [4096, 4096]
M = 1, 2, 4, ..., 2048
```

## Dynamic scaled-matmul LX experiment

`run_qo_fp8_lx_contract_sweep.sh` compares three complete timed paths:

- matched FP16 matmul;
- baseline FP8 with dynamic per-row activation-scale derivation, activation
  normalization/packing, FP8 matmul, and both output-scale applications; and
- the same FP8 value contract with the DD2 Q/O LX/work-division PoC enabled.

Static weight prepacking is performed in a separate compiled graph and excluded
from timing. Set both dynamic-scale controls for the production-shaped run:

```bash
FP8_DERIVE_ACTIVATION_SCALE=1 \
FP8_FUSED_ACTIVATION_SCALE=1 \
STUDY_ROOT=/path/to/results \
bash benchmarks/torch_spyre_fp8_matmul/run_qo_fp8_lx_contract_sweep.sh
```

The optimized path fixes the Q/O matmul at `M:8 x N:4 x K:1`, permits explicit
LX ownership relayouts only for source tensors between 64 KiB and 4 MiB, and
uses DD2's specialized per-row FP8 scale reduction. Override those experimental
bounds with:

```text
FP8_LX_RELAYOUT_MIN_SOURCE_BYTES
FP8_LX_RELAYOUT_MAX_SOURCE_BYTES
```

The runner refuses an environment containing `1p5`, executes cases serially,
and never launches two benchmarks on one device. Accepted results and emitted
bundle evidence are documented in
[`../../docs/results/granite_e2e/torch_spyre_fp8_lx_qscale_poc_20260801/`](../../docs/results/granite_e2e/torch_spyre_fp8_lx_qscale_poc_20260801/).

## Earlier path-isolation sweep

`run_qo_fp8_poc_sweep.sh` runs:

- FP16 matmul;
- automatic FP8 with static weight prepacking and activation conversion in the
  timed graph; and
- raw FP8 with both activation and weight prepacked outside the timed graph.

The raw control can be disabled with `FP8_INCLUDE_RAW_CONTROL=0`. Optional
diagnostics are `FP8_INCLUDE_SCALED=1` for the explicit scale-application path
and `FP8_FORCE_ORACLE_AB=1` for the old forced work-division A/B.

Effective throughput is `2*M*K*N` divided by Kineto `cat == "kernel"` time.
Compilation, transfers, and prepacking graphs are not included. Scale derivation
is excluded and unit FP16 quantization scales are supplied in this earlier
sweep, so it is a path-isolation experiment rather than the dynamic
scaled-matmul contract.

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

Accepted results and limitations are documented in
[`../../docs/results/granite_e2e/torch_spyre_fp8_planner_20260731/`](../../docs/results/granite_e2e/torch_spyre_fp8_planner_20260731/).
