# Torch-Spyre FP8 Q/O matmul PoC

This directory contains the standalone benchmark prepared for the experimental
DD2 Torch-Spyre FP8 path. The target Granite Q/O operation is:

```text
[M, 4096] @ [4096, 4096]
M = 1, 2, 4, ..., 2048
```

The implementation patch and investigation handoff are in
[`../../docs/results/granite_e2e/torch_spyre_fp8_qo_poc_20260730/`](../../docs/results/granite_e2e/torch_spyre_fp8_qo_poc_20260730/).
The optimized QFP8MB path is still blocked in DeepTools code generation, so
this runner is prepared but has not produced an accepted optimized sweep.

## Measurement contract

`bench_qo_fp8_poc.py` reports effective matmul throughput as:

```text
2 * M * K * N / total Kineto cat=="kernel" time
```

The default sweep measures:

- FP16 matmul;
- baseline FP8 matmul; and
- optimized FP8 matmul with the experimental M/output work division.

Both FP8 variants prepack the static weight in a separate compiled graph. The
timed FP8 graph includes supplied-scale activation conversion and packing, the
FP8 matmul, explicit row/column output scaling, and FP16 output production. It
excludes weight prepacking, scale derivation, compilation, and transfer events.

The supplied scales are unit-valued. This isolates the execution structure but
does **not** validate the production non-unit `_scaled_mm` scale contract.

## Run

After applying the PoC patches and activating the pinned DD2 stack:

```bash
STUDY_ROOT=/path/to/results \
bash benchmarks/torch_spyre_fp8_matmul/run_qo_fp8_poc_sweep.sh
```

Useful overrides:

```text
FP8_M_VALUES="512"
FP8_WARMUPS=5
FP8_REPS=20
FP8_CORELET_PRELOAD=/path/to/libdeeptools_fp8mb_corelet_poc.so
```

The runner is serialized and refuses environments whose selected stack is
labeled `1p5`.
