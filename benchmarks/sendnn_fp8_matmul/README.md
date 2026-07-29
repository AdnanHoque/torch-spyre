# Standalone SenDNN FP8 matmul sweep

These helpers reproduce the Granite KV projection sweep archived in
[`docs/results/granite_e2e/sendnn_scaled_fp8_vs_fp16_kv_m_sweep_20260729`](../../docs/results/granite_e2e/sendnn_scaled_fp8_vs_fp16_kv_m_sweep_20260729).

The fixed operation is:

```text
[M, 4096] @ [4096, 1024]
M = 1, 2, 4, ..., 2048
```

Files:

- `direct_sendnn_fp16_fp8_pair_benchmark_per_axis.py` builds the rank-4
  direct-SenDNN FP16 and scaled-FP8 graphs and checks them against a CPU
  reference.
- `direct_sendnn_kineto_wrapper.py` profiles 20 device-kernel executions after
  warmup and preserves the Kineto trace and result JSON.
- `run_granite_kv_m_sweep.sh` sets the pinned production environment and runs
  every case serially, alternating FP16/FP8 order by M.
- `summarize_granite_kv_m_sweep.py` validates all results, computes effective
  TFLOP/s, and writes CSV, JSON, Markdown, and SVG outputs.

The benchmark and Kineto wrapper are preserved byte-for-byte from the measured
run so their SHA-256 values match `provenance.txt`. `ruff check` passes; a
format-only rewrite of those two measured files is intentionally not applied.

The FP8 graph uses per-row activation scales and per-output-channel weight
scales with fixed unit values. Its kernel includes Qfp8 cast/packing, the FP8
matmul, relayouts, and two scale-recovery stages. It excludes scale derivation
and activation normalization.

Run on a Spyre pod with the production reference tree present:

```bash
STUDY_ROOT=/home/adnan/codex-isolated/fp8_sendnn_kineto_current_20260729 \
bash benchmarks/sendnn_fp8_matmul/run_granite_kv_m_sweep.sh
```

Summarize a completed run:

```bash
python3 benchmarks/sendnn_fp8_matmul/summarize_granite_kv_m_sweep.py \
  /path/to/run-root \
  --output-dir /path/to/summary
```
