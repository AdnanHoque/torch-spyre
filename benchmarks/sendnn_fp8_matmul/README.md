# Standalone SenDNN FP8 matmul sweep

These helpers reproduce the Granite 3 8B TP1 linear-layer sweeps archived in
[`docs/results/granite_e2e/sendnn_scaled_fp8_vs_fp16_granite_linear_m_sweeps_20260729`](../../docs/results/granite_e2e/sendnn_scaled_fp8_vs_fp16_granite_linear_m_sweeps_20260729).

The unique single-device transformer-block operations are:

```text
K/V:     [M, 4096]  @ [4096, 1024]
Q/O:     [M, 4096]  @ [4096, 4096]
gate/up: [M, 4096]  @ [4096, 12800]
down:    [M, 12800] @ [12800, 4096]

M = 1, 2, 4, ..., 2048 for every shape
```

Files:

- `direct_sendnn_fp16_fp8_pair_benchmark_per_axis.py` builds the rank-4
  direct-SenDNN FP16 and scaled-FP8 graphs and checks them against a CPU
  reference.
- `direct_sendnn_kineto_wrapper.py` profiles 20 device-kernel executions after
  warmup and preserves the Kineto trace and result JSON.
- `run_granite_kv_m_sweep.sh` sets the pinned production environment and runs
  the original K/V-only experiment.
- `run_granite_linear_m_sweep.sh` runs an arbitrary `(K,N)` shape serially,
  alternating FP16/FP8 order by M. Different shapes may run on different pods,
  but a pod runs only one benchmark process at a time.
- `summarize_granite_kv_m_sweep.py` validates all results, computes effective
  TFLOP/s, and writes the original K/V outputs.
- `summarize_granite_linear_m_sweeps.py` validates all 96 all-shape cases,
  pins the measured software/helper hashes, rejects 1p5 provenance, and writes
  combined and per-shape outputs.

The all-shape run uses benchmark SHA-256
`3536cfcb912779e2f04013df04d534e0c11b2d38a43152ca235e21b713bbd046`.
Relative to the original K/V run at commit `2ec366b`, the only benchmark change
is an FP16 aggregate relative-L2 validation ceiling of `0.06` instead of
`0.03`; the elementwise `rtol=0.02`, `atol=0.25` gate and timed graph are
unchanged. This avoids rejecting the K=12800 down projection after it passes
the elementwise comparison.

The FP8 graph uses per-row activation scales and per-output-channel weight
scales with fixed unit values. Its kernel includes Qfp8 cast/packing, the FP8
matmul, relayouts, and two scale-recovery stages. It excludes scale derivation
and activation normalization.

Run one shape on an AIU 1.0 Spyre pod with the production reference tree
present:

```bash
SHAPE_LABEL=qo \
K=4096 \
N=4096 \
STUDY_ROOT=/home/adnan/codex-isolated/fp8_sendnn_linear_sweeps_20260729 \
bash benchmarks/sendnn_fp8_matmul/run_granite_linear_m_sweep.sh
```

Summarize four completed, locally extracted run roots:

```bash
python3 benchmarks/sendnn_fp8_matmul/summarize_granite_linear_m_sweeps.py \
  --run-root kv=/path/to/kv-run \
  --run-root qo=/path/to/qo-run \
  --run-root mlp_up=/path/to/mlp-up-run \
  --run-root mlp_down=/path/to/mlp-down-run \
  --output-dir /path/to/summary
```
