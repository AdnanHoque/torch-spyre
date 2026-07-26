# Full-model SenDNN versus Torch-Spyre Granite runbook

This runbook reproduces the July 25, 2026 full 40-layer comparison of the
SenDNN and Torch-Spyre Granite stacks.

## 1. Headline result

Both fresh runs used five measured generations after compile and device
warmup. Each generation contains one 512-token prefill and three decode calls.

| Trace-derived device-program time | SenDNN | Torch-Spyre | SenDNN change | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 190.406 ms | 301.442 ms | 36.84% lower | 1.5832x |
| Decode average | 123.961 ms | 153.592 ms | 19.29% lower | 1.2390x |

The decode average covers the first decode and two steady decode calls. The
per-call means were:

| Phase | SenDNN | Torch-Spyre |
| --- | ---: | ---: |
| First decode | 123.920 ms | 154.546 ms |
| Steady decode 1 | 124.025 ms | 153.172 ms |
| Steady decode 2 | 123.937 ms | 153.057 ms |

## 2. Matched workload

```text
model:                  ibm-granite/granite-3.3-8b-instruct
decoder layers:         40
batch size:             1
fixed prompt length:    512
generated-token phases: 4 (one prefill plus three decode calls)
measured iterations:    5
dtype:                  fp16
weights:                unfused
attention:              SDPA
compile/warmup:          excluded from the trace
```

The checkpoint metadata hashes are identical in both environments. See
`results/2026-07-25/full_model_comparison/environment.json` for the hashes,
source revisions, runtime identities, absolute run roots, and pod UIDs.

## 3. Execution targets

SenDNN ran on the idle owned pod `adnan-clc-spyre-dev-pf` on
`p1-worker-53`. Torch-Spyre ran immediately afterward on
`adnan-cdx-spyre-dev-pf` on `p1-worker-35`, where the exact historical Antoni
environment was preserved. Both pods used this container image digest:

```text
sha256:913f394b4b3f03740a9d35f70f273b1cb799d4cba55f7fdaff108d7749d77964
```

The different pods and backend-specific Python/Torch/FMS environments mean
the result is an end-to-end stack comparison. It is not an isolated
SenDNN-versus-Torch-Spyre library A/B in a single Python process.

## 4. Run SenDNN

Verify or restage the pinned SenDNN environment as described in
`runbooks/sendnn_antoni_equivalent.md`, then run:

```bash
SENDNN_POD=adnan-clc-spyre-dev-pf \
SENDNN_ITERS=5 \
SENDNN_RUN_TAG=<unique-tag> \
scripts/run_sendnn_full_model.sh
```

To retain every final SDSC after LX optimization for operation-level peer-LX
attribution, add the optional debug selector:

```bash
SENDNN_POD=adnan-clc-spyre-dev-pf \
SENDNN_ITERS=1 \
SENDNN_RUN_TAG=<unique-tag> \
SENDNN_PERFDSC_DEBUG=sdsc,lxopt,lxAnalysis,dsg,isg,bo \
scripts/run_sendnn_full_model.sh
```

The launcher places the dump in `${SENDNN_RUN_ROOT}/perfdsc_debug` unless
`SENDNN_PERFDSC_DUMP_DIR` overrides it. `execute_itr0` is the `mb=512`
prefill graph and `execute_itr256` is the `mb=1` decode graph. The measured
capture produced 227 prefill and 243 decode SDSCs. See
`runbooks/sendnn_vs_torch_spyre_smc_study.md` for the exact LX analysis.

The full-model launcher intentionally omits both `--run_block 1` and
`--prefill_only`. It preserves the required production runtime isolation:

```text
PATH=/opt/ibm/spyre/runtime/bin:...
LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:...
TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

Without those pins, the pod can mix a workspace `compile_graph` with
production libraries and fail on an undefined `DTCompiler` symbol.

The checked-in run root is:

```text
/home/adnan/codex-isolated/sendnn_granite_antoni_20260725/runs/full_40_layer_b1_s512_5x4_measured_20260725_1045
```

## 5. Run Torch-Spyre

Verify or restage the exact historical environment as described in
`runbooks/antoni_exact_reproduction.md`, then run:

```bash
ANTONI_POD=adnan-cdx-spyre-dev-pf \
ANTONI_ITERS=5 \
ANTONI_RUN_TAG=<unique-tag> \
scripts/run_torch_spyre_full_model.sh
```

The launcher unsets `ANTONI_LAYER_LIMIT`, so all 40 layers execute. Its
unprofiled generation immediately before Kineto materializes the lazy graphs;
do not remove it.

The checked-in run root is:

```text
/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724/runs/full_40_layer_b1_s512_5x4_measured_20260725_1050
```

## 6. Analyze

Decompress the three log artifacts to a temporary directory, then run:

```bash
mkdir -p /tmp/granite_full_model_analysis
gzip -dc results/2026-07-25/full_model_comparison/logs/sendnn_run.log.gz \
  > /tmp/granite_full_model_analysis/sendnn_run.log
gzip -dc results/2026-07-25/full_model_comparison/logs/sendnn_old_stack_compiler.log.gz \
  > /tmp/granite_full_model_analysis/sendnn_old_stack_compiler.log
gzip -dc results/2026-07-25/full_model_comparison/logs/torch_spyre_run.log.gz \
  > /tmp/granite_full_model_analysis/torch_spyre_run.log

python3 tools/analyze_full_model_comparison.py \
  --sendnn-trace results/2026-07-25/full_model_comparison/traces/sendnn_full_40_layer_b1_s512_5x4.pt.trace.json.gz \
  --sendnn-run-log /tmp/granite_full_model_analysis/sendnn_run.log \
  --sendnn-compiler-log /tmp/granite_full_model_analysis/sendnn_old_stack_compiler.log \
  --torch-spyre-trace results/2026-07-25/full_model_comparison/traces/torch_spyre_full_40_layer_b1_s512_5x4.pt.trace.json.gz \
  --torch-spyre-run-log /tmp/granite_full_model_analysis/torch_spyre_run.log \
  --output /tmp/full_model_metrics.json

diff -u \
  results/2026-07-25/full_model_comparison/metrics.json \
  /tmp/full_model_metrics.json
```

Expected structural checks:

```text
SenDNN generations:             5
SenDNN fused kernel events:    20
SenDNN DtoH events:            20
Torch-Spyre generations:        5
Torch-Spyre decoder layers:    40 per phase
Torch-Spyre kernel events:   4880
```

## 7. Measurement boundary

The comparison uses trace-derived device-program time:

- SenDNN: the fused `embedding` event for each complete model phase. The name
  is a graph-entry label; the compiler export and execution order contain all
  40 decoder layers, final norm, output head, and sampling division.
- Torch-Spyre: the sum of all device kernels from phase input through the
  sampling division.

SenDNN's DtoH event spans the fused kernel interval (99.998% mean coverage),
so adding it would double-count the device program. The profiled Python wall
times are retained in `metrics.json` as diagnostics but are not compared;
Kineto and host dispatch inflate Torch-Spyre wall time disproportionately.

SenDNN compiler totals are 127,146,112 cycles for prefill and 257,152 cycles
for decode. At the launcher's 1100 MHz assumption, the prefill compiler-cycle
proxy is 60.706%. It is a compiler-derived proxy, not a hardware utilization
counter. The decode proxy is not a useful performance explanation because the
compiler estimate accounts for only a tiny fraction of the observed fused
decode duration.

## 8. Checked-in artifacts

```text
results/2026-07-25/full_model_comparison/metrics.json
results/2026-07-25/full_model_comparison/environment.json
results/2026-07-25/full_model_comparison/SHA256SUMS
results/2026-07-25/full_model_comparison/summary.md
results/2026-07-25/full_model_comparison/traces/sendnn_full_40_layer_b1_s512_5x4.pt.trace.json.gz
results/2026-07-25/full_model_comparison/traces/torch_spyre_full_40_layer_b1_s512_5x4.pt.trace.json.gz
results/2026-07-25/full_model_comparison/logs/sendnn_run.log.gz
results/2026-07-25/full_model_comparison/logs/sendnn_old_stack_compiler.log.gz
results/2026-07-25/full_model_comparison/logs/torch_spyre_run.log.gz
results/2026-07-25/full_model_comparison/exports/sendnn_full_40_layer_compiler_export.tar.gz
```
