# Exact Antoni-equivalent Granite runbook

This is the Torch-Spyre Granite runbook. It reproduces the July 24, 2026
one-layer measurement that was matched against Antoni Viros i Martin's July 2
full 40-layer trace. The corresponding SenDNN run is documented in
`runbooks/sendnn_antoni_equivalent.md`.

## 1. Immutable identities

OpenShift target:

```text
namespace: a6-quantization
pod:       adnan-cdx-spyre-dev-pf
```

Absolute pod paths:

```text
base:
/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724

python:
/tmp/adnan-cdx-costmodel-kineto/bin/python

model:
/tmp/models/granite-3.3-8b-instruct

runner:
/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724/run_historical_98ac91e/antoni_inference_profile.py

historical Torch-Spyre Python/compiler:
/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724/torch-spyre-98ac91e

successful reproduction trace:
/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724/run_historical_98ac91e/exact_one_layer_s512_20x4_20260724/trace_warm/adnan-cdx-spyre-dev-pf_88928.1784945025741518406.pt.trace.json
```

Source revisions:

```text
test-spyre-scripts       afda166e58b23519d0b4ca871350b011b56d91a3
foundation-model-stack   61bc991b175103e80cb8202b24a66ba7dbe79d1b
aiu-fms-testing-utils    dbb1617525844651e7a2c5afcdec27fe163caa5f
Torch-Spyre              98ac91e7823919e410b20dc2d0a1ee0ed6a620fa
```

The historical Torch-Spyre Python/compiler source is paired with the
Kineto-compatible `_C.so` built for Torch `2.11.0+aiu.kineto.1.1.2`. The exact
binary SHA-256 is
`1f4d328fbce73cb2e96b819437e4aebb7b760b6f8cbe361782a8ce1526f22db1`.
Historical binaries built against the earlier runtime ABI do not work with the
current pod runtime.

## 2. Host prerequisites

- `oc` authenticated to the cluster containing the target pod;
- access from the pod to `github.ibm.com` and `github.com` if restaging;
- an AIU assigned to the named pod;
- the Granite checkpoint at the exact model path above;
- the Kineto Python environment at the exact Python path above.

The checkpoint is `ibm-granite/granite-3.3-8b-instruct`. Verification checks:

```text
config.json SHA-256:
1313edf0d39dcf7ed35a072d341ec11b516c12acc4267cfb6d248c6bdcdddcb7

model.safetensors.index.json SHA-256:
c3a88218300666c35343b129857d4e8583ee1b15bf68d90ab976f51744560379

tokenizer.json SHA-256:
91168e938f05796aa6dcca7e485e4b30ab52785320c7a6391ecef86e6c84681e
```

## 3. Verify the preserved environment

From any clone of this repository:

```bash
scripts/verify_pod_environment.sh
```

This fails closed on any source revision, runner, runtime binary, model
metadata, or Torch version mismatch.

## 4. Restage from pinned source if needed

If the absolute base tree is absent, create it from the checked-in
implementation and pinned repositories:

```bash
scripts/stage_pinned_environment.sh
```

The script is non-destructive toward unexpected dirty checkouts. It clones the
four pinned repositories, uploads the exact FMS runner and Torch-Spyre overlay,
and builds the Kineto-compatible extension only if `_C.so` is absent.

To deliberately rebuild the extension in place:

```bash
ANTONI_REBUILD_RUNTIME=1 scripts/stage_pinned_environment.sh
```

The build uses these absolute AIU image paths:

```text
RUNTIME_INSTALL_DIR=/opt/ibm/spyre/runtime
SENLIB_INSTALL_DIR=/opt/ibm/spyre/senlib
DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
SPYRE_COMMS_INSTALL_DIR=/opt/ibm/spyre/spyre-comms
LIBAIUPTI_INSTALL_DIR=/opt/ibm/spyre/runtime
SEN_COMMON_HEADERS=/opt/ibm/spyre/runtime/include/flex
```

## 5. Run the exact one-layer workload

```bash
scripts/run_antoni_exact_one_layer.sh
```

Optional overrides:

```bash
ANTONI_ITERS=20 \
ANTONI_RUN_TAG=verification_20260724 \
ANTONI_LOCAL_TRACE_DIR="$PWD/local_runs" \
scripts/run_antoni_exact_one_layer.sh
```

Default output root:

```text
/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724/runs/one_layer_<timestamp>
```

Each run has isolated `cache/`, `export/`, and `trace_warm/` directories. The
script prints the absolute trace path and SHA-256 when it finishes.

The per-token wall times printed while Kineto is active include profiler and
host overhead and are not comparable to Antoni's reported uninstrumented E2E
numbers. Use the trace-derived device-kernel values produced in step 6.

Do not remove the unprofiled `infer(...)` immediately before the profiler in
`implementation/antoni_inference_profile.py`. It materializes all lazy graphs
before Kineto begins. Profiling cold compilation previously produced a 10 GB
trace and does not measure the intended runtime.

## 6. Analyze against the canonical reference

```bash
python3 tools/analyze_granite_trace.py \
  --reference results/2026-07-24/traces/antoni_july2_full_40_layer_2x4.pt.trace.json.gz \
  --reproduction results/2026-07-24/traces/antoni_equivalent_one_layer_20x4.pt.trace.json.gz \
  --output /tmp/granite_metrics.json

diff -u results/2026-07-24/metrics.json /tmp/granite_metrics.json
```

Expected structural checks:

```text
reference kernel events:       1960
reference generation runs:     2
reference decoder layers:      40
reproduction kernel events:    800
reproduction generation runs:  20
reproduction decoder layers:   1
prefill kernels per layer:      5
first-decode kernels per layer: 7
steady-decode kernels/layer:    6
```

Expected headline values:

```text
prefill one-layer block:        7.2840598 ms
prefill block x40:              291.362392 ms
prefill projected device phase: 300.106249 ms
prefill reference device phase: 297.706132 ms

decode one-layer average:       3.734517 ms
decode projected device phase:  153.486818 ms
decode reference device phase:  150.088898 ms
```

## 7. Artifact provenance

Original local inputs retained during the investigation:

```text
/Users/adnan/Downloads/aviros-spyre-test-2_865491.1783015374401061696.pt.trace.json
/Users/adnan/Documents/Codex/2026-07-24/st/work/antoni_exact_one_layer_20x4_20260724.pt.trace.json
/Users/adnan/Documents/Codex/2026-07-24/st/work/antoni_exact_one_layer_reproduction.md
```

Canonical repository artifacts:

```text
results/2026-07-24/traces/antoni_july2_full_40_layer_2x4.pt.trace.json.gz
results/2026-07-24/traces/antoni_equivalent_one_layer_20x4.pt.trace.json.gz
results/2026-07-24/traces/repository_validation_one_layer_20x4.pt.trace.json.gz
results/2026-07-24/metrics.json
results/2026-07-24/validation_metrics.json
results/2026-07-24/environment.json
results/2026-07-24/SHA256SUMS
```

The gzip files were created with `gzip -9 -n`; their decompressed SHA-256 values
are recorded in `environment.json` and `SHA256SUMS`.
