# Exact SenDNN Antoni-equivalent Granite runbook

This runbook reproduces the July 25, 2026 SenDNN measurement corresponding to
the repository's Torch-Spyre one-layer Granite experiment. It runs one Granite
3.3 8B decoder layer with batch 1, a fixed 512-token prompt, FP16 unfused
weights, and 20 measured prefill iterations.

For the measured full 40-layer prefill-and-decode comparison, see
`runbooks/full_model_sendnn_vs_torch_spyre.md`.

## 1. What “SenDNN” means here

[`ai-chip-toolchain/sendnn`](https://github.ibm.com/ai-chip-toolchain/sendnn)
is the low-level graph and runtime library; it is not a standalone model
launcher. The model path is:

```text
FMS Granite model
  -> torch.compile(backend="sendnn")
  -> torch_sendnn
  -> production SenDNN / Flex runtime
  -> Spyre AIU
```

The run pins the linked SenDNN repository as a source/API reference and pins
the measured production runtime independently by RPM revision and binary hash.
It uses the SenDNN Granite launcher from `spyre-perf-suite`.

## 2. Immutable identities

OpenShift target used for the checked-in result:

```text
namespace: a6-quantization
pod:       adnan-spyre-dev-pf
node:      p1-worker-53
image:     sha256:913f394b4b3f03740a9d35f70f273b1cb799d4cba55f7fdaff108d7749d77964
```

The pod was selected after checking all four owned pods for active processes;
it had only the pod init and SSH listener. Check again before a rerun and set
`SENDNN_POD` to another idle owned pod if necessary.

Absolute pod paths:

```text
base:
/home/adnan/codex-isolated/sendnn_granite_antoni_20260725

python:
/home/adnan/codex-isolated/sendnn_granite_antoni_20260725/venv/bin/python

runner:
/home/adnan/codex-isolated/sendnn_granite_antoni_20260725/spyre-perf-suite/utils/inference_granite_os.py

model:
/home/adnan/hub/models--ibm-granite--granite-3.3-8b-instruct/snapshots/51dd4bc2ade4059a6bd87649d68aa11e4fb2529b

successful run:
/home/adnan/codex-isolated/sendnn_granite_antoni_20260725/runs/one_layer_b1_s512_20x4_20260725_0905
```

Source revisions:

```text
SenDNN source/API reference  8cc4fe436f161f72e9bb4b76b8252d9bea981da6
spyre-perf-suite             cbde23adbe606775bf7fdad5c63e3ba32aa5e01d
FMS sendnn_fms               7a66f2f34ff95c2b9ad2e49b615e918f8aa85031
aiu-fms-testing-utils        0325bd64662a8537803a2e3890294138ea17238a
```

Runtime identities:

```text
Torch:          2.10.0+aiu.kineto.1.1.1
torch_sendnn:   1.3.0+main.1.1bef083.0
ibm-flex RPM:   2.0.0-0.main.1+388.81385a4_0.el10.x86_64
ibm-deeptools:  2.0.0-0.main.1+1401.ee2f97a_0.el10.x86_64

compile_graph SHA-256:
805f83f452c4f12c42a28b6d6fa8c1a573102a22607bd74b62f1e44df655ebbf

libsendnn.so SHA-256:
496da8a8f666a0f8f52501cd8535bd85a59c531f9292fdf831004ad582ba5eac

sendnn Python extension SHA-256:
95f7348c99801d35ee92ffbdf7166439b689666ef1d657a56e1020d32a85ca4f
```

The full package and binary manifest is in
`results/2026-07-25/sendnn/environment.json`.

## 3. Runtime isolation is required

The pod's interactive environment points `PATH`, `PYTHONPATH`, and
`LD_LIBRARY_PATH` at a workspace build under `/home/adnan/dt-inductor`. The
checked-in run deliberately uses the production runtime under
`/opt/ibm/spyre` for both executables and libraries.

Mixing the workspace `compile_graph` with production libraries fails before
device execution with:

```text
undefined symbol: _ZN6sendnn10DTCompilerC1ESt8optionalIbE
```

The run script therefore resets all runtime paths and sets
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` so the unrelated Torch-Spyre extension is not
autoloaded. Do not remove those settings.

## 4. Verify the checkpoint

The checkpoint is `ibm-granite/granite-3.3-8b-instruct`, identical to the
Torch-Spyre reproduction. Required hashes:

```text
config.json:
1313edf0d39dcf7ed35a072d341ec11b516c12acc4267cfb6d248c6bdcdddcb7

model.safetensors.index.json:
c3a88218300666c35343b129857d4e8583ee1b15bf68d90ab976f51744560379

tokenizer.json:
91168e938f05796aa6dcca7e485e4b30ab52785320c7a6391ecef86e6c84681e
```

The verifier checks these hashes, every source revision, the production
runtime binaries, package versions, AIU assignment, and active Granite
processes:

```bash
scripts/verify_sendnn_pod_environment.sh
```

## 5. Restage the pinned environment

From a clone of this repository:

```bash
scripts/stage_pinned_sendnn_environment.sh
```

This creates an isolated virtual environment with system site packages, pins
the Kineto-enabled Torch wheel and Python dependencies from
`results/2026-07-25/sendnn/requirements.freeze.txt`, and installs the two exact
editable source revisions. It does not modify the pod's default Python
environment. Dirty or unexpected source checkouts fail closed.

Useful overrides:

```bash
SENDNN_POD=<another-idle-owned-pod> \
SENDNN_BASE=/home/adnan/codex-isolated/sendnn_granite_antoni_20260725 \
scripts/stage_pinned_sendnn_environment.sh
```

The alternative pod must have the same image, AIU assignment, production
binary hashes, user home layout, and checkpoint path; the verifier enforces the
measured contract.

## 6. Run the exact one-layer workload

```bash
scripts/run_sendnn_antoni_one_layer.sh
```

To copy the complete raw run back to the host:

```bash
SENDNN_RUN_TAG=verification_20260725 \
SENDNN_LOCAL_ARTIFACT_DIR="$PWD/local_runs" \
scripts/run_sendnn_antoni_one_layer.sh
```

The exact model arguments are:

```text
architecture:           hf_pretrained
decoder layers:         1
batch size:             1
fixed prompt length:    512
max-new-token contract: 4
mode:                   prefill_only
iterations:             20
dtype:                  fp16
weights:                unfused
attention:              SDPA
backend:                sendnn
dynamic compile:        enabled for Torch and SenDNN
```

FMS compiles and performs an AIU initialization warmup before the profiler is
started. The measured trace therefore contains runtime executions, not compile
time. The compiler still emits one prefill and one decode program while warming
the generation contract; only prefill is executed in the measured loop.

## 7. Analyze the artifacts

```bash
python3 tools/analyze_sendnn_trace.py \
  --trace results/2026-07-25/sendnn/traces/sendnn_one_layer_b1_s512_20x4.pt.trace.json.gz \
  --run-log results/2026-07-25/sendnn/logs/run.log.gz \
  --compiler-log results/2026-07-25/sendnn/logs/old_stack_compiler.log.gz \
  --torch-metrics results/2026-07-24/metrics.json \
  --output /tmp/sendnn_metrics.json

diff -u results/2026-07-25/sendnn/metrics.json /tmp/sendnn_metrics.json
```

Expected structural checks:

```text
measured iterations:                  20
fused kernel events:                  20
fused kernel name:                    embedding
H2D events per iteration:             3
DtoH events per iteration:            1
memset events per iteration:          4
prefill compiler execution-order ops: 87
prefill compiler cycles:               3,184,768
```

`embedding` is the fused program label, not an embedding-only workload. The
compiler execution order proves that the program also contains layer norm,
attention QK/AV matmuls, safe softmax, SwiGLU/MLP matmuls, final norm, output
head, and sampling division.

Expected headline values:

```text
fused device-program mean:       8.05597855 ms
fused device-program median:     8.06986400 ms
profiled first-token wall mean:  9.80235000 ms
compiler-cycle utilization:     35.93906834%
```

## 8. Interpretation boundaries

The DtoH event spans essentially the same interval as the fused kernel. Do not
add their durations. The launcher's `Spyre TIME` sum of 16.191 ms double-counts
that overlap; use the 8.056 ms kernel event for device-program time.

The launcher also prints 0.142% PT utilization because it chooses the first
compiler `Total`, which is the 12,544-cycle decode program. The measured trace
is prefill. The analyzer selects the matching 3,184,768-cycle prefill section:

```text
ideal prefill duration = 3,184,768 / 1100 MHz = 2.895244 ms
utilization proxy      = 2.895244 / 8.055979 = 35.9391%
```

This is a compiler-cycle-derived proxy, not a physical utilization counter.

The fused SenDNN program includes embedding, one decoder layer, final norm, and
output head. Its matched Torch-Spyre scope is the one-layer `phase_total`,
16.027917 ms, not the 7.284060 ms decoder block alone. On that matched scope,
SenDNN is 49.74% lower and the Torch-Spyre/SenDNN time ratio is 1.9896x. This
run is prefill-only; it does not claim a SenDNN decode result or a 40-layer
projection.

## 9. Checked-in artifacts

```text
results/2026-07-25/sendnn/environment.json
results/2026-07-25/sendnn/metrics.json
results/2026-07-25/sendnn/summary.md
results/2026-07-25/sendnn/requirements.freeze.txt
results/2026-07-25/sendnn/SHA256SUMS
results/2026-07-25/sendnn/logs/run.log.gz
results/2026-07-25/sendnn/logs/old_stack_compiler.log.gz
results/2026-07-25/sendnn/traces/sendnn_one_layer_b1_s512_20x4.pt.trace.json.gz
results/2026-07-25/sendnn/exports/sendnn_compiler_export.tar.gz
```

The export archive contains the four pre-/post-compiler CBOR graphs plus the
77-file DeepRT export tree. The trace and logs use deterministic `gzip -9 -n`.
