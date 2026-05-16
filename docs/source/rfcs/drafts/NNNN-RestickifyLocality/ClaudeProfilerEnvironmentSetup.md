# Claude Profiler Environment Setup

This note is written for a second Spyre pod running Claude. It sets up the same
counter/profiler environment used for the Stage 13-16 restickify measurements.

The goal is not to build a new compiler stack from scratch. The goal is to get a
known-good environment that can run:

```text
tools/restickify_aiusmi_marker_probe.py
```

and produce:

```text
aiusmi_marker_rows.jsonl
aiusmi_marker_rows.csv
aiusmi_marker_summary.svg
aiusmi_marker_summary.html
```

## Assumptions

- You are inside an OpenShift Spyre pod with access to the AIU device.
- Python 3.12 is available as `python3.12`.
- The IBM Spyre runtime is installed under `/opt/ibm/spyre`.
- The `aiu-monitor` wheel is available, either copied into the pod or accessible
  from the internal artifact source.
- This setup is separate from any production or Claude code-editing checkout.

If any assumption fails, stop and report the exact command/output that failed.

## One-Time Setup

Pick a separate profiler workspace:

```sh
export HOME=${HOME:-/home/adnan-cdx}
export PROF_ROOT=$HOME/dt-inductor-profiler
mkdir -p "$PROF_ROOT"
cd "$PROF_ROOT"
```

Create and activate a dedicated Python environment:

```sh
python3.12 -m venv "$PROF_ROOT/.venv-py312"
source "$PROF_ROOT/.venv-py312/bin/activate"
python -m pip install --upgrade pip setuptools wheel
```

Install the `aiu-monitor` wheel. If the wheel has already been copied into the
pod:

```sh
python -m pip install /tmp/ibm_aiu_monitor-1.2.1+torch.spyre-py312-none-linux_x86_64.whl
```

If the wheel is still on the Mac, copy it from the Mac side first:

```sh
oc cp \
  ~/Downloads/ibm_aiu_monitor-1.2.1+torch.spyre-py312-none-linux_x86_64.whl \
  <pod-name>:/tmp/ibm_aiu_monitor-1.2.1+torch.spyre-py312-none-linux_x86_64.whl
```

Then verify:

```sh
which aiu-smi
aiu-smi --version || aiu-smi -h | head
test -f "$PROF_ROOT/.venv-py312/etc/senlib_config_aiusmi.json"
```

Expected:

```text
.../.venv-py312/bin/aiu-smi
.../etc/senlib_config_aiusmi.json
```

## Torch-Spyre Checkout

Clone Adnan's working branch:

```sh
cd "$PROF_ROOT"
git clone https://github.com/AdnanHoque/torch-spyre.git torch-spyre-profiler-stage3b
cd "$PROF_ROOT/torch-spyre-profiler-stage3b"
git checkout AdnanHoque/rfc-restickify-first-principles
```

If HTTPS auth fails, use SSH instead:

```sh
git clone git@github.com:AdnanHoque/torch-spyre.git torch-spyre-profiler-stage3b
```

## Runtime Environment

Use this block for every run:

```sh
source "$PROF_ROOT/.venv-py312/bin/activate"

export HOME=${HOME:-/home/adnan-cdx}
export SENTIENT_BASE_INSTALL_DIR=/opt/ibm/spyre
export DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export PATH="$PROF_ROOT/.venv-py312/bin:/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/runtime/bin:/opt/ibm/spyre/senlib/bin:$PATH"
export LD_LIBRARY_PATH="/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/senlib/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROF_ROOT/torch-spyre-profiler-stage3b:${PYTHONPATH:-}"

export SENCORES=32
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

export SENLIB_DEVEL_CONFIG_FILE="$PROF_ROOT/.venv-py312/etc/senlib_config_aiusmi.json"
export AIUPTI_ENABLE_METRICS=1
export AIUSMI_ENABLE_METRICS=1
export ENABLE_AIUPTI_ACTIVITY_KIND_EVENT=1
export ENABLE_AIUPTI_ACTIVITY_KIND_METRIC=1
export AIUPTI_SAMPLER_INTERVAL=1

cd "$PROF_ROOT/torch-spyre-profiler-stage3b"
```

If `torch` or `torch_spyre` cannot import, the pod does not yet have a matching
Torch-Spyre Python environment. In that case, source the project's documented
development environment instead, then reinstall `aiu-monitor` into that venv:

```sh
source "$DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh"
python -m pip install /tmp/ibm_aiu_monitor-1.2.1+torch.spyre-py312-none-linux_x86_64.whl
```

## Smoke Test

First check Python imports:

```sh
python3.12 - <<'PY'
import torch
import torch_spyre
print(torch.__version__)
print("torch_spyre import ok")
PY
```

Then run the marker-separated counter probe:

```sh
python3.12 -u tools/restickify_aiusmi_marker_probe.py \
  --case adds_then_matmul_x \
  --size 512 \
  --mode baseline \
  --warmup 2 \
  --iters 500 \
  --output-dir /tmp/restickify-aiusmi-marker-smoke \
  --fail-on-error
```

Expected signs of success:

```text
status ok
traffic_samples > 0
rd_peak > 0
wr_peak > 0
Wrote .../aiusmi_marker_summary.html
```

Inspect the row:

```sh
python3.12 - <<'PY'
import json, pathlib
p = pathlib.Path("/tmp/restickify-aiusmi-marker-smoke/aiusmi_marker_rows.jsonl")
for row in map(json.loads, p.read_text().splitlines()):
    print(row["mode"], row["size"], row["median_ms"], row.get("ring_total_byte_hops"))
    print("traffic", row.get("aiusmi_traffic_samples"), "rd", row.get("aiusmi_peak_rdmem_GiB_per_s"), "wr", row.get("aiusmi_peak_wrmem_GiB_per_s"))
    print("html", pathlib.Path("/tmp/restickify-aiusmi-marker-smoke/aiusmi_marker_summary.html").exists())
PY
```

## Stage 15 Reproduction Command

Use this to reproduce the high-signal case:

```sh
python3.12 -u tools/restickify_aiusmi_marker_probe.py \
  --case adds_then_matmul_x \
  --size 2048 \
  --mode baseline \
  --mode stage3b \
  --warmup 10 \
  --iters 5000 \
  --sample-interval 0.1 \
  --output-dir /tmp/restickify-aiusmi-marker-fused-2048 \
  --fail-on-error
```

Expected directional result:

```text
baseline: byte_hops ~= 67,108,864
stage3b:  byte_hops == 0
restickify count unchanged
bytes moved unchanged
aiu-smi read/write traffic still nonzero in both modes
```

## Known Gotchas

1. Do not delete `/tmp/metrics.0000:aa:00.0` after any tensor has touched
   `device="spyre"`. The runtime may already have the file open. The marker
   probe clears it before Spyre runtime initialization.

2. `SENLIB_DEVEL_CONFIG_FILE` is required. Without it, `aiu-smi` can run but all
   device counters stay at zero.

3. Custom `AIUPTI_METRIC_PATH` and `SPYRE_METRIC_PATH` did not redirect the
   metric file in the current runtime. Use the default
   `/tmp/metrics.0000:aa:00.0`.

4. `aiu-smi` read/write counters are aggregate device-memory counters for the
   measured loop. They do not by themselves isolate one restickify edge inside a
   fused SDSC bundle.

5. If `traffic_samples == 0`, check in this order:
   - `SENLIB_DEVEL_CONFIG_FILE` path exists;
   - `aiu-smi -s -g A -d 0.1 -f /tmp/test.csv` can run;
   - `/tmp/metrics.0000:aa:00.0` appears during a workload;
   - the workload actually runs long enough to overlap the sample interval;
   - no code deleted the metric file after Spyre runtime initialization.

## What To Send Back

After setup, send back:

```text
which aiu-smi
aiu-smi version/help first line
torch version
torch_spyre import result
smoke JSON row
path to aiusmi_marker_summary.html
```

