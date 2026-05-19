# Stage 71: Input-Fetch Neighbor Probe Harness

## Summary

Stage 71 adds a small, default-off probe harness for the Stage 70 hypothesis:

```text
Deeptools InputFetchNeighbor can express the restickify movement we want while
preserving the producer tensor's real LX allocation identity.
```

This stage does not change compiler lowering. It only makes the experiment
repeatable.

## Code Changes

Two probe-only changes were added.

First, `tools/restickify_scenario_probe.py` can now preserve the generated
Torch-Spyre SDSC bundle directories:

```sh
--copy-kernel-code
```

When enabled, each launched kernel's code directory is copied under:

```text
<output-dir>/kernel_code/<case>_<size>/
```

The kernel launch JSONL also records `copied_code_dir`.

Second, a new staging/running tool was added:

```text
tools/restickify_input_fetch_neighbor_probe.py
```

It reads either a copied code directory or a kernel-launch JSONL, finds a
restickify SDSC, stages the neighboring SDSCs as:

```text
producer_pre.json
restickify_reference.json
consumer_main.json
```

and optionally runs:

```text
dcg_inpfetch_standalone \
  -initSdscMain consumer_main.json \
  -initSdscPre producer_pre.json \
  -d dataDSC/relayout.json \
  -s
```

The tool writes JSONL and JSON summaries with producer/restickify/consumer
metadata and, when Deeptools runs, generated SDSC and senprog token counts.

## Commands

Once pod auth is restored, capture the clean Stage 62/69 fixture:

```sh
export HOME=/home/adnan-cdx
export DTI_PROJECT_ROOT=$HOME/dt-inductor
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PYTHONPATH=/tmp/torch-spyre-stage2:${PYTHONPATH:-}
export SENTIENT_BASE_INSTALL_DIR=/opt/ibm/spyre
export DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export PATH=/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/runtime/bin:$PATH
export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}
export SENCORES=32
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
cd /tmp/torch-spyre-stage2
```

Capture generated SDSCs:

```sh
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --kernel-launch-log \
  --copy-kernel-code \
  --output-dir /tmp/restickify-input-fetch-capture \
  --fail-on-error
```

Stage and run the input-neighbor probe:

```sh
python tools/restickify_input_fetch_neighbor_probe.py \
  --kernel-launch-log /tmp/restickify-input-fetch-capture/kernel_launches/computed_transpose_adds_then_matmul_2048.jsonl \
  --output-dir /tmp/restickify-input-fetch-neighbor \
  --run \
  --senprog \
  --fail-on-error
```

If `dcg_inpfetch_standalone` is not on `PATH`, pass it explicitly:

```sh
--dcg-inpfetch-standalone /opt/ibm/spyre/deeptools/bin/dcg_inpfetch_standalone
```

## What To Check

Success at this stage means the generated InputFetchNeighbor output has:

```text
returncode = 0
HBM = 0
L3LU = 0
L3SU = 0
LXLU > 0
LXSU > 0
```

and the source-side piece addresses come from the producer SDSC's
`coreStateInit_.lbrInit_`, not from an artificial all-zero compact map.

That would prove the allocation-identity blocker from Stage 69 has a plausible
Deeptools-native solution.

## Current Blocker

Live pod execution is blocked by OpenShift auth from this desktop session:

```text
You must be logged in to the server (Unauthorized)
```

The harness was validated locally with a synthetic three-SDSC directory and by
`py_compile`; the real Deeptools run needs pod auth restored.

## Validation

Local static validation:

```text
python3 -m py_compile \
  tools/restickify_scenario_probe.py \
  tools/restickify_input_fetch_neighbor_probe.py
```

Synthetic staging validation:

```text
tools/restickify_input_fetch_neighbor_probe.py --code-dir <synthetic> --output-dir <tmp>
```

Result:

```text
staged producer_pre.json, restickify_reference.json, consumer_main.json
```

## Next Step

After re-auth, run the two pod commands above. If the generated senprog confirms
no HBM/L3 tokens and real producer LX addresses, the next implementation step is
to wire this shape into Torch-Spyre behind a default-off flag:

```text
SPYRE_RESTICKIFY_INPUT_FETCH_NEIGHBOR=1
```

That should replace the diagnostic compact DDL bridge as the serious path for
correct LX-to-LX in-graph restickification.
