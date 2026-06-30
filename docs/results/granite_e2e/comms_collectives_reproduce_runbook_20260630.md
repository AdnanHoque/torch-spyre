# Granite Prefill LX Relayout Reproduction Runbook

Date: 2026-06-30

Branch: `ah/comms-collectives`

This runbook records the working Granite causal-prefill setup used to reproduce
the LX relayout speedup on `adnan-spyre-dev-pf`.  The important lesson from the
latest debugging pass is that DXP behavior did not differ between pods.  The
run environment differed.

## Summary

The known-good full-LX run requires split frontend/backend handling of
`DXP_LX_FRAC_AVAIL`:

- Torch/frontend sees `DXP_LX_FRAC_AVAIL=0`, which means Torch can plan using
  full frontend LX.
- DXP/backend sees `DXP_LX_FRAC_AVAIL=1`, which gives backend chunk planning
  usable LX space.
- A wrapper maps `DXP_BACKEND_LX_FRAC_AVAIL` into `DXP_LX_FRAC_AVAIL` only for
  the DXP subprocess.

Without the wrapper, DXP sees `DXP_LX_FRAC_AVAIL=0` and can fail with:

```text
initial chunk parameters must fit in LX
```

## Known-Good Results

Archived split-env result:

| variant | kernel ms/iter | wall ms | kernel speedup |
|---|---:|---:|---:|
| baseline off | 14.6977 | 34.8575 | 1.000x |
| full Torch LX + backend LX=1 | 12.3391 | 32.4521 | 1.191x |

Archived summary:

```text
/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114/runs/granite_prefill_isolated_20260629_171937/isolated_run_summary.md
```

Latest literal replay of the archived setup:

| run | kernel ms/iter | wall ms | status |
|---|---:|---:|---|
| literal replay full LX | 12.0625 | 32.0718 | pass |
| comms branch replay full LX | 12.0129 | 32.2778 | pass |
| comms branch collectives enabled | 12.3147 | 32.5027 | pass |

Latest run roots:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/literal_replay_full_lx_20260630_034958
/home/adnan/codex-isolated/comms_collectives_20260629/runs/comms_replay_full_lx_runtimefix_20260630_035214
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_splitenv_20260630_040302
```

Both runs completed successfully.  They emitted a `RuntimeStream::synchronize()`
warning after 60000 ms but finished successfully.

## Pod And Device

Use `adnan-spyre-dev-pf` in namespace `a6-quantization` for AIU validation.

After the latest restart the device mapping was:

```text
/dev/vfio/31
AIU PCI address: 0000:3c:00.0
```

The pod was manually restarted from:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/tmp/pod-restart/adnan-spyre-dev-pf.restart.json
```

The CDX pod was also manually restarted from:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/tmp/pod-restart/adnan-cdx-spyre-dev-pf.restart.json
```

If the device is wedged, prefer recreating the pod from the saved manifest.
`aiu_dd2_hot_reset -t chip` reached the VFIO device but aborted with:

```text
RISCV config not found.
```

`aiu_dd2_hot_reset -t linux` requires elevated privileges.

## Checkout Paths

Known-good archived scatter environment:

```text
ROOT=/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114
TORCH=$ROOT/torch-spyre
DEEPTOOLS_PATH=$ROOT/deeptools
BENCH=$ROOT/spyre-granite-e2e-bench
FMS=/home/adnan/dt-inductor/foundation-model-stack
PYTHON=/home/adnan/dt-inductor/.venv/bin/python3
```

Latest comms-collectives environment:

```text
ROOT=/home/adnan/codex-isolated/comms_collectives_20260629
TORCH=$ROOT/torch-spyre
DEEPTOOLS_PATH=$ROOT/deeptools
BENCH=/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114/spyre-granite-e2e-bench
FMS=/home/adnan/dt-inductor/foundation-model-stack
PYTHON=/home/adnan/dt-inductor/.venv/bin/python3
```

Important SHAs from the archived environment:

```text
Torch: 3a222ecc5dcd6c8448c5753e94e13e9c1a1d5d5b
Deeptools: b8c09743c46505b4cac46b434b9eb3243ae0b685
spyre-granite-e2e-bench: 76cd51426ba1de6e99dd8fbf613cb0f32b71e87f
```

## DXP Split Wrapper

Use this wrapper first in `PATH`:

```text
/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114/tools/dxp-split-wrapper/dxp_standalone
```

Wrapper behavior:

```bash
if [[ -n "${DXP_BACKEND_LX_FRAC_AVAIL:-}" ]]; then
  export DXP_LX_FRAC_AVAIL="$DXP_BACKEND_LX_FRAC_AVAIL"
fi
```

Manual verification:

| replay | backend `DXP_LX_FRAC_AVAIL` | result |
|---|---:|---|
| dev-pf SDSC replay | 0 | fail |
| dev-pf SDSC replay | 0.01 | fail |
| dev-pf SDSC replay | 0.2 | pass |
| dev-pf SDSC replay | 1 | pass |
| cdx full-LX SDSC replay | 0 | fail |
| cdx full-LX SDSC replay | 1 | pass |

## Required Environment

Use this environment for full-LX runs:

```bash
export PYTHONPATH="$TORCH:$TORCH/tests/inductor:$FMS:${PYTHONPATH:-}"
export PATH=/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114/tools/dxp-split-wrapper:$PATH
export DEEPTOOLS_PATH="$DEEPTOOLS_PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

export SPYRE_LX_PLANNER_RELAYOUT=1
unset SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES
export LX_BOUNDARY_CLONES=1

export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1

export LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:${LD_LIBRARY_PATH:-}
```

The `LD_LIBRARY_PATH` ordering matters.  Earlier failing runs used
`/home/adnan/dt-inductor/flex-pr1019-install/lib` before the installed runtime
libraries and hit attention hardware errors even when the SDSCs otherwise
matched.

## Granite Prefill Command

From the benchmark checkout:

```bash
cd "$BENCH"

RUN="$ROOT/runs/granite_prefill_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

"$PYTHON" benchmarks/granite_block_layer_probe.py \
  --fms-root "$FMS" \
  --run-root "$RUN" \
  --case prefill \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 5 \
  --warmups 1 \
  --profile \
  --no-profile-memory
```

Expected successful output includes:

```text
returncode: 0
output shape: [1, 512, 4096]
cache shapes: [[1, 8, 512, 128], [1, 8, 512, 128]]
```

## Minimal Device Smoke

Use a small MLP-core run after pod restart to confirm the device is healthy
before running the full block:

```bash
cd "$BENCH"

RUN="$ROOT/runs/device_smoke_mlp_core_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

"$PYTHON" benchmarks/granite_block_probe.py \
  --fms-root "$FMS" \
  --run-root "$RUN" \
  --part mlp_core \
  --regime prefill \
  --fused-weights \
  --iters 1
```

Latest post-restart smoke result:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/device_smoke_mlp_core_20260630_033248
median_ms: 17.1621
status: pass
```

## Known Failure Modes

### Backend LX Chunk Failure

Symptom:

```text
initial chunk parameters must fit in LX
```

Cause:

DXP saw `DXP_LX_FRAC_AVAIL=0`, which reserves all LX for frontend planning and
leaves no backend chunk space.

Fix:

Use the split wrapper and set:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

### Attention Hardware Error

Symptom:

```text
RAS::RUNTIMESCHEDULER::ComputeHardwareError
StreamInErrorState
```

Observed causes:

- running with wrong frontend LX env;
- using the wrong runtime library ordering;
- device state left poisoned after a failed run.

Fixes:

- use the exact split env above;
- use `/opt/ibm/spyre/runtime/lib` and `/opt/ibm/spyre/spyre-comms/lib` first in
  `LD_LIBRARY_PATH`;
- if needed, recreate the pod from the saved manifest.

### GraphEditor ReinterpretView Boundary Clone Failure

Symptom:

```text
AssertionError: unexpected buffer type <class 'torch._inductor.ir.ReinterpretView'> while replacing 'buf11'
```

Cause:

Current main can wrap graph outputs as `ReinterpretView(StorageBox(...))`.
Boundary clone replacement must unwrap and rewrap that view instead of assuming
the wrapper stack only contains `TensorBox` and `StorageBox`.

Local patch:

```text
/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114/runs/granite_prefill_isolated_20260629_171937/local_graph_editor_reinterpretview_patch.diff
```

The same compatibility fix is currently applied locally on this artifact branch
in:

```text
torch_spyre/_inductor/scratchpad/graph_editor.py
```

## Interpretation

The valid conclusion from the latest runs is:

- DXP itself behaves consistently across pods.
- The old cdx result worked because it used a DXP wrapper that split the Torch
  and DXP meanings of `DXP_LX_FRAC_AVAIL`.
- With the split env and runtime library ordering reproduced on
  `adnan-spyre-dev-pf`, the full-LX Granite causal-prefill speedup is recovered.
- Earlier comms-collectives failures should not be interpreted as a branch
  regression until rerun under this exact environment.

## Next Reproduction Step

To evaluate the collectives prototype, rerun the same command with the corrected
environment and explicitly enable collectives:

```bash
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
```

The latest collectives-enabled run passed but emitted no
`lxRelayoutClassifications_` metadata.  That means the collectives classifier
did not fire on the full Granite block.  The artifact comparison showed:

| metric | baseline off | full Torch LX | collectives enabled |
|---|---:|---:|---:|
| `ReStickifyOpHBM` rows | 5 | 5 | 5 |
| SDSCs with `lxRelayoutClassifications_` | 0 | 0 | 0 |
| LX allocate rows | 53 | 66 | 66 |
| HBM allocate rows | 61 | 54 | 54 |

So the valid interpretation is:

- the current full-LX win comes from more intermediate LX residency inside fused
  chains;
- the explicit HBM restickify rows remain;
- the full Granite block hides the next communication opportunities behind
  already-inserted `ReStickifyOpHBM` nodes;
- the next prototype must intercept or replace those layout-restickify HBM
  nodes before revisiting loop-scoped matmul operand collectives.
