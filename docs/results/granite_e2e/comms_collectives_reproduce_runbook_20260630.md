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

## Restickify-Output Diagnostic

Date: 2026-06-30

This diagnostic answered a narrower question:

```text
If we simply make synthetic spyre.restickify outputs eligible for LX planning,
do the remaining ReStickifyOpHBM rows become LX-backed?
```

Answer: no.  The run still passes and keeps the same performance range, but the
five explicit `ReStickifyOpHBM` rows remain.  The allocator rejects those
buffers because of `core div mismatch`, not because the `restickify` op name is
missing from the LX allowlist.

### Local Diagnostic Patch

This patch was applied locally in the comms checkout for the diagnostic run.  It
is intentionally recorded here as an experiment, not as a production feature.

```diff
diff --git a/torch_spyre/_inductor/config.py b/torch_spyre/_inductor/config.py
@@
 lx_planner_relayout_collectives: bool = (
     os.environ.get("SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES", "0") == "1"
 )
+
+# Experimental research lane for layout-restickify spills.  This only makes
+# synthetic spyre.restickify outputs eligible for LX planning; it does not by
+# itself change the backend relayout carrier.
+lx_planner_relayout_restickify_outputs: bool = (
+    os.environ.get("SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS", "0") == "1"
+)
+
 dxp_lx_frac_avail: float = float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2"))
diff --git a/torch_spyre/_inductor/scratchpad/allocator.py b/torch_spyre/_inductor/scratchpad/allocator.py
@@
                 config.allow_all_ops_in_lx_planning
                 or (self._get_op_name(op) in OP_OUTPUT_GOOD_FOR_LX_REUSE)
+                or (
+                    config.lx_planner_relayout_restickify_outputs
+                    and self._get_op_name(op) == "restickify"
+                )
                 or (config.lx_boundary_clones and self._get_op_name(op) == "clone")
```

### Diagnostic Environment

The diagnostic used the same split frontend/backend LX setup as the known-good
Granite run:

```bash
export ROOT=/home/adnan/codex-isolated/comms_collectives_20260629
export TORCH=$ROOT/torch-spyre
export DEEPTOOLS_PATH=$ROOT/deeptools
export BENCH=/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114/spyre-granite-e2e-bench
export FMS=/home/adnan/dt-inductor/foundation-model-stack
export PYTHON=/home/adnan/dt-inductor/.venv/bin/python3

export PYTHONPATH="$TORCH:$TORCH/tests/inductor:$FMS:${PYTHONPATH:-}"
export PATH=/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114/tools/dxp-split-wrapper:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
export LX_BOUNDARY_CLONES=1

export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1

export LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:${LD_LIBRARY_PATH:-}
```

Run command:

```bash
cd "$BENCH"

RUN="$ROOT/runs/granite_prefill_restickify_outputs_lx_$(date +%Y%m%d_%H%M%S)"
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

Archived run:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_restickify_outputs_lx_20260630_041911
```

Result:

| run | kernel ms/iter | wall ms | status |
|---|---:|---:|---|
| restickify-output diagnostic | 12.3526 | 32.2778 | pass |

Artifact counts:

| metric | count |
|---|---:|
| LX allocate rows | 66 |
| HBM allocate rows | 54 |
| `ReStickifyOpHBM` rows | 5 |
| SDSCs with `lxRelayoutClassifications_` | 0 |

This matches the previous full-LX profile shape: no new restickify rows were
removed.

### Debug Logging Run

To see why the diagnostic did not fire, rerun with legacy Spyre logging:

```bash
export SPYRE_INDUCTOR_LOG=1
export SPYRE_INDUCTOR_LOG_LEVEL=DEBUG

RUN="$ROOT/runs/granite_prefill_restickify_debug_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
export SPYRE_LOG_FILE="$RUN/spyre_debug.log"

cd "$BENCH"
"$PYTHON" benchmarks/granite_block_layer_probe.py \
  --fms-root "$FMS" \
  --run-root "$RUN" \
  --case prefill \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 1 \
  --warmups 0 \
  --no-profile-memory
```

Archived debug run:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_restickify_debug_20260630_042244
```

Useful grep commands:

```bash
rg "Injecting restickify|restickify plan|lx_pinning: buf4[5-9]|ReStickifyOpHBM" "$RUN/spyre_debug.log"
```

Key evidence from the debug log:

```text
Injecting restickify on buf6 input arg2_1: [64, 4096, 1] -> [262144, 1, 4096]
Injecting restickify on buf14 input buf13: [128, -1, 64, 65536, 1] -> [1, -1, 8192, 65536, 128]
Injecting restickify on buf24 input arg5_1: [64, 4096, 1] -> [262144, 1, 4096]
Injecting restickify on buf33 input arg7_1: [64, 4096, 1] -> [262144, 1, 4096]
Injecting restickify on buf36 input arg8_1: [64, 12800, 1] -> [819200, 1, 12800]
```

The five restickify rows are:

| restickify output | source | generated graph evidence | scope |
|---|---|---|---|
| `buf45` | `arg2_1 -> buf6` | `arg2_1` has shape `[6144,4096]`, the attention QKV projection weight | out of scope: weight prelayout |
| `buf46` | `mul_6` / `buf13 -> buf14` | `restickify_default_1 = spyre.restickify(mul_6)` | in scope: computed activation layout restickify |
| `buf47` | `arg5_1 -> buf24` | `arg5_1` has shape `[4096,4096]`, the attention output projection weight | out of scope: weight prelayout |
| `buf48` | `arg7_1 -> buf33` | `arg7_1` has shape `[25600,4096]`, the fused FFN gate/up projection weight | out of scope: weight prelayout |
| `buf49` | `arg8_1 -> buf36` | `arg8_1` has shape `[4096,12800]`, the FFN down-projection weight | out of scope: weight prelayout |

The benchmark uses empty Spyre parameters to avoid copying real checkpoint
weights, but these tensors are still model parameters in the compiled graph.
Their restickifies should be owned by offline/preload weight layout work.  This
branch should not chase those four rows.

The latest selective-retry run keeps that contract explicit in the generated
metadata:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_selective_relayout_retry_20260630_044850

ReStickifyOpHBM rows: 5
layout_restickify_weight classifications: 4
disabled runtime relayout reservations: buf14:buf46, buf22:buf21
```

Each weight row is classified as:

```text
kind = layout_restickify_weight
communication_pattern = offline_weight_prelayout
unsupported_reason = graph-input/parameter restickify is owned by offline weight prelayout, not runtime LX relayout
```

The one non-weight restickify remains `buf13 -> buf14`, generated from
`mul_6`.  That row is a computed attention activation restickify, not a model
parameter restickify.  It is still coupled to the downstream `buf46 -> buf14`
matmul operand broadcast, which is why making `restickify` outputs LX-eligible
by itself is not enough to remove the spill.

### Latest Classification Run

After adding explicit computed activation restickify classification, reproduce
with the same diagnostic environment above and:

```bash
RUN="$ROOT/runs/granite_prefill_layout_restickify_class_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
echo "$RUN" > "$ROOT/latest_layout_restickify_class_run.txt"

"$PYTHON" benchmarks/granite_block_layer_probe.py \
  --fms-root "$FMS" \
  --run-root "$RUN" \
  --case prefill \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 3 \
  --warmups 1 \
  --profile \
  --no-profile-memory 2>&1 | tee "$RUN/output.log"
```

Archived run:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_layout_restickify_class_20260630_050148
```

Result:

| metric | value |
|---|---:|
| `kernel_ms_per_iter` | 12.0335 |
| median wall ms | 32.5332 |
| `ReStickifyOpHBM` rows | 5 |
| SDSCs with `lxRelayoutClassifications_` | 15 |
| `scatter` classes | 14 |
| `layout_restickify_weight` classes | 4 |
| `layout_restickify_activation` classes | 1 |
| `matmul_operand_broadcast` classes | 1 |

The non-weight unrealized classes are:

| source | consumer SDSC | class | communication pattern | current gap |
|---|---|---|---|---|
| `buf46` | attention `sdsc_10.json` | `layout_restickify_activation` | `layout_transform_then_operand_broadcast` | needs LX layout-restickify contract plus loop-scoped matmul operand lowering |
| `buf21` | attention `sdsc_18.json` | `matmul_operand_broadcast` | `all_gather_replicate` | full resident reservation does not fit; needs staged/loop-scoped lowering |

Use this scanner to regenerate the class counts:

```bash
RUN=$(cat "$ROOT/latest_layout_restickify_class_run.txt")
RUN_PATH="$RUN" python3 - <<'PY'
import collections, json, os, pathlib

run = pathlib.Path(os.environ["RUN_PATH"])
root = run / "block_prefill/cache/inductor-spyre"
counts = collections.Counter()
for f in root.rglob("sdsc_*.json"):
    data = json.load(open(f))
    text = json.dumps(data)
    root_obj = next(iter(data.values()))
    rel = root_obj.get("lxRelayoutClassifications_", {})
    if rel:
        counts["class_files"] += 1
        for plan in rel.values():
            counts["class:" + plan.get("kind", "")] += 1
    if "ReStickifyOpHBM" in text:
        counts["restickify"] += 1
print(dict(counts))
PY
```

### Backend Source Pointers For The Next Patch

The latest evidence says the next patch is not another Torch allocator tweak.
The remaining non-weight classes need Deeptools to use staged input movement
instead of full resident materialization.

Current resident-relayout insertion is here:

```text
deeptools/dxp/dxp.cpp
  Dxp::runDsmRelayout(sdsc, executionStep, memTrackers, relayout_sdscs)

deeptools/dxp/SdscRelayoutInsertion.cpp
  Dxp::insertRelayoutSdsc(...)
```

This path inserts a standalone relayout before the consumer and reserves a full
post-relayout resident view.  It is the right class for resident `scatter`, but
not for `matmul_operand_broadcast`.

Existing staged input-neighbor fetch support is here:

```text
deeptools/dcg/dcg_manager/dcg_manager.cpp
  DcgManager::runDcgForInputFetchNeighbor(SuperDsc& mySDscMain, SuperDsc* mySDscPre)

deeptools/dcg/dcg_fe/pcfg_gen/pcfg_gen.cpp
  DcgFE::generatePcfgIRForDataOpInpFetch(...)

deeptools/dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp
  DcgFE::fillDataDSCForInputFetchNeighbor(...)
```

That path already creates an `STCDPOpLx` data op with producer/consumer
subpieces, inferred segment groups, chunk-rank metadata, multicast metadata, and
traffic-per-chunk accounting.

The next Deeptools prototype should:

1. detect `lxRelayoutClassifications_` with
   `kind == matmul_operand_broadcast`;
2. find the producer SDSC for that source tensor;
3. call the existing input-neighbor fetch path for the consumer/prod pair;
4. hard-fail on missing producer/consumer metadata instead of falling back to
   HBM;
5. leave resident `scatter` on the current `SdscRelayoutInsertion` path.

Do not try to remove the computed activation restickify by simply forcing its
output into LX.  That would make the downstream batchmatmul read a local shard
without the required operand broadcast.  The correct implementation is computed
LX layout restickify plus staged operand movement.

### Deeptools Schema Patch

Current Deeptools `SuperDsc` does not preserve unknown top-level JSON fields.
That means Torch's `lxRelayoutClassifications_` metadata is dropped on import
unless Deeptools grows an explicit schema field.

The local Deeptools branch:

```text
/Users/adnan/torch-spyre-work/deeptools-comms-collectives
branch: ah/comms-collectives
```

adds:

```text
dsc/superdsc.h
  std::map<std::string, std::map<std::string, std::string>>
      lxRelayoutClassifications_;

dsc/superdsc.cpp
  import/export for scalar fields under lxRelayoutClassifications_
```

This is not the full lowering.  It is the schema hook required for DXP to route
`scatter` to resident relayout, `matmul_operand_broadcast` to staged
input-neighbor fetch, and future classes to their own lowering paths.

Validation status: patched locally, not yet rebuilt in a Deeptools build tree in
this runbook.  Run a narrow Deeptools build before treating this backend patch
as ready.

Allocator rejection:

```text
lx_pinning: buf45 (restickify) -> core div mismatch
lx_pinning: buf46 (restickify) -> core div mismatch
lx_pinning: buf47 (restickify) -> core div mismatch
lx_pinning: buf48 (restickify) -> core div mismatch
lx_pinning: buf49 (restickify) -> core div mismatch
```

The corresponding `ReStickifyOpHBM` iteration spaces:

```text
ReStickifyOpHBM iteration_space={c0: (6144, 32), c1: (4096, 1)}
ReStickifyOpHBM iteration_space={c0: (32, 32), c1: (512, 1), c2: (128, 1)}
ReStickifyOpHBM iteration_space={c0: (4096, 32), c1: (4096, 1)}
ReStickifyOpHBM iteration_space={c0: (25600, 25), c1: (4096, 1)}
ReStickifyOpHBM iteration_space={c0: (4096, 1), c1: (12800, 25)}
```

### What This Proves

The remaining explicit HBM rows are not solved by widening the LX output
allowlist.  They are rejected because the synthetic restickify buffers have a
different core division from the surrounding producer/consumer views.

For this branch, the next real feature should only target the computed
activation restickify.  It needs to treat that row as a layout-changing relayout
edge, not as an ordinary op output:

1. keep graph-input/weight restickifies out of scope, since weight preloading is
   expected to handle that class;
2. start with the computed activation restickify `mul_6` / `buf13 -> buf14`;
3. preserve both pre-restickify and post-restickify layout metadata;
4. decide whether Torch can emit an LX-to-LX `ReStickifyOpHBM` equivalent using
   existing SDSC fields, or whether Deeptools needs an explicit pre/post layout
   contract for LX restickify;
5. only after layout-restickify is working, return to loop-scoped matmul operand
   collectives such as attention value-side broadcast/all-gather.

### Minimal Sanity Probe For LX ReStickify SDSC Encoding

Before adding more passes, check whether Torch SDSC codegen can describe a
restickify op with both input and output allocated in LX:

```bash
cd "$TORCH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$TORCH:$TORCH/tests/inductor:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:${LD_LIBRARY_PATH:-}

"$PYTHON" - <<'PY'
from sympy import Integer, Symbol, Mod, floor
from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import OpSpec, TensorArg

mb = Symbol("x0")
out = Symbol("x1")
args = [
    TensorArg(True, 0, DataFormats.SEN169_FP16, [512, 200, 64],
              [mb, floor(out / 64), Mod(out, 64)], {"lx": 0}),
    TensorArg(False, 1, DataFormats.SEN169_FP16, [200, 512, 64],
              [floor(out / 64), mb, Mod(out, 64)], {"lx": 4096}),
]
op = OpSpec(
    op=RESTICKIFY_OP,
    is_reduction=False,
    iteration_space={mb: (Integer(512), 32), out: (Integer(12800), 1)},
    args=args,
    op_info={},
)
sdsc, *_ = compile_op_spec(0, op, [])
root = next(iter(sdsc.values()))
allocs = []
for node in root["dscs_"][0][RESTICKIFY_OP]["scheduleTree_"]:
    if node.get("nodeType_") == "allocate":
        address = node.get("startAddressCoreCorelet_", {}).get("data_", {}).get("[0, 0, 0]")
        allocs.append((node["name_"], node["component_"], node.get("layoutDimOrder_"), address))
print(allocs)
PY
```

If this emits valid `lx` allocations, the next prototype can try a computed-only
LX restickify.  If it cannot, the gap is in SDSC/schema/codegen before runtime.
