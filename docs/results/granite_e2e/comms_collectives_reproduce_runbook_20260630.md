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

## Current-Source DXP Replay

The current `ah/comms-collectives` validation uses a DXP binary rebuilt from the
current Deeptools checkout, not the archived scatter DXP binary.

Current paths:

```text
ROOT=/home/adnan/codex-isolated/comms_collectives_20260629
TORCH=$ROOT/torch-spyre
DEEPTOOLS_PATH=$ROOT/deeptools
BENCH=$ROOT/spyre-granite-e2e-bench
DXP_BUILD=$DEEPTOOLS_PATH/build-dxp-comms-current
DXP_INSTALL=$DEEPTOOLS_PATH/install-comms-current
DXP_WRAPPER=$ROOT/tools/dxp-split-wrapper-current/dxp_standalone
```

Current-source runs:

```text
$ROOT/runs/granite_prefill_collectives_current_dxp_20260630_060731
$ROOT/runs/granite_prefill_collectives_current_dxp_runtimefix_20260630_061024
$ROOT/runs/granite_prefill_collectives_current_dxp_routingfix_20260630_061426
$ROOT/runs/debug_relayout_replay_20260630_062207
```

Lessons from those runs:

- Put `/opt/ibm/spyre/deeptools/lib` before inherited `dt-inductor` Deeptools
  libraries.  Otherwise Python can import a stale `libdvs.so` and fail before
  compilation.
- Route only true mixed SDSCs through `runDcgForDataOpsDlOps`: a populated
  `coreIdToDscSchedule` is not enough; `dataOpdscs_` must also be nonempty.
- Current Deeptools automatic relayout insertion successfully inserts LX
  `STCDPOpLx` relayouts for resident `Tensor0` scatter-like mismatches.
- The attention value-side `Tensor1` operand is a different class:
  `matmul_operand_broadcast` / `all_gather_replicate`, `read_index=1`,
  consumer ds type `KERNEL`.  It needs staged operand movement, not full
  resident consumer-view materialization.

## Attention Operand Broadcast Replay

The current active collective reproducer is the attention `sdsc_18` bundle from:

```text
$ROOT/runs/granite_prefill_collectives_current_dxp_routingfix_20260630_061426/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_1kh00ffo
```

Replay command:

```bash
export DXP_LX_FRAC_AVAIL=1
export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
$DXP --bundle -d "$BUNDLE"
```

The SDSC confirms this is not a resident scatter:

```text
sdsc_18 root op: 18_batchmatmul
consumer work division: {x:1, mb:32, out:1, in:1}
classification: matmul_operand_broadcast
communication_pattern: all_gather_replicate
read_index: 1
consumer ds type: KERNEL
producer shards: 32
consumer tensor split dims: {}
```

The producer has 32 operand shards, while each consumer core wants the full
operand.  That makes this a true all-gather/replicate class.  It should not be
treated as PR1's resident scatter/permutation class.

Known failed lowerings:

| attempt | artifact | result | interpretation |
|---|---|---|---|
| full resident materialization | debug replay before input-neighbor routing | `LX_MODLRFIMM :: lrfimm:-4161536` | full per-consumer materialization is too large for LX and produces invalid LX immediates |
| chunked loop-scoped input-neighbor, original chunking | replay before chunk override | `dtTable=4096 inpSP=4096 outSP=4096`, timeout | one all-gather row is too monolithic |
| grouped destination pieces plus `x=16` chunk override | `attention_broadcast_replay_x16_20260630_092727` | `Max IBUFF(256) Current IBUFF(745)` for L3LU | representation reaches DCC but still over-grows L3LU instruction buffer |
| disabling `STCDPOpLx::enSubPieceReuse` for the broadcast | `attention_broadcast_replay_noreuse_20260630_093208` | `Max IBUFF(256) Current IBUFF(1729)` for L3LU | reuse/interleaving is not the cause; disabling it makes the program larger |
| forcing `launchDCC` to O3 | `attention_broadcast_replay_dcc_o3_20260630_094121` | `Max IBUFF(256) Current IBUFF(745)` for L3LU | existing DCC optimization level is not the missing legalization |
| compact grouped L3LU cap=16 | `attention_broadcast_replay_compact_l3lu2_20260630_094844` | `Max IBUFF(256) Current IBUFF(1377)` for L3LU | bounded grouping still emits too much nested L3LU control flow |
| compact grouped L3LU cap=4 | `attention_broadcast_replay_compact_l3lu_cap4_20260630_095205` | `Max IBUFF(256) Current IBUFF(1346)` for L3LU | smaller groups do not recover a legal program |
| compact grouped L3LU cap=2 | `attention_broadcast_replay_compact_l3lu_cap2_20260630_095518` | DXP/DCC timeout after 180s, still `dtTable=256` | cap tuning can make compile time worse without reducing the logical transfer table |
| staged input-neighbor, 4 producer cores per row | `attention_broadcast_replay_staged4_20260630_101812` | passes DXP replay; 8 rows with `dtTable=4 inpSP=4 outSP=4 maxL3SU=1 maxL3LU=4` | legalizes the first Granite all-gather-like operand broadcast by splitting one oversized movement row into small scheduled phases |
| staged input-neighbor, 4 producer cores per row, compact coordinate-sorted L3SU/L3LU | `attention_broadcast_replay_staged4_sorted_20260630_102627` | passes DXP replay; same 8 staged rows | stronger runtime candidate because compact rows use coordinate-sorted send/receive ordering |

Latest DCC/runtime status:

- A Deeptools DCC index fix is required for the staged rows: use the current
  data-op descriptor via `dataOpdscs_.at(datadscIdx)`.
- With that fix, the all-gather/replicate lowering now compiles.
- The stage-size compile sweep passes for
  `DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=4`, `8`, `16`, and `32`.
- Full Granite currently compiles and then bus-fences at runtime.  Treat the
  current branch state as runtime-safety blocked, not DCC-compile blocked.
- Next diagnostic: run full Granite with
  `DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=32`.

The useful current conclusion is that `matmul_operand_broadcast` needs a real
collective/staged-input lowering.  Expressing it as one large input-neighbor
`STCDPOpLx` row is functionally close but not codegen-legal because the
generated L3LU program exceeds IBUFF.  The grouped-cap experiments show this is
not a one-knob DCC tuning issue; the emitted collective needs a more compact
ring program or multiple legal staged phases.  The staged-four replay proves the
multiple-phase shape is viable for the isolated `sdsc_18` attention operand
broadcast.

The first full Granite execution attempt with staged-four append ordering
compiled, emitted the staged rows, and then hit a runtime PCIe bus fence during
the first block iteration:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_staged4_20260630_102000
RAS::PCI::BusFence, code 0xa35e
```

That is a runtime-safety issue, not an SDSC import or DCC legality issue.  The
current next diagnostic keeps the now-compiling staged movement path and runs
full Granite with `DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=32`.

If the device fences, the hot-reset utility requires the PCI device id:

```bash
/opt/ibm/spyre/senlib/bin/aiu_dd2_hot_reset -t chip -d 3c:00.0
```

On `adnan-spyre-dev-pf` this can initialize the chip path but the unprivileged
Linux reset is blocked; use another pod for continued experiments if reset is
not clean.

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
BENCH=$ROOT/spyre-granite-e2e-bench
FMS=/home/adnan/dt-inductor/foundation-model-stack
PYTHON=/home/adnan/dt-inductor/.venv/bin/python3
```

Do not validate this work from older incidental Deeptools/Torch clones under
`/home/adnan/dt-inductor/deeptools-*`.  They may contain unrelated historical
experiments and can make DXP behavior look different when the actual difference
is the checkout or environment.

Also do not use the old split wrapper as-is for this branch:

```text
/home/adnan/codex-isolated/pr_lx_scatter_20260629_170114/tools/dxp-split-wrapper/dxp_standalone
```

That wrapper is still useful as a template for the frontend/backend
`DXP_LX_FRAC_AVAIL` split, but it points at the older
`pr_lx_scatter_20260629_170114/deeptools/build-dxp-relayout-isolated` binary.
For comms-collectives validation, create a fresh wrapper that points at a DXP
build produced from:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/deeptools
```

When configuring that build, pass `-DMANAGE_LLVM=false` with the existing
`LLVM_PROJ_SRC` and `LLVM_PROJ_BUILD` paths.  Without that flag, Deeptools CMake
tries to manage the shared LLVM checkout through `ExternalProject`, including a
destructive source checkout/update step.  The safe intent is to consume the
already-built LLVM/MLIR package configs, not rebuild or reclone LLVM.

Local Mac setup is intentionally lightweight:

```text
/Users/adnan/.codex/venvs/torch-spyre-cpu
```

That venv currently has `pytest` for syntax/helper checks.  Meaningful
Torch-Spyre tests still run on the pod because the local Mac does not have the
compiled `torch_spyre._C` extension or an AIU runtime.

Important SHAs from the archived environment:

```text
Torch: 3a222ecc5dcd6c8448c5753e94e13e9c1a1d5d5b
Deeptools: b8c09743c46505b4cac46b434b9eb3243ae0b685
spyre-granite-e2e-bench: 76cd51426ba1de6e99dd8fbf613cb0f32b71e87f
```

## DXP Split Wrapper

For current comms-collectives validation, use this wrapper first in `PATH`:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/tools/dxp-split-wrapper-current/dxp_standalone
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
export PATH=/home/adnan/codex-isolated/comms_collectives_20260629/tools/dxp-split-wrapper-current:$PATH
export DEEPTOOLS_PATH="$DEEPTOOLS_PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

export SPYRE_LX_PLANNER_RELAYOUT=1
unset SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES
export LX_BOUNDARY_CLONES=1

export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1

export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
```

The `LD_LIBRARY_PATH` ordering matters.  Earlier failing runs used
`/home/adnan/dt-inductor/flex-pr1019-install/lib` before the installed runtime
libraries and hit attention hardware errors even when the SDSCs otherwise
matched.

On CDX, use a clean runtime `LD_LIBRARY_PATH` with the installed Spyre
Deeptools/runtime/comms libraries first, as shown above.  Do not inherit stale
`dt-inductor`, flex, or older Deeptools library paths ahead of
`/opt/ibm/spyre/deeptools/lib`, `/opt/ibm/spyre/runtime/lib`, and
`/opt/ibm/spyre/spyre-comms/lib`.

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
export BENCH=$ROOT/spyre-granite-e2e-bench
export FMS=/home/adnan/dt-inductor/foundation-model-stack
export PYTHON=/home/adnan/dt-inductor/.venv/bin/python3

export PYTHONPATH="$TORCH:$TORCH/tests/inductor:$FMS:${PYTHONPATH:-}"
export PATH=$ROOT/tools/dxp-split-wrapper-current:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
export LX_BOUNDARY_CLONES=1

export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1

export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
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
export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

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

## 2026-06-30 Guarded Collective Reproduction Update

The all-gather-style attention operand experiment produced three useful
outcomes:

1. DXP compile was unblocked by fixing mixed data-op PCFG generation to use the
   scheduled data-op index:

   ```text
   dcg/dcg_fe/pcfg_gen/pcfg_gen.cpp
     mySDscMain.dataOpdscs_.at(datadscIdx)
   ```

2. Full resident all-gather is unsafe for the attention operand.  Both compact
   and explicit/non-compact variants fenced:

   ```text
   CDX compact stage32:
   /home/adnan-cdx/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_stage32_20260630_112057
   PROC_RC=255, RAS::PCI::BusFence

   CLC non-compact stage32:
   /home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_noncompact_stage32_20260630_113822
   PROC_RC=255, RAS::PCI::BusFence
   ```

3. The safe current behavior is to keep resident collectives guarded and
   require tiled/loop-scoped lowering for large operands:

   ```bash
   export SPYRE_LX_PLANNER_RELAYOUT=1
   export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
   export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVE_MAX_BYTES=1048576
   export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
   ```

The successful guarded Granite prefill run is:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_20260630_114425
returncode=0
wall median=24.495 ms
trace kernel_ms_per_iter=12.0539
trace=/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_20260630_114425/block_prefill/trace/adnan-spyre-dev-pf_41462.1782819965394707728.pt.trace.json
```

Fresh replay after the CLC pod reset and the `ReStickifyOpLx` frontend
contract/test patch:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_clc_20260630_122458
returncode=0
wall median=23.9255 ms
trace kernel_ms_per_iter=12.0628
trace=/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_clc_20260630_122458/block_prefill/trace/adnan-clc-spyre-dev-pf_489.1782822400344090752.pt.trace.json
```

Focused frontend contract tests on CLC:

```text
tests/inductor/test_lx_relayout_dldsc.py
14 passed in 0.16s
```

The CLC replay preserves the same guarded communication inventory:

```text
ReStickifyOpHBM rows: 5
SDSCs with lxRelayoutClassifications_: 15
scatter realized: 14
layout_restickify_weight unrealized: 4
layout_restickify_activation unrealized: 1
matmul_operand_broadcast unrealized: 1
```

The remaining `matmul_operand_broadcast` is the attention value-side operand in:

```text
.../sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_l66aaivz/sdsc_18.json
estimated_tensor_bytes = 4194304
realized = false
unsupported_reason = resident all-gather exceeds SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVE_MAX_BYTES=1048576; needs tiled/streamed lowering
```

## Staged Broadcast After-Sync Probe

After reviewing the Deeptools staged input-neighbor path, one likely runtime
safety issue was that the synthesized data-op schedule ordered movement before
DL compute but did not request a runtime sync:

```cpp
schedule.push_back(DscScheduleStep(dataDscIdx, -1, false, false));
```

The local experiment changed the staged broadcast rows to request `after_sync`:

```cpp
schedule.push_back(DscScheduleStep(dataDscIdx, -1, false, true));
```

Build on CLC:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/deeptools/build-dxp-comms-current/dxp/dxp_standalone
target dxp_standalone rebuilt successfully
```

Compile-only replay still passes with the patched DXP:

```text
BUNDLE=/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_clc_20260630_122458/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_l66aaivz
DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=4
dxp_standalone --bundle -d "$BUNDLE"
exit code 0
```

Full Granite with the resident all-gather guard raised to force the staged
broadcast path still bus-fenced:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_staged4_aftersync_clc_20260630_123407
DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=4
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVE_MAX_BYTES=999999999
return code 255
RAS::PCI::BusFence code 0xa35e
```

Conclusion: missing schedule fencing was a real concern and worth fixing, but
it is not the only runtime-safety issue.  The remaining likely problems are in
the staged input-neighbor realization itself: grouped output pieces,
multi-consumer `cIDXs`, GTR/GTRIMM behavior for non-`INPUT` consumers, or the
fact that the current resident form still writes a full consumer operand region
instead of a matmul-loop-scoped tile.

The proof that the unsafe attention collective was skipped is in:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_20260630_114425/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_7yt9sfzx/sdsc_18.json
```

Expected fields:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
estimated_tensor_bytes = 4194304
realized = false
unsupported_reason contains "needs tiled/streamed lowering"
```

For a compact shareable inventory of the guarded run, use:

```text
docs/results/granite_e2e/comms_collectives_guarded_spill_inventory_20260630.md
docs/results/granite_e2e/comms_collectives_guarded_spill_inventory_20260630.csv
```

That inventory records:

```text
scatter realized: 14
layout_restickify_weight unrealized: 4
layout_restickify_activation unrealized: 1
matmul_operand_broadcast unrealized: 1
```

Interpretation:

- the four `layout_restickify_weight` rows are graph-input/parameter layout
  preparation and should stay with offline weight prelayout work;
- the one `layout_restickify_activation` row is the remaining non-weight
  HBM restickify in attention;
- the one `matmul_operand_broadcast` row is the attention value-side
  all-gather/replicate class skipped by the resident-size guard.

### Pod Runtime Notes

Do not assume all pods have identical `/opt/ibm/spyre` runtime contents:

- `adnan-spyre-dev-pf`: `/opt/ibm/spyre/runtime/lib/libflex.so` has the
  required `AllocationDirective(... MemoryType)` symbol.  Use `/opt` first.
- `adnan-clc-spyre-dev-pf`: `/opt/ibm/spyre/runtime/lib/libflex.so` lacks that
  symbol.  Use `/home/adnan/opt-newer` first for Torch/Spyre imports.
- `adnan-cdx-spyre-dev-pf`: use the clean stack runtime first:
  `/home/adnan-cdx/dt-inductor-codex-clean/install/runtime/lib`.

After the unsafe all-gather bus-fence experiments, reset CLC before using it
for hardware validation.  The reset procedure is:

```bash
kubectl get pod -n a6-quantization adnan-clc-spyre-dev-pf -o json \
  > tmp/pod-restart/adnan-clc-spyre-dev-pf.raw.json
# create sanitized restart spec:
# tmp/pod-restart/adnan-clc-spyre-dev-pf.restart.json
kubectl delete pod -n a6-quantization adnan-clc-spyre-dev-pf --wait=true
kubectl apply -f tmp/pod-restart/adnan-clc-spyre-dev-pf.restart.json
kubectl wait -n a6-quantization --for=condition=Ready pod/adnan-clc-spyre-dev-pf --timeout=10m
```

Do not run new hardware tests on CLC until the reset completes and a `ps -ef`
check shows no stale Python/DXP/senprog/AIU jobs.

Historical reset result from the first 2026-06-30 CLC recreate:

```text
pod/adnan-clc-spyre-dev-pf created
pod/adnan-clc-spyre-dev-pf condition met
STATUS=Running
IP=10.128.14.133
NODE=p1-worker-28
creationTimestamp=2026-06-30T12:02:34Z
AIU=/dev/vfio/73
stale python/dxp/senprog/aiu processes: none
```

Current reset result after the later 2026-06-30 CLC recreate:

```text
pod/adnan-clc-spyre-dev-pf created
pod/adnan-clc-spyre-dev-pf condition met
STATUS=Running
IP=10.128.18.230
NODE=p1-worker-43
creationTimestamp=2026-06-30T14:57:xxZ
AIU=/dev/vfio/25
stale python/dxp/senprog/aiu processes: none
```

The split LX env remains required:

```bash
export DXP_LX_FRAC_AVAIL=0          # frontend/Torch gets full-LX planning
export DXP_BACKEND_LX_FRAC_AVAIL=1  # wrapper maps this to DXP_LX_FRAC_AVAIL for backend DXP
```

## 2026-06-30 Staged Broadcast Debug Runs

Do not put FMS on startup `PYTHONPATH` for the Granite layer probe.  FMS has a
lightweight `fms/triton` package that shadows Triton and lacks
`triton.language`, which makes `torch._dynamo` fail before compilation:

```text
AttributeError: module 'triton' has no attribute 'language'
```

Pass FMS via `--fms-root` instead.  A run-local `sitecustomize.py` that calls
`torch_spyre._autoload()` is sufficient when the runtime library path is
ordered correctly.

### Compact Finite-Burst Replay

After patching `finalizeBurstInfo()` to ceil-divide sub-stick dimensions and
clamp burst/transaction factors to at least one, the compact staged broadcast
data-op dump is finite:

```bash
ROOT=/home/adnan/codex-isolated/comms_collectives_20260629
RUN=$ROOT/runs/granite_prefill_collectives_staged4_aftersync_clc_20260630_123407
BUNDLE=$RUN/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_imhpskvs
DUMP=$ROOT/runs/dxp_dataop_dump_ceilburst_20260630_125425

export DEEPTOOLS_PATH=$ROOT/deeptools
export DXP_LX_FRAC_AVAIL=1
export DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=4
export DXP_DUMP_RELAYOUT_DATAOPS_DIR=$DUMP
$ROOT/deeptools/build-dxp-comms-current/dxp/dxp_standalone --bundle -d "$BUNDLE"
```

Expected transfer summary:

```text
18_batchmatmul-dataop-{0..7}.json
entries=4
inf=0
maxBurst={32: 4}
numTransactions={512: 4}
cMemIDs=[32]
cIDXs=[1]
```

Hardware result with this compact path:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_staged4_ceilburst_clc_nofmspath_20260630_125846
return code 255
RAS::PCI::BusFence code 0xa35e
```

### Noncompact Partial-Route Replay

The noncompact path needs a separate backend marker for staged partial-output
coverage.  `compactInputNeighborBroadcast=0` must not route the data-op through
ordinary STCDP, because ordinary STCDP expects every data-op row to fully cover
the output LDS piece.  The exploratory patch uses `partialOutputCoverage=true`
to keep the row on the InputFetchNeighbor route while disabling compact grouped
output behavior.

DXP replay command:

```bash
ROOT=/home/adnan/codex-isolated/comms_collectives_20260629
RUN=$ROOT/runs/granite_prefill_collectives_staged4_ceilburst_clc_nofmspath_20260630_125846
BUNDLE=$RUN/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_s70orcsi
DUMP=$ROOT/runs/dxp_dataop_dump_noncompact_partialroute_20260630_130859

export DEEPTOOLS_PATH=$ROOT/deeptools
export DXP_LX_FRAC_AVAIL=1
export DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=4
export DXP_LX_RELAYOUT_BROADCAST_COMPACT=0
export DXP_DUMP_RELAYOUT_DATAOPS_DIR=$DUMP
$ROOT/deeptools/build-dxp-comms-current/dxp/dxp_standalone --bundle -d "$BUNDLE"
```

Expected transfer summary:

```text
18_batchmatmul-dataop-{0..7}.json
entries=4
inf=0
maxBurst={1: 4}
numTransactions={16384: 4}
cMemIDs=[32]
cIDXs=[32]
```

Hardware result:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_noncompact_partialroute_clc_20260630_130928
return code 255
RAS::PCI::BusFence code 0xa35e
```

The same noncompact path without `--profile` also fenced:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_noncompact_partialroute_noprofile_clc_20260630_131206
return code 255
RAS::PCI::BusFence code 0xa35e
```

Conclusion: the failure is not caused by Kineto/AIUPTI profiling, and it is not
only caused by compact grouped output pieces.  The current staged broadcast
still behaves like whole-operand precompute setup.  Attention needs a true
matmul-loop-scoped operand movement primitive tied to the consumer matmul
schedule.

Latest CLC reset after the risky runs:

```text
pod/adnan-clc-spyre-dev-pf created
pod/adnan-clc-spyre-dev-pf condition met
IP=10.128.18.230
NODE=p1-worker-43
AIU=/dev/vfio/25
stale python/dxp/senprog/aiu processes: none
```

### Current Paired IFN Replay Result

The guarded Granite run above remains the safe CLC success baseline:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_clc_20260630_122458
returncode=0
kernel_ms_per_iter=12.0627818
```

For the attention `sdsc_18` broadcast/all-gather
(`matmul_operand_broadcast`, `all_gather_replicate`), the current paired
InputFetchNeighbor experiment is DXP-negative: paired compact and paired
noncompact both reach DCC, then fail L3LU IBUFF.

| paired IFN form | DXP/DCC result |
|---|---|
| compact | `Max IBUFF(256) Current IBUFF(651)` |
| noncompact | `Max IBUFF(256) Current IBUFF(745)` |

`DXP_LX_RELAYOUT_BROADCAST_IFN_COALESCE` values `1`, `2`, `4`, and `8` did not
change the compact result.  Do not treat the paired IFN path as a solved
Granite runtime candidate until it gets past DCC and then hardware validation.

### Latest DXP/Runtime Controls

After the CLC pod was recreated, the valid restored paired compact IFN replay
was:

```bash
ROOT=/home/adnan/codex-isolated/comms_collectives_20260629
BUNDLE=$ROOT/runs/granite_prefill_collectives_staged4_ceilburst_clc_nofmspath_20260630_125846/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_s70orcsi
OUT=$ROOT/runs/dxp_paired_ifn_affine_restored_clc_20260630_142351

DEEPTOOLS_PATH=$ROOT/deeptools \
DXP_LX_FRAC_AVAIL=1 \
DXP_DUMP_RELAYOUT_DATAOPS_DIR=$OUT/dataops \
DXP_LX_RELAYOUT_BROADCAST_PAIRED_IFN=1 \
DXP_LX_RELAYOUT_BROADCAST_COMPACT=1 \
DXP_LX_RELAYOUT_IFN_AFFINE_DTKEYS=1 \
$ROOT/deeptools/build-dxp-comms-current/dxp/dxp_standalone --bundle -d "$BUNDLE"
```

Result:

```text
rc=134
Max IBUFF(256) Current IBUFF(651)
full 32-producer coverage in dataops/18_batchmatmul-dataop-0.json
```

The affine destination-address compression did not reduce the dominant L3LU
control shape.  The dumped data-op still contains 8192 each of
`coreIDForRingCondAndVal`, `GTRAndBurstCondAndVal`, `destStartCondAndVal`, and
`bigStAddrOffsets`.

Forced-unicast controls are useful diagnostics but not viable:

```text
dxp_paired_ifn_affine_unicast_forced_clc_20260630_143338
dxp_paired_ifn_noncompact_unicast_forced_clc_20260630_143411
```

Both fail DCC with `coreids_ring.size() == 1`.  This means unicast cannot be
applied to the current grouped multi-destination IFN piece without a different
lowering that splits each destination into its own legal transfer.

The staged non-paired no-sync DXP replay is:

```bash
ROOT=/home/adnan/codex-isolated/comms_collectives_20260629
BUNDLE=$ROOT/runs/granite_prefill_collectives_staged4_ceilburst_clc_nofmspath_20260630_125846/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_s70orcsi
OUT=$ROOT/runs/dxp_nonpaired_stage32_nostagedsync_clc_20260630_143615

DEEPTOOLS_PATH=$ROOT/deeptools \
DXP_LX_FRAC_AVAIL=1 \
DXP_DUMP_RELAYOUT_DATAOPS_DIR=$OUT/dataops \
DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=32 \
DXP_LX_RELAYOUT_BROADCAST_COMPACT=1 \
DXP_LX_RELAYOUT_BROADCAST_NO_STAGED_SYNC=1 \
$ROOT/deeptools/build-dxp-comms-current/dxp/dxp_standalone --bundle -d "$BUNDLE"
```

Result:

```text
rc=0
IBUFF=none
```

Hardware result with the same schedule-level no-sync idea:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_stage32_nostagedsync_dev_optfirst_20260630_144557
returncode=255
RAS::PCI::BusFence code 0xa35e
```

The run used `DXP_LX_RELAYOUT_BROADCAST_NO_STAGED_SYNC=1` and
`DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=32`.  The DXP gate passed, but hardware
still fenced.  Therefore, the staged path's broad schedule-level `after_sync`
is not the primary runtime failure.  The current whole-operand staged
all-gather is itself unsafe for the Granite attention operand.

Runtime path gotcha: after pod recreation, CLC could not load the isolated
`torch_spyre._C.so` and `/opt/ibm/spyre/spyre-comms/lib/libspyre_comms.so.1`
with one obvious flex runtime.  On dev, putting `/opt/ibm/spyre/runtime/lib`
first allowed import and reached the hardware fence:

```bash
export LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/sentient/runtime/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/opt/ibm/spyre/deeptools/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
```

Always archive `env.txt` and verify import failures separately from relayout
failures.

## Frontend Loop-Scoped Contract Knob

Use this when generating a new bundle that should represent the attention
`matmul_operand_broadcast` edge as a backend-synthesized operand movement,
rather than as resident full-operand replication:

```bash
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVE_REALIZATION=loop_scoped
```

This records:

```text
communication_pattern = all_gather_replicate
realization_strategy = loop_scoped_input_fetch
```

For CLC unit-test validation after pod recreation, use the newer runtime tree
first:

```bash
cd /home/adnan/codex-isolated/comms_collectives_20260629/torch-spyre
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_LIBRARY_PATH=/home/adnan/opt-newer/runtime/lib:/home/adnan/opt-newer/spyre-comms/lib:/home/adnan/opt-newer/deeptools/lib:/home/adnan/opt-newer/senlib/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
/home/adnan/dt-inductor/.venv/bin/python -m pytest tests/inductor/test_lx_relayout_dldsc.py -q
```

Latest result:

```text
14 passed in 4.18s
```

DXP metadata control after the latest CLC recreate:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_metadata_control_clc_20260630_150250
bundle:
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_stage32_nostagedsync_dev_optfirst_20260630_144557/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_g_umxry2
result:
  sdsc_18 has matmul_operand_broadcast
  DXP rc=0
```

## Subpiece-Reuse Diagnostic For Grouped IFN

The recreated CLC pod for this run was:

```text
pod=adnan-clc-spyre-dev-pf
IP=10.128.18.231
node=p1-worker-43
AIU=/dev/vfio/25
```

First sync the local Deeptools diagnostic change and rebuild DXP:

```bash
COPYFILE_DISABLE=1 tar -C /Users/adnan/torch-spyre-work/deeptools-comms-collectives \
  -cf /tmp/deeptools_sdsc_relayout_insertion.tar dxp/SdscRelayoutInsertion.cpp
kubectl cp /tmp/deeptools_sdsc_relayout_insertion.tar \
  a6-quantization/adnan-clc-spyre-dev-pf:/tmp/deeptools_sdsc_relayout_insertion.tar

kubectl exec -n a6-quantization adnan-clc-spyre-dev-pf -- bash -lc '
set -euo pipefail
ROOT=/home/adnan/codex-isolated/comms_collectives_20260629
cd "$ROOT/deeptools"
tar xf /tmp/deeptools_sdsc_relayout_insertion.tar
find dxp -name "._*" -delete
grep -n "BROADCAST_SUBPIECE_REUSE" dxp/SdscRelayoutInsertion.cpp
cmake --build build-dxp-comms-current --target dxp_standalone -j$(nproc)
'
```

Replay the known attention bundle without running hardware:

```bash
ROOT=/home/adnan/codex-isolated/comms_collectives_20260629
BUNDLE=$ROOT/runs/granite_prefill_collectives_stage32_nostagedsync_dev_optfirst_20260630_144557/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_g_umxry2
DXP=$ROOT/deeptools/build-dxp-comms-current/dxp/dxp_standalone
RUNROOT=$ROOT/runs/dxp_subpiece_reuse_controls_clc_20260630_151848

DEEPTOOLS_PATH=$ROOT/deeptools \
DXP_LX_FRAC_AVAIL=1 \
DXP_DUMP_RELAYOUT_DATAOPS_DIR=$RUNROOT/paired_compact_affine/dataops \
DXP_LX_RELAYOUT_BROADCAST_PAIRED_IFN=1 \
DXP_LX_RELAYOUT_BROADCAST_COMPACT=1 \
DXP_LX_RELAYOUT_IFN_AFFINE_DTKEYS=1 \
$DXP --bundle -d "$BUNDLE"

DEEPTOOLS_PATH=$ROOT/deeptools \
DXP_LX_FRAC_AVAIL=1 \
DXP_DUMP_RELAYOUT_DATAOPS_DIR=$RUNROOT/paired_compact_affine_unicast_noreuse/dataops \
DXP_LX_RELAYOUT_BROADCAST_PAIRED_IFN=1 \
DXP_LX_RELAYOUT_BROADCAST_COMPACT=1 \
DXP_LX_RELAYOUT_IFN_AFFINE_DTKEYS=1 \
DXP_LX_RELAYOUT_BROADCAST_UNICAST=1 \
DXP_LX_RELAYOUT_BROADCAST_SUBPIECE_REUSE=0 \
$DXP --bundle -d "$BUNDLE"
```

Observed results:

```text
paired_compact_affine rc=134
  Max IBUFF(256) Current IBUFF(651)

paired_compact_affine_unicast_reuse rc=134
  DtException: coreids_ring.size() == 1

paired_compact_affine_unicast_noreuse rc=134
  DtException: coreids_ring.size() == 1

paired_noncompact_affine_unicast_noreuse rc=134
  Max IBUFF(256) Current IBUFF(1481)
  Max IBUFF(256) Current IBUFF(1412)
```

Takeaway: `DXP_LX_RELAYOUT_BROADCAST_SUBPIECE_REUSE=0` does not unblock the
grouped IFN route.  The compact unicast path fails in DCC ring lowering; the
noncompact path remains IBUFF-heavy.  Do not spend more time on this exact
whole-operand broadcast family unless Deeptools changes the IFN legality model.

## Parallel Pod Lane Notes

CDX is ready as a second DXP replay lane:

```text
pod=adnan-cdx-spyre-dev-pf
IP=10.129.19.183
node=p1-worker-44
AIU=/dev/vfio/80
experiment root=/home/adnan-cdx/codex-isolated/comms_collectives_20260629
torch branch=ah/comms-collectives
torch sha=df26b2ec7c14159e835a288d3369e7971661c43b
deeptools branch=ah/comms-collectives
deeptools sha=c4cf6a4356dee851f1b1c73de8aafcb3f6e1f643
dxp=/home/adnan-cdx/codex-isolated/comms_collectives_20260629/deeptools/build-dxp-comms-current/dxp/dxp_standalone
wrapper=/home/adnan-cdx/codex-isolated/comms_collectives_20260629/tools/dxp-split-wrapper-current/dxp_standalone
```

Known CDX attention bundle:

```text
/home/adnan-cdx/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_indexfix_cleanrt_20260630_110841/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_hpoq6qop
```

Safe DXP-only replay result:

```text
output=/home/adnan-cdx/codex-isolated/comms_collectives_20260629/runs/dxp_only_metadata_replay_20260630_152132
command:
  /home/adnan-cdx/codex-isolated/comms_collectives_20260629/deeptools/build-dxp-comms-current/dxp/dxp_standalone --bundle -d .../bundle_input
result:
  rc=0
  stderr=empty
```

Use CDX for compile-only/DXP experiments unless a run specifically needs the
new local CLC-only Deeptools source patch.  Sync source and rebuild there before
assuming a diagnostic knob is available.

Dev is ready as the primary hardware/profile lane:

```text
pod=adnan-spyre-dev-pf
IP=10.128.11.76
node=p1-worker-19
AIU=/dev/vfio/31
experiment root=/home/adnan/codex-isolated/comms_collectives_20260629
torch branch=ah/comms-collectives
torch sha=df26b2ec7c14159e835a288d3369e7971661c43b
deeptools branch=ah/comms-collectives
deeptools sha=c4cf6a4356dee851f1b1c73de8aafcb3f6e1f643
dxp=/home/adnan/codex-isolated/comms_collectives_20260629/deeptools/build-dxp-comms-current/dxp/dxp_standalone
stale workload processes=none seen in lightweight scan
```

## Next Implementation Target

Do not continue broad sweeps over the current staged whole-operand broadcast
rows.  The current evidence says that family is the wrong abstraction for the
Granite attention operand:

```text
resident replication: too large / unsafe
one grouped IFN row: IBUFF overflow
unicast grouped IFN: DCC coreids_ring.size() == 1
staged whole-operand rows: DXP can pass, hardware bus-fences
```

The next backend implementation should reuse the existing DL scheduler's
LX-neighbor path:

```text
dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
  isLabeledDsLXNeighbor
  createAllocationAndTransfer
  createSynchronization

dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp
  createPcfgForInputFetchNeighbor
  createSubPieces

dcg/dcg_fe/pcfg_gen/stcdpOp.cpp
  transformToPcfgSTCDPLxUnrolled
```

Target behavior:

1. DXP reads the Torch classification:

   ```text
   kind = matmul_operand_broadcast
   communication_pattern = all_gather_replicate
   realization_strategy = loop_scoped_input_fetch
   read_index = <consumer operand index>
   ```

2. Deeptools marks that consumer operand as an LX-neighbor tensor in the DL
   schedule tree.
3. `L3DlOpsScheduler` inserts the `NO_COMPONENT -> LX` `_lx_neighbor` marker
   inside the consumer matmul loop, before the tile/subchunk compute.
4. Existing InputFetchNeighbor/STCDP lowering realizes the ring movement for
   that tile-scoped marker.

This preserves the frontend/backend split we want: Torch classifies and costs
the communication class; Deeptools realizes a legal scheduled movement using
the coordinates and the matmul schedule it owns.

### IFN-With-DL Diagnostic

An experimental Deeptools knob was added locally:

```bash
export DXP_LX_RELAYOUT_IFN_WITH_DLOP=1
```

It changes `DcgManager::runDcgForDataOpsDlOps` so a paired input-fetch schedule
also attempts normal DL PCFG generation.  Rebuild on CLC:

```bash
COPYFILE_DISABLE=1 tar -C /Users/adnan/torch-spyre-work/deeptools-comms-collectives \
  -cf /tmp/deeptools_dcg_manager_ifn_with_dlop.tar dcg/dcg_manager/dcg_manager.cpp
kubectl cp /tmp/deeptools_dcg_manager_ifn_with_dlop.tar \
  a6-quantization/adnan-clc-spyre-dev-pf:/tmp/deeptools_dcg_manager_ifn_with_dlop.tar

kubectl exec -n a6-quantization adnan-clc-spyre-dev-pf -- bash -lc '
set -euo pipefail
ROOT=/home/adnan/codex-isolated/comms_collectives_20260629
cd "$ROOT/deeptools"
tar xf /tmp/deeptools_dcg_manager_ifn_with_dlop.tar
find dcg -name "._*" -delete
cmake --build build-dxp-comms-current --target dxp_standalone -j$(nproc)
'
```

DXP-only replay:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_ifn_with_dlop_controls_clc_20260630_153441
```

Result:

```text
paired_compact_affine_old rc=134
  Max IBUFF(256) Current IBUFF(651)

paired_compact_affine_with_dlop rc=134
  DtException: unit already set for associated schedule step
  dcc/src/Stitcher/ModuleStitcher.cpp line 279
```

This rules out the naive "generate IFN and DL independently for the same paired
step" approach.  DCC's `ModuleStitcher` indexes program units by schedule step,
and the independent IFN/DL modules collide for the same unit and step.  The next
backend patch should avoid independent module stitching and instead make the
LX-neighbor transfer marker part of the DL-generated module, or explicitly
teach DCC how a paired IFN+DL step is represented.

Marker-only control:

```bash
DEEPTOOLS_PATH=$ROOT/deeptools \
DXP_LX_FRAC_AVAIL=1 \
DXP_DUMP_RELAYOUT_DATAOPS_DIR=$OUT/dataops \
DXP_LX_RELAYOUT_BROADCAST_PAIRED_IFN=1 \
DXP_LX_RELAYOUT_BROADCAST_COMPACT=1 \
DXP_LX_RELAYOUT_IFN_AFFINE_DTKEYS=1 \
DXP_LX_RELAYOUT_IFN_WITH_DLOP=1 \
DXP_LX_RELAYOUT_IFN_DLOP_MARKER_ONLY=1 \
$DXP --bundle -d "$BUNDLE"
```

Result:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_ifn_dlop_marker_only_clc_20260630_153749
rc=134
std::out_of_range: vector::_M_range_check: __n (which is 0) >= this->size()
```

This rules out the opposite naive approach too.  The DL scheduler cannot use
the marker alone; it needs IFN metadata from `generatePcfgIRForDataOpInpFetch`.
The missing backend abstraction is therefore more specific:

```text
build IFN metadata and tile-scoped transfer marker
without stitching a separate IFN module into the same schedule step
```

### Metadata-Only IFN Plus DCC Skip

The closest diagnostic to the desired architecture is:

```bash
export DXP_LX_RELAYOUT_IFN_WITH_DLOP=1
export DXP_LX_RELAYOUT_IFN_METADATA_ONLY=1
```

Local guarded changes:

```text
dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp
  populate IFN metadata, then return before createPcfgsSTCDPOp

dcc/src/Conversion/PCFGToDataflowIR/PCFGToDFManager.cpp
  do not convert metadata-only IFN dataops as standalone empty modules
```

This moved past the previous errors:

```text
old paired IFN: IBUFF overflow
IFN + DL independent modules: unit already set for associated schedule step
DL marker only: empty IFN metadata / out_of_range
metadata-only IFN + DCC skip: reaches DL program verification
```

Runs:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_ifn_metadata_dccskip_clc_20260630_154401
  rc=134
  LX_MODLRFIMM :: lrfimm:-4161536 src0:0

/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_ifn_metadata_dccskip_allocreset_clc_20260630_154546
  rc=134
  DtException: allocNode, dsc/dsc2.cpp line 3999

/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_ifn_metadata_dccskip_alloczero_clc_20260630_154802
  rc=134
  LX_MODLRFIMM :: lrfimm:-4161536 src0:0
```

The `allocreset` run proves the consumer LX allocation node cannot simply be
removed; DDC needs it to recover layout dimensions.  The `alloczero` run proves
that replacing the base address alone is not enough.  The DL address-generation
path still sees a full logical 4 MB neighbor operand and creates an out-of-range
LRF immediate.  The next backend implementation has to make the neighbor
allocation chunk-local, not just LX-local.

## Communication Classifier Unit Validation

The artifact branch now records a frontend communication class for each
producer/consumer LX mismatch:

```text
scatter
broadcast
multicast
gather
all_gather
reduce
all_reduce
unsupported
```

The focused unit test was validated on `adnan-spyre-dev-pf` in a throwaway copy
of the current comms-collectives checkout:

```text
/home/adnan/codex-isolated/lx_classifier_validate_20260630_230533
```

The first run failed before collection with a stale runtime library binding:

```text
ImportError: /opt/ibm/spyre/spyre-comms/lib/libspyre_comms.so.1:
undefined symbol: _ZN4flex19AllocationDirectiveC1ENS_15PlacementPolicyE...
```

The fix is the same library pin used for Granite profiling: put the installed
Spyre Deeptools/runtime/comms libraries first, and include the Torch shared
libraries from the pod venv.

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$DST:$DST/tests/inductor:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

/home/adnan/dt-inductor/.venv/bin/python3 \
  -m pytest tests/inductor/test_lx_relayout_dldsc.py -q
```

Result:

```text
21 passed in 7.76s
```

The test includes the concrete attention sub-stick case:

```text
producer: 32 producer slices over tensor dim out
consumer: {x:16, out:2}
expected communication_class: gather
```
