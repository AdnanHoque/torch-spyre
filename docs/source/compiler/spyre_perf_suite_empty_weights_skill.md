# Spyre Perf-Suite Empty Weights Skill

Use this runbook when benchmarking parameterized FMS modules in
`spyre-perf-suite` for performance, profiler, or SDSC investigation where real
checkpoint values are irrelevant.

## Problem

The standard `fms_granite_micro.swiglu` perf-suite path constructs random CPU
`nn.Parameter` weights and then moves the module to Spyre. For Granite 3 8B
SwiGLU this copies about 629 MB of parameters from host to device before
Inductor artifacts exist. That copy can stall in flex DMA and make the run look
like a compiler hang.

For microbenchmarks, the weights only need correct shapes, dtypes, and device
placement. They do not need initialized values.

## Preferred Pattern

Build the FMS module normally, cast it on CPU, then materialize uninitialized
parameters directly on AIU:

```python
module = _make_swiglu_module(config, torch, fused_weights=True)
module = module.to(dtype=torch.float16)
module = module.to_empty(device=torch.device("spyre"))
module.requires_grad_(False)
module.eval()
```

Keep the `to(dtype=torch.float16).to_empty(device=torch.device("spyre"))` order.
The dtype cast is still needed, but `to_empty` avoids the host-to-device
parameter copy.  Freeze parameters for benchmark-only runs so AOTAutograd does
not trace a useless backward graph through the Spyre PrivateUse1 hooks path.

## Existing Op-File Wrapper

Use the existing wrapper when running FMS SwiGLU through perf-suite:

```bash
cd /tmp/torch-spyre-co-remap-native

$PY212 /home/adnan-cdx/spyre-perf-suite/benchmark.py \
  --stack torch-spyre \
  --op fms_swiglu_empty \
  --op-file /tmp/torch-spyre-co-remap-native/tools/perf_suite_fms_swiglu_empty_params_op.py \
  --shape 1 512 4096 \
  --runs 5 \
  --without-compilation \
  --with-profiling \
  --output /tmp/fms_swiglu_empty_perf.txt
```

The same op-file also supports the unfused path:

```bash
--op fms_swiglu_unfused_empty
```

By default the wrapper uses `to_empty` when perf-suite invokes the op-file with
`stack == "tsp"` for the torch-spyre path. Set
`SPYRE_FMS_SWIGLU_TO_EMPTY=0` only when deliberately reproducing the standard
CPU-parameter transfer behavior.

## Granite Block Wrapper

For a full single-block prefill probe, use the Granite block wrapper.  It
constructs one FMS Granite block, casts it to fp16, and materializes parameters
with `to_empty(device="spyre")`.

```bash
cd /tmp/torch-spyre-co-remap-native

$PY212 tools/run_coordinate_remap_bench.py \
  --output-root /tmp/granite_block_coordinate_remap_profile \
  --torch-root /tmp/torch-spyre-co-remap-native \
  --deeptools-root /tmp/deeptools-coordinate-remap-mainport-lean \
  --perf-suite-root /home/adnan-cdx/spyre-perf-suite \
  --variant branch-baseline \
  --variant coordinate-remap \
  --op fms_granite_block_empty \
  --op-file /tmp/torch-spyre-co-remap-native/tools/perf_suite_fms_granite_block_empty_params_op.py \
  --shape 1 512 4096 \
  --runs 5 \
  --env LD_LIBRARY_PATH=/home/adnan-cdx/dt-inductor-codex-clean/install/libaiupti/lib:/home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib:${LD_LIBRARY_PATH} \
  --env SPYRE_FMS_GRANITE_BLOCK_SCOPE=full \
  --env SPYRE_FMS_GRANITE_BLOCK_ATTN_NAME=sdpa_bidirectional
```

`SPYRE_FMS_GRANITE_BLOCK_SCOPE` can be set to `mlp` or `attention` to isolate
the block's feed-forward or attention submodules without the normalization
prefix.  Use `mlp_with_norm` or `attention_with_norm` when debugging norm
lowering as part of the block.  The default attention mode is
`sdpa_bidirectional`, matching the existing FMS Granite attention microbench
choice to avoid the currently problematic causal-mask lowering.

Each run writes `artifacts/onchip_move_edge_report.md` and `.csv`.  Use those
files to determine whether coordinate remap fired only in the SwiGLU/MLP path
or also on attention/residual/norm edges.

## Standalone GraniteBlock Probe

When perf-suite is too heavy for blocker isolation, use the standalone
`benchmarks/granite_block_probe.py` probe.  It is derived from the Granite
cost-model probe and defaults to fake/empty Spyre weights.

```bash
cd /tmp/torch-spyre-co-remap-native
DEE=/tmp/deeptools-coordinate-remap-mainport-lean

export PYTHONPATH=/tmp/torch-spyre-co-remap-native:/tmp/torch-spyre-co-remap-native/tests/inductor:${PYTHONPATH:-}
export PATH="$DEE/build-swiglu-dxp-main-lean/dxp:$DEE/build/dxp:${PATH}"
export LD_LIBRARY_PATH=/home/adnan-cdx/dt-inductor-codex-clean/install/libaiupti/lib:/home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib:${LD_LIBRARY_PATH:-}

$PY212 benchmarks/granite_block_probe.py \
  --part mlp_core \
  --regime prefill \
  --fused-weights \
  --iters 1
```

Use `mlp_core` for the FFN/SwiGLU core, `mlp_norm` to isolate the norm prefix,
`mlp_residual` to isolate residual scale/add, and `mlp` for the full MLP path.
Coordinate-remap runs also need:

```bash
export SPYRE_ONCHIP_MOVE_PLANNER=1
export SPYRE_ONCHIP_MOVE_REALIZE=1
export SPYRE_ONCHIP_MOVE_CARRIER=coordinate_remap
export SPYRE_ONCHIP_MOVE_JSONL=/tmp/granite_probe/onchip_move.jsonl
export SPYRE_ONCHIP_MOVE_DEBUG_DIR=/tmp/granite_probe/onchip_move_debug
```

Do not prepend the lean Deeptools build libraries to `LD_LIBRARY_PATH` for this
probe on the current pod; doing so can make the profiler `_C.so` overlay fail
to load with a Flex symbol mismatch.

## When To Use

Use empty weights for kernel timing, SDSC structure checks, coordinate-remap
experiments, profiler traces, and other shape-only microbenchmarks.

Do not use empty weights when validating numerical correctness, checkpoint
loading, serialization, or behavior that depends on actual parameter values.

## Failure Signature

If a parameterized FMS perf-suite benchmark hangs before Inductor artifacts,
check whether it is using `module.to("spyre")` on CPU-created parameters. A hang
or long pause in flex DMA before compilation is a strong signal to switch to the
empty-weight wrapper.
