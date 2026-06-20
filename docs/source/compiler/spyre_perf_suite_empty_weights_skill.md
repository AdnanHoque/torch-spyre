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
module.eval()
```

Keep the `to(dtype=torch.float16).to_empty(device=torch.device("spyre"))` order.
The dtype cast is still needed, but `to_empty` avoids the host-to-device
parameter copy.

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
