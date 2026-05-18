# Stage 52: DDL Bridge Runtime Blocker

## Summary

Stage 52 pushed the DDL bridge from compile-only validation into real hardware
execution. The result is a useful blocker:

```text
baseline HBM restickify at size 1280 runs
DDL bridge restickify at size 1280 times out during program/artifact load
```

So the bridge is no longer blocked by DDC/DCC/DXP. It is blocked by the runtime
contract for loading/executing a bundle that contains the DDL-shaped LX-local
restickify SDSC.

## Prototype Fixes Applied

Two cleanup fixes were made before the hardware retry.

### Do not force `target_ = "senulator"`

The Stage 50/51 bridge inherited an old fixture detail and set:

```json
"target_": "senulator"
```

on the root and DSC. A patched bridge with no `target_` still passed standalone
DXP and still had no HBM/L3 work tokens:

```json
{"HBM":0,"L3LU":0,"L3SU":0,"LXLU":0,"LXSU":20,"PT":5780,"SFP":560}
```

The generator now preserves the normal Torch-Spyre hardware target behavior by
not forcing `senulator`.

### Preserve Original Labeled DS Roles

The first bridge prototype forced:

```text
INPUT -> OUTPUT
```

For the real matmul-side restickify, the source SDSC is:

```text
OUTPUT -> KERNEL
```

because the restickified value feeds the matmul kernel operand. The generator now
preserves the original labeled DS roles and primary DS keys. The emitted 1280
bridge now looks like:

```text
primaryDsInfo = ["OUTPUT", "KERNEL"]
labeledDs = [("Tensor0", "OUTPUT", ["lx"]), ("Tensor1", "KERNEL", ["lx"])]
target = None
numCores = 20
work_slices = {"mb": 20, "out": 1}
```

Standalone DXP accepts both the `INPUT -> KERNEL` and `OUTPUT -> KERNEL`
variants.

## Compile-Only Size Sweep

We reran `adds_then_matmul` with hardware launch monkeypatched out and captured
which sizes emit the DDL bridge:

| Size | Bridge emitted? | Skip reasons | Ring byte-hops |
|---:|---:|---|---:|
| 768 | no | `source-not-in-graph-computed`, `expected-one-split-dim` | 6,930,432 |
| 1024 | no | `source-not-in-graph-computed`, `expected-one-split-dim` | 11,141,120 |
| 1280 | yes | `source-not-in-graph-computed` | 26,214,400 |
| 1536 | yes | `source-not-in-graph-computed` | 37,748,736 |
| 1792 | yes | `source-not-in-graph-computed` | 51,380,224 |
| 2048 | yes | `source-not-in-graph-computed` | 67,108,864 |

The smallest bridge-emitting size in this family is `1280`.

## Runtime Comparison

### Baseline

Command shape:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=0 \
python3 tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 1280 \
  --ring-telemetry \
  --skip-correctness \
  --output-dir /tmp/stage52-e2e-baseline-1280 \
  --fail-on-error
```

Result:

```text
ok size=1280 case=adds_then_matmul restickifies=2 bytes=6,553,600 byte_hops=26,214,400
```

The baseline mm bundle has no DDL bridge and loads/runs:

```text
sdsc_0_ReStickifyOpHBM.json
sdsc_1_batchmatmul.json
segment output size = 25,600
```

### DDL Bridge

Command shape:

```sh
DEEPTOOLS_PATH=/tmp/stage50-template-share \
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1 \
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1 \
python3 tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 1280 \
  --ring-telemetry \
  --output-dir /tmp/stage52-e2e-correctness-1280-roles \
  --fail-on-error
```

Result:

```text
timeout after 120s
```

A launch wrapper confirmed the first add bundle launches, and the timeout happens
while launching/loading the mm bundle:

```text
[stage52] launching .../sdsc_fused_add_t_0_g2oznjte
  sdsc_0_ReStickifyOpHBM.json
  sdsc_1_add.json
  sdsc_2_add.json

[stage52] launching .../sdsc_fused_mm_1_qc3lprln
  sdsc_0_ReStickifyOpHBM_ddl_bridge.json
  sdsc_1_batchmatmul.json
```

The stack is consistently in artifact/program loading:

```text
flex::RuntimeScheduler::schedule(RuntimeOperationH2DE)
spyre::SpyreStream::copyAsyncImpl
spyre::getOrLoadArtifacts
spyre::launchKernel
```

This is before we can observe numerical correctness.

## Segment Difference

The generated bridge bundle differs from the baseline bundle in a notable way:

```text
baseline mm bundle:
  output segment size = 25,600
  input/model/stack = 25,600

bridge mm bundle:
  output segment size = 0
  input/model/stack = 25,600
```

Preserving `OUTPUT -> KERNEL` roles did not change this. That suggests the zero
output segment is caused by the LX-only bridge contract and bundle-level memory
planning, not just by a mislabeled DS role.

This may or may not be the direct cause of the load hang, but it is the most
concrete structural difference between the runnable HBM bundle and the hanging
DDL bridge bundle.

## Validation

Static:

```text
python3 -m py_compile \
  torch_spyre/_inductor/codegen/restickify_ddl_bridge.py \
  tests/inductor/test_restickify_ddl_bridge.py
```

Pod:

```text
python3 -m pytest tests/inductor/test_restickify_ddl_bridge.py -q
```

Result:

```text
6 passed in 0.03s
```

Local copied artifacts:

```text
artifacts/stage52_e2e_runtime_blocker/compile_only_size_sweep_summary.json
artifacts/stage52_e2e_runtime_blocker/preserve_roles_compile_row.json
artifacts/stage52_e2e_runtime_blocker/preserve_roles_compile_audit.jsonl
artifacts/stage52_e2e_runtime_blocker/baseline_1280_row.jsonl
artifacts/stage52_e2e_runtime_blocker/bridge_1280_timeout_audit.jsonl
```

## Interpretation

We have now proven:

```text
DDL bridge can be emitted inside a real Torch-Spyre bundle.
DDL bridge passes DXP inside that bundle.
Standalone DXP for the emitted bridge has no HBM/L3 work tokens.
The normal HBM restickify version of the same size runs.
The DDL bridge version times out while loading the mm bundle.
```

We have not proven:

```text
the DDL bridge reads the producer's LX-resident values correctly
the DDL bridge can execute on hardware
the DDL bridge improves runtime
```

## Next Blocker

The next task is not more shape sweeping. The next task is to understand the
Torch-Spyre/Flex runtime contract for an SDSC whose input/output labeled DS are
LX-only and intended to be consumed by the next SDSC in the same bundle.

Concrete questions:

1. Why does the bridge bundle have `output` segment size `0` while the baseline
   HBM restickify bundle has `25,600`?
2. Does `SpyreSDSCKernelRunner` or Flex require every launched bundle to have a
   nonzero output segment, even if the first SDSC output is only an intermediate?
3. Is there a supported way for an SDSC output to be LX-only and consumed by a
   later SDSC in the same bundle?
4. Do we need an explicit bundle-level `ldsShareInfo_`, `prodConsList`, or
   data-stage relation so the bridge output aliases the matmul kernel input?

Until those questions are answered, the DDL bridge remains a compile-time
prototype, not a runnable optimization.

