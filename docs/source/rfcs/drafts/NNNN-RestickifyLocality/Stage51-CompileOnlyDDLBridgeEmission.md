# Stage 51: Compile-Only DDL Bridge Emission

## Summary

Stage 51 fixed the immediate Stage 50 blocker: the default-off DDL bridge now
emits for the known high-signal `adds_then_matmul` 2048 in-graph restickify in a
compile-only run.

This stage still does not prove numerical correctness, because the hardware
kernel launch was intentionally skipped. It does prove that Torch-Spyre can emit
the `_ddl_bridge` SDSC inside a real compiled bundle and that DXP accepts it
with the Stage 49 patched `restickify.ddl` contract.

## What Changed

The previous DDL bridge eligibility gate rejected the mirrored 2048 direction:

```text
reason = output-stick-is-not-split-dim
```

That gate was correct for Stage 42, where the mirrored direction still failed
DCC/DXP register-bound checks. Stage 49 changed the contract by patching the
DDL template source spelling to:

```ddl
ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_input"}
```

With that template, both directions pass DDC/DCC/DXP. Stage 51 therefore removes
the old direction-specific gate and keeps the conservative gates that still
matter:

```text
source_kind == in_graph_computed
one input / one output
no constants, padding, or coordinate masking
exactly one split dimension
split dimension covers all cores
per-core LX contract <= 512 KiB
```

The change remains behind:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1
```

No default behavior changes.

## Compile-Only Method

To avoid another hardware timeout, the probe monkeypatched the Python kernel
runner in-process:

```python
import torch_spyre.execution.kernel_runner as kr

def fake_launch_kernel(code_dir, args):
    print(f"skipped launch_kernel for {code_dir}")

kr.launch_kernel = fake_launch_kernel
```

That lets `torch.compile`:

1. generate the Torch-Spyre wrapper,
2. generate SDSCs,
3. run DXP,
4. instantiate the kernel runner,
5. skip only the final hardware launch.

This gives us real compile artifacts without executing the device program.

## Command Shape

Environment:

```sh
export DEEPTOOLS_PATH=/tmp/stage50-template-share
export SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1
export SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1
export SPYRE_RESTICKIFY_DDL_BRIDGE_AUDIT_JSONL=/tmp/stage51-compile-only-relaxed/audit.jsonl
```

Probe case:

```text
tools/restickify_scenario_probe.py
case = adds_then_matmul
size = 2048
device = spyre
correctness = skipped
hardware launch = monkeypatched no-op
```

## Results

The compile-only run completed:

```text
status = ok
compile_run_ms = 6210.85
restickify_count = 2
total_bytes = 16,777,216
ring_total_byte_hops = 67,108,864
```

Two bundles were generated:

```text
/tmp/torchinductor_1000800000/tmplktehcbx/inductor-spyre/sdsc_fused_add_t_0_mtzd12w4
  sdsc_0_ReStickifyOpHBM.json
  sdsc_1_add.json
  sdsc_2_add.json

/tmp/torchinductor_1000800000/tmplktehcbx/inductor-spyre/sdsc_fused_mm_1_32ttq2er
  sdsc_0_ReStickifyOpHBM_ddl_bridge.json
  sdsc_1_batchmatmul.json
```

The audit confirms the expected split:

```json
{"source_kind":"graph_input_or_weight","source_name":"arg1_1","status":"skipped","reason":"source-not-in-graph-computed"}
{"source_kind":"in_graph_computed","source_name":"buf1","status":"emitted","reason":null}
```

That is exactly the desired v0 behavior: graph-input/weight restickify remains
out of scope, while the in-graph producer-to-restickify edge emits the LX-local
DDL bridge.

## DXP Recheck Of Emitted Bridge

The emitted bridge SDSC was then run through the standalone preload DXP probe:

```text
/tmp/torchinductor_1000800000/tmplktehcbx/inductor-spyre/sdsc_fused_mm_1_32ttq2er/sdsc_0_ReStickifyOpHBM_ddl_bridge.json
```

Result:

```text
DXP rc = 0
pre-DDC schedule nodes = 7
post-DDC schedule nodes = 71
final senprog bytes = 943,126
```

Final `senprog.txt` token summary:

```json
{"HBM":0,"L3LU":0,"L3SU":0,"LXLU":0,"LXSU":32,"PT":8928,"SFP":896}
```

So the actual Torch-Spyre-emitted bridge has the same no-HBM generated-program
shape as the Stage 50 artifact-only bridge.

## Validation

Local static validation:

```text
python3 -m py_compile \
  torch_spyre/_inductor/codegen/restickify_ddl_bridge.py \
  tests/inductor/test_restickify_ddl_bridge.py
```

Pod focused tests:

```text
python3 -m pytest tests/inductor/test_restickify_ddl_bridge.py -q
```

Result:

```text
5 passed in 0.03s
```

## Interpretation

The prototype has now crossed a useful line:

```text
real Torch graph
  -> real Torch-Spyre compile
  -> real in-graph restickify selected
  -> emitted ReStickifyOpHBM_ddl_bridge SDSC
  -> DXP accepts the bundle
  -> standalone DXP of emitted bridge shows no HBM/L3 work tokens
```

The remaining missing proof is runtime semantics.

We still need to prove that the DDL bridge input aliases the previous op's
LX-resident output correctly. That requires a hardware run with correctness
enabled, or a lower-level runtime binding inspection showing that the bridge
input labeled DS is wired to the producer output allocation.

## Next Step

The next experiment should be a minimal e2e correctness run that preserves the
same emitted bridge but reduces unrelated risk:

1. Use the monkeypatch path only to identify a bridge-emitting shape.
2. Try the same shape with real hardware launch and correctness enabled.
3. If it times out, reduce the graph while preserving:
   - `source_kind = in_graph_computed`
   - one split dimension over 32 cores
   - per-core LX contract under 512 KiB
   - emitted `_ddl_bridge` audit row
4. Compare the generated bundle before/after reduction so we do not accidentally
   fall back to a skipped HBM restickify.

Only after numerical correctness passes should we return to kernel timing and
memory-counter comparisons.

