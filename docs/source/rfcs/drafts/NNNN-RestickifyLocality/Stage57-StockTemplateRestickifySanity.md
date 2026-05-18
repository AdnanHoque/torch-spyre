# Stage 57: Stock Template ReStickify Sanity Check

## Summary

Stage 57 reran the minimal Stage 55 restickify reproducer with the stock
Deeptools template path:

```sh
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
```

The same source pattern:

```python
def fn(a):
    return a.t().contiguous()
```

at size `128` still emits a single `ReStickifyOpHBM` SDSC, but it now launches
and synchronizes successfully. This changes the interpretation of Stage 55: the
timeout was caused by the experimental `/tmp/stage50-template-share` template
directory, not by stock Torch-Spyre or stock Deeptools restickify execution.

## Command

```sh
timeout 120s python3 tools/restickify_scenario_probe.py \
  --case isolated_transpose_contiguous \
  --size 128 \
  --ring-telemetry \
  --skip-correctness \
  --sync-after-kernel \
  --kernel-launch-log \
  --output-dir /tmp/stage57-stock-isolated-contiguous-128 \
  --fail-on-error
```

## Result

The launch log has the full launch/sync sequence:

```json
{"phase":"before_launch","kernel_name":"sdsc_fused_clone_t_0","sdsc_files":["sdsc_0_ReStickifyOpHBM.json"]}
{"phase":"after_launch","kernel_name":"sdsc_fused_clone_t_0","sdsc_files":["sdsc_0_ReStickifyOpHBM.json"]}
{"phase":"before_sync","kernel_name":"sdsc_fused_clone_t_0","sdsc_files":["sdsc_0_ReStickifyOpHBM.json"]}
{"phase":"after_sync","kernel_name":"sdsc_fused_clone_t_0","sdsc_files":["sdsc_0_ReStickifyOpHBM.json"]}
```

The generated code directory contains:

```text
bundle.mlir
sdsc_0_ReStickifyOpHBM.json
segment_size.json
execute/
loadprogram_to_device/
```

The generated SDSC has the expected minimal HBM restickify shape:

```text
numCoresUsed_: 4
numWkSlicesPerDim_: mb:2, out:2
coreIdToWkSlice_: 0:(mb0,out0), 1:(mb1,out0), 2:(mb0,out1), 3:(mb1,out1)
input layout:  mb, out   stick_dim=out
output layout: out, mb   stick_dim=mb
computeOp: ReStickifyOpHBM on sfp
labeled DS memOrg: hbm and lx present for both input and output
```

The probe-level ring telemetry reports zero rows because this is a standalone
graph-input materialization. It has no in-graph producer edge for the Stage 3B
byte-hop estimator to attribute.

## Template Difference

The failing Stage 55 run used `/tmp/stage50-template-share`. Compared with the
stock template, that directory changed one line in `restickify.ddl`:

```diff
-%src_inp_lxsfp = ddl.unit(%inptensor, %inptensor_lx_allocation) {unit="lxlu", data_connect="lxlu_input"}
+%src_inp_lxsfp = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_input"}
```

That mutation was part of the earlier DDL bridge experiment. It is not safe to
use for runtime validation of stock restickify behavior.

## Interpretation

Stock `ReStickifyOpHBM` can retire on the pod. The next measurements should use
the stock template path unless the experiment is explicitly testing a Deeptools
template change.

This also narrows the current restickify question:

- Stock HBM restickify is runnable.
- The SDSC and lowered code still show that the operation is not simply a direct
  HBM copy; the labeled data spaces are present in both HBM and LX, and Deeptools
  lowering involves HBM/LX plus SFP/L0/PT/PE work.
- Stage 3B remains about changing core ownership for eligible in-graph
  producer-to-restickify edges. It should be evaluated on stock templates before
  drawing runtime conclusions.

## Next Step

Rerun the high-signal `adds_then_matmul` size `2048` guardrail with stock
templates:

1. baseline flags off;
2. Stage 3B flags on;
3. kernel launch logging enabled;
4. optional timing once both modes retire cleanly.

The acceptance check is unchanged: restickify count and bytes moved should stay
the same, while eligible in-graph byte-hops should drop to zero or near-zero
under Stage 3B.
