# Stage 143: Mixed Graph Export Retry

## Summary

This stage retried the mixed Deeprt export path without launching generated
kernels. The goal was to export:

```text
producer add compute SDSC
  -> scheduled LX data-op restickify SDSC
  -> consumer add compute SDSC
```

using the full-bundle consumer endpoint evidence from Stage 142.

Result: still not a working export path. The failure is in Deeptools graph
composition/scheduling, before any hardware execution.

## Inputs

Producer and consumer came from the Stage 142 full-bundle compile-only artifact:

```text
/tmp/stage142-full-bundle-consumer-lx-sweep/full_consumer_lx_output_no_corestate/sdsc_0_add.json
/tmp/stage142-full-bundle-consumer-lx-sweep/full_consumer_lx_output_no_corestate/sdsc_2_add.json
```

The restickify data-op was the scheduled Stage3B two-step artifact:

```text
sdsc_stage3b_TwoStepReStickifyLxStcdp_2048_scheduled.json
```

The mixed-graph probe was rebuilt with optional edge-index arguments so we
could test whether the internal restickify edge could connect to the consumer's
second input slot.

## Results

### Consumer input slot 1

Command shape:

```text
producer:0 -> restickify:0
restickify:0 -> consumer:1
```

Result:

```text
SIGSEGV in sengraph::Graph::insertEdge(...)
```

Backtrace:

```text
sengraph::Graph::insertEdge(...)
sengraph::Graph::insertCtrlEdge(...)
populate_chain(...)
main(...)
```

This suggests the prepared-op graph port index is not the same thing as the
consumer LDS index. In other words, `consumer_input_index=1` is not a valid way
to express "connect to `sdsc_2_add` LDS 1" through this `DscSenGraph` path.

### Consumer input slot 0

Command shape:

```text
producer:0 -> restickify:0
restickify:0 -> consumer:0
```

This gets past graph construction and reaches the older Stage 76 failure:

```text
producer=0_add dldscs=1 dataops=0
restickify=0_TwoStepReStickifyLxStcdp_stage3b_dataop dldscs=0 dataops=2
consumer=2_add dldscs=1 dataops=0
graph_nodes=3
edges=producer:0->restickify:0, restickify:0->consumer:0
[DeepRT] ===== Calling Vertical Compilation for node: 0_add =====
...
terminate called after throwing an instance of 'std::out_of_range'
what(): unordered_map::at
```

So the old mixed-graph blocker remains: already-prepared Torch-Spyre compute
SDSCs plus a Deeptools data-op SDSC do not become a valid shared-memory Deeprt
graph simply by inserting them into a `DscSenGraph` chain.

## Interpretation

Stage 143 closes off one tempting shortcut:

```text
Use Deeprt graph edges to connect the data-op output directly to consumer LDS 1
```

That is not currently expressed by the simple graph API in this probe. The graph
edge port indices are not the prepared SDSC's internal `labeledDs_` indices, and
using `1` crashes during graph construction.

The viable paths now look like:

1. **DXP/fused-bundle path:** keep the normal Torch-Spyre bundle ABI and make the
   producer, LX restickify movement, and consumer endpoint agree inside the
   existing bundle/codegen path.
2. **Compound SDSC path:** generate a single bridge-aware SDSC or DLDSc where
   DDC/DCC see the producer output, restickify movement, and consumer input as
   one internal schedule.
3. **Deeptools API path:** find the real API that maps graph edges to a prepared
   SDSC's internal LDS role, if such an API exists. The naive `insertDataEdge`
   port index is not it.

## Device Safety

No `launch_kernel` call was made in this stage. The failures happened in C++
compiler/export probes while building or scheduling a Deeptools graph.

## Artifacts

Pod:

```text
/tmp/stage143-mixed-graph-input
/tmp/stage143-mixed-0_0_0_0.stdout
/tmp/stage143-mixed-0_0_0_0.stderr
/tmp/stage143-mixed-0_0_0_1.stdout
/tmp/stage143-mixed-0_0_0_1.stderr
/tmp/stage143-mixed-graph-gdb.stdout
```

Local copy:

```text
artifacts/stage143_mixed_graph_export_retry/
```
