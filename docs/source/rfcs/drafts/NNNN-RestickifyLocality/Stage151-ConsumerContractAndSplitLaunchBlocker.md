# Stage 151: Consumer Contract And Split-Launch Blocker

## Summary

This stage changed the current LX-to-LX restickify goal from "fix one consumer
LX metadata field" to "package the replacement inside the normal fused runtime
artifact."  The decisive result is that launching the consumer SDSC as a
standalone `launch_kernel` bundle is not a valid model for this case, even when
the consumer SDSC is left in its original HBM-backed form.

## What We Tested

The target case remained:

```text
computed_transpose_adds_then_matmul_tuple, size=2048
```

The known good compile/packaging path still works in prepare-only mode when the
schema-v4 descriptor is emitted:

```text
restickifies=1
bytes=8,388,608
byte_hops=0
dataop_contract_source=schema-v4-lx-materialization-contract
dataop_export_returncode=0
```

We then isolated the consumer side with `SPYRE_RESTICKIFY_LX_SPLIT_STAGES=consumer`.

## Findings

### 1. Constant low LX addresses were not the only issue

The original split consumer prototype placed the bridge input at LX base `8192`.
That could overlap ordinary DXP-created LX allocations, so we also tested the
higher base `1572864`, which came from the data-op seed before patching.  The
consumer still emitted `RAS::RUNTIMESCHEDULER::ComputeHardwareError`.

### 2. INPUT role metadata was not enough

Forcing the bridge input to `dsType_="INPUT"` and copying
`primaryDsInfo_["OUTPUT"]` to `primaryDsInfo_["INPUT"]` still emitted the same
compute hardware error.

### 3. DDL-like LX metadata was not enough

Deeptools DDL templates represent LX-only tensors with fields such as:

```text
segment_="stack"
isExternal_=0
dataTransfers_=[]
hbmStartAddress_=-1
hbmSize_=UINT64_MAX
lxStartAddress_=-1
lxBufferSize_=UINT64_MAX
```

Adding a diagnostic DDL-like input mode to the consumer patch did not remove the
compute hardware error.  The generated schedule still contained an external
LX local transfer:

```text
transfer_lds1_src:no_component_dst:lx_lx_local
```

That transfer shape exists in Deeptools templates, so it is not automatically
invalid by itself.

### 4. Keeping the removed bridge argument was not enough

We tested passing the bridge tensor argument to the split consumer launch even
after marking the SDSC input LX-only:

```text
consumer_arg_indices=[2,4,5]
```

The consumer still emitted the compute hardware error.

### 5. Standalone consumer launch is the real blocker

The decisive test launched the original, unpatched consumer SDSC by itself:

```text
SPYRE_RESTICKIFY_LX_SPLIT_STAGES=consumer
SPYRE_RESTICKIFY_LX_SPLIT_SKIP_CONSUMER_PATCH=1
```

This triggered a PCIe bus fence:

```text
RAS::PCI::BusFence
```

That means the split-launch harness is not a valid runtime model for this fused
producer/restickify/consumer bundle.  The consumer frame appears to depend on
the normal fused artifact context, frame ordering, or runtime setup.  Therefore,
consumer-only `launch_kernel` results cannot be used to validate the LX input
contract.

## Code Changes

The data-op payload patcher no longer injects the diagnostic
`addressPreservingProbe_` object into generated SDSCs.  That debug-only field
made DeeRT export order-sensitive in prior runs; the same information remains
available through `summary.json`.

Two diagnostic flags were added to the probe harness:

```text
SPYRE_RESTICKIFY_LX_SPLIT_KEEP_BRIDGE_ARG=1
SPYRE_RESTICKIFY_LX_SPLIT_SKIP_CONSUMER_PATCH=1
```

One diagnostic metadata mode was added:

```text
SPYRE_RESTICKIFY_LX_SPLIT_DDL_LIKE_INPUT=1
SPYRE_RESTICKIFY_LX_SPLIT_DDL_LX_SIZE=<bytes>
```

These are probe-only flags and should not be treated as production API.

## Recommendation

Stop using separate producer/data-op/consumer launches for hardware validation.
The next prototype should operate on the normal fused runtime artifact:

1. Generate the LX-to-LX data-op / InputFetchNeighbor frame.
2. Replace or splice the restickify frame inside the original fused artifact.
3. Keep producer and consumer execution in the original fused artifact ordering.
4. Only then test whether the consumer reads the materialized LX view without a
   `ReStickifyOpHBM` frame and without an HBM reload for that logical edge.

In short: the goal is still LX-to-LX restickify, but the route should be
same-artifact packaging, not split launches.
