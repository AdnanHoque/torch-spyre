# Stage 53: Bundle Runtime Segment Contract

## Summary

Stage 53 investigated why the Stage 52 DDL bridge bundle compiled but timed out
on hardware. The main finding is that the Torch-Spyre/Flex runtime path does not
use semantic SDSC role names to bind live tensors. It binds runtime tensors by
argument order to fixed graph-free xlat segments.

That means an LX-only DDL bridge must preserve the original runtime segment for
each restickify argument. Hardcoding both DDL bridge labeled data streams to the
`stack` segment is wrong for real fused bundles.

## Runtime Contract

Torch-Spyre launches a compiled SDSC bundle with the Python tensor arguments:

```cpp
std::vector<const flex::CompositeAddress*> tensor_allocs;
for (size_t i = 0; i < args.size(); ++i) {
  auto* ctx = static_cast<SharedOwnerCtx*>(
      args[i].storage().data_ptr().get_context());
  tensor_allocs.push_back(&ctx->composite_addr);
}

flex::RuntimeOperationCompute compute_op(
    &ctx->composite_addr, std::move(tensor_allocs), arts.bundle_mlir_path);
```

In Flex PF runtime, those tensor addresses are passed to graph-free compute in
the same order:

```cpp
for(size_t i = 0; i < num_tensors; ++i) {
  tensor_paddrs.push_back(getDmvaAddress(inp_out_allocs[i]));
  tensor_sizes.push_back(SEGMENT_SIZE);
}

cbs->CreateGraphFreeCompute(compute_name, tensor_paddrs, tensor_sizes,
                            prog_paddr, prog_size);
```

The Flex control-block comments state the same contract: tensor paddrs are
inputs first, output last, and each element index maps directly to its segment.

For the 1280 `adds_then_matmul` mm bundle, the generated wrapper launches:

```python
sdsc_fused_mm_1.run(buf1, arg3_1, buf4, buf2)
```

and the restickify op is:

```text
input:  arg_index=0  -> buf1
output: arg_index=2  -> buf4
matmul input uses arg_index=2
```

So the DDL bridge source must use runtime segment 0 (`output`) and the DDL bridge
destination must use runtime segment 2 (`model`). The old bridge emitted both as
`stack`, which pointed both source and destination at segment 3.

## Code Change

The bridge now derives the runtime segment from `SDSCArg.start_address`.
`superdsc.py` already sets that address from `SEGMENT_OFFSETS[arg_index]`, so the
bridge can recover the segment that Flex will bind at launch:

```text
SEGMENT_OFFSETS[0] -> output
SEGMENT_OFFSETS[1] -> input
SEGMENT_OFFSETS[2] -> model
SEGMENT_OFFSETS[3] -> stack
...
```

If a restickify argument does not map to one of these runtime segments, the
default-off DDL bridge now skips with:

```text
unsupported-runtime-segment
```

This keeps the prototype conservative.

## Artifact Probe

I copied the Stage 52 generated mm bundle and patched only the DDL bridge labeled
DS segments to match the runtime arg-index contract:

```text
Tensor0 OUTPUT -> segment output
Tensor1 KERNEL -> segment model
```

DXP still compiles the mixed bundle:

```text
dxp_standalone --bundle -d /tmp/stage53-ddl-segment-variants/arg-index-0-2
rc=0
```

The DDL bridge restickify senprog remains HBM-free:

```text
HBM=0, L3LU=0, L3SU=0, LXLU=0, LXSU=20, SFP=560, PT=5780
```

The full mixed bundle still reports the expected segment table shape for the
1280 case:

```text
input: 25600
model: 25600
stack: 25600
const: 124
```

## Hardware Smoke

After applying the runtime-segment fix in Torch-Spyre, the generated DDL bridge
for `adds_then_matmul`, size 1280, emitted:

```text
Tensor0 OUTPUT segment=output memOrg=lx
Tensor1 KERNEL segment=model  memOrg=lx
```

The DDL audit row shows the in-graph restickify was emitted:

```json
{"source_kind":"in_graph_computed","source_name":"buf1","status":"emitted"}
```

However, the hardware smoke still timed out. The stack was again in
`PfRuntimeScheduler::issueBarrier` while loading the next program. A baseline
run in the same current pod state also timed out in the same barrier path, while
a tiny Spyre tensor transfer still worked. So this specific hardware smoke is
not clean evidence against the segment fix.

## Current Conclusion

Stage 53 fixed a real bug in the prototype DDL bridge contract: the bridge must
bind LX-only labeled data streams to the original runtime argument segments, not
to a hardcoded `stack` segment.

What is proven:

- Torch-Spyre/Flex graph-free runtime binds tensor addresses by runtime argument
  index.
- The DDL bridge can preserve that binding using existing `SEGMENT_OFFSETS`.
- The corrected bridge still compiles through DXP.
- The corrected restickify senprog remains HBM/L3-free.

What is not proven yet:

- The corrected bridge executes successfully on hardware.
- The bridge reads the producer's LX-resident values correctly.
- The bridge improves runtime.

## Next Step

Do not broaden the compiler change yet. The next validation should isolate the
runtime issue:

1. Run the known-good baseline 1280 case from a clean pod/runtime state.
2. Add a launch wrapper that synchronizes after the add bundle and before the mm
   bundle, so we can tell whether the add bundle or mm program load is the
   actual blocker.
3. Build a smaller hardware fixture that launches only the mm bundle shape with
   explicit input/temp/output tensors, avoiding the preceding add bundle.
4. Only if that fixture passes should we rerun the full DDL bridge probe and
   compare timing.

