# Stage 67: Tensor-Binding Data-Op Probe

## Summary

Stage 67 tried to turn the Stage 65/66 Deeprt data-op export into a
tensor-bearing runtime proof.

Result:

- Stock DXP-generated `ReStickifyOpHBM` bundles do bind runtime tensor
  arguments and can write real output values.
- The Deeprt-exported standalone data-op bundle launches, but the output tensor
  remains zero even when the data-op LDS records are marked as `input` and
  `output`.
- Deeptools source inspection explains why the earlier LX-only segment tweak did
  not work: DSI only broadcasts runtime LDS segment addresses into data-op LDS
  records when the LDS is HBM-pinned.
- A patched HBM-pinned `STCDPOpHBM` control compiled far enough to export an
  `init.txt`, but still did not bind runtime tensor values through the handmade
  package.

So the current status is:

```text
LX-local data-op compile/export proof: yes
Torch-Spyre/Flex can launch the Deeprt init program: yes
Runtime tensor values wired into standalone Deeprt data-op: not yet
Core-to-core restickify value proof without HBM: not yet
```

This is still useful. It narrows the blocker from "can Deeptools generate an
LX-local restickify program?" to "how do we generate the full runtime DSG/segment
binding metadata for a standalone data-op, or embed the data-op inside a normal
compiled graph where producer and consumer tensors already have real runtime
addresses?"

## What Was Tested

### 1. LX-only `STCDPOpLx` With Runtime Segment Labels

I generated a one-core same-stick `STCDPOpLx` control and patched the input and
output labeled data structures:

```text
input LDS:
  segment_ = "input"
  PlacementInfo = LX

output LDS:
  segment_ = "output"
  PlacementInfo = LX
```

Deeprt compiled and exported the node far enough to emit:

```text
execute/0_STCDPOpLx_dataop/init.txt
execute/0_STCDPOpLx_dataop/sdsc.json
execute/0_STCDPOpLx_dataop/senprog.txt
execute/0_STCDPOpLx_dataop/smc.txt
loadprogram_to_device_dsg.txt
```

The harness then segfaulted during/after export, but the emitted files were
usable for a runtime-shaped package.

Launching with real Spyre tensors succeeded:

```python
launch_kernel("/tmp/stage67-stcdp-value/runtime-shape", [out, inp])
torch.accelerator.synchronize()
```

But the output remained zero:

```json
{"order": "out_in", "nonzero": 0, "maxdiff_vs_input": 4096.0}
{"order": "in_out", "nonzero": 0, "maxdiff_vs_input": 4096.0}
```

### 2. Deeptools Source Check

The key source path is Deeptools `dsi.cpp` in `broadcast_lds`.

For data-op DSCs, runtime segment rebasing is guarded by `isHbmPinned()`:

```cpp
if (myLds.isHbmPinned()) {
  if (myLds.segment_ != ldsProp.at(myLds.ldsName_).ldsLoc) {
    myLds.segment_ = ldsProp.at(myLds.ldsName_).ldsLoc;
  }
  myLds.setHbmStartAddress(ldsProp.at(myLds.ldsName_).startAddr);
  for (auto& kv : myLds.pieces_) {
    auto it = kv.second.placement.find(SenComponents::HBM);
    ...
    it->second.updateStAddr(
        it->second.getStartAddr().at(0) + myLds.getHbmStartAddress(),
        0);
  }
}
```

For an LX-only data-op LDS:

```cpp
bool LdsInfo::isLxPinned() const {
  ...
  if (isLx) {
    return (!isHbmPinned());
  }
  ...
}
```

That means setting `segment_ = "input"` or `segment_ = "output"` on an LX-only
data-op LDS does not make the current DSI path rebase the LX start address to a
runtime tensor address. The generated data-op still reads/writes its own
program-local LX addresses.

### 3. HBM-Pinned Data-Op Control

I also patched a same-stick `STCDPOpLx` payload into an HBM-pinned
`STCDPOpHBM` control:

```text
op.name = STCDPOpHBM
input segment_ = "input"
output segment_ = "output"
PlacementInfo = LX + HBM
hbmSize_ = 64 sticks for a 64x64 fp16 tensor
coreIDtoANInfo = one non-analytical p1 -> p1 movement
```

Deeprt compiled through DCG, DCC, DIP, frame filling, and export:

```text
Running DCG for DataOp: Node-name:0_STCDPOpHBM_dataop
Computing transfer function metaData..
Creating pcfg for coreID:0 : L3SU : L3LU : LX : PE0 ...
Calling DCC
Calling DIP
Calling Program FramePtr Filler
Calling Export...
```

It emitted:

```text
execute/0_STCDPOpHBM_dataop/init.txt
execute/0_STCDPOpHBM_dataop/sdsc.json
execute/0_STCDPOpHBM_dataop/senprog.txt
execute/0_STCDPOpHBM_dataop/smc.txt
```

But the runtime launch still left output zero, both with the no-argument
Stage66-style package and with a more normal Torch-Spyre-shaped package:

```text
bundle.mlir
sdsc_0_STCDPOpHBM.json
segment_size.json
loadprogram_to_device/runtime-real-SenProgSend/init.txt
```

Observed output:

```json
{"order": "out_in", "nonzero": 0, "maxdiff_vs_input": 4096.0}
{"order": "in_out", "nonzero": 0, "maxdiff_vs_input": 4096.0}
```

## Stock Runtime Control

To make sure this was not a misunderstanding of `launch_kernel`, I re-ran a
stock DXP-generated `ReStickifyOpHBM` bundle captured from the
`isolated_transpose_contiguous_128` probe.

The real DXP bundle has the normal Torch-Spyre shape:

```text
bundle.mlir
execute/<kernel>/pagi.json
execute_dsg.txt
loadmodel_to_device_dsg.txt
loadmodel_to_spad_dsg.txt
loadprogram_to_device/<kernel>-SenProgSend/init.txt
loadprogram_to_device_dsg.txt
loadprogram_to_spad_dsg.txt
sdsc_0_ReStickifyOpHBM.json
segment_size.json
```

Launching it with the input tensor first and output tensor second produced real
values:

```json
{
  "order": "in_out",
  "nonzero": 16383,
  "maxdiff_transpose": 8.0,
  "first8": [0.0, 128.0, 256.0, 384.0, 512.0, 640.0, 768.0, 896.0]
}
```

The small `maxdiff_transpose` is not important for this stage; the control proves
that a normal DXP/Torch-Spyre bundle can bind runtime tensor arguments and write
visible output values. The failure is specific to our handmade Deeprt data-op
runtime package.

## Interpretation

The Deeprt data-op export currently gives us a real program binary, but not the
full runtime binding contract that DXP/Torch-Spyre normally creates around a
compiled graph.

For external runtime tensors, the normal path needs more than:

```text
init.txt + bundle.mlir + segment_size.json
```

It also needs the correctly generated DSG/segment/DCI metadata that tells Flex
how tensor arguments map to LDS names and how to patch or interpret those
addresses for the program.

For LX-only data-op movement, there is an additional conceptual issue: external
PyTorch tensors do not naturally arrive as "already resident in this exact LX
address." They enter the compiled graph as runtime tensor allocations. To prove
true LX-to-LX restickify with values, we probably need an in-graph producer that
writes the LX buffer, then the data-op restickify reads that LX buffer, then an
in-graph consumer or store makes the result visible.

## Conclusion

Stage 67 does not prove value-correct core-to-core restickify yet.

It does prove three narrower facts:

1. The Stage65/66 Deeprt data-op program launch was a real loader/execution
   smoke, but not a tensor-value proof.
2. LX-only `segment_` patching cannot bind external runtime tensors by itself,
   because Deeptools only rebases data-op runtime segment addresses for
   HBM-pinned LDS records.
3. Stock DXP bundles do bind runtime tensor arguments correctly, so the next
   blocker is the data-op packaging/integration path, not `launch_kernel`
   itself.

## Recommended Next Step

The next useful experiment is one of:

1. Build a fuller Deeprt DSG harness that supplies the same LDS/DCI/segment
   metadata DXP has for `input` and `output`, then retry the HBM-pinned
   `STCDPOpHBM` value-control.
2. Stop trying to make an external LX-only data-op read PyTorch tensors directly
   and instead embed the data-op into a real compiled graph:
   producer compute op -> LX-local data-op restickify -> consumer/store.

Option 2 is closer to the real Stage 3B goal, because Stage 3B is about
in-graph producer-to-restickify locality, not graph-input tensor loading.

