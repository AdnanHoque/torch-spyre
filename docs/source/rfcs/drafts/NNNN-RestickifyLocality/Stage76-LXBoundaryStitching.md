# Stage 76: LX Boundary Stitching Probe

## Summary

Stage 76 tried to move the LX-local restickify work from isolated program
artifacts toward a real producer -> restickify -> consumer prototype.

The result is not a working correctness prototype yet, but it materially narrows
the blocker:

```text
producer compute SDSC -> LX-local restickify -> consumer compute SDSC
```

cannot be made correct by replacing only the restickify SDSC. The consumer SDSC
is still compiled as an independent HBM-boundary op and will load its restickify
input from HBM unless we also change the consumer boundary contract.

## Deeprt Probe Update

`tools/deeprt_dataop_export_probe.cpp` now initializes Deeprt's memory tracker
before calling the vertical scheduler/codegen/export pipeline. The installed
Deeptools SDK exposes `DeepRt::memTrackers` but does not install
`sharedtools/mem_track_bundle.h`, so the probe uses a narrow ABI declaration for
`MemTrackBundle::initializeMemoryTrackers`.

With that change, a normal compute SDSC such as `sdsc_2_add.json` can pass the
one-node Deeprt path and emit runtime-style artifacts:

```text
execute/2_add/init.txt
execute/2_add/sdsc.json
execute/2_add/senprog.txt
execute/2_add/smc.txt
```

This fixed the previous `map::at` scheduler failure for one-node compute SDSCs.

## Mixed Graph Attempt

I added `tools/deeprt_mixed_graph_export_probe.cpp`, a throwaway harness that
constructs a three-node `DscSenGraph`:

```text
0_add -> 0_TwoStepReStickifyLxStcdp_stage3b_dataop -> 2_add
```

The harness successfully compiled/codegenned:

1. the producer `add` SDSC;
2. the Stage 74 address-preserving LX data-op restickify SDSC.

It then failed while scheduling the consumer `add` SDSC:

```text
terminate called after throwing an instance of 'std::out_of_range'
what(): unordered_map::at
```

A `catch throw` backtrace shows the throw in shared memory tracking:

```text
DsTrackInMem::removeDs(...)
L3DlOpsScheduler::allocAllMem(...)
L3DlOpsScheduler::setChunkDataStageParams(SuperDsc&)
DeepRt::runDlOpsScheduler(...)
```

Namespacing the repeated `Tensor0`/`Tensor1`/`Tensor2` labels did not fix it.
The likely interpretation is that already-prepared Torch-Spyre SDSCs are not a
valid Deeprt shared-memory graph just by putting them in one `DscSenGraph`; the
normal Torch-Spyre/DXP bundle path treats each SDSC as a boundary op and stitches
them through runtime/segment metadata.

## DDL Bridge Boundary Finding

The DDL bridge path gave the most useful new information.

The earlier `compact-lxlu` bridge launches but is numerically wrong. Stage 76
explains why: the following consumer still starts with an HBM load for the same
logical edge.

From the DXP debug dump for the real fixture:

```text
sdsc_0_add producer output Tensor2 local source offset: 12954
sdsc_2_add consumer input Tensor1 local destination offset: 12826
```

Patching only the DDL bridge source address to either `16384` or `12954` still
failed correctness, because `sdsc_2_add` continued to emit:

```text
transfer_lds1_src:hbm_dst:lx
```

for the consumer input.

Then I patched the consumer boundary as well:

- bridge input allocation/read -> producer local offset `12954`
- bridge output allocation/write -> consumer local offset `12826`
- consumer `Tensor1` memOrg -> LX-only
- consumer `allocate-Tensor1_hbm` -> `allocate-Tensor1_lx`

This made DXP generate the desired consumer read shape:

```text
transfer_lds1_src:no_component_dst:lx_lx_local
transfer_lds1_src:lxlu_dst:sfp
```

and the restickify bridge program still had the desired no-HBM/no-L3 signature:

```text
ReStickify bridge: HBM=0, L3LU=0, L3SU=0, LXLU=32, LXSU=32
```

However, the hardware run did not succeed. The first fused add/restickify/add
bundle launched, then the stream entered a compute hardware error state before
the matmul bundle:

```text
Compute CB hardware error detected
Cannot schedule operation on stream in error state
```

So this is closer than the previous correctness mismatch, but still not a
working prototype.

## What We Learned

The missing contract is now concrete:

```text
producer-owned LX buffer
  -> restickify LXLU source
  -> restickify LXSU destination
  -> consumer LXLU source
```

must be represented as one internal in-graph edge with legal synchronization and
aliasing. A standalone restickify replacement is insufficient because the next
SDSC boundary re-materializes the tensor from HBM.

The Spyre KB supports this reading. The runtime-facing contract is not just
`init.txt` plus SDSC JSON; Flex consumes a full SpyreCode/runtime metadata
package with memory segment layout, per-operation descriptors, program frame
pointers, and correction metadata. Hand-splicing binaries or changing one SDSC
cannot reliably express internal producer/consumer aliasing.

## Conclusion

Still not proven:

```text
value-correct LX-to-LX restickify in a Torch-Spyre graph
```

Now proven or strongly indicated:

1. Deeptools can generate HBM-free/L3-free LX data-op restickify programs.
2. The current DDL bridge can emit explicit LXLU and LXSU and launch.
3. Correctness requires patching the consumer side too; otherwise it reloads the
   restickify input from HBM.
4. Patching the consumer side through JSON is enough for DXP to emit an LXLU
   consumer read, but the resulting program hits a hardware control-block error.

## Next Step

Stage 77 should stop treating the restickify as an isolated SDSC replacement and
instead prototype an internal-edge contract. Two viable directions:

1. Deeptools-side DDL/DLDSc contract: add or discover a first-class internal LX
   alias edge so producer, bridge, and consumer agree on one LX allocation and
   sync plan.
2. Torch-Spyre-side fused boundary prototype: generate a single bridge-aware
   compound SDSC for the small fixture so DDC/DCC see producer output,
   restickify movement, and consumer input in one schedule rather than three
   independent HBM-boundary SDSCs.

The success criterion for the next stage is modest:

```text
one small in-graph restickify fixture
  + no restickify HBM/L3 traffic
  + explicit bridge LXLU/LXSU
  + consumer reads via LXLU, not HBM
  + first bundle retires without stream error
```
