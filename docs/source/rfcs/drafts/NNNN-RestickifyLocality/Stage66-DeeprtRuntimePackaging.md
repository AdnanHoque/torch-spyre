# Stage 66: Deeprt Runtime Packaging Probe

## Summary

Stage 66 compared the normal DXP/Torch-Spyre runtime bundle shape against the
Stage 65 Deeprt export, then built a tiny runtime-shaped directory from the
Deeprt Sentient artifacts.

The important result:

- The staged Deeprt data-op program launched through the existing
  `torch_spyre._C.launch_kernel(...)` path and synchronized successfully.
- No DXP changes were required.
- No Deeptools changes were required.
- For this no-argument staged program, the runtime only required:
  - `bundle.mlir`
  - `loadprogram_to_device/<code_dir_basename>-SenProgSend/init.txt`

This is not yet a tensor-correctness proof for a real restickify edge, but it is
a real runtime-packaging proof: a Deeprt-generated data-op program can be shaped
so the current Torch-Spyre/Flex runner accepts and executes it.

## Why This Stage Was Needed

Stage 65 proved that Deeprt can compile and export the scheduled
`ReStickifyOpLx -> STCDPOpLx` data-op SDSC:

```text
scheduled data-op SDSC
  -> Deeprt one-node DscSenGraph injection
  -> DataOp scheduler
  -> DCC
  -> DIP / program frame
  -> Sentient/Senulator export
```

The remaining question was whether the Deeprt export tree matched the current
Torch-Spyre runtime contract closely enough to launch.

Torch-Spyre currently does:

```python
generate_bundle(kernel_name, output_dir, op_specs)
dxp_standalone --bundle -d output_dir
return SpyreSDSCKernelRunner(kernel_name, output_dir)
```

and the C++ runner loads:

```cpp
std::string bundle_path = code_dir / "bundle.mlir";
std::string init_path =
    code_dir / "loadprogram_to_device/<code_dir_basename>-SenProgSend/init.txt";
```

Then it passes the uploaded program pointer, runtime tensor addresses, and the
`bundle.mlir` path to Flex:

```cpp
flex::RuntimeOperationCompute compute_op(
    &program_addr, std::move(tensor_allocs), arts.bundle_mlir_path);
```

So the concrete packaging question was: can a Deeprt exported program be placed
under that directory contract?

## Normal DXP Bundle Shape

A normal DXP-compiled single restickify bundle has this runtime shape:

```text
bundle.mlir
execute/<kernel_name>/pagi.json
execute_dsg.txt
loadmodel_to_device_dsg.txt
loadmodel_to_spad_dsg.txt
loadprogram_to_device/<kernel_name>-SenProgSend/init.txt
loadprogram_to_device_dsg.txt
loadprogram_to_spad_dsg.txt
sdsc_0_ReStickifyOpHBM.json
segment_size.json
```

For example, a stock `ReStickifyOpHBM` bundle had:

```text
bundle.mlir                                                 126 bytes
execute/.../pagi.json                                      219 bytes
loadprogram_to_device/...-SenProgSend/init.txt            6682 bytes
sdsc_0_ReStickifyOpHBM.json                              14861 bytes
segment_size.json                                          395 bytes
```

The `bundle.mlir` simply names the root SDSC JSON:

```mlir
module {
	func.func @sdsc_bundle() {
		sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0_ReStickifyOpHBM.json"}
		return
	}
}
```

## Deeprt Export Shape

Stage 65 Sentient export produced:

```text
after_pipeline.json
execute/0_TwoStepReStickifyLxStcdp_stage3b_dataop/init.txt
execute/0_TwoStepReStickifyLxStcdp_stage3b_dataop/sdsc.json
execute/0_TwoStepReStickifyLxStcdp_stage3b_dataop/senprog.txt
execute/0_TwoStepReStickifyLxStcdp_stage3b_dataop/smc.txt
execute_dsg.txt
ldsToDciPath.json
ldsToIsgInfo.json
loadmodel_to_device_dsg.txt
loadmodel_to_spad_dsg.txt
prog_size.json
segment_size.json
```

The key mismatch is that Deeprt places `init.txt` under the execute node:

```text
execute/<node>/init.txt
```

whereas Torch-Spyre's runner expects:

```text
loadprogram_to_device/<code_dir_basename>-SenProgSend/init.txt
```

## Runtime-Shaped Probe

I created a staged directory:

```text
/tmp/stage66-deeprt-runtime-shape/
  bundle.mlir
  execute_dsg.txt
  segment_size.json
  sdsc_0_TwoStepReStickifyLxStcdp_stage3b_dataop.json
  loadprogram_to_device/stage66-deeprt-runtime-shape-SenProgSend/init.txt
```

The `init.txt` came directly from the Stage 65 Deeprt Sentient export:

```text
/tmp/stage65-deeprt-export-sentient/execute/0_TwoStepReStickifyLxStcdp_stage3b_dataop/init.txt
```

The staged `bundle.mlir` was:

```mlir
module {
	func.func @sdsc_bundle() {
		sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0_TwoStepReStickifyLxStcdp_stage3b_dataop.json"}
		return
	}
}
```

First launch attempt failed before packaging was tested because no Flex runtime
context existed:

```text
RuntimeContext not created
```

After initializing the runtime with a tiny Spyre tensor, the same staged
directory launched and synchronized:

```python
import torch
import torch_spyre
from torch_spyre._C import launch_kernel

x = torch.empty((1,), dtype=torch.float16, device="spyre")
torch.accelerator.synchronize()

launch_kernel("/tmp/stage66-deeprt-runtime-shape", [])
torch.accelerator.synchronize()
```

Output:

```text
initializing runtime
launching staged Deeprt data-op bundle
launch returned
sync returned
```

## Minimal File Probe

I then removed files from the staged directory to see what this no-argument
runtime path actually needs.

| Variant | Files removed | Result |
|---|---|---|
| `full` | none | ok |
| `no_execute_dsg` | `execute_dsg.txt` | ok |
| `no_segment_size` | `segment_size.json` | ok |
| `min_bundle_sdsc_init` | `execute_dsg.txt`, `segment_size.json` | ok |
| `no_root_sdsc` | root SDSC JSON | ok |
| `no_bundle` | `bundle.mlir` | failed |

The failure was exactly the Torch-Spyre C++ guard:

```text
Bundle not found: /tmp/stage66-contract-min/no_bundle/bundle.mlir
```

So for this staged no-argument Deeprt program, the launch-time minimum was:

```text
bundle.mlir
loadprogram_to_device/<code_dir_basename>-SenProgSend/init.txt
```

The root SDSC JSON, `execute_dsg.txt`, and `segment_size.json` were not required
by this specific launch.

## What This Proves

This stage proves a narrower but very important point:

```text
Deeprt data-op export init.txt
  -> Torch-Spyre loadprogram_to_device path
  -> existing launch_kernel()
  -> Flex RuntimeOperationCompute
  -> stream synchronize returns
```

In other words, we do not need to make raw `dxp_standalone --bundle` accept
`dataOpdscs_` before we can launch a data-op-generated program through
Torch-Spyre's current runtime.

## What This Still Does Not Prove

This is not yet a full restickify correctness proof:

- The staged program was launched with no runtime tensor arguments.
- It does not verify that a producer tensor's LX-resident values are read.
- It does not verify that a consumer sees the restickified output.
- It does not compare numerical output against stock `ReStickifyOpHBM`.
- It does not yet integrate Deeprt export into `SpyreAsyncCompile.sdsc()`.

The result should be read as a runtime-packaging proof, not yet as a complete
LX-to-LX restickify replacement.

## Current Best Interpretation

The data-op path is now much closer than the earlier DDL bridge path:

- Stage 64 showed the scheduled data-op SDSC can reach stitched Dataflow IR and
  ProgIR without HBM/L3 textual traffic.
- Stage 65 showed Deeprt can export Sentient/Senulator runtime-style artifacts
  for that SDSC.
- Stage 66 showed the Deeprt `init.txt` can be repackaged and launched through
  the current Torch-Spyre/Flex runtime.

The remaining hard work is not "can Flex launch a Deeprt data-op program?".
It can.

The remaining hard work is binding real runtime tensor segments into a
data-op-generated restickify program and validating values.

## Recommended Next Step

Build a tensor-bearing Stage 67 fixture:

1. Generate a two-step data-op restickify program whose source and destination
   correspond to real Torch-Spyre runtime argument segments.
2. Package the Deeprt `init.txt` under the existing
   `loadprogram_to_device/<kernel>-SenProgSend/init.txt` path.
3. Launch with actual Spyre tensors as arguments, not `[]`.
4. Validate that the destination tensor changes as expected.
5. Only then wire a default-off `SpyreAsyncCompile` prototype that dispatches
   eligible pure data-op restickify bundles through Deeprt export instead of raw
   DXP.

The smallest useful acceptance test is:

```text
input tensor on Spyre
  -> Deeprt data-op restickify program
  -> output tensor on Spyre
  -> copy output to CPU
  -> compare against expected restickified logical values
```

That would be the first real LX-to-LX restickify correctness proof.
