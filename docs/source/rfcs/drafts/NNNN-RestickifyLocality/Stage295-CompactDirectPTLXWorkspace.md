# Stage 295: Compact Direct PT-LX Workspace

## Summary

This stage moved the direct PT-LX sidecar from a Deeptools
`ReStickifyOpWithPTLx` valid-gap failure to a late export/runtime-packaging
failure. The important compiler-side change is that the direct bridge now
describes its temporary gather buffer as a compact tile-local 64x64 workspace,
and its gather input fragments use compact tile coordinates plus per-fragment
LX start-address offsets back into the producer allocation.

The stock `ReStickifyOpHBM` path remains the fallback. The PT-LX path is still
uncertified and default-off.

## Why This Was Needed

Stage 293 selected the correct direct bridge direction for
`computed_transpose_adds_then_matmul_tuple` at size 512:

```text
direction = output-to-kernel
bridge_kind = direct-ptlx-layout-transform
```

But Deeptools rejected later tiles because the local PT restickify input kept
global tile coordinates. For tile 1, the gather output/restickify input had an
`out_` layout size of 128 while the local output tile had `out_ = 64`.

That mixed two coordinate systems:

- source tensor coordinates, used to locate the producer data
- tile-local workspace coordinates, needed by the local 64x64 PT transform

The new lowering separates them:

- gather input: compact tile coordinates, with LX `startAddr` adjusted to the
  source fragment offset
- gather output: compact 64x64 tile workspace
- PT restickify input/output: compact 64x64 tile-local descriptors

## Code Changes

- `generate_streaming_ptlx_direct_tile_bridge_sdsc` now compacts direct gather
  inputs and outputs before invoking `ReStickifyOpWithPTLx`.
- `_tile_piece_info` accepts a per-fragment `start_addr` override for LX
  fragments.
- `_compact_lx_read_fragments` computes compact tile coordinates while
  preserving the producer LX source location with `_lx_fragment_offset_bytes`.
- `_streaming_value_flow_contract` now treats direct PT-LX producer starts as a
  producer allocation range rather than a single base address, because compact
  reads legitimately start at offsets inside the producer allocation.

## Validation

Focused unit/import checks in the pod:

```sh
python -m py_compile \
  torch_spyre/_inductor/codegen/restickify_lx_dataop.py \
  torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py \
  tests/inductor/test_restickify_lx_dataop.py

python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  -q
```

Result:

```text
55 passed in 8.97s
```

The new focused unit test verifies that tile 1 is described as a compact 64x64
workspace instead of carrying global valid-gap state into
`ReStickifyOpWithPTLx`.

## Deeptools Export Probe

Probe:

```sh
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code \
  --output-dir /tmp/stage295-descriptor-512 \
  --fail-on-error
```

The compiler-side descriptor still reports:

```text
restickifies=1 bytes=524288 byte_hops=0 device_events=0
```

Exporting the generated sidecar with `/tmp/stage65-deeprt-dataop-probe` reached
DCG, DCC, ProgIR verification, DIP, frame-pointer fill, and export. It produced:

```text
/tmp/stage295-direct-sidecar-export-512/execute/1_LXNeighborStreamingReStickifyOpWithPTLx/senprog.txt
```

A simple opcode-prefix count over that `senprog.txt` showed:

```text
HBM=0
SFP=896
PT=256
```

The exporter then returned `SIGSEGV` late in the export path after writing the
program files. This is progress relative to the previous blocker:

- before: Deeptools rejected `ReStickifyOpWithPTLx` valid-gap metadata
- now: the full direct PT-LX sidecar compiles far enough to emit an HBM-free
  program, then fails during late export/packaging

## Runtime Packaging Smoke

Even though the throwaway Deeprt export probe returned `SIGSEGV`, it had already
written the execute-node `init.txt`. Following the Stage 66 packaging shape, the
file was staged as:

```text
/tmp/stage295-direct-runtime-shape/
  bundle.mlir
  loadprogram_to_device/
    stage295-direct-runtime-shape-SenProgSend/
      init.txt
```

The first launch attempt failed with the expected setup error:

```text
RuntimeContext not created
```

After initializing the runtime with a tiny Spyre tensor, the no-argument direct
PT-LX sidecar launched and synchronized:

```python
import torch
import torch_spyre
from torch_spyre._C import launch_kernel

x = torch.empty((1,), dtype=torch.float16, device="spyre")
torch.accelerator.synchronize()

launch_kernel("/tmp/stage295-direct-runtime-shape", [])
torch.accelerator.synchronize()
```

Output:

```text
runtime_initialized
launch_returned
direct_ptlx_runtime_smoke_ok
```

A post-launch stock Spyre smoke also passed:

```text
post_sidecar_stock_smoke_ok spyre:0
```

This proves the generated direct PT-LX sidecar program can retire through the
normal Torch-Spyre `launch_kernel` runtime shape. It does **not** prove tensor
value correctness yet because this smoke does not bind producer/consumer tensors
around the sidecar.

## Non-Signal Probe

A manually truncated 2-tile sidecar failed earlier in DCC with:

```text
DtException: dsc_schedule.size() > 0
```

That result is not treated as evidence against the descriptor shape because the
manual pruning also invalidated schedule structure.

## Current Blocker

The next blocker is not the PT local valid-gap contract anymore, and not basic
runtime retirement of the HBM-free sidecar. It is value-correct integration:
the normal Torch-Spyre producer/restickify/consumer bundle still needs to bind
the producer output, direct PT-LX sidecar, and consumer input to the same
internal LX value-flow contract.

The next useful step is to package the generated HBM-free program into a
producer-sidecar-consumer graph and run a value-correct 512 hardware test. A
clean non-segfaulting exporter remains desirable, but the staged runtime smoke
shows the late harness/export crash is not by itself a hardware-retirement
blocker.
