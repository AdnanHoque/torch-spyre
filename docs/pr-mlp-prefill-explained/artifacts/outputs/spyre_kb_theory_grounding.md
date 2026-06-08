# Spyre KB Grounding for the Shared-Weight Unit-BMM Prefill Fix

This note connects the `pr-mlp-fix` result to the Spyre knowledgebase model of AIU matmul execution. It is written as PR/Claude explanation material, not as a new benchmark report.

## Short Version

The PR did not change the math of the MLP projection. It fixed the compiler representation of the wide shared-weight prefill projection so DeepTools sees the layout/work shape that AIU matmul expects.

For the prefill projection:

```text
activation: [1, 512, 4096]
weight:     [4096, 12800]
output:     [1, 512, 12800]
```

the important hardware fact is that AIU matmul/BMM uses a KG3 weight-stationary dataflow:

- weights are block-loaded into PT XRF and held stationary;
- activations stream through LX/L0/PT rows;
- partial sums accumulate in PT ARF;
- a weight tile is reused across the M dimension.

That means PT utilization depends on presenting a clean stream of M rows and a good N split, not merely on having a mathematically equivalent 2D matmul. The bug was that the size-1 BMM dimension got flattened/squeezed before codegen layout construction, so the SDSC/device loop nest no longer looked like the sendnn-style shared-weight unit-BMM. The fix preserves that logical unit-BMM axis and emits the intended primary layouts:

```text
INPUT:  [mb, in, x]
KERNEL: [in, out]
OUTPUT: [mb, out, x]
```

With the matching work split:

```text
x=1, mb=4, out=8, in=1
```

the prefill MLP-proj matmul moves from low PT utilization to near sendnn parity:

```text
main:        3.749 ms kernel, 29.794% PT
pr-mlp-fix:  1.023 ms kernel, 72.799% PT
speedup:     3.66x kernel
```

## Why Weight-Stationary Matters

The knowledgebase describes AIU matmul/BMM as KG3, a weight-stationary dataflow. For matmul/BMM, the stationary operand is the weight tile in PT XRF. Activations stream through the array and partial sums accumulate in PT.

This directly explains why a bad layout can look like "the array is starving":

- If weights are resident in XRF, the inner loop wants to reuse a weight tile across many M rows.
- The compiler should therefore expose enough M rows per core to feed PT, while also splitting/tiling N because `12800 / 64 = 200` output sticks is too wide for one PT tile.
- If the layout metadata collapses the unit-BMM dimension and presents the wrong loop/order to the bundle compiler, the operation is still mathematically correct, but the generated schedule can underfill PT.

In KB terms, the matmul projection wants:

```text
M = prefill rows
K = reduction dimension, in sticks
N = output/generated dimension, in sticks
```

For the Granite-sized MLP projection, the KB example states:

```text
global:  [512, 4096] x [4096, 12800]
sticks:  K=64, N=200
per-core prefill target: M=16 rows, K=64 sticks, N=200 sticks
tiling: K_TILE=8, N_TILE=25, K_ITERS=8, N_ITERS=8
```

So a healthy prefill projection streams activation sticks for multiple M rows while reusing weight tiles and walking N tiles. Our cost model change matches this: once the per-core M tile is large enough to feed PT, prefer spending remaining split capacity on `out`/N rather than over-splitting M or splitting K.

## What a Stick Is

The knowledgebase uses "stick" as the software-facing form of the AIU DataStick:

```text
1 DL16/BF16 stick = 64 elements = 128 bytes
```

Device memory is not just ordinary PyTorch strided memory. It is described by:

- `device_size`: the loop ranges after stick tiling;
- `stride_map`: how each device loop advances through host elements;
- dense row-major device memory over `device_size`.

The stick dimension appears twice:

- once as the outer stick-count dimension;
- once as the innermost 64-element stick dimension.

For a 2D `[512, 240]` tensor, the KB example shows the device loop:

```python
for s in range(4):        # stick groups
  for r in range(512):    # rows
    for e in range(64):   # elements within stick
      ...
```

This is why "which logical dimension becomes the stick dimension" is not cosmetic. It determines the DMA loop nest, the device memory order, and what dimensions the work-distribution passes can split across cores.

For matmul `X=[M,K]`, `W=[K,N]`, `O=[M,N]`, the KB gives the stick constraints:

```text
X:      stick on K
W:      stick on N
Output: stick on N
```

The weight beyond-stick order is `["in", "out"]`, giving physical memory shaped like:

```text
[stick(64), K, N/64]
```

with `N/64` outermost. That outer N-stick dimension can be split across cores into contiguous N partitions without cross-core reduction.

## How to Say the Layout Bug Precisely

The diagram language "bad layout: M buried next to the stick" is directionally right, but for a technical PR explanation I would make it more precise:

The old path flattened/squeezed the logical unit-BMM axis before the Spyre OpSpec/SDSC layout was built. That caused the shared-weight prefill projection to lose the sendnn-like unit-BMM loop structure. DeepTools still got a legal matmul, but not the layout/work-distribution shape that naturally exposes `mb`/M rows and `out`/N sticks to the KG3 scheduler.

The fix preserves the singleton BMM axis as layout metadata and emits:

```text
INPUT:  [mb, in, x]
KERNEL: [in, out]
OUTPUT: [mb, out, x]
```

That gives DeepTools a clean non-stick M/`mb` dimension plus an outer N/`out` stick partition. The result is not "better math"; it is a better tiled-device loop nest and better core split for the same math.

## Claude-Ready Explanation

The fix is best explained in AIU dataflow terms. Matmul/BMM on AIU is KG3 weight-stationary: the weight tile is loaded into PT XRF and held there while activation sticks stream through the PE/PT pipeline and partial sums accumulate. For the MLP prefill projection, `K=4096` is 64 sticks and `N=12800` is 200 output sticks, so the healthy schedule needs both M-row streaming and N tiling/splitting. Upstream torch-spyre was flattening away the size-1 BMM axis before layout/codegen, so the SDSC no longer represented the operation like sendnn's shared-weight unit-BMM. It was mathematically correct but presented a weaker device loop nest and work split, which underfilled PT. The PR preserves the unit-BMM axis and emits the sendnn-like layout `input [mb,in,x]`, `kernel [in,out]`, `output [mb,out,x]`, then adjusts the split cost so cores are spent on N once M is sufficient. That is why the same projection improves from 3.749 ms / 29.8% PT to 1.023 ms / 72.8% PT.

## Source Anchors

Private repo: `github.ibm.com/msrivats/spyre-knowledgebase`

- `wiki/concepts/dataflow-architecture.md:24-32`: KG3 weight-stationary matmul/BMM, weights in PT XRF, activations stream, partial sums accumulate.
- `wiki/concepts/dataflow-architecture.md:61-86`: XRF removes the weight read stream from LX during inner compute and reuses a weight tile across M.
- `wiki/concepts/dataflow-architecture.md:88-100`: matmul stick constraints and default N distribution.
- `wiki/concepts/stride-map.md:18-20`: `device_size` and `stride_map` define tiled device memory and DMA traversal.
- `wiki/concepts/stride-map.md:27-36`: the stick dimension appears as both stick-count and stick-elements.
- `wiki/concepts/stride-map.md:64-85`: example showing device memory grouping same-position sticks for PE processing.
- `wiki/concepts/stride-map.md:137-146`: device memory order, DMA pattern, and work distribution are driven by the stride map.
- `wiki/artifacts/rfcs/0047-tiled-tensors.md:10-14`: ordinary PyTorch strides cannot express AIU tiled layouts.
- `wiki/artifacts/rfcs/0047-tiled-tensors.md:29-33`: stick/DataStick is 128 bytes; compiler stick-dim choice determines dataflow compatibility.
- `examples/README.md:44-56`: 32 cores per AIU; prefill splits along non-stick dimensions; decode uses batch/groups.
- `examples/dataflowir/level4c_matmul_mlp.mlir:9-19`: Granite MLP projection stick sizes and prefill per-core shape.
- `examples/dataflowir/level4c_matmul_mlp.mlir:42-70`: prefill N/K tiling and activation/weight/output stick counts.
- `examples/schedule_ir/level4c_matmul_mlp.mlir:4-10`: prefill per-core schedule requires tiling both K and N.
