# Buf21 Small-Fit Boundary Result

This directory is derived from the full `buf21` attention value repro, but it shrinks the value operand while preserving the important producer/consumer mismatch:

- Tensor1 producer residency: `out` sharded across 32 cores.
- Consumer AV matmul compute: `mb` sharded across 32 cores.
- Tensor1 layout dimensions: `out,in,x`; `mb` is not a Tensor1 tensor dimension.

The reduced shape is `x=1, out=128, in=16`, so the full Tensor1 operand is only `4096` bytes.

Observed result:

- `dxp_stderr.log` is empty.
- DXP compile returned success.
- The input dldsc still carries the mismatched Tensor1 allocation coordinates and consumer compute coordinates.

Interpretation:

This proves the current backend mechanism can compile an AV-shaped tensor/compute mismatch when the resident post-relayout value operand fits. The full Granite prefill failure is therefore a capacity/materialization-scope problem: current DXP relayout insertion tries to materialize the full value operand for each consumer core instead of staging/broadcasting the operand through the matmul transfer loop.

This does not recover the missing Granite speedup by itself. The missing communication class remains a matmul operand broadcast/all-gather, not another `scatter` case.
