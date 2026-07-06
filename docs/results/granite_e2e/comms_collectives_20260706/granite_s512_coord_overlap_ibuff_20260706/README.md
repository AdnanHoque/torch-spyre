# Granite S512 Coordinate-Overlap Checkpoint

Date: 2026-07-06
Pod: adnan-cdx-spyre-dev-pf
Torch source branch: gather-restickify
Torch source head: cc29ba259ec190e62f760f032ffd3355581a57df
Deeptools source branch: ah/comms-collectives plus local coordinate-overlap-only diff
Deeptools base head: b594b3afc725b693d074a64fc027ac7a6024d5fd

## What this checkpoint proves

The matmul operand broadcast planner previously grouped source and destination cores by equal coordinate ids. That was wrong for the Granite SwiGLU edge where producer split is 32-way and consumer split is 8-way replicated across destination cores.

The local Deeptools diff in this directory changes grouping to use coordinate interval overlap. The corrected logical transfer expansion starts with:

- source core 0 -> destination cores 0, 8, 16, 24
- source core 1 -> destination cores 0, 8, 16, 24
- source core 2 -> destination cores 0, 8, 16, 24

That is the expected all-gather/replicate topology for this edge.

## Focused tests

- LayoutAllgatherRestickify.*: 32 passed
- DxpTestFixture.CoreWorkDivIncomptLxRelayout*: 2 passed

Logs are included in this directory.

## Current blocker

The corrected full Granite S512 replay still fails in DCC:

- SDSC: sdsc_fused_add_linear_mul_silu_split_with_sizes_3_clt9lx2o
- Root op: 5_batchmatmul
- Input: Tensor0 / matmul_operand_broadcast
- Logical transfers: 128
- Lowering selected: gather_then_restickify
- Failure: DCC instruction-buffer estimate exceeds limit, Max IBUFF 128, Current IBUFF 134

This means the logical DLDSC communication contract is now correct for this edge, but the current data-op plus ReStickifyOpLx physical carrier is too heavy for full Granite shape.

## Negative prototype results

Two local-only prototypes were tried and intentionally reverted:

1. Compact replicated intermediate gather pieces.
   - Result: still failed with the same 134/128 IBUFF pressure.
   - Read: STCDP fanout piece count is not the dominant issue.

2. One final ReStickify output shard per destination core.
   - Result: failed earlier in apeOp.cpp at iPieceOrder.size() == oPieceOrder.size().
   - Read: current ReStickifyOpLx lowering expects one-to-one input/output piece pairing and does not support many-input-to-one-output assembly yet.

## Next backend work

The likely implementation options are now narrower:

1. Extend the DL-neighbor path to support activation INPUT operands and staged layout conversion, not only KERNEL operands.
2. Extend local ReStickifyOpLx/layout conversion to support many-input-piece to one-output-piece assembly.
3. Add a lower-instruction physical carrier for all-gather/replicate plus local layout conversion so DCC does not exceed IBUFF.

The current coordinate-overlap diff is useful independently, but it is not sufficient for full end-to-end Granite until one of those carrier gaps is closed.
