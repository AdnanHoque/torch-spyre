# Deeptools Candidate Patch Summary

Branch: `Adnan-Hoque1/deeptools:gather-restickify`
Commit: `e3e265d22c7283054dd36e147a7e7ec919606441`

## Why this patch exists

The Torch side can describe matmul operand broadcast/multicast/all-gather-like edges through DLDSC relayout metadata, but the large Granite attention operand should not be materialized as a full resident copied tensor. Dense gather/restickify lowered semantically, but failed either IBUFF or LX capacity on Granite S256/S512.

## What the patch changes

- In `L3DlOpsScheduler.cpp`, KERNEL-neighbor destination allocations are seeded from the consumer SuperDSC work division. This fixes a DDC coordinate-capture crash where the destination allocation inherited incomplete producer coordinates.
- In `L3DlOpsScheduler.cpp` and `inputNeighFetchOp.cpp`, mixed HBM-pinned tensors plus LX input-neighbor tensors are allowed only for the explicitly enabled matmul operand kernel-neighbor path.

## What it does not change

- It does not globally allow mixed HBM and IFN for arbitrary ops.
- It does not claim value correctness for the known flash zero-stride broadcast issue.
- It does not implement reduce/all-reduce.

## Evidence

- Focused Deeptools tests pass: `LayoutAllgatherRestickify.*` 27/27 and `DxpTestFixture.CoreWorkDivIncomptLxRelayout*` 2/2.
- Granite S256 and S512 structural prefill compile with loop-scoped matmul operand plans and no diagnostic override.
- Flash compile probe emits 32 `loop_scoped_input_fetch` matmul operand plans.
