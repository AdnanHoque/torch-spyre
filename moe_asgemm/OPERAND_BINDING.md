# Expert-loop operand binding

## Purpose

An expert loop is useful only if each iteration can bind the correct weights
and routing payload without materializing selected tensors. The binding is a
compiler contract, not a property of the matmul DDL.

The source boundary is:

```text
torch_spyre/_inductor/spyre_kernel.py::LoopOperandBinding
torch_spyre/_inductor/spyre_kernel.py::SpyreKernel._resolve_loop_operand_binding
```

## Implemented form

The current implementation supports `sequential_affine` binding:

```text
operand_address(step) = base_address + affine_device_element_offset(step)
```

Coarse tiling captures each operand's pre-division host advance. Codegen
reprojects it into device-element space, retains the exact enclosing-loop
symbol and trip count, and emits an affine address inside the static loop.

If an expert dimension becomes statically one inside the loop body, its
ordinary iteration symbol may disappear. The preserved operand advance remains
authoritative. Missing advancement fails compilation instead of reusing expert
zero.

## Deliberately unsupported forms

The current contract does not implement:

- an expert ID loaded from an index table;
- a data-dependent loop trip count;
- a routed-row base or row-index vector;
- a per-expert dynamic row count; or
- inverse output placement and combine metadata.

Active-dense execution principally needs the first two. Grouped execution also
needs routed-row work extent, ownership, and output mapping. They may share the
binding interface, but indexed expert selection alone is not a grouped GEMM
implementation.

## Extension rule

A future binding kind must add an explicit OpSpec and SuperDSC representation,
validation, and a fail-closed backend capability check. It must not overload
the sequential affine expression or degrade to an invariant address when its
metadata is incomplete.
