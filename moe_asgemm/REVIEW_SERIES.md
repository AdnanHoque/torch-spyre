# Activation-stationary expert matmul review series

## Scope

This branch contains the compiler mechanism for a dense all-expert MoE
program. It is deliberately separate from the evidence branch and contains no
large result or tensor artifacts.

The implemented operation is:

```text
Y[t,h] = sum over e of alpha[e,t,0] * Wd[e](
    gelu(X[t,:] * Wg[e]) * (X[t,:] * Wu[e]))
```

The accepted representative shape is:

```text
E=128, T=512, H=2816, F=704, C=32, FP16
```

## Commit dependencies

The series is ordered so each layer exposes the contract required by the next:

1. `Add activation-stationary expert matmul contracts`

   Adds shared-LHS and prepacked expert-matmul operators, lowering, metadata,
   and layout propagation.

2. `Form flat expert loops with invariant activation state`

   Builds one static expert loop, compacts and hoists the shared activation,
   preserves expert slab advances, constructs fixed reduction state, and
   removes the stale unit-size local reduction after coarse tiling.

3. `Keep expert-loop activations and accumulator in LX`

   Adds the narrowly marked invariant and accumulator lifetimes, ownership
   alignment, capacity checks, alias checks, and fail-closed LX planning.

4. `Preserve expert advances and core maps through codegen`

   Serializes symbols that exist only in HBM argument advances, emits the
   selected physical core map, and adds an opt-in timeout for the large static
   bundle.

5. `Add activation-stationary compiler regressions`

   Covers compact copies, loop liveness, unit-dimension expert strides,
   fixed-state allocation, reduction normalization, affine arguments, exact
   C32 core maps, rollback, and ordinary-path negative controls.

6. `Add self-contained activation-stationary validation probes`

   Adds reduced and representative compile/correctness/timing probes plus a
   strict bundle checker. No measured result is embedded in this branch.

## Execution contract

The representative emitted program must have:

- one wrapper call and one bundle;
- one flat expert loop and no temporal token loop;
- one shared activation HBM-to-LX preheader;
- gate, up, and the activation on the same all-core ownership map;
- direct HBM Wg, Wu, Wd, and alpha operands with expert-loop advances;
- runtime alpha shaped `[E,T,1]` and applied after down;
- all intermediate activations and the fixed accumulator in LX;
- zero HBM-pool intermediates;
- zero HBM restickify operations; and
- one final LX-to-HBM output drain.

## Validation order

```text
1. affected host compiler suites
2. reduced E2/T64/H64/F64/C1 source structure
3. reduced real DeepTools compilation
4. reduced same-callable two-alpha correctness
5. representative E128/T512/H2816/F704/C32 structure
6. representative correctness
7. device timing only after every earlier gate passes
```

The exact reduced checker is `moe_asgemm/tools/dasx_c1_gate.py`. The exact
probes are under `experiments`.

## Evidence boundary

This series implements the compiler-generated kernel. It does not claim model
integration, router-logit computation, energy results, a native custom DDL, or
universal superiority over every grouped schedule.
