# Activation-stationary expert matmul review series

## Scope

This branch contains the compiler realization for dense all-expert MoE
execution. It is based on Torch-Spyre commit
`3fd7a0f954a84817a417b6b45639b0d5f3499575` and is deliberately separate
from measured evidence and tensor artifacts.

The semantic operation is:

```text
Y[t,h] = sum over e of alpha[e,t,0] * Wd[e](
    gelu(X[t,:] * Wg[e]) * (X[t,:] * Wu[e]))
```

The accepted representative shape is:

```text
E=128, T=512, H=2816, F=704, C=32, FP16
```

## Production commits

The commits follow the compiler pipeline. Each production commit includes its
own focused tests and is intended to import and test independently.

1. `c0400085` `Add activation-stationary expert matmul contracts`

   Adds shared-LHS and prepacked expert-matmul operators, fake contracts,
   lowering, metadata, and layout propagation. The activation remains 2-D by
   construction while the expert coordinate indexes only weights and output.

2. `d259fa79` `Hoist compact invariant expert-loop operands`

   Builds compact read copies, preserves planning-time expert-slab strides,
   omits the expert loop from invariant activation copies, and records the
   owning loop group without fabricating a read.

3. `795ba5ca` `Build a fixed expert-loop reduction accumulator`

   Forms separate fixed accumulation and final-output storage, fills before
   the expert loop, combines once per expert, drains once afterward, and
   replaces a stale local unit sum with its sole contribution.

4. `1ded55d4` `Model loop-carried LX lifetimes explicitly`

   Adds a general exclusive lifetime-end override. It affects address overlap
   without changing the access list, read count, residency score, or spill
   benefit.

5. `b06ec81d` `Align activation-stationary ownership and LX residency`

   Registers the alignment passes, admits only the marked fixed accumulator,
   aligns the invariant activation and FFN path to one legal ownership, checks
   capacity and aliases, and rolls back unsafe projections.

6. `dffd639f` `Emit expert operand bindings and physical core maps`

   Names the loop-operand binding boundary, implements sequential affine
   expert rebinding, serializes symbols that survive only in operand advances,
   emits the selected physical core map, and adds an opt-in compile timeout for
   the large static bundle.

The following validation commit contains reduced and representative probes, a
strict bundle checker, and compact correctness evidence from one AIU. It
contains no timing result.

## Operand-binding variation point

`torch_spyre/_inductor/spyre_kernel.py::LoopOperandBinding` is the named
strategy boundary between loop formation and runtime operand addressing.

The only supported binding today is:

```text
sequential_affine(step) -> device-element operand offset
```

That is sufficient for dense all-expert execution. An indexed expert table,
data-dependent trip count, routed-row extent, or output-combine map is not
implemented and must extend the contract explicitly. It must never silently
fall back to a fixed base address.

## Execution contract

The representative emitted program must have:

- one wrapper call and one bundle;
- one flat expert loop and no temporal token loop;
- one shared activation HBM-to-LX preheader;
- gate, up, down, pointwise operations, and the accumulator on the accepted
  all-core ownership map;
- direct HBM Wg, Wu, Wd, and alpha operands with expert-loop advances;
- runtime alpha shaped `[E,T,1]` and applied after down;
- all intermediate activations and the fixed accumulator in LX;
- zero HBM-pool intermediates;
- zero HBM restickify operations; and
- one final LX-to-HBM output drain.

## Validation order

```text
1. import and focused tests at each production commit
2. full affected compiler suite at the final production commit
3. reduced E2/T64/H64/F64/C1 source structure
4. reduced real DeepTools compilation
5. reduced same-callable two-alpha correctness
6. representative E128/T512/H2816/F704/C32 structure
7. representative correctness
8. device timing only after every earlier gate passes
```

The exact reduced checker is `moe_asgemm/tools/dasx_c1_gate.py`. The exact
probes are under `experiments`. The completed recut validation is recorded in
`moe_asgemm/RECUT_VALIDATION.md`.

## Evidence boundary

This series implements the compiler-generated kernel. It does not claim model
integration, router-logit computation, energy results, indexed or grouped
binding, a native custom DDL, or universal superiority over every grouped
schedule.
