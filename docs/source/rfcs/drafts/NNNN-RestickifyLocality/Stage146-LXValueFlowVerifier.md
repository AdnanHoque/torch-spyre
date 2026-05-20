# Stage 146: LX Value-Flow Verifier

## Goal

Turn the Stage145 wrong-values failure into an explicit compiler-side contract
check.  The target is still:

```text
producer output LX -> restickify bridge -> consumer input LX
```

inside one normal Torch-Spyre bundle.  The new requirement is stronger than
matching LX addresses: producer, restickify, and consumer must agree on the
same logical value ownership contract.

## Change

Added a default-off diagnostic flag:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_VALUE_FLOW_ASSERT=1
```

When the DDL bridge boundary patch is active, the patch now computes a
`value_flow_contract` over:

- producer output view,
- bridge input view,
- bridge output view,
- consumer input view.

The contract compares:

- `numWkSlicesPerDim_`,
- `coreIdToWkSlice_`,
- primary layout/stick metadata.

With the assert flag enabled, a mismatch raises before bundle files are written
or launched.  With the flag disabled, existing prototype behavior is unchanged,
but the patch row records whether the value-flow contract passed.

## Validation

Static and unit validation in the pod:

```text
python -m py_compile torch_spyre/_inductor/codegen/restickify_lx_boundary.py \
  tests/inductor/test_restickify_ddl_bridge.py
python -m pytest tests/inductor/test_restickify_ddl_bridge.py -q
```

Result:

```text
24 passed
```

The first pytest run hit unrelated Mac AppleDouble `._*.yaml` files in the
temporary pod copy.  Removing those disposable files fixed test collection.

## High-Signal 2048 Probe

I reran the Stage145 `computed_transpose_adds_then_matmul_tuple` no-launch
probe with:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_OPFUNC=ReStickifyOpLx
SPYRE_RESTICKIFY_DDL_BRIDGE_ALLOW_MULTI_SPLIT=1
SPYRE_RESTICKIFY_DDL_BRIDGE_VALUE_FLOW_ASSERT=1
DXP_LX_FRAC_AVAIL=1
```

Output:

```text
ValueError: restickify DDL bridge 1 failed LX value-flow contract:
producer->bridge mismatches=['num_work_slices_per_dim', 'primary'],
bridge->consumer mismatches=['core_id_to_work_slice']
```

This is the expected result.  It catches the same case that previously launched
and produced wrong values.

## Core Mapping Diagnostic

I then enabled the existing core-mapping propagation knob:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_PROPAGATE_CORE_MAPPING=1
```

The failure narrowed to:

```text
producer->bridge mismatches=['primary'],
bridge->consumer mismatches=[]
```

Interpretation: core ownership can be made to line up, but the producer still
writes a different primary layout/stick view than the bridge reads.

## Producer Layout Diagnostic

Finally, I added:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_PATCH_PRODUCER_OUTPUT_LAYOUT=1
```

The value-flow assert no longer fired; compilation advanced to Deeptools and
failed in the producer add DDL:

```text
broadcast_ops.ddl:37:1: error: slice size does not match
Ddl constraints not met
```

This is useful evidence: the missing semantic piece is exactly the producer
output contract, but post-hoc producer layout mutation is too blunt for
Deeptools.  The producer op's DDL, allocation layout, coordinate metadata, and
work slicing need to be planned together rather than patched after scheduling.

## Conclusion

We have not yet achieved the final goal, but the blocker is now precise:

```text
endpoint aliasing is solved;
core mapping can be reconciled;
producer primary/source-view agreement is still missing.
```

The next implementation should stop trying to alias the stock bridge blindly.
It should either:

1. plan the producer output as the exact LX source view the bridge consumes, in
   a Deeptools-compatible way; or
2. use an explicit remote-LX movement/data-op bridge that reads the producer's
   real physical output view and materializes the consumer/restickify view.

The second path is still the more general solution for real restickification,
because a restickify often exists precisely because producer and consumer
ownership/layout differ.
