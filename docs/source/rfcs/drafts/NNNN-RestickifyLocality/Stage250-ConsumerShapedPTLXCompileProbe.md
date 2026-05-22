# Stage 250: Consumer-Shaped PT-LX Compile Probe

## Summary

Stage 249 showed that the direct-tile PT-LX bridge writes an `out/mb` output
descriptor while the real `adds_then_matmul` consumer expects an `mb/in`
matmul input descriptor. Stage 250 ran compile-only Deeptools probes to test
whether a simple consumer-shaped `ReStickifyOpWithPTLx` data-op can satisfy
that contract.

It does not, at least not as a simple 2D rename.

## Probes

### Probe 1: Output Only Renamed To Consumer Axis

The first probe kept the bridge input as `mb/out` stick `out`, then changed the
bridge output to `mb/in` stick `in`.

Deeptools rejected the descriptor during transfer metadata validation. The
important rule is visible in `transfer_compute.cpp`: for
`ReStickifyOpWithPTLx`, every output dimension must also be present in the
input descriptor.

In plain terms: `ReStickifyOpWithPTLx` cannot create a new `in` dimension if
the input descriptor only has `mb/out`.

### Probe 2: Both Sides Presented Under Consumer Axis

The second probe renamed both bridge input and bridge output to `mb/in`, with
stick `in` on both sides. This also failed:

```text
DtException: op->inpLds->stickDimOrder_ != op->outLds->stickDimOrder_
```

So `ReStickifyOpWithPTLx` also expects the input and output stick dimensions to
differ.

## Interpretation

The next production-shaped bridge cannot be a simple 2D descriptor rewrite.
For the matmul-input case, the data-op likely needs a PT-style expanded
descriptor where:

- the source axis that Torch-Spyre calls `out` is proven equivalent to the
  consumer axis `in`;
- the output stick dimension `in` is present in the input descriptor, satisfying
  Deeptools' subset rule;
- the input and output stick dimensions remain distinct, satisfying
  `ReStickifyOpWithPTLx`;
- the bridge output descriptor matches the consumer input descriptor.

This is consistent with the earlier native 4D diagnostic: Deeptools is
comfortable with expanded PT-like descriptors, but the bridge must still map
that expanded internal shape back into the exact layout the consumer reads.

## Artifacts

- `artifacts/stage250_consumer_shaped_ptlx/consumer_shaped_first_dcg.log`
- `artifacts/stage250_consumer_shaped_ptlx/consumer_shaped_fixed_stick_dcg.log`
- `artifacts/stage250_consumer_shaped_ptlx/consumer_axis_dcg.log`
- `artifacts/stage250_consumer_shaped_ptlx/consumer_axis.json`

## Next Step

The next implementation attempt should generate a consumer-aware expanded
descriptor rather than a direct 2D descriptor:

1. derive a symbol correspondence from producer `out` to consumer `in`;
2. build a PT-LX bridge input descriptor that includes both the source stick
   axis and the consumer output stick axis required by Deeptools;
3. emit only if the consumer descriptor verifier proves the final bridge output
   matches the consumer input layout/stick/piece contract;
4. otherwise keep stock `ReStickifyOpHBM`.
