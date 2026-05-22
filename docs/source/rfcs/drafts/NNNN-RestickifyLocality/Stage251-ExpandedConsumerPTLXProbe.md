# Stage 251: Expanded Consumer PT-LX Probe

## Summary

Stage 250 showed that a simple 2D consumer-shaped descriptor does not satisfy
Deeptools' `ReStickifyOpWithPTLx` contract for the `adds_then_matmul`
matmul-input consumer. Stage 251 tested expanded descriptors where the bridge
input includes both:

- the source stick axis, `out`; and
- the consumer stick axis, `in`.

This is closer to the contract Deeptools expects: every output dimension is
present in the input descriptor, and input/output stick dimensions differ.

## Deeptools Rules Confirmed

From the compile probes and Deeptools checks:

1. `ReStickifyOpWithPTLx` cannot create an output dimension that is absent from
   the input descriptor.
2. The input and output stick dimensions must differ.
3. Input pieces must carry at least one full output-stick span as well as one
   full input-stick span.

These explain why the direct 2D bridge could have a plausible endpoint contract
but still not be a valid matmul-consumer bridge.

## Compile Results

### Row-Slice Expanded Descriptor

Input:

```text
layout: out, mb, in
stick:  out
piece:  out=64, mb=1, in=64
```

Output:

```text
layout: mb, in
stick:  in
piece:  mb=1, in=64
```

Result:

```text
Computing Re-StickifyOpWithPT (Special re-stickify) transfer function..
Writing DataDsc to ...
Writing PCFG to ...
Writing SenPrograms to dataDSC/senprog.txt..
```

This proves the expanded descriptor shape can compile.

### Full-Tile Expanded Descriptor

Input:

```text
layout: out, mb, in
stick:  out
piece:  out=64, mb=64, in=64
```

Output:

```text
layout: mb, in
stick:  in
piece:  mb=64, in=64
```

Result: Deeptools also compiles this descriptor.

However, this version is not semantically certified. The input piece describes
more logical elements than the original 64x64 source tile, so it cannot be used
as a production lowering without a proof that the expanded dimension is an
alias/contract dimension rather than real extra data.

### Sparse/Fake Consumer Axis

Input variants that tried to make the extra `in` axis size `1` while preserving
an output `in=64` tile failed Deeptools checks:

```text
myInpPiece.second.dimToSize_.at(dimName) >=
  op->outLds->dimToStickSize_.at(dimName) - ...
```

So the extra consumer-stick axis must be present at full stick width in the
input piece. We cannot use a compact fake axis to satisfy the checker.

## Interpretation

The expanded descriptor is a promising compile contract but not yet a
value-correct lowering. The production path needs one more proof layer:

- Either show that the expanded `in` dimension is a legal alias of the source
  `out` dimension and does not cause extra reads, or
- Generate a row-slice bridge where each expanded input descriptor represents a
  value-correct subset of the source tile and the collection of row slices
  reconstructs the consumer input exactly.

Until that proof exists, the compiler must keep the stock `ReStickifyOpHBM`
fallback.

## Artifacts

- `artifacts/stage251_expanded_consumer_ptlx/expanded_consumer_row_slice.json`
- `artifacts/stage251_expanded_consumer_ptlx/expanded_consumer_row_slice_dcg.log`
- `artifacts/stage251_expanded_consumer_ptlx/mb_in_full.json`
- `artifacts/stage251_expanded_consumer_ptlx/mb_in_full_dcg.log`
- `artifacts/stage251_expanded_consumer_ptlx/input_in1_gap63_layout64_dcg.log`

## Next Step

The next implementation step should be a hardware-free semantic checker for
expanded descriptors:

1. Map producer logical coordinates to expanded input coordinates.
2. Map expanded output coordinates to the consumer input coordinates.
3. Assert that each consumer element has exactly one producer source element.
4. Reject descriptors that introduce extra live coordinates or duplicate reads.

Only after that checker passes should the expanded descriptor be wired into the
runtime PT-LX bridge selector.
