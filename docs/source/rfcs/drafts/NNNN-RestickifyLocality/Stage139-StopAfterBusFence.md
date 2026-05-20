# Stage 139: Stop After Consumer Baseline Bus Fence

## Summary

After Stage 138 showed that the patched consumer alone poisons the stream, I
tried one final baseline: compile the original unpatched consumer SDSC as a
single-SDSC bundle and launch it with HBM-backed inputs.

That attempt triggered a PCIe bus fence:

```text
RAS::PCI::BusFence
The card encountered a problem and triggered a PCIe bus fence operation
```

I stopped hardware experiments immediately after this. The device/runtime needs
a reset before more measurements are trustworthy.

## What This Means

Do not over-interpret the original-consumer single-SDSC result yet. A single
consumer extracted out of its original bundle may not be a valid standalone
runtime artifact without matching the original argument/layout/bundle contract.
The bus fence tells us the fixture is unsafe, not that normal consumer add is
bad.

The reliable conclusion remains Stage 138:

- producer-only check passed
- data-op-only-after-producer check passed
- patched consumer launch poisoned the stream

## Next Step After Reset

Resume with an offline/compile-only consumer metadata sweep before launching
anything:

1. Generate consumer variants.
2. Run `dxp_standalone` only.
3. Inspect generated program summaries for obvious HBM/LX/corelet/cardinality
   mismatches.
4. Launch only the variants that compile cleanly and preserve a sane runtime
   contract.

The first launched baseline after reset should be a known-good unmodified
Torch-Spyre generated bundle, not an extracted consumer, to confirm the device
is healthy again.
