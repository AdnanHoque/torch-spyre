# Stage 164: Consumer LX Contract Blocker

## Summary

The same-artifact LX bridge now generates and launches HBM-free bridge frames, but the value-correctness blocker moved to the consumer-side contract.  A diagnostic run showed the tuple output was effectively `a` alone, meaning the bridge contribution was not visible to the consumer.

The likely cause is that replacing only the `ReStickifyOpHBM` frame is insufficient: the following consumer frame still follows the original HBM input path for the restickified edge.  That can overwrite or bypass the LX bridge value.

## Results

- `STCDPOpLx -> ReStickifyOpLx` with an explicit source-view intermediate was semantically cleaner, but Deeptools rejected it because `STCDPOpLx` requires matching input/output stick dimensions.
- `ReStickifyOpLx -> STCDPOpLx -> ReStickifyOpLx` satisfies the `STCDPOpLx` same-stick rule and exports as an HBM-free frame.
- The three-op same-artifact bridge launched, but tuple-prefix correctness still failed.
- A permutation diagnostic compared the returned tuple value with several CPU candidates.  The returned tensor matched `a` with zero mismatches at tolerance `0.1`, confirming the consumer did not see the `(b + c).t()` bridge contribution.
- HBM-free bridge token counts for the three-op frame were:
  - `HBM=0`
  - `L3LU=96`
  - `L3SU=96`
  - `LXLU=64`
  - `LXSU=64`
  - `PT=0`
  - `SFP=0`

## Unsafe Attempt

I tried replacing the consumer frame with a standalone consumer frame compiled against an LX input.  That was not a valid fused-bundle contract: the standalone frame can disagree with the fused bundle's runtime metadata and argument/DCI mapping.

That run triggered a PCIe bus fence.  Do not use `SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_CONSUMER_FRAME=1` as a runtime path.  The code now requires an explicit unsafe override before that path can run.

## Safer Next Path

The safer prototype is:

1. Generate the LX bridge frame from the original producer/restickify/consumer descriptor.
2. Patch the consumer SDSC in place so the restickified input is LX-resident at the bridge output address.
3. Recompile the full fused bundle so all runtime metadata remains coherent.
4. Splice only the restickify frame with the HBM-free LX bridge frame.

This is implemented behind:

```sh
SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_CONSUMER_SDSC=1
```

Compile/splice-only validation passed on an existing generated code directory:

- consumer SDSC patched in place
- full fused bundle recompiled successfully
- restickify frame spliced with the HBM-free LX bridge
- patched artifact size: `32512` bytes, `254` 128-byte flits
- bridge tokens: `HBM=0`, `L3LU=96`, `L3SU=96`, `LXLU=64`, `LXSU=64`

It has not been hardware-validated yet because the card needs recovery after the bus fence.

## Next Validation

After a device reset/recovery:

1. Recover/reset the AIU after the bus fence.
2. Inspect the fused `loadprogram_to_device` frame sizes from the safe dry run and confirm only the restickify frame is replaced after full-bundle recompile.
3. Run the 2048 tuple-prefix correctness check with:

```sh
SPYRE_RESTICKIFY_LX_DATAOP_DIRECTION=restickify-stcdp-restickify
SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_CONSUMER_SDSC=1
```

Do not set `SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_CONSUMER_FRAME=1`.
