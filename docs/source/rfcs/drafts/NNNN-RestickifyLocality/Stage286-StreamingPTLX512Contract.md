# Stage 286: Streaming PT-LX 512 Contract Findings

## Question

Can the streaming tiled PT-LX prototype handle the non-2048 `computed_transpose_adds_then_matmul` shape at size 512 without falling back to `ReStickifyOpHBM`?

## Result

Not yet.  The run made progress from "candidate silently falls back" to "candidate can be forced into the emitted bundle", but the forced no-HBM bundles are not value-correct at size 512.

## What Changed

- Added force-only descriptor overrides for native tiled and validGap consumer tiled candidates so hardware validation can run the candidate instead of falling back during audit.
- Made the native tiled PT-LX descriptor record and honor the inferred logical direction (`kernel-to-output` vs `output-to-kernel`) at the stick-dimension level.
- Added focused unit coverage for:
  - native `output-to-kernel` stick selection;
  - force-validating the native descriptor despite its internal 4D tile descriptor;
  - force-validating the validGap consumer descriptor despite the temporary `in_` alias for the consumer `out_` axis.

These force paths are validation-only.  They are not production certificates.

## Evidence

Native 4D tiled path:

- Artifact: `/tmp/stage282-native-dir-nolaunch-512`
- Audit: `/tmp/stage282-native-dir-nolaunch-512-audit.jsonl`
- No-launch result: patched, no `ReStickifyOpHBM`, `coalescing=native-64x64-tiles`, `direction=output-to-kernel`.
- Hardware tuple result before the direction fix:
  - Artifact: `/tmp/stage281-native-force-tuple-hw-512`
  - Failed first tuple output with about 50% mismatched elements.
- Patterned-input hardware after the direction fix still showed scrambled tile coordinates:
  - expected `(b + c).t()`; actual first tile began with values from the wrong source rows and later tile columns were zero.

ValidGap consumer-shaped path:

- Artifact: `/tmp/stage284-validgap-force-nolaunch-512`
- Audit: `/tmp/stage284-validgap-force-nolaunch-512-audit.jsonl`
- No-launch result: patched, no `ReStickifyOpHBM`, `coalescing=validgap-consumer-64x64-tiles`, 128 data ops.
- Hardware tuple result:
  - Artifact: `/tmp/stage284-validgap-force-tuple-hw-512`
  - Failed first tuple output with about 42.7% mismatched elements.
- Patterned-input hardware showed the sparse `in_` alias is not a valid production interpretation of the consumer `out_` axis.

Consumer-destination planning experiment:

- Changing the streaming planner to target the consumer work split directly made Deeptools reject the descriptor:
  - `restickifyOp.cpp line 576`
  - `iPieceOrder.size()`
- This was backed out.

Device health after the failed descriptor experiment:

- A tiny stock Spyre tensor smoke completed successfully:
  - `torch.ones((16, 16), device="spyre") + ...`
  - CPU result contained `2.0`.

## Interpretation

The current blocker is no longer "can we emit and launch a no-HBM candidate?"  We can.

The blocker is the producer-bridge-consumer descriptor contract:

- The native 4D tile path uses a Deeptools-native restickify shape internally, but its final scatter descriptor is not the consumer-visible 2D layout.
- The validGap path tries to make the PT-LX op write a consumer-shaped output directly, but the temporary `in_` alias does not produce value-correct data on hardware.
- Planning the streaming destination directly against the consumer split is closer to the desired contract, but the current descriptor shape is not accepted by Deeptools.

## Next Step

The next production-shaped attempt should keep the accepted restickify-shaped transform and add an explicit consumer-compatible remap stage:

1. gather producer fragments into a bounded per-core tile workspace;
2. run `ReStickifyOpWithPTLx` in a Deeptools-accepted restickify-shaped tile descriptor;
3. run a separate `STCDPOpLx` remap whose output descriptor matches the real consumer input layout/stick names;
4. certify the producer input, bridge output, and consumer input as one value-flow contract before replacing `ReStickifyOpHBM`.

The stock HBM fallback must remain the default until that contract is value-correct across 512, 1024, 1536, and 2048.
