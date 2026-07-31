# Torch-Spyre FP8 activation-quantization reuse PoC

## Scope

This branch adds a post-grad FX pass for the repeated dynamic activation
quantization used by Granite Q/K/V and gate/up projections.  It recognizes the
decomposed activation path

```text
activation, scale -> reciprocal -> multiply -> clamp -> qfp8ch/qfp8mb
```

and common-subexpression-eliminates only that path and the allowlisted dynamic
scale ancestors (`amin`, `amax`, zero clipping, absolute maximum, division, and
scale clamping).  The upstream activation node is an identity boundary.  The
pass is not general graph CSE and explicitly excludes `qfp8wt` weight packing.

## Structural result

The focused graph test constructs three independently-emitted Granite-style
quantization chains from one activation.  The pass changes:

```text
qfp8mb: 3 -> 1
amin:   3 -> 1
amax:   3 -> 1
```

The returned quantized activation and scale nodes are shared by all three
consumers.  Negative tests retain separate chains for distinct activation
sources and distinct quantization parameters.  A decode-style `qfp8ch` pair is
also merged, while an otherwise-identical `qfp8wt` pair remains separate.

On `adnan-spyre-current-pf`, all four focused graph tests passed with the stack
activated by:

```bash
source /home/adnan/spyre-envs/image-913f394b4b3f/activate.sh
```

A controlled compiler integration probe reached generated Torch-Spyre source.
With the pass disabled at runtime, the source contained two `qfp8ch` calls; with
the pass enabled, it contained one `qfp8ch` node with two users.  Final
DeepTools compilation did not complete: the M=64 probe hit `QFP8MB codegen
requires exactly two logical dimensions`, and the M=1 probe hit `Expect
parameter value is multiple of stick size`.  Therefore this PoC has causal
structural compiler evidence, not a device timing or end-to-end Granite result.

## Exact limitations

1. The pass starts from decomposed `spyre.qfp8ch` or `spyre.qfp8mb`.  A frontend
   that leaves activation conversion as native `.to(float8)` must first lower
   that conversion to the Spyre QFP8 path.
2. It recognizes reciprocal-plus-multiply normalization.  A frontend that emits
   direct division needs one additional matcher arm.
3. The dynamic-scale operation allowlist is intentionally based on the checked
   Granite chain.  A different quantizer formulation will not be merged until
   its pure operations are reviewed and added.
4. This removes duplicate conversion and scale computation; it does not fuse
   scale computation, clamping, and packing into one backend operation.
5. One FX producer does not prove that its packed bytes remain in LX scratchpad
   for every consumer.  Final emitted bundles must still establish whether the
   shared value is retained on chip or written to and reloaded from HBM.
6. The PoC keeps the first canonical node's recovered Spyre hint metadata.  A
   production version must define how conflicting per-consumer hints are
   reconciled.

## Integration

The implementation is isolated on branch `ah/fp8-quant-reuse-poc`.  Its changes
are confined to:

```text
torch_spyre/_inductor/reuse_fp8_quantization.py
torch_spyre/_inductor/passes.py
tests/inductor/test_reuse_fp8_quantization.py
```

It can be rebased or cherry-picked after the scaled-matmul contract changes;
the only likely textual overlap is the import/pass-list edit in `passes.py`.
